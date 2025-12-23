"""
Single-Omics Analysis Module

This module provides:
- Robust gene matching across study tables
- Extraction of AUROC + logFC + direction
- A composite evidence score focused on the main task:
    (i) discriminative ability using AUC-disc = max(AUC, 1−AUC) (label-invariant),
    (ii) stability across studies (IQR of AUC-disc),
    (iii) direction agreement (MAFLD vs Healthy),
    (iv) a smooth study-count weight based on the number of valid AUROCs.
- Visualisations (lollipop AUROC plot, AUROC vs logFC concordance) and a results table

Designed to be imported by streamlit_app.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from robust_data_loader import load_single_omics_studies


# =============================================================================
# ROBUST GENE MATCHING
# =============================================================================

def _get_gene_lookup_cached(study_df: pd.DataFrame, gene_col: str) -> Dict[str, int]:
    """
    Build (and memoise) an exact-match lookup: lower(gene) -> first row index.
    Stored in study_df.attrs to avoid recomputing on every gene query.
    """
    if study_df is None or study_df.empty:
        return {}

    try:
        cached = study_df.attrs.get("_gene_lookup_cache", None)
        if isinstance(cached, dict) and cached.get("gene_col") == gene_col and isinstance(cached.get("lookup"), dict):
            return cached["lookup"]
    except Exception:
        cached = None

    ser = study_df[gene_col].astype(str).str.strip().str.lower()
    lookup: Dict[str, int] = {}
    # keep first occurrence if duplicates exist
    for i, g in enumerate(ser.tolist()):
        if g and g not in lookup:
            lookup[g] = i

    try:
        study_df.attrs["_gene_lookup_cache"] = {"gene_col": gene_col, "lookup": lookup}
    except Exception:
        pass

    return lookup


def find_gene_in_study(
    gene_name: str,
    study_df: pd.DataFrame,
    *,
    allow_substring: bool = True,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Find gene in study dataframe with robust matching.

    Exact match uses a cached lookup (fast).
    Substring match is optional and is expensive on large tables.
    Returns (matched_row, gene_col_name) or (None, None) if not found.
    """
    if study_df is None or study_df.empty:
        return None, None

    gene_col = None
    if "Gene" in study_df.columns:
        gene_col = "Gene"
    elif "gene" in study_df.columns:
        gene_col = "gene"
    else:
        return None, None

    gkey = str(gene_name).strip().lower()
    if not gkey:
        return None, None

    lookup = _get_gene_lookup_cached(study_df, gene_col)
    if gkey in lookup:
        try:
            return study_df.iloc[int(lookup[gkey])], gene_col
        except Exception:
            pass

    if allow_substring:
        # Fallback: expensive scan. Keep it for interactive gene explorer use-cases.
        col = study_df[gene_col].astype(str)
        sub_mask = col.str.contains(str(gene_name), case=False, na=False)
        sub_match = study_df[sub_mask]
        if not sub_match.empty:
            return sub_match.iloc[0], gene_col

    return None, None


def _to_float(x) -> Optional[float]:
    """Robust numeric conversion: returns None for NA / non-numeric strings."""
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def extract_metrics_from_row(row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Extract AUC, logFC, and direction (direction from explicit column if present, else infer from logFC).

    direction is:
      - "MAFLD" if the gene is higher in disease (logFC > 0)
      - "Healthy" if the gene is higher in control (logFC < 0)
    """
    if row is None or row.empty:
        return None, None, None

    auc = None
    for col in ["AUC", "auc", "AUC_score", "AUROC", "auroc", "roc_auc", "ROC_AUC"]:
        if col in row.index:
            auc = _to_float(row[col])
            if auc is not None:
                break

    lfc = None
    for col in ["avg_logFC", "avg_LFC", "logFC", "log2FC", "avg_log2FC"]:
        if col in row.index:
            lfc = _to_float(row[col])
            if lfc is not None:
                break

    direction = None
    if "direction" in row.index and pd.notna(row["direction"]):
        dir_val = str(row["direction"]).lower()
        if any(x in dir_val for x in ["nash", "nafld", "mafld", "mash"]):
            direction = "MAFLD"
        elif any(x in dir_val for x in ["healthy", "control", "chow", "ctrl"]):
            direction = "Healthy"
    else:
        if lfc is not None:
            direction = "MAFLD" if lfc > 0 else "Healthy"

    return auc, lfc, direction


def _build_score_help() -> Dict[str, str]:
    """One-sentence explanations surfaced in the Streamlit UI."""
    return {
        "Evidence Score": "Overall evidence across studies (geometric mean of Strength, Stability, Direction Factor, Study Weight).",
        "Direction Agreement": "Fraction of studies where the gene’s direction (MAFLD vs Healthy) matches the majority.",
        "Direction Factor": "Soft version of direction agreement (floored so direction noise does not zero-out the score).",
        "Median AUC (disc)": "Median discriminative AUC across studies: AUC-disc = max(AUC, 1−AUC).",
        "Studies Found": "Number of studies where the gene is present (even if AUROC is missing).",
        "Strength": "How far the median AUC-disc is above 0.5 (0=no signal; 1=perfect).",
        "Stability": "Cross-study consistency of AUC-disc (1=very consistent; 0=very variable; based on IQR).",
        "Study Weight": "Downweights scores supported by very few AUROC values (increases smoothly with n_auc).",
        "Valid AUROC (n_auc)": "Number of studies with a usable AUROC value for this gene.",
        "Median AUC (raw)": "Median of the raw AUROC values as stored in the study tables (diagnostic).",
        "Median AUC (oriented)": "Median AUROC after aligning direction so MAFLD is treated as ‘positive’ (diagnostic).",
        "AUC-disc IQR": "Interquartile range of AUC-disc across studies (lower = more stable).",
        "AUC-orient IQR": "Interquartile range of oriented AUROC across studies (diagnostic).",
    }


# =============================================================================
# CONSISTENCY / EVIDENCE SCORING
# =============================================================================

def compute_consistency_score(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None
) -> Optional[Dict[str, Any]]:
    """
    Composite evidence score for cross-study support.

    We compute:
      - AUC_raw: as reported in each study table (may be label-dependent)
      - AUC_oriented: aligned so MAFLD is treated as positive:
            if direction == MAFLD:   AUC_oriented = AUC_raw
            if direction == Healthy: AUC_oriented = 1 - AUC_raw
      - AUC_disc: max(AUC_raw, 1 - AUC_raw) (label-invariant discriminative ability)
      - Strength uses median(AUC_disc)
      - Stability uses IQR(AUC_disc)
      - Direction agreement is computed from direction labels (or inferred from logFC)
      - n_weight downweights small numbers of valid AUROCs

    Evidence Score is intentionally conservative but not punishing:
      - We use a soft Direction Factor (floored) rather than a hard gate
      - We combine components via a geometric mean to avoid score collapse
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    if not studies_data:
        return None

    auc_values: List[float] = []
    auc_oriented_vals: List[float] = []
    auc_disc_vals: List[float] = []
    directions: List[str] = []
    found_count = 0

    for _, df in studies_data.items():
        row, _ = find_gene_in_study(gene_name, df, allow_substring=False)
        if row is None:
            continue

        found_count += 1
        auc, lfc, direction = extract_metrics_from_row(row)

        if direction is not None:
            directions.append(direction)

        if auc is None:
            continue

        auc_raw = _to_float(auc)
        if auc_raw is None:
            continue
        if not (0.0 <= float(auc_raw) <= 1.0):
            continue

        auc_raw = float(auc_raw)
        auc_values.append(auc_raw)

        if direction == "MAFLD":
            auc_oriented = auc_raw
        elif direction == "Healthy":
            auc_oriented = float(1.0 - auc_raw)
        else:
            auc_oriented = auc_raw

        if 0.0 <= float(auc_oriented) <= 1.0:
            auc_oriented_vals.append(float(auc_oriented))

        auc_disc = float(max(auc_raw, 1.0 - auc_raw))
        if 0.0 <= auc_disc <= 1.0:
            auc_disc_vals.append(auc_disc)

    if found_count == 0:
        return None

    mafld_n = directions.count("MAFLD")
    healthy_n = directions.count("Healthy")
    total_dir = len(directions)

    if total_dir >= 2:
        direction_agreement = (max(mafld_n, healthy_n) / total_dir)
        if mafld_n > healthy_n:
            consensus_direction = "MAFLD"
        elif healthy_n > mafld_n:
            consensus_direction = "Healthy"
        else:
            consensus_direction = "Mixed"
    elif total_dir == 1:
        direction_agreement = 1.0
        consensus_direction = directions[0]
    else:
        direction_agreement = 1.0
        consensus_direction = "Unknown"

    n_auc = len(auc_values)

    median_auc_raw = float(np.median(auc_values)) if auc_values else 0.0
    median_auc_oriented = float(np.median(auc_oriented_vals)) if auc_oriented_vals else 0.5
    median_auc_disc = float(np.median(auc_disc_vals)) if auc_disc_vals else 0.5

    strength = max(0.0, (median_auc_disc - 0.5) / 0.5)

    disc_iqr = 0.0
    if len(auc_disc_vals) > 1:
        disc_iqr = float(np.subtract(*np.percentile(auc_disc_vals, [75, 25])))
        stability = max(0.0, 1.0 - (disc_iqr / 0.5))
    else:
        stability = 1.0

    orient_iqr = 0.0
    if len(auc_oriented_vals) > 1:
        orient_iqr = float(np.subtract(*np.percentile(auc_oriented_vals, [75, 25])))

    # Study-count weight: slightly less harsh than /3.0
    n_weight = float(1.0 - np.exp(-n_auc / 2.0))

    # Soft direction: floors the penalty so direction noise does not collapse the score
    direction_factor = float(0.75 + 0.25 * float(direction_agreement))  # in [0.75, 1.0]

    # Less harsh combination: geometric mean of components (still conservative)
    eps = 1e-12
    comp = np.array([strength, stability, direction_factor, n_weight], dtype=float)
    comp = np.clip(comp, 0.0, 1.0)

    if strength <= 0.0:
        evidence_score = 0.0
    else:
        evidence_score = float(np.exp(np.mean(np.log(comp + eps))))

    if strength > 0.7 and stability > 0.7 and direction_agreement > 0.7:
        interpretation = "Highly consistent: strong, stable, and directionally aligned"
    elif strength > 0.7 and stability > 0.7:
        interpretation = "Strong and stable, but mixed direction"
    elif strength > 0.7 and direction_agreement > 0.7:
        interpretation = "Strong and directionally aligned, but variable AUC"
    elif stability > 0.7 and direction_agreement > 0.7:
        interpretation = "Stable and directionally aligned, but weak signal"
    else:
        interpretation = "Weak or inconsistent evidence"

    return {
        "auc_values": auc_values,
        "auc_oriented_vals": auc_oriented_vals,
        "auc_disc_vals": auc_disc_vals,
        "auc_median": median_auc_raw,
        "auc_median_oriented": median_auc_oriented,
        "auc_median_discriminative": median_auc_disc,
        "disc_iqr": float(disc_iqr),
        "orient_iqr": float(orient_iqr),
        "strength": float(strength),
        "stability": float(stability),
        "direction_agreement": float(direction_agreement),
        "direction_factor": float(direction_factor),
        "direction_counts": {"MAFLD": int(mafld_n), "Healthy": int(healthy_n), "Total": int(total_dir)},
        "consensus_direction": consensus_direction,
        "evidence_score": float(evidence_score),
        "n_weight": float(n_weight),
        "found_count": int(found_count),
        "n_auc": int(n_auc),
        "interpretation": interpretation,
        "score_help": _build_score_help(),
    }


# =============================================================================
# PLOTS + TABLE
# =============================================================================

def create_lollipop_plot(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None,
    auc_mode: str = "oriented"  # "raw" or "oriented"
) -> Optional[go.Figure]:
    """
    Lollipop plot of AUROC per study.

    auc_mode:
      - "raw": show AUROC as reported
      - "oriented": show AUROC aligned so MAFLD is treated as positive
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    plot_data = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df, allow_substring=False)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)
        if auc is None:
            continue
        if not (0.0 <= float(auc) <= 1.0):
            continue

        auc_raw = float(auc)
        if direction == "MAFLD":
            auc_oriented = auc_raw
        elif direction == "Healthy":
            auc_oriented = float(1.0 - auc_raw)
        else:
            auc_oriented = auc_raw

        auc_plot = auc_oriented if auc_mode.lower() == "oriented" else auc_raw

        if direction == "MAFLD":
            symbol = "triangle-up"
            color = "#2E86AB"
        elif direction == "Healthy":
            symbol = "triangle-down"
            color = "#A23B72"
        else:
            symbol = "circle"
            color = "#999999"

        size = 10 + abs(lfc if lfc is not None else 0.0) * 1.5
        size = min(size, 16)

        plot_data.append({
            "study": study_name,
            "auc_raw": auc_raw,
            "auc_oriented": float(auc_oriented),
            "auc_plot": float(auc_plot),
            "lfc": float(lfc) if lfc is not None else np.nan,
            "direction": direction,
            "symbol": symbol,
            "color": color,
            "size": size
        })

    if not plot_data:
        return None

    plot_data = sorted(plot_data, key=lambda x: x["auc_plot"])

    fig = go.Figure()

    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[0.5, item["auc_plot"]],
            y=[item["study"], item["study"]],
            mode="lines",
            line=dict(color="#cccccc", width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ))

    for item in plot_data:
        lfc_txt = f"{item['lfc']:.3f}" if pd.notna(item["lfc"]) else "N/A"
        auc_label = "AUC (oriented)" if auc_mode.lower() == "oriented" else "AUC (raw)"
        auc_val = item["auc_plot"]

        hover = (
            f"<b>{item['study']}</b>"
            f"<br>AUC (raw): {item['auc_raw']:.3f}"
            f"<br>AUC (oriented): {item['auc_oriented']:.3f}"
            f"<br>{auc_label}: {auc_val:.3f}"
            f"<br>logFC: {lfc_txt}"
            f"<br>Direction: {item['direction']}"
        )

        fig.add_trace(go.Scatter(
            x=[item["auc_plot"]],
            y=[item["study"]],
            mode="markers",
            marker=dict(
                size=item["size"],
                symbol=item["symbol"],
                color=item["color"],
                line=dict(color="white", width=1),
            ),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False
        ))

    title = "AUROC Across Studies" if auc_mode.lower() == "raw" else "MAFLD-oriented AUROC Across Studies"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#000000")),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color="#000000")),
        height=300,
        hovermode="closest",
        xaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0"
        ),
        yaxis=dict(tickfont=dict(color="#000000", size=11)),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white"
    )

    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.6, line_dash="dot", line_color="#bbbbbb", line_width=1.0)
    fig.add_vline(x=0.7, line_dash="dot", line_color="#bbbbbb", line_width=1.0)
    fig.add_vline(x=0.8, line_dash="dot", line_color="#bbbbbb", line_width=1.0)

    return fig


def create_auc_logfc_scatter(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None,
    auc_mode: str = "oriented"  # "raw" or "oriented"
) -> Optional[go.Figure]:
    """
    Scatter of AUROC vs logFC across studies.

    auc_mode:
      - "raw": x = raw AUROC
      - "oriented": x = MAFLD-oriented AUROC
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    plot_data = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df, allow_substring=False)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)
        if auc is None or lfc is None:
            continue
        if not (0.0 <= float(auc) <= 1.0):
            continue

        auc_raw = float(auc)
        if direction == "MAFLD":
            auc_oriented = auc_raw
        elif direction == "Healthy":
            auc_oriented = float(1.0 - auc_raw)
        else:
            auc_oriented = auc_raw

        x_auc = auc_oriented if auc_mode.lower() == "oriented" else auc_raw

        if direction == "MAFLD":
            symbol = "triangle-up"
            color = "#2E86AB"
        elif direction == "Healthy":
            symbol = "triangle-down"
            color = "#A23B72"
        else:
            symbol = "circle"
            color = "#999999"

        plot_data.append({
            "study": study_name,
            "auc_raw": auc_raw,
            "auc_oriented": float(auc_oriented),
            "auc_plot": float(x_auc),
            "lfc": float(lfc),
            "direction": direction,
            "symbol": symbol,
            "color": color
        })

    if len(plot_data) < 2:
        return None

    fig = go.Figure()

    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[item["auc_plot"]],
            y=[item["lfc"]],
            mode="markers",
            marker=dict(size=10, color=item["color"], symbol=item["symbol"], line=dict(width=0)),
            hovertext=(
                f"<b>{item['study']}</b>"
                f"<br>AUC (raw): {item['auc_raw']:.3f}"
                f"<br>AUC (oriented): {item['auc_oriented']:.3f}"
                f"<br>logFC: {item['lfc']:.3f}"
                f"<br>Direction: {item['direction']}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)

    xlab = "AUROC" if auc_mode.lower() == "raw" else "MAFLD-oriented AUROC"
    title = "Concordance: AUC vs logFC" if auc_mode.lower() == "raw" else "Concordance: Oriented AUROC vs logFC"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#000000")),
        xaxis_title=dict(text=xlab, font=dict(size=12, color="#000000")),
        yaxis_title=dict(text="logFC (MAFLD vs Healthy)", font=dict(size=12, color="#000000")),
        height=350,
        hovermode="closest",
        xaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0"
        ),
        yaxis=dict(
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0"
        ),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white"
    )

    return fig


def create_results_table(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None
) -> Optional[pd.DataFrame]:
    """
    Per-study table including:
      - AUC_raw, AUC_oriented, AUC_disc
      - logFC and direction
      - Used_in_score + reason (debug + trust)
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    results = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df, allow_substring=False)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)

        used = True
        reason = ""

        auc_raw = _to_float(auc) if auc is not None else None
        auc_oriented = None
        auc_disc = None

        if auc_raw is None:
            used = False
            reason = "Missing AUROC"
        elif not (0.0 <= float(auc_raw) <= 1.0):
            used = False
            reason = "AUROC out of range"
        else:
            auc_raw = float(auc_raw)
            if direction == "MAFLD":
                auc_oriented = float(auc_raw)
            elif direction == "Healthy":
                auc_oriented = float(1.0 - auc_raw)
            else:
                auc_oriented = float(auc_raw)

            auc_disc = float(max(auc_raw, 1.0 - auc_raw))

        results.append({
            "Study": study_name,
            "AUC_raw": f"{auc_raw:.3f}" if auc_raw is not None else "N/A",
            "AUC_oriented": f"{auc_oriented:.3f}" if auc_oriented is not None else "N/A",
            "AUC_disc": f"{auc_disc:.3f}" if auc_disc is not None else "N/A",
            "logFC": f"{float(lfc):.3f}" if lfc is not None else "N/A",
            "Direction": direction if direction else "Unknown",
            "Used_in_score": "Yes" if used else "No",
            "Exclusion_reason": reason
        })

    if not results:
        return None

    return pd.DataFrame(results)
