"""
Single-Omics Analysis Module

This module provides:
- Robust gene matching across study tables
- Extraction of AUROC + logFC + direction
- A composite evidence score that reflects:
    (i) discriminative power (orientation-invariant AUROC),
    (ii) stability (IQR of discriminative AUROC),
    (iii) direction agreement,
    (iv) number of studies with valid AUROC
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

def find_gene_in_study(gene_name: str, study_df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Find gene in study dataframe with robust matching.
    Returns (matched_row, gene_col_name) or (None, None) if not found.
    Exact match preferred over substring match.
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

    col = study_df[gene_col].astype(str)

    exact_mask = col.str.lower() == str(gene_name).lower()
    exact_match = study_df[exact_mask]
    if not exact_match.empty:
        return exact_match.iloc[0], gene_col

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
    """
    One-sentence explanations surfaced in the Streamlit UI.
    Keep these short and non-technical.
    """
    return {
        "Evidence Score": "Overall evidence (0–100%) combining discriminative AUROC, cross-study stability, direction agreement, and how many studies have valid AUROC.",
        "Direction Agreement": "How consistently the gene is up in MAFLD vs up in Healthy across studies (1.0 means all studies agree).",
        "Median AUC (disc)": "Median discriminative AUROC across studies, where AUC < 0.5 is treated as label-flipped signal via max(AUC, 1−AUC).",
        "Studies Found": "How many studies contained the gene (even if AUROC is missing).",
        "Strength": "Scaled discriminative performance: 0 means no better than random (AUC-disc=0.5), 1 means perfect separation (AUC-disc=1.0).",
        "Stability": "Cross-study consistency of AUC-disc: 1 means very consistent, lower values mean high between-study spread (IQR).",
        "Study Weight": "More studies with valid AUROC increase confidence; this is a smooth weight that grows with n_auc.",
        "Valid AUROC (n_auc)": "Number of studies contributing valid AUROC to the score.",
        "Median AUC (raw)": "Median AUROC as reported in the study tables (can be <0.5 if the labels are flipped).",
        "Median AUC (oriented)": "Median AUROC after orienting so higher values correspond to MAFLD as the ‘positive’ side (for diagnostics).",
        "disc_full": "Reference ‘good signal’ line: AUC-disc=0.65 corresponds to a moderate discriminative signal.",
    }


# =============================================================================
# CONSISTENCY / EVIDENCE SCORING (DISCRIMINATIVE AUC)
# =============================================================================

def compute_consistency_score(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None
) -> Optional[Dict[str, Any]]:
    """
    Composite evidence score.

    Strength is based on discriminative AUROC: max(AUC, 1-AUC)
    so AUROC < 0.5 counts as predictive (but label-flipped).

    Note: "auc_oriented_vals" is kept for diagnostics only; it is NOT used
    to compute the discriminative AUROC.
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
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        found_count += 1
        auc, lfc, direction = extract_metrics_from_row(row)

        if direction is not None:
            directions.append(direction)

        if auc is None:
            continue

        if not (0.0 <= auc <= 1.0):
            continue

        auc_values.append(float(auc))

        # Discriminative AUC (orientation-invariant) MUST be from raw auc
        auc_disc_vals.append(float(max(auc, 1.0 - auc)))

        # Oriented AUC (diagnostic): align towards MAFLD as "positive"
        if direction == "MAFLD":
            auc_oriented = float(auc)
        elif direction == "Healthy":
            auc_oriented = float(1.0 - auc)
        else:
            auc_oriented = float(auc)

        if 0.0 <= auc_oriented <= 1.0:
            auc_oriented_vals.append(float(auc_oriented))

    if found_count == 0:
        return None

    mafld_n = directions.count("MAFLD")
    healthy_n = directions.count("Healthy")
    total_dir = len(directions)

    direction_agreement = (max(mafld_n, healthy_n) / total_dir) if total_dir else 0.0
    consensus_direction = None
    if total_dir:
        if mafld_n > healthy_n:
            consensus_direction = "MAFLD"
        elif healthy_n > mafld_n:
            consensus_direction = "Healthy"
        else:
            consensus_direction = "Mixed"

    n_auc = len(auc_disc_vals)

    median_auc_discriminative = float(np.median(auc_disc_vals)) if auc_disc_vals else 0.5
    median_auc_raw = float(np.median(auc_values)) if auc_values else 0.0
    median_auc_oriented = float(np.median(auc_oriented_vals)) if auc_oriented_vals else 0.5

    # Strength: 0.5 -> 0, 1.0 -> 1
    strength = max(0.0, (median_auc_discriminative - 0.5) / 0.5)

    # Stability: 1 - IQR/0.5 on discriminative AUC
    disc_iqr = 0.0
    if len(auc_disc_vals) > 1:
        disc_iqr = float(np.subtract(*np.percentile(auc_disc_vals, [75, 25])))
        stability = max(0.0, 1.0 - (disc_iqr / 0.5))
    else:
        stability = 1.0

    # Weight by number of valid AUCs
    n_weight = float(1.0 - np.exp(-n_auc / 3.0))

    evidence_score = float(strength * stability * float(direction_agreement) * n_weight)

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
        "auc_median_discriminative": median_auc_discriminative,
        "disc_iqr": float(disc_iqr),
        "strength": float(strength),
        "stability": float(stability),
        "direction_agreement": float(direction_agreement),
        "direction_counts": {"MAFLD": int(mafld_n), "Healthy": int(healthy_n), "Total": int(total_dir)},
        "consensus_direction": consensus_direction,
        "evidence_score": float(evidence_score),
        "n_weight": float(n_weight),
        "found_count": int(found_count),
        "n_auc": int(n_auc),
        "interpretation": interpretation,
        "disc_full": 0.65,
        "score_help": _build_score_help(),
    }


# =============================================================================
# PLOTS + TABLE
# =============================================================================

def create_lollipop_plot(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None,
    auc_mode: str = "raw"  # "raw" or "disc"
) -> Optional[go.Figure]:
    """
    Lollipop plot of AUROC per study.

    auc_mode:
      - "raw": show AUROC as reported
      - "disc": show discriminative AUROC = max(AUC, 1-AUC)
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    plot_data = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)
        if auc is None:
            continue
        if not (0.0 <= float(auc) <= 1.0):
            continue

        auc_raw = float(auc)
        auc_disc = float(max(auc_raw, 1.0 - auc_raw))
        auc_plot = auc_disc if auc_mode.lower() == "disc" else auc_raw

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
            "auc_disc": auc_disc,
            "auc_plot": auc_plot,
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
            hovertext=(
                f"<b>{item['study']}</b>"
                f"<br>AUC (raw): {item['auc_raw']:.3f}"
                f"<br>AUC (disc): {item['auc_disc']:.3f}"
                f"<br>logFC: {item['lfc']:.3f}" if pd.notna(item["lfc"]) else f"<br>logFC: N/A"
            ) + (
                f"<br>Direction: {item['direction']}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    title = "AUROC Across Studies" if auc_mode.lower() == "raw" else "Discriminative AUROC Across Studies (max(AUC, 1−AUC))"

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

    # Guides
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.6, line_dash="dot", line_color="#bbbbbb", line_width=1.0)
    fig.add_vline(x=0.7, line_dash="dot", line_color="#bbbbbb", line_width=1.0)
    fig.add_vline(x=0.8, line_dash="dot", line_color="#bbbbbb", line_width=1.0)

    return fig


def create_auc_logfc_scatter(
    gene_name: str,
    studies_data: Optional[Dict[str, pd.DataFrame]] = None,
    auc_mode: str = "raw"  # "raw" or "disc"
) -> Optional[go.Figure]:
    """
    Scatter of AUROC vs logFC across studies.

    auc_mode:
      - "raw": x = raw AUROC
      - "disc": x = discriminative AUROC = max(AUC, 1-AUC)
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    plot_data = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)
        if auc is None or lfc is None:
            continue
        if not (0.0 <= float(auc) <= 1.0):
            continue

        auc_raw = float(auc)
        auc_disc = float(max(auc_raw, 1.0 - auc_raw))
        x_auc = auc_disc if auc_mode.lower() == "disc" else auc_raw

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
            "auc_disc": auc_disc,
            "auc_plot": x_auc,
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
                f"<br>AUC (disc): {item['auc_disc']:.3f}"
                f"<br>logFC: {item['lfc']:.3f}"
                f"<br>Direction: {item['direction']}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)

    xlab = "AUROC" if auc_mode.lower() == "raw" else "Discriminative AUROC (max(AUC, 1−AUC))"
    title = "Concordance: AUC vs logFC" if auc_mode.lower() == "raw" else "Concordance: AUC-disc vs logFC"

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


def create_results_table(gene_name: str, studies_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[pd.DataFrame]:
    """
    Per-study table including:
      - AUC_raw, AUC_disc, AUC_oriented
      - logFC and direction
      - Used_in_score + reason (debug + trust)
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    results = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)

        used = True
        reason = ""

        auc_raw = None
        auc_disc = None
        auc_oriented = None

        if auc is None:
            used = False
            reason = "Missing AUROC"
        else:
            auc_raw = float(auc) if auc is not None else None
            if auc_raw is None or not (0.0 <= auc_raw <= 1.0):
                used = False
                reason = "AUROC out of range"
            else:
                auc_disc = float(max(auc_raw, 1.0 - auc_raw))
                if direction == "MAFLD":
                    auc_oriented = float(auc_raw)
                elif direction == "Healthy":
                    auc_oriented = float(1.0 - auc_raw)
                else:
                    auc_oriented = float(auc_raw)

        results.append({
            "Study": study_name,
            "AUC_raw": f"{auc_raw:.3f}" if auc_raw is not None else "N/A",
            "AUC_disc": f"{auc_disc:.3f}" if auc_disc is not None else "N/A",
            "AUC_oriented": f"{auc_oriented:.3f}" if auc_oriented is not None else "N/A",
            "logFC": f"{float(lfc):.3f}" if lfc is not None else "N/A",
            "Direction": direction if direction else "Unknown",
            "Used_in_score": "Yes" if used else "No",
            "Exclusion_reason": reason
        })

    if not results:
        return None

    return pd.DataFrame(results)
