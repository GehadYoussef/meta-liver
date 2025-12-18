"""
Single-Omics Analysis Module

This module provides:
- Robust gene matching across study tables
- Extraction of AUROC + logFC + direction
- A composite evidence score that reflects (i) discriminative power, (ii) stability,
  (iii) direction agreement, and (iv) the number of studies with valid AUROC
- Visualisations (lollipop AUROC plot, AUROC vs logFC concordance) and a results table

It is designed to be imported by streamlit_app.py so the single-omics logic lives in one place.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from robust_data_loader import load_single_omics_studies


# ============================================================================
# ROBUST GENE MATCHING
# ============================================================================

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

    # Exact match first (case-insensitive)
    exact_mask = col.str.lower() == str(gene_name).lower()
    exact_match = study_df[exact_mask]
    if not exact_match.empty:
        return exact_match.iloc[0], gene_col

    # Substring fallback
    sub_mask = col.str.contains(str(gene_name), case=False, na=False)
    sub_match = study_df[sub_mask]
    if not sub_match.empty:
        return sub_match.iloc[0], gene_col

    return None, None


def extract_metrics_from_row(row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Extract AUC, logFC, and direction from a row (direction from explicit column if present, else infer from logFC)."""
    if row is None or row.empty:
        return None, None, None

    # AUC
    auc = None
    for col in ["AUC", "auc", "AUC_score"]:
        if col in row.index:
            try:
                auc = float(row[col])
                break
            except Exception:
                pass

    # logFC
    lfc = None
    for col in ["avg_logFC", "avg_LFC", "logFC", "log2FC", "avg_log2FC"]:
        if col in row.index:
            try:
                lfc = float(row[col])
                break
            except Exception:
                pass

    # Direction
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


# ============================================================================
# CONSISTENCY / EVIDENCE SCORING
# ============================================================================

def compute_consistency_score(gene_name: str, studies_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[Dict[str, Any]]:
    """
    Composite evidence score.

    Key design choices:
    - Direction agreement is based on inferred/declared direction per study.
    - Strength is based on *discriminative AUROC* (max(AUC, 1-AUC)) so that AUROC < 0.5
      is treated as 'still predictive but flipped label orientation'.
    - Stability is based on the IQR of discriminative AUROC values.
    - n_weight depends on number of studies with a valid AUROC (not just gene presence).
    """
    if studies_data is None:
        studies_data = load_single_omics_studies()

    if not studies_data:
        return None

    auc_values: List[float] = []
    auc_oriented_vals: List[float] = []
    directions: List[str] = []
    found_count = 0

    for _, df in studies_data.items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        found_count += 1
        auc, lfc, direction = extract_metrics_from_row(row)

        if auc is not None and not np.isnan(auc):
            auc_values.append(float(auc))

            # Oriented AUC (kept for debugging/plotting; not used for strength after patch)
            if direction == "MAFLD":
                auc_oriented = float(auc)
            elif direction == "Healthy":
                auc_oriented = 1.0 - float(auc)
            else:
                auc_oriented = float(auc)

            auc_oriented_vals.append(float(auc_oriented))

        if direction is not None:
            directions.append(direction)

    if found_count == 0:
        return None

    # Direction agreement (majority fraction)
    direction_agreement = (
        max(directions.count("MAFLD"), directions.count("Healthy")) / len(directions)
        if directions else 0.0
    )

    # Patch: strength/stability computed on discriminative AUROC (orientation-invariant)
    n_auc = len(auc_oriented_vals)
    auc_disc_vals = [max(a, 1.0 - a) for a in auc_oriented_vals]
    median_auc_discriminative = float(np.median(auc_disc_vals)) if auc_disc_vals else 0.5
    median_auc_raw = float(np.median(auc_values)) if auc_values else 0.0
    median_auc_oriented = float(np.median(auc_oriented_vals)) if auc_oriented_vals else 0.5

    # Strength: map 0.5->0, 1.0->1
    strength = max(0.0, (median_auc_discriminative - 0.5) / 0.5)

    # Stability: 1 - IQR/0.5 on discriminative AUROC
    if len(auc_disc_vals) > 1:
        iqr = float(np.subtract(*np.percentile(auc_disc_vals, [75, 25])))
        stability = max(0.0, 1.0 - (iqr / 0.5))
    else:
        stability = 1.0

    # Sample weight: use valid AUC count
    n_weight = float(1.0 - np.exp(-n_auc / 3.0))

    evidence_score = float(strength * stability * direction_agreement * n_weight)

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
        "auc_median": median_auc_raw,
        "auc_median_oriented": median_auc_oriented,
        "auc_median_discriminative": median_auc_discriminative,
        "strength": strength,
        "stability": stability,
        "direction_agreement": float(direction_agreement),
        "evidence_score": evidence_score,
        "n_weight": n_weight,
        "found_count": int(found_count),
        "n_auc": int(n_auc),
        "interpretation": interpretation,
    }


# ============================================================================
# PLOTS + TABLE
# ============================================================================

def create_lollipop_plot(gene_name: str, studies_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[go.Figure]:
    """Create horizontal lollipop plot with direction cues (triangle markers)."""
    if studies_data is None:
        studies_data = load_single_omics_studies()

    plot_data = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)
        if auc is None or np.isnan(auc):
            continue

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
            "auc": float(auc),
            "lfc": float(lfc) if lfc is not None else 0.0,
            "direction": direction,
            "symbol": symbol,
            "color": color,
            "size": size
        })

    if not plot_data:
        return None

    plot_data = sorted(plot_data, key=lambda x: x["auc"])

    fig = go.Figure()

    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[0.5, item["auc"]],
            y=[item["study"], item["study"]],
            mode="lines",
            line=dict(color="#cccccc", width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ))

    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[item["auc"]],
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
                f"<br>AUC: {item['auc']:.3f}"
                f"<br>logFC: {item['lfc']:.3f}"
                f"<br>Direction: {item['direction']}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text="AUROC Across Studies", font=dict(size=14, color="#000000")),
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
        yaxis=dict(
            tickfont=dict(color="#000000", size=11)
        ),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white"
    )

    return fig


def create_auc_logfc_scatter(gene_name: str, studies_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[go.Figure]:
    """Create AUROC vs logFC scatter plot with direction symbols."""
    if studies_data is None:
        studies_data = load_single_omics_studies()

    plot_data = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)
        if auc is None or lfc is None or np.isnan(auc) or np.isnan(lfc):
            continue

        if direction == "MAFLD":
            symbol = "triangle-up"
        elif direction == "Healthy":
            symbol = "triangle-down"
        else:
            symbol = "circle"

        plot_data.append({
            "study": study_name,
            "auc": float(auc),
            "lfc": float(lfc),
            "direction": direction,
            "symbol": symbol
        })

    if len(plot_data) < 2:
        return None

    fig = go.Figure()

    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[item["auc"]],
            y=[item["lfc"]],
            mode="markers",
            marker=dict(
                size=10,
                color="#333333",
                symbol=item["symbol"],
                line=dict(width=0),
            ),
            hovertext=(
                f"<b>{item['study']}</b>"
                f"<br>AUC: {item['auc']:.3f}"
                f"<br>logFC: {item['lfc']:.3f}"
                f"<br>Direction: {item['direction']}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)

    fig.update_layout(
        title=dict(text="Concordance: AUC vs logFC", font=dict(size=14, color="#000000")),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color="#000000")),
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
    """Create per-study AUROC/logFC/direction table for the selected gene."""
    if studies_data is None:
        studies_data = load_single_omics_studies()

    results = []

    for study_name, df in (studies_data or {}).items():
        row, _ = find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = extract_metrics_from_row(row)

        results.append({
            "Study": study_name,
            "AUC": f"{auc:.3f}" if (auc is not None and not np.isnan(auc)) else "N/A",
            "logFC": f"{lfc:.3f}" if (lfc is not None and not np.isnan(lfc)) else "N/A",
            "Direction": direction if direction else "Unknown"
        })

    if not results:
        return None

    return pd.DataFrame(results)
