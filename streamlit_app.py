"""
Meta Liver - Interactive Streamlit App for Liver Genomics Analysis
Three-tab interface: Single-Omics Evidence | MAFLD Knowledge Graph | Co-expression and PPI Networks
"""

import sys
import importlib
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from robust_data_loader import load_single_omics_studies, load_kg_data, load_ppi_data
from kg_analysis import (
    get_gene_kg_info, get_cluster_genes, get_cluster_drugs, get_cluster_diseases, interpret_centrality
)
from wgcna_ppi_analysis import (
    load_wgcna_module_data, get_gene_module, get_module_genes,
    get_coexpressed_partners, find_ppi_interactors, get_network_stats,
    load_wcgna_mod_trait_cor, load_wcgna_mod_trait_pval, load_wcgna_pathways
)

# -----------------------------------------------------------------------------
# IMPORTANT: always import the *local* single_omics_analysis.py from this app folder
# and always call its functions (no duplicated scoring logic in this file).
# This prevents stale/other-module versions silently "winning".
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import single_omics_analysis as soa
soa = importlib.reload(soa)


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_all_data():
    single_omics = load_single_omics_studies()
    kg_data = load_kg_data()
    wgcna_module_data = load_wgcna_module_data()
    ppi_data = load_ppi_data()

    wcgna_cor = load_wcgna_mod_trait_cor()
    wcgna_pval = load_wcgna_mod_trait_pval()
    wcgna_pathways = load_wcgna_pathways()

    return single_omics, kg_data, wgcna_module_data, ppi_data, wcgna_cor, wcgna_pval, wcgna_pathways


try:
    single_omics_data, kg_data, wgcna_module_data, ppi_data, wcgna_cor, wcgna_pval, wcgna_pathways = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    single_omics_data = {}
    kg_data = {}
    wgcna_module_data = {}
    ppi_data = {}
    wcgna_cor = pd.DataFrame()
    wcgna_pval = pd.DataFrame()
    wcgna_pathways = {}


# =============================================================================
# PLOT HELPERS (RAW vs DISC) — NO NEED TO MODIFY single_omics_analysis.py
# =============================================================================

def _collect_gene_metrics(gene_name: str, studies_data: dict) -> list[dict]:
    """
    Returns per-study dicts: study, auc_raw, auc_disc, auc_oriented, lfc, direction
    Uses soa.find_gene_in_study + soa.extract_metrics_from_row for consistency.
    """
    out = []
    for study_name, df in (studies_data or {}).items():
        row, _ = soa.find_gene_in_study(gene_name, df)
        if row is None:
            continue

        auc, lfc, direction = soa.extract_metrics_from_row(row)

        auc_raw = None
        if auc is not None and not np.isnan(auc) and 0.0 <= float(auc) <= 1.0:
            auc_raw = float(auc)

        lfc_val = None
        if lfc is not None and not np.isnan(lfc):
            lfc_val = float(lfc)

        # Discriminative (orientation-invariant)
        auc_disc = None
        if auc_raw is not None:
            auc_disc = float(max(auc_raw, 1.0 - auc_raw))

        # Oriented (align MAFLD as "positive") — diagnostic only
        auc_oriented = None
        if auc_raw is not None:
            if direction == "Healthy":
                auc_oriented = float(1.0 - auc_raw)
            else:
                auc_oriented = float(auc_raw)

        out.append({
            "study": study_name,
            "auc_raw": auc_raw,
            "auc_disc": auc_disc,
            "auc_oriented": auc_oriented,
            "lfc": lfc_val,
            "direction": direction
        })

    return out


def _marker_style(direction: str):
    if direction == "MAFLD":
        return dict(symbol="triangle-up", color="#2E86AB")
    if direction == "Healthy":
        return dict(symbol="triangle-down", color="#A23B72")
    return dict(symbol="circle", color="#777777")


def make_lollipop(metrics: list[dict], auc_key: str, title: str, subtitle: str | None = None) -> go.Figure | None:
    vals = [m for m in metrics if m.get(auc_key) is not None]
    if not vals:
        return None

    vals = sorted(vals, key=lambda x: x[auc_key])

    fig = go.Figure()

    for m in vals:
        fig.add_trace(go.Scatter(
            x=[0.5, m[auc_key]],
            y=[m["study"], m["study"]],
            mode="lines",
            line=dict(color="#cccccc", width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ))

    for m in vals:
        style = _marker_style(m.get("direction"))
        lfc = m.get("lfc")
        size = 10 + abs(lfc if lfc is not None else 0.0) * 1.5
        size = float(min(size, 16))

        hover = f"<b>{m['study']}</b><br>{auc_key}: {m[auc_key]:.3f}"
        if m.get("auc_raw") is not None:
            hover += f"<br>AUC_raw: {m['auc_raw']:.3f}"
        if m.get("auc_disc") is not None:
            hover += f"<br>AUC_disc: {m['auc_disc']:.3f}"
        if m.get("lfc") is not None:
            hover += f"<br>logFC: {m['lfc']:.3f}"
        if m.get("direction") is not None:
            hover += f"<br>Direction: {m['direction']}"

        fig.add_trace(go.Scatter(
            x=[m[auc_key]],
            y=[m["study"]],
            mode="markers",
            marker=dict(
                size=size,
                symbol=style["symbol"],
                color=style["color"],
                line=dict(color="white", width=1),
            ),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False
        ))

    title_text = title if subtitle is None else f"{title}<br><span style='font-size:11px;color:#666'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#000000")),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color="#000000")),
        height=320,
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
    return fig


def make_scatter_auc_logfc(metrics: list[dict], auc_key: str, title: str, subtitle: str | None = None) -> go.Figure | None:
    pts = [m for m in metrics if m.get(auc_key) is not None and m.get("lfc") is not None]
    if len(pts) < 2:
        return None

    fig = go.Figure()
    for m in pts:
        style = _marker_style(m.get("direction"))
        fig.add_trace(go.Scatter(
            x=[m[auc_key]],
            y=[m["lfc"]],
            mode="markers",
            marker=dict(size=10, symbol=style["symbol"], color="#333333", line=dict(width=0)),
            hovertext=(
                f"<b>{m['study']}</b>"
                f"<br>{auc_key}: {m[auc_key]:.3f}"
                f"<br>logFC: {m['lfc']:.3f}"
                f"<br>Direction: {m.get('direction', 'Unknown')}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)

    title_text = title if subtitle is None else f"{title}<br><span style='font-size:11px;color:#666'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#000000")),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color="#000000")),
        yaxis_title=dict(text="logFC (MAFLD vs Healthy)", font=dict(size=12, color="#000000")),
        height=340,
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


def make_auc_disc_distribution(metrics: list[dict]) -> go.Figure | None:
    vals = [m["auc_disc"] for m in metrics if m.get("auc_disc") is not None]
    if not vals:
        return None

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=vals,
        boxpoints="all",
        jitter=0.25,
        pointpos=0,
        name="AUC-disc",
        marker=dict(size=8),
        line=dict(width=1)
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.update_layout(
        title=dict(text="AUC-disc distribution (stability view)", font=dict(size=14, color="#000000")),
        yaxis_title=dict(text="AUC-disc = max(AUC, 1−AUC)", font=dict(size=12, color="#000000")),
        height=320,
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        yaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0"
        ),
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000", size=11))
    )
    return fig


# =============================================================================
# KG DISPLAY HELPERS (UI ONLY)
# =============================================================================

def _is_nanlike(x: object) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except Exception:
        return x is None


def _fmt_pct(p: object) -> str:
    if _is_nanlike(p):
        return "N/A"
    try:
        return f"{float(p):.1f}%"
    except Exception:
        return "N/A"


def _fmt_num(x: object, decimals: int = 4, sci_if_small: bool = False) -> str:
    if _is_nanlike(x):
        return "N/A"
    try:
        v = float(x)
        if sci_if_small and 0 < abs(v) < 10 ** (-(decimals)):
            return f"<{10 ** (-(decimals)):.0e}"
        return f"{v:.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_num_commas(x: object, decimals: int = 2) -> str:
    if _is_nanlike(x):
        return "N/A"
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return "N/A"


def _pct_str_to_float(s: object) -> float:
    if s is None:
        return np.nan
    try:
        t = str(s).strip().replace("%", "")
        return float(t)
    except Exception:
        return np.nan


def _num_str_to_float(s: object) -> float:
    if s is None:
        return np.nan
    try:
        t = str(s).strip().replace(",", "")
        return float(t)
    except Exception:
        return np.nan


def _prepare_cluster_table(df: pd.DataFrame, sort_key: str, top_n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    tmp = df.copy()

    pct_cols = ["Composite %ile", "PR %ile", "Bet %ile", "Eigen %ile"]
    num_cols = ["PageRank", "Betweenness", "Eigen"]
    for c in pct_cols:
        if c in tmp.columns:
            tmp[f"__{c}_num__"] = tmp[c].map(_pct_str_to_float)
    for c in num_cols:
        if c in tmp.columns:
            tmp[f"__{c}_num__"] = tmp[c].map(_num_str_to_float)

    sort_map = {
        "Composite %ile": "__Composite %ile_num__",
        "PR %ile": "__PR %ile_num__",
        "Bet %ile": "__Bet %ile_num__",
        "Eigen %ile": "__Eigen %ile_num__",
        "PageRank (raw)": "__PageRank_num__",
        "Betweenness (raw)": "__Betweenness_num__",
        "Eigen (raw)": "__Eigen_num__",
        "Name": "Name",
    }

    col = sort_map.get(sort_key, sort_key)
    if col in tmp.columns:
        asc = True if sort_key == "Name" else False
        tmp = tmp.sort_values(col, ascending=asc, na_position="last")

    tmp = tmp.head(int(top_n)).copy()
    tmp = tmp[[c for c in tmp.columns if not c.startswith("__")]]
    return tmp


# =============================================================================
# WGCNA DISPLAY HELPERS (MODULE–TRAIT + PATHWAYS)
# =============================================================================

def _match_module_row(df: pd.DataFrame, module: str) -> str | None:
    if df is None or df.empty:
        return None
    m = str(module).strip()
    candidates = [m, m.lower(), m.upper(), f"ME{m}", f"ME{m.lower()}", f"ME{m.upper()}"]
    for c in candidates:
        if c in df.index:
            return c
    idx_lower = {str(i).lower(): i for i in df.index}
    for c in candidates:
        key = str(c).lower()
        if key in idx_lower:
            return idx_lower[key]
    return None


def _module_trait_table(module_name: str, cor_df: pd.DataFrame, pval_df: pd.DataFrame) -> pd.DataFrame | None:
    if cor_df is None or cor_df.empty:
        return None

    row_key = _match_module_row(cor_df, module_name)
    if row_key is None:
        return None

    cor_row = cor_df.loc[row_key].copy()
    out = cor_row.reset_index()
    out.columns = ["Trait", "Correlation"]

    if pval_df is not None and not pval_df.empty:
        p_key = _match_module_row(pval_df, module_name)
        if p_key is not None:
            p_row = pval_df.loc[p_key].reset_index()
            p_row.columns = ["Trait", "P-value"]
            out = out.merge(p_row, on="Trait", how="left")

    out["__abs_corr__"] = pd.to_numeric(out["Correlation"], errors="coerce").abs()
    out = out.sort_values("__abs_corr__", ascending=False).drop(columns=["__abs_corr__"])
    return out


# =============================================================================
# MAIN APP
# =============================================================================

st.sidebar.markdown("## 🔬 Meta Liver")
search_query = st.sidebar.text_input("Search gene:", placeholder="e.g., SAA1, TP53, IL6").strip().upper()

st.sidebar.caption(f"single_omics_analysis loaded from: {getattr(soa, '__file__', 'unknown')}")

if data_loaded:
    if single_omics_data:
        st.sidebar.success(f"✓ {len(single_omics_data)} studies loaded")
        with st.sidebar.expander("📊 Studies:"):
            for study_name in sorted(single_omics_data.keys()):
                st.write(f"• {study_name}")
    else:
        st.sidebar.warning("⚠ No single-omics studies found")

    if kg_data:
        st.sidebar.success("✓ Knowledge graph loaded")
    else:
        st.sidebar.warning("⚠ Knowledge graph not available")

    if wgcna_module_data:
        st.sidebar.success(f"✓ WGCNA modules loaded ({len(wgcna_module_data)} modules)")
    else:
        st.sidebar.warning("⚠ WGCNA modules not available")

    if isinstance(wcgna_cor, pd.DataFrame) and not wcgna_cor.empty:
        st.sidebar.success("✓ WGCNA moduleTraitCor loaded")
    else:
        st.sidebar.warning("⚠ moduleTraitCor not available")

    if isinstance(wcgna_pval, pd.DataFrame) and not wcgna_pval.empty:
        st.sidebar.success("✓ WGCNA moduleTraitPvalue loaded")
    else:
        st.sidebar.warning("⚠ moduleTraitPvalue not available")

    if isinstance(wcgna_pathways, dict) and len(wcgna_pathways) > 0:
        st.sidebar.success(f"✓ WGCNA pathways loaded ({len(wcgna_pathways)} modules)")
    else:
        st.sidebar.warning("⚠ WGCNA pathways not available")

    if ppi_data:
        st.sidebar.success("✓ PPI networks loaded")
    else:
        st.sidebar.warning("⚠ PPI networks not available")
else:
    st.sidebar.error("✗ Error loading data")

st.sidebar.markdown("---")

if not search_query:
    st.title("🔬 Meta Liver")
    st.markdown("*Hypothesis Engine for Liver Genomics*")

    st.markdown("""
    ## Single-Omics Analysis

    Search for a gene to see:
    - Evidence Score (discriminative AUROC + stability + direction agreement + study count)
    - AUROC Across Studies (raw + discriminative)
    - Concordance: AUROC vs logFC
    - Detailed per-study results

    ### Try searching for:
    - SAA1
    - TP53
    - IL6
    - TNF
    """)
else:
    st.title(f"🔬 {search_query}")

    if not single_omics_data:
        st.error("No studies data found!")
    else:
        consistency = soa.compute_consistency_score(search_query, single_omics_data)

        if consistency is None:
            st.warning(f"Gene '{search_query}' not found in any study")
        else:
            tab_omics, tab_kg, tab_coexpr = st.tabs([
                "Single-Omics Evidence",
                "MAFLD Knowledge Graph",
                "Co-expression and PPI Networks"
            ])

            # -----------------------------------------------------------------
            # TAB 1: SINGLE-OMICS EVIDENCE
            # -----------------------------------------------------------------
            with tab_omics:
                st.markdown("""
                This tab summarises gene-level evidence across the single-omics data sets.
                AUC is per-study discriminative performance, logFC indicates direction (MAFLD vs Healthy), and the Evidence Score
                summarises strength, stability, direction agreement, and study support.
                """)
                st.markdown("---")

                help_text = {
                    "Evidence Score": "Overall evidence across studies (Strength × Stability × Direction Agreement × Study Weight).",
                    "Direction Agreement": "Fraction of studies where the gene’s direction (MAFLD vs Healthy) matches the majority.",
                    "Median AUC (disc)": "Median discriminative AUC across studies: AUC-disc = max(AUC, 1−AUC).",
                    "Studies Found": "Number of studies where the gene is present (even if AUROC is missing).",
                    "Strength": "How far the median AUC-disc is above 0.5 (0=no signal; 1=perfect).",
                    "Stability": "Cross-study consistency of AUC-disc (1=very consistent; 0=very variable).",
                    "Study Weight": "Downweights scores supported by very few AUROC values (increases with n_auc).",
                    "Valid AUROC (n_auc)": "Number of studies with a usable AUROC value for this gene.",
                    "Median AUC (raw)": "Median of the raw AUROC values as stored in the study tables (diagnostic).",
                    "Median AUC (oriented)": "Median AUROC after aligning direction so MAFLD is treated as ‘positive’ (diagnostic).",
                    "AUC-disc IQR": "Interquartile range of AUC-disc across studies (lower = more stable).",
                }

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Evidence Score", f"{consistency['evidence_score']:.1%}")
                    st.caption(help_text["Evidence Score"])

                with col2:
                    st.metric("Direction Agreement", f"{consistency['direction_agreement']:.1%}")
                    st.caption(help_text["Direction Agreement"])

                with col3:
                    med_disc = consistency.get("auc_median_discriminative", None)
                    st.metric("Median AUC (disc)", "missing" if med_disc is None else f"{med_disc:.3f}")
                    st.caption(help_text["Median AUC (disc)"])

                with col4:
                    st.metric("Studies Found", f"{consistency['found_count']}")
                    st.caption(help_text["Studies Found"])

                st.info(f"📊 **{consistency['interpretation']}**")
                st.markdown("---")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("Strength", f"{consistency.get('strength', np.nan):.3f}")
                    st.caption(help_text["Strength"])

                with c2:
                    st.metric("Stability", f"{consistency.get('stability', np.nan):.3f}")
                    st.caption(help_text["Stability"])

                with c3:
                    st.metric("Study Weight", f"{consistency.get('n_weight', np.nan):.3f}")
                    st.caption(help_text["Study Weight"])

                with c4:
                    st.metric("Valid AUROC (n_auc)", f"{consistency.get('n_auc', 'N/A')}")
                    st.caption(help_text["Valid AUROC (n_auc)"])

                st.markdown("---")

                d1, d2, d3 = st.columns(3)

                auc_disc_vals = []
                try:
                    auc_disc_vals = [max(a, 1.0 - a) for a in consistency.get("auc_values", []) if a is not None and not np.isnan(a)]
                except Exception:
                    auc_disc_vals = []

                disc_iqr = np.nan
                if len(auc_disc_vals) > 1:
                    disc_iqr = float(np.subtract(*np.percentile(auc_disc_vals, [75, 25])))

                with d1:
                    st.metric("Median AUC (raw)", f"{consistency.get('auc_median', np.nan):.3f}")
                    st.caption(help_text["Median AUC (raw)"])

                with d2:
                    st.metric("Median AUC (oriented)", f"{consistency.get('auc_median_oriented', np.nan):.3f}")
                    st.caption(help_text["Median AUC (oriented)"])

                with d3:
                    st.metric("AUC-disc IQR", "NA" if np.isnan(disc_iqr) else f"{disc_iqr:.3f}")
                    st.caption(help_text["AUC-disc IQR"])

                st.markdown("---")

                metrics = _collect_gene_metrics(search_query, single_omics_data)

                left, right = st.columns(2)

                with left:
                    st.markdown("**AUROC Across Studies (raw)**")
                    fig_raw = make_lollipop(
                        metrics,
                        auc_key="auc_raw",
                        title="AUROC Across Studies (raw)",
                        subtitle="Raw AUROC values as stored in each study table."
                    )
                    if fig_raw:
                        st.plotly_chart(fig_raw, use_container_width=True)
                    else:
                        st.info("No AUROC values available for this gene.")

                with right:
                    st.markdown("**AUROC Across Studies (discriminative)**")
                    fig_disc = make_lollipop(
                        metrics,
                        auc_key="auc_disc",
                        title="AUROC Across Studies (discriminative)",
                        subtitle="AUC-disc = max(AUC, 1−AUC), so AUROC < 0.5 counts as label-flipped signal."
                    )
                    if fig_disc:
                        st.plotly_chart(fig_disc, use_container_width=True)
                    else:
                        st.info("No AUROC values available for this gene.")

                left2, right2 = st.columns(2)

                with left2:
                    st.markdown("**Concordance: AUC-disc vs logFC**")
                    fig_scatter_disc = make_scatter_auc_logfc(
                        metrics,
                        auc_key="auc_disc",
                        title="Concordance: AUC-disc vs logFC",
                        subtitle="Checks whether discriminative signal aligns with up/down regulation."
                    )
                    if fig_scatter_disc:
                        st.plotly_chart(fig_scatter_disc, use_container_width=True)
                    else:
                        st.info("Not enough data for concordance plot (need AUROC + logFC in ≥2 studies).")

                with right2:
                    st.markdown("**AUC-disc distribution (stability view)**")
                    fig_dist = make_auc_disc_distribution(metrics)
                    if fig_dist:
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.info("Not enough AUROC values to show a distribution.")

                st.markdown("---")

                st.markdown("**Detailed Results**")
                results_df = soa.create_results_table(search_query, single_omics_data)
                if results_df is not None:
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No per-study rows found for this gene.")

            # -----------------------------------------------------------------
            # TAB 2: MAFLD KNOWLEDGE GRAPH
            # -----------------------------------------------------------------
            with tab_kg:
                st.markdown("""
                This tab places the selected gene in its network context within the MAFLD MASH subgraph.
                We report whether the gene is present in the subgraph, its assigned cluster, and centrality
                metrics (PageRank, betweenness, eigenvector) to indicate whether it behaves as a hub or a
                peripheral node. The cluster view lists co-clustered genes, drugs, and disease annotations.
                """)
                st.markdown("---")

                if kg_data:
                    kg_info = get_gene_kg_info(search_query, kg_data)

                    if kg_info:
                        cluster_id = kg_info.get("cluster", None)

                        h1, h2, h3, h4 = st.columns(4)
                        with h1:
                            st.metric("Cluster", "N/A" if cluster_id is None else str(cluster_id))
                        with h2:
                            st.metric("Composite centrality", _fmt_pct(kg_info.get("composite_percentile")))
                            st.caption(f"raw: {_fmt_num(kg_info.get('composite'), decimals=6)}")
                        with h3:
                            st.metric("Betweenness", _fmt_pct(kg_info.get("bet_percentile")))
                            st.caption(f"raw: {_fmt_num_commas(kg_info.get('betweenness'), decimals=2)}")
                        with h4:
                            st.metric("PageRank", _fmt_pct(kg_info.get("pagerank_percentile")))
                            st.caption(f"raw: {_fmt_num(kg_info.get('pagerank'), decimals=4)}")

                        h5, h6, _, _ = st.columns(4)
                        with h5:
                            st.metric("Eigenvector", _fmt_pct(kg_info.get("eigen_percentile")))
                            st.caption(f"raw: {_fmt_num(kg_info.get('eigen'), decimals=6, sci_if_small=True)}")
                        with h6:
                            st.caption("Percentiles computed across the MASH subgraph nodes table.")

                        st.markdown("---")

                        interpretation = interpret_centrality(
                            kg_info.get("pagerank", np.nan),
                            kg_info.get("betweenness", np.nan),
                            kg_info.get("eigen", np.nan),
                            pagerank_pct=kg_info.get("pagerank_percentile"),
                            betweenness_pct=kg_info.get("bet_percentile"),
                            eigen_pct=kg_info.get("eigen_percentile"),
                            composite_pct=kg_info.get("composite_percentile"),
                        )
                        st.info(f"📍 {interpretation}")

                        with st.expander("Show raw centrality values and subgraph ranges"):
                            st.markdown("**Composite Centrality (Weighted Geo-Mean of Percentiles)**")
                            st.write(f"Composite score: {_fmt_num(kg_info.get('composite'), decimals=6)}")
                            st.write(
                                f"Min/Max (subgraph): "
                                f"{_fmt_num(kg_info.get('composite_min'), decimals=6)} – {_fmt_num(kg_info.get('composite_max'), decimals=6)}"
                            )
                            st.write(f"Percentile: {_fmt_pct(kg_info.get('composite_percentile'))}")
                            st.write("*Weights: PageRank 50%, Betweenness 25%, Eigenvector 25%*")

                            st.markdown("---")
                            st.markdown("**Individual Metrics – MASH Subgraph Context**")

                            c1, c2, c3 = st.columns(3)

                            with c1:
                                st.write("**PageRank**")
                                st.write(f"Min: {_fmt_num(kg_info.get('pagerank_min'), decimals=4)}")
                                st.write(f"Max: {_fmt_num(kg_info.get('pagerank_max'), decimals=4)}")
                                st.write(f"Your node: {_fmt_num(kg_info.get('pagerank'), decimals=4)}")
                                st.write(f"Percentile: {_fmt_pct(kg_info.get('pagerank_percentile'))}")

                            with c2:
                                st.write("**Betweenness**")
                                st.write(f"Min: {_fmt_num(kg_info.get('bet_min'), decimals=4)}")
                                st.write(f"Max: {_fmt_num(kg_info.get('bet_max'), decimals=4)}")
                                st.write(f"Your node: {_fmt_num_commas(kg_info.get('betweenness'), decimals=4)}")
                                st.write(f"Percentile: {_fmt_pct(kg_info.get('bet_percentile'))}")

                            with c3:
                                st.write("**Eigenvector**")
                                st.write(f"Min: {_fmt_num(kg_info.get('eigen_min'), decimals=6)}")
                                st.write(f"Max: {_fmt_num(kg_info.get('eigen_max'), decimals=6)}")
                                st.write(f"Your node: {_fmt_num(kg_info.get('eigen'), decimals=6, sci_if_small=True)}")
                                st.write(f"Percentile: {_fmt_pct(kg_info.get('eigen_percentile'))}")

                        st.markdown("---")

                        if cluster_id is None or str(cluster_id).strip() == "" or str(cluster_id).lower() == "nan":
                            st.warning("Cluster ID missing for this node; cannot display cluster neighbours.")
                        else:
                            st.markdown("**Nodes in Cluster**")

                            ctl1, ctl2 = st.columns(2)
                            with ctl1:
                                top_n = st.slider("Show top N per table", 10, 300, 50, step=10, key="kg_top_n")
                            with ctl2:
                                sort_key = st.selectbox(
                                    "Sort by",
                                    ["Composite %ile", "PR %ile", "Bet %ile", "Eigen %ile", "PageRank (raw)", "Betweenness (raw)", "Eigen (raw)", "Name"],
                                    index=0,
                                    key="kg_sort_key"
                                )

                            tab_genes, tab_drugs, tab_diseases = st.tabs(["Genes/Proteins", "Drugs", "Diseases"])

                            with tab_genes:
                                genes_df = get_cluster_genes(cluster_id, kg_data)
                                if genes_df is not None and not genes_df.empty:
                                    st.dataframe(_prepare_cluster_table(genes_df, sort_key, top_n), use_container_width=True, hide_index=True)
                                else:
                                    st.write("No genes/proteins in this cluster")

                            with tab_drugs:
                                drugs_df = get_cluster_drugs(cluster_id, kg_data)
                                if drugs_df is not None and not drugs_df.empty:
                                    st.dataframe(_prepare_cluster_table(drugs_df, sort_key, top_n), use_container_width=True, hide_index=True)
                                else:
                                    st.write("No drugs in this cluster")

                            with tab_diseases:
                                diseases_df = get_cluster_diseases(cluster_id, kg_data)
                                if diseases_df is not None and not diseases_df.empty:
                                    st.dataframe(_prepare_cluster_table(diseases_df, sort_key, top_n), use_container_width=True, hide_index=True)
                                else:
                                    st.write("No diseases in this cluster")
                    else:
                        st.warning(f"⚠ '{search_query}' not found in MASH subgraph")
                else:
                    st.warning("⚠ Knowledge graph data not loaded")

            # -----------------------------------------------------------------
            # TAB 3: CO-EXPRESSION AND PPI NETWORKS
            # -----------------------------------------------------------------
            with tab_coexpr:
                st.markdown("""
                This tab summarises the gene's systems-level context from WGCNA module membership
                and protein–protein interaction (PPI) networks. We report the WGCNA module assignment
                and highlight co-expressed partners, then show direct PPI interactors and local network stats.
                """)
                st.markdown("---")

                st.markdown("**WGCNA Co-expression Module**")
                if wgcna_module_data:
                    gene_module_info = get_gene_module(search_query, wgcna_module_data)
                    if gene_module_info:
                        module_name = gene_module_info["module"]
                        st.markdown(f"**Module Assignment:** {module_name}")

                        module_genes = get_module_genes(module_name, wgcna_module_data)
                        if module_genes is not None:
                            st.markdown(f"Top genes in module {module_name}:")
                            st.dataframe(module_genes.head(15), use_container_width=True)
                        else:
                            st.info(f"No other genes found in module {module_name}")

                        with st.expander("Module–trait relationships (WGCNA)"):
                            mt = _module_trait_table(module_name, wcgna_cor, wcgna_pval)
                            if mt is None or mt.empty:
                                st.info("Module–trait tables not available for this module (check module index names).")
                            else:
                                st.dataframe(mt, use_container_width=True, hide_index=True)

                        with st.expander("Pathways / enrichment (module)"):
                            key = str(module_name).strip().lower()
                            dfp = (wcgna_pathways or {}).get(key)
                            if dfp is None or dfp.empty:
                                st.info(f"No enrichment table found for module '{module_name}' in wcgna/pathways/")
                            else:
                                st.dataframe(dfp, use_container_width=True, hide_index=True)

                    else:
                        st.info(f"⚠ '{search_query}' not found in WGCNA module assignments")
                else:
                    st.info("⚠ WGCNA module data not available")

                st.markdown("---")
                st.markdown("**Protein-Protein Interaction Network**")
                if ppi_data:
                    ppi_df = find_ppi_interactors(search_query, ppi_data)
                    if ppi_df is not None:
                        net_stats = get_network_stats(search_query, ppi_data)
                        if net_stats:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Direct Interactors", net_stats["degree"])
                            with col2:
                                st.write(f"**Network Property:** {net_stats['description']}")
                        st.markdown(f"Direct interaction partners of {search_query}:")
                        st.dataframe(ppi_df, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"⚠ '{search_query}' not found in PPI networks")
                else:
                    st.info("⚠ PPI network data not available")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 11px;'>"
    "<p>Meta Liver - Three-tab interface: Single-Omics Evidence | MAFLD Knowledge Graph | Co-expression and PPI Networks</p>"
    "</div>",
    unsafe_allow_html=True
)
