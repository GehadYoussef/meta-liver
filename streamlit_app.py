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
    get_coexpressed_partners, find_ppi_interactors, get_network_stats
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
    return single_omics, kg_data, wgcna_module_data, ppi_data


try:
    single_omics_data, kg_data, wgcna_module_data, ppi_data = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    single_omics_data = {}
    kg_data = {}
    wgcna_module_data = {}
    ppi_data = {}


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

    # stems
    for m in vals:
        fig.add_trace(go.Scatter(
            x=[0.5, m[auc_key]],
            y=[m["study"], m["study"]],
            mode="lines",
            line=dict(color="#cccccc", width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ))

    # points
    for m in vals:
        style = _marker_style(m.get("direction"))
        lfc = m.get("lfc")
        size = 10 + abs(lfc if lfc is not None else 0.0) * 1.5
        size = float(min(size, 16))

        hover = (
            f"<b>{m['study']}</b>"
            f"<br>{auc_key}: {m[auc_key]:.3f}"
        )
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
# MAIN APP
# =============================================================================

st.sidebar.markdown("## 🔬 Meta Liver")
search_query = st.sidebar.text_input("Search gene:", placeholder="e.g., SAA1, TP53, IL6").strip().upper()

# Debug: confirm which single_omics_analysis.py is actually being imported
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
        # Always use the module's scoring logic (no duplicated scoring logic here)
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

                # Headline metrics
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

                # Score components shown directly (no pull-down)
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

                # Diagnostics (useful for debugging)
                d1, d2, d3 = st.columns(3)

                # AUC-disc IQR (if not returned, compute quickly from values)
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

                # NEW PLOTS (raw + disc + distribution)
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

                    if kg_info and kg_info.get("found", False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cluster", kg_info["cluster"])
                        with col2:
                            st.metric("PageRank", f"{kg_info['pagerank']:.4f}")
                        with col3:
                            st.metric("Betweenness", f"{kg_info['betweenness']:.4f}")

                        col4, col5, col6 = st.columns(3)
                        with col4:
                            st.metric("Eigenvector", f"{kg_info['eigen']:.4f}")
                        with col5:
                            st.metric("Composite", f"{kg_info['composite']:.6f}")
                        with col6:
                            st.write("")

                        st.markdown("---")

                        st.markdown("**Centrality Metrics - Whole Graph Context**")
                        st.markdown("**Composite Centrality (Weighted Geo-Mean of Percentiles)**")
                        st.write(f"Composite score: {kg_info['composite']:.6f}")
                        st.write(f"Min/Max (whole graph): {kg_info['composite_min']:.6f} – {kg_info['composite_max']:.6f}")
                        st.write(f"Percentile: {kg_info['composite_percentile']:.1f}%")
                        st.write("*Weights: PageRank 50%, Betweenness 25%, Eigenvector 25%*")

                        st.markdown("---")
                        st.markdown("**Individual Metrics - Whole Graph Context**")

                        context_col1, context_col2, context_col3 = st.columns(3)

                        with context_col1:
                            st.write("**PageRank**")
                            st.write(f"Min: {kg_info['pr_min']:.4f}")
                            st.write(f"Max: {kg_info['pr_max']:.4f}")
                            st.write(f"Your node: {kg_info['pagerank']:.4f}")
                            st.write(f"Percentile: {kg_info['pr_percentile']:.1f}%")

                        with context_col2:
                            st.write("**Betweenness**")
                            st.write(f"Min: {kg_info['bet_min']:.4f}")
                            st.write(f"Max: {kg_info['bet_max']:.4f}")
                            st.write(f"Your node: {kg_info['betweenness']:.4f}")
                            st.write(f"Percentile: {kg_info['bet_percentile']:.1f}%")

                        with context_col3:
                            st.write("**Eigenvector**")
                            st.write(f"Min: {kg_info['eigen_min']:.4f}")
                            st.write(f"Max: {kg_info['eigen_max']:.4f}")
                            st.write(f"Your node: {kg_info['eigen']:.4f}")
                            st.write(f"Percentile: {kg_info['eigen_percentile']:.1f}%")

                        st.markdown("---")

                        interpretation = interpret_centrality(
                            kg_info["pagerank"],
                            kg_info["betweenness"],
                            kg_info["eigen"]
                        )
                        st.info(f"📍 {interpretation}")

                        st.markdown("**Nodes in Cluster**")
                        tab_genes, tab_drugs, tab_diseases = st.tabs(["Genes/Proteins", "Drugs", "Diseases"])

                        with tab_genes:
                            genes_df = get_cluster_genes(kg_info["cluster"], kg_data)
                            if genes_df is not None:
                                st.dataframe(genes_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("No genes/proteins in this cluster")

                        with tab_drugs:
                            drugs_df = get_cluster_drugs(kg_info["cluster"], kg_data)
                            if drugs_df is not None:
                                st.dataframe(drugs_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("No drugs in this cluster")

                        with tab_diseases:
                            diseases_df = get_cluster_diseases(kg_info["cluster"], kg_data)
                            if diseases_df is not None:
                                st.dataframe(diseases_df, use_container_width=True, hide_index=True)
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
                        st.markdown(f"**Module Assignment:** {gene_module_info['module']}")
                        module_genes = get_module_genes(gene_module_info["module"], wgcna_module_data)
                        if module_genes is not None:
                            st.markdown(f"Top genes in module {gene_module_info['module']}:")
                            st.dataframe(module_genes.head(15), use_container_width=True)
                        else:
                            st.info(f"No other genes found in module {gene_module_info['module']}")
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
