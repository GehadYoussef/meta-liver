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
from kg_analysis import get_gene_kg_info, get_cluster_genes, get_cluster_drugs, get_cluster_diseases, interpret_centrality
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
    - AUROC Across Studies
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
                For the selected gene, we report study-specific AUROC values and differential expression
                direction (logFC), together with a composite evidence score reflecting discriminative power,
                stability across studies, direction agreement, and the number of studies with valid AUROC.
                """)
                st.markdown("---")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Evidence Score", f"{consistency['evidence_score']:.1%}")

                with col2:
                    st.metric("Direction Agreement", f"{consistency['direction_agreement']:.1%}")

                # Show discriminative median (this is the “AUC-disc” you care about)
                with col3:
                    med_disc = consistency.get("auc_median_discriminative", None)
                    if med_disc is None:
                        st.metric("Median AUC (disc)", "missing")
                    else:
                        st.metric("Median AUC (disc)", f"{med_disc:.3f}")

                with col4:
                    st.metric("Studies Found", f"{consistency['found_count']}")

                st.info(f"📊 **{consistency['interpretation']}**")

                with st.expander("Show scoring components"):
                    st.write(f"Strength: {consistency.get('strength', np.nan):.3f}")
                    st.write(f"Stability: {consistency.get('stability', np.nan):.3f}")
                    st.write(f"n_auc (valid AUROC): {consistency.get('n_auc', 'N/A')}")
                    st.write(f"n_weight: {consistency.get('n_weight', np.nan):.3f}")
                    st.write(f"Median AUC (raw): {consistency.get('auc_median', np.nan):.3f}")
                    st.write(f"Median AUC (oriented): {consistency.get('auc_median_oriented', np.nan):.3f}")

                st.markdown("**AUROC Across Studies**")
                fig = soa.create_lollipop_plot(search_query, single_omics_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Concordance: AUROC vs logFC**")
                fig_scatter = soa.create_auc_logfc_scatter(search_query, single_omics_data)
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough data for concordance plot")

                st.markdown("**Detailed Results**")
                results_df = soa.create_results_table(search_query, single_omics_data)
                if results_df is not None:
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

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
