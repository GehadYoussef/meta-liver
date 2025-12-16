"""
Gene Dossier Template for Meta Liver
Creates interactive one-page reports for genes with three linked panels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from evidence_scoring import compute_evidence_fingerprint
from data_loaders import (
    load_wgcna_expr, load_wgcna_mes, load_wgcna_mod_trait_cor,
    load_gene_mapping, load_module_genes, load_pathway_enrichment,
    load_kg_mash_nodes, load_kg_mash_drugs, get_modules,
    load_gse210501, load_gse212837, load_gse189600, load_auc_coassolo
)

# ============================================================================
# PANEL 1: WGCNA CONTEXT
# ============================================================================

def render_wgcna_panel(gene_name: str, fingerprint: Dict):
    """
    Render WGCNA panel: module, kME, co-expressing genes, module-level drug proximity
    """
    
    st.subheader("🧬 WGCNA Context")
    
    microenv = fingerprint['microenv_score']
    module = microenv['module']
    kme = microenv['kme']
    
    if module is None:
        st.warning(f"Gene not found in WGCNA expression matrix")
        return
    
    # Module info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Module", module.upper(), f"kME: {kme:.3f}")
    
    with col2:
        st.metric("Module Annotation", microenv['module_annotation'].title())
    
    with col3:
        trait_cor = microenv['module_trait_cor']
        st.metric("Module-Trait Correlation", f"{trait_cor:.3f}")
    
    # Top co-expressing genes in module
    st.markdown("**Top Co-Expressing Genes in Module**")
    
    expr_data = load_wgcna_expr()
    mes_data = load_wgcna_mes()
    
    if not expr_data.empty and not mes_data.empty:
        # Get module eigenvector
        me_col = f"ME{module}"
        if me_col in mes_data.columns:
            module_me = mes_data[me_col].values
            
            # Compute correlation of all genes with module eigenvector
            correlations = {}
            for col in expr_data.columns:
                gene_expr = expr_data[col].values
                corr = np.corrcoef(gene_expr, module_me)[0, 1]
                correlations[col] = corr
            
            # Get top genes (excluding the query gene)
            top_genes = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_genes = [g for g in top_genes if gene_name.lower() not in g[0].lower()][:10]
            
            # Display as table
            coexpr_df = pd.DataFrame({
                'Gene': [g[0] for g in top_genes],
                'kME': [f"{g[1]:.3f}" for g in top_genes]
            })
            
            st.dataframe(coexpr_df, use_container_width=True, hide_index=True)
    
    # Pathway enrichment for module
    st.markdown("**Module Enrichment (Top Terms)**")
    
    enrichment = load_pathway_enrichment(module)
    if not enrichment.empty:
        # Show top 5 enriched terms
        top_enrichment = enrichment.nlargest(5, 'Count')[['Description', 'Count', 'P-value']]
        
        # Format p-value
        top_enrichment = top_enrichment.copy()
        top_enrichment['P-value'] = top_enrichment['P-value'].apply(lambda x: f"{x:.2e}")
        
        st.dataframe(top_enrichment, use_container_width=True, hide_index=True)
    else:
        st.info("No enrichment data available for this module")

# ============================================================================
# PANEL 2: KNOWLEDGE GRAPH CONTEXT
# ============================================================================

def render_kg_panel(gene_name: str, fingerprint: Dict):
    """
    Render KG panel: cluster neighbors + best mechanistic routes to drugs
    """
    
    st.subheader("🕸️ Knowledge Graph Context")
    
    kg = fingerprint['kg_prior']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("KG Centrality", f"{kg['centrality']:.3f}")
    
    with col2:
        st.metric("Cluster", kg['cluster'] if kg['cluster'] else "Unknown")
    
    with col3:
        st.metric("Node Type", kg['node_type'].title())
    
    # Cluster neighbors
    st.markdown("**Cluster Neighbors (Top 10)**")
    
    mash_nodes = load_kg_mash_nodes()
    if not mash_nodes.empty and kg['cluster'] is not None:
        cluster_nodes = mash_nodes[mash_nodes['Cluster'] == kg['cluster']]
        
        # Sort by PageRank
        if 'PageRank Score' in cluster_nodes.columns:
            cluster_nodes = cluster_nodes.nlargest(10, 'PageRank Score')
        
        # Display relevant columns
        display_cols = ['Name', 'Type', 'PageRank Score', 'Cluster']
        available_cols = [c for c in display_cols if c in cluster_nodes.columns]
        
        neighbor_df = cluster_nodes[available_cols].copy()
        st.dataframe(neighbor_df, use_container_width=True, hide_index=True)
    else:
        st.info("No cluster information available")
    
    # Mechanistic paths to drugs
    st.markdown("**Best Mechanistic Routes to Drugs**")
    
    # This is a simplified version - in full implementation, would compute actual paths
    st.info("""
    Top mechanistic routes would be computed here based on:
    - Direct target edges (highest priority)
    - Co-mention edges
    - Generic pathway edges
    
    Example: Gene → Protein Complex → Pathway → Drug
    """)

# ============================================================================
# PANEL 3: CROSS-STUDY AUC FOREST PLOT
# ============================================================================

def render_auc_panel(gene_name: str, fingerprint: Dict):
    """
    Render AUC forest plot across studies with heterogeneity badge
    """
    
    st.subheader("📊 Cross-Study Evidence")
    
    hep = fingerprint['hepatocyte_score']
    
    # Collect AUC data across studies
    studies_auc = []
    studies_names = []
    
    for study_name, study_data in hep['studies'].items():
        if 'auc' in study_data:
            studies_auc.append(study_data['auc'])
            studies_names.append(study_name)
    
    if not studies_auc:
        st.warning("No AUC data available for this gene")
        return
    
    # Create forest plot
    fig = go.Figure()
    
    # Add points for each study
    fig.add_trace(go.Scatter(
        x=studies_auc,
        y=studies_names,
        mode='markers',
        marker=dict(size=12, color='#1f77b4'),
        name='AUC',
        hovertemplate='<b>%{y}</b><br>AUC: %{x:.3f}<extra></extra>'
    ))
    
    # Add median line
    median_auc = np.median(studies_auc)
    fig.add_vline(x=median_auc, line_dash="dash", line_color="red", 
                  annotation_text=f"Median: {median_auc:.3f}")
    
    # Add reference line at 0.5 (random)
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", 
                  annotation_text="Random (0.5)")
    
    fig.update_layout(
        title="AUC Across Studies",
        xaxis_title="AUC",
        yaxis_title="Study",
        height=300,
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Heterogeneity badge
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Median AUC", f"{hep['auc_median']:.3f}")
    
    with col2:
        st.metric("Direction Agreement", f"{hep['direction_agreement']:.1%}")
    
    with col3:
        heterogeneity = fingerprint['heterogeneity']
        st.metric("Heterogeneity", heterogeneity.replace('_', ' ').title())
    
    # Study details table
    st.markdown("**Study Details**")
    
    details = []
    for study_name, study_data in hep['studies'].items():
        details.append({
            'Study': study_name,
            'AUC': f"{study_data.get('auc', np.nan):.3f}" if 'auc' in study_data else 'N/A',
            'logFC': f"{study_data.get('logfc', np.nan):.3f}" if 'logfc' in study_data else 'N/A',
        })
    
    details_df = pd.DataFrame(details)
    st.dataframe(details_df, use_container_width=True, hide_index=True)

# ============================================================================
# EVIDENCE FINGERPRINT VISUALIZATION
# ============================================================================

def render_evidence_fingerprint(fingerprint: Dict):
    """
    Render side-by-side evidence fingerprint with discordance flags
    """
    
    st.markdown("---")
    st.subheader("📋 Evidence Fingerprint")
    
    hep_score = fingerprint['hepatocyte_score']['score']
    microenv_score = fingerprint['microenv_score']['score']
    kg_score = fingerprint['kg_prior']['score']
    composite = fingerprint['composite_score']
    
    # Create fingerprint visualization
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hepatocyte Score", f"{hep_score:.3f}", 
                 delta=f"Intent: {fingerprint['intent']}")
    
    with col2:
        st.metric("Microenv Score", f"{microenv_score:.3f}")
    
    with col3:
        st.metric("KG Prior", f"{kg_score:.3f}")
    
    with col4:
        st.metric("Composite", f"{composite:.3f}")
    
    # Discordance and taxonomy
    col1, col2 = st.columns(2)
    
    with col1:
        discordance = fingerprint['discordance']
        st.info(f"**Discordance:** {discordance}")
    
    with col2:
        taxonomy = fingerprint['taxonomy']
        st.success(f"**Taxonomy:** {taxonomy}")
    
    # Radar chart for scores
    fig = go.Figure(data=go.Scatterpolar(
        r=[hep_score, microenv_score, kg_score],
        theta=['Hepatocyte', 'Microenv', 'KG Prior'],
        fill='toself',
        name='Evidence'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        title="Evidence Profile"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN GENE DOSSIER
# ============================================================================

def render_gene_dossier(gene_name: str, intent: str = "hepatocyte"):
    """
    Render complete gene dossier with all three panels and evidence fingerprint
    """
    
    # Compute evidence fingerprint
    fingerprint = compute_evidence_fingerprint(gene_name, intent)
    
    # Title with fingerprint badge
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(f"🔬 {gene_name}")
    
    with col2:
        taxonomy = fingerprint['taxonomy']
        st.markdown(f"**{taxonomy}**")
    
    # Plain-language summary
    st.markdown(f"""
    ### Why This Gene Matters
    
    **{gene_name}** shows {fingerprint['discordance'].lower()} evidence in liver disease context.
    
    - **Hepatocyte Signal:** {fingerprint['hepatocyte_score']['score']:.1%} confidence
    - **Microenvironment Signal:** {fingerprint['microenv_score']['score']:.1%} confidence
    - **Disease Relevance:** {fingerprint['kg_prior']['score']:.1%} (KG prior)
    
    Overall evidence strength: **{fingerprint['composite_score']:.1%}**
    """)
    
    st.markdown("---")
    
    # Three linked panels
    tab1, tab2, tab3 = st.tabs(["WGCNA", "Knowledge Graph", "Cross-Study Evidence"])
    
    with tab1:
        render_wgcna_panel(gene_name, fingerprint)
    
    with tab2:
        render_kg_panel(gene_name, fingerprint)
    
    with tab3:
        render_auc_panel(gene_name, fingerprint)
    
    # Evidence fingerprint
    render_evidence_fingerprint(fingerprint)
    
    # Action items
    st.markdown("---")
    st.subheader("💡 Suggested Actions")
    
    if fingerprint['taxonomy'] == 'hepatocyte driver':
        st.success("""
        **High-confidence hepatocyte target**
        - Prioritize for functional validation in hepatocytes
        - Consider for drug screening
        - Validate with hepatocyte-specific perturbations
        """)
    elif fingerprint['taxonomy'] == 'microenvironment hub':
        st.info("""
        **Microenvironment/fibrosis marker**
        - Investigate fibroblast/immune cell involvement
        - Consider for fibrosis/NASH biology studies
        - May be secondary to hepatocyte drivers
        """)
    elif fingerprint['taxonomy'] == 'broad disease node':
        st.warning("""
        **Disease-associated node (broad signal)**
        - Likely hub in disease network
        - May be downstream of primary drivers
        - Useful for pathway understanding
        """)
    else:
        st.warning("""
        **Context-specific / mixed signal**
        - Evidence is study or context-dependent
        - Requires careful interpretation
        - Consider study-specific factors
        """)
    
    return fingerprint
