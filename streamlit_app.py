"""
Meta Liver v4 - Simplified with Inline Forest Plot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from kg_analysis import load_kg_data, get_gene_kg_info, get_cluster_genes, get_cluster_drugs, get_cluster_diseases, interpret_centrality

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean scientific look
st.markdown("""
<style>
    /* Clean white/light gray theme */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 14px;
        padding: 10px 20px;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_resource
def load_all_data():
    """Load all data from Google Drive"""
    from robust_data_loader import load_single_omics_data
    
    data_dir = Path.home() / "meta_liver_data"
    
    # Load single-omics data
    single_omics = load_single_omics_data(data_dir)
    
    # Load KG data
    kg_data = load_kg_data(data_dir)
    
    return single_omics, kg_data


try:
    single_omics_data, kg_data = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# ============================================================================
# TITLE & DESCRIPTION
# ============================================================================

st.markdown("# 🔬 Meta Liver")
st.markdown("""
Explore liver genomics data across multiple studies, including single-omics gene 
expression analysis and knowledge graph context.
""")

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Single-Omics Search", "WGCNA Modules", "Knowledge Graph"]
)

# ============================================================================
# PAGE 1: SINGLE-OMICS SEARCH
# ============================================================================

if page == "Single-Omics Search":
    st.markdown("## Single-Omics Gene Search")
    
    if not data_loaded:
        st.error("Could not load data. Please check data directory.")
    else:
        # Search input
        search_query = st.text_input(
            "Enter gene name",
            placeholder="e.g., APOB, ALB, HMGCR",
            key="gene_search"
        ).strip()
        
        if search_query:
            # Find matching genes
            matching_genes = []
            for study_name, study_df in single_omics_data.items():
                genes_in_study = study_df['Gene'].unique()
                matches = [g for g in genes_in_study if search_query.lower() in g.lower()]
                for match in matches:
                    if match not in [m['gene'] for m in matching_genes]:
                        matching_genes.append({'gene': match, 'study': study_name})
            
            if not matching_genes:
                st.warning(f"No genes matching '{search_query}' found in any study")
            else:
                # If multiple matches, let user select
                if len(matching_genes) > 1:
                    gene_options = [f"{m['gene']}" for m in matching_genes]
                    selected_gene = st.selectbox("Select gene:", gene_options)
                else:
                    selected_gene = matching_genes[0]['gene']
                
                # Collect data for selected gene across all studies
                results = []
                for study_name, study_df in single_omics_data.items():
                    gene_data = study_df[study_df['Gene'].str.lower() == selected_gene.lower()]
                    if not gene_data.empty:
                        for _, row in gene_data.iterrows():
                            results.append({
                                'Study': study_name,
                                'Gene': row['Gene'],
                                'AUC': float(row['AUC']),
                                'logFC': float(row['avg_logFC']),
                                'Direction': row['direction']
                            })
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Calculate consistency scores
                    auc_values = results_df['AUC'].values
                    auc_consistency = 1 - (auc_values.std() / auc_values.mean()) if auc_values.mean() > 0 else 0
                    auc_consistency = max(0, min(1, auc_consistency))
                    
                    direction_counts = results_df['Direction'].value_counts()
                    direction_agreement = direction_counts.max() / len(results_df)
                    
                    overall_consistency = (auc_consistency + direction_agreement) / 2
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Studies Found", len(results_df))
                    with col2:
                        st.metric("Mean AUC", f"{results_df['AUC'].mean():.3f}")
                    with col3:
                        st.metric("AUC Consistency", f"{auc_consistency:.2%}")
                    with col4:
                        st.metric("Direction Agreement", f"{direction_agreement:.0%}")
                    
                    # Lollipop plot
                    st.markdown("---")
                    st.markdown("**AUC Across Studies**")
                    
                    fig = go.Figure()
                    
                    # Color by direction
                    colors = ['#2E86AB' if d == 'MAFLD' else '#A23B72' for d in results_df['Direction']]
                    
                    # Marker symbols by direction
                    symbols = ['triangle-up' if d == 'MAFLD' else 'triangle-down' for d in results_df['Direction']]
                    
                    for i, (idx, row) in enumerate(results_df.iterrows()):
                        fig.add_trace(go.Scatter(
                            x=[row['AUC']],
                            y=[row['Study']],
                            mode='markers+lines',
                            marker=dict(
                                size=12,
                                symbol=symbols[i],
                                color=colors[i],
                                line=dict(color='white', width=1)
                            ),
                            line=dict(color='#cccccc', width=1),
                            name=row['Direction'],
                            hovertemplate=f"<b>{row['Study']}</b><br>AUC: {row['AUC']:.3f}<br>logFC: {row['logFC']:.2f}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title=f"<b>{selected_gene}</b> - AUC Across Studies",
                        xaxis_title="AUC Score",
                        yaxis_title="Study",
                        template="plotly_white",
                        height=400,
                        showlegend=False,
                        hovermode='closest',
                        font=dict(size=12),
                        xaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Concordance scatter plot
                    st.markdown("---")
                    st.markdown("**AUC vs logFC Concordance**")
                    
                    fig2 = go.Figure()
                    
                    for i, (idx, row) in enumerate(results_df.iterrows()):
                        fig2.add_trace(go.Scatter(
                            x=[row['AUC']],
                            y=[row['logFC']],
                            mode='markers',
                            marker=dict(
                                size=14,
                                symbol=symbols[i],
                                color=colors[i],
                                line=dict(color='white', width=1)
                            ),
                            name=row['Study'],
                            hovertemplate=f"<b>{row['Study']}</b><br>AUC: {row['AUC']:.3f}<br>logFC: {row['logFC']:.2f}<extra></extra>"
                        ))
                    
                    fig2.update_layout(
                        title=f"<b>{selected_gene}</b> - AUC vs logFC",
                        xaxis_title="AUC Score",
                        yaxis_title="log2(Fold Change)",
                        template="plotly_white",
                        height=400,
                        hovermode='closest',
                        font=dict(size=12),
                        xaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Data table
                    st.markdown("---")
                    st.markdown("**Raw Data**")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Knowledge Graph Section
                    st.markdown("---")
                    st.markdown("**Knowledge Graph Context**")
                    
                    if kg_data:
                        kg_info = get_gene_kg_info(selected_gene, kg_data)
                        
                        if kg_info and kg_info['found']:
                            # Gene's position in subgraph
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Cluster", kg_info['cluster'])
                            with col2:
                                st.metric("PageRank", f"{kg_info['pagerank']:.4f}")
                            with col3:
                                st.metric("Betweenness", f"{kg_info['betweenness']:.4f}")
                            
                            # Interpretation
                            interpretation = interpret_centrality(
                                kg_info['pagerank'],
                                kg_info['betweenness'],
                                kg_info['eigen']
                            )
                            st.info(f"📍 {interpretation}")
                            
                            # Nodes in same cluster - three tabs
                            st.markdown("**Nodes in Cluster**")
                            
                            tab_genes, tab_drugs, tab_diseases = st.tabs(["Genes/Proteins", "Drugs", "Diseases"])
                            
                            with tab_genes:
                                genes_df = get_cluster_genes(kg_info['cluster'], kg_data)
                                if genes_df is not None:
                                    st.dataframe(genes_df, use_container_width=True, hide_index=True)
                                else:
                                    st.write("No genes/proteins in this cluster")
                            
                            with tab_drugs:
                                drugs_df = get_cluster_drugs(kg_info['cluster'], kg_data)
                                if drugs_df is not None:
                                    st.dataframe(drugs_df, use_container_width=True, hide_index=True)
                                else:
                                    st.write("No drugs in this cluster")
                            
                            with tab_diseases:
                                diseases_df = get_cluster_diseases(kg_info['cluster'], kg_data)
                                if diseases_df is not None:
                                    st.dataframe(diseases_df, use_container_width=True, hide_index=True)
                                else:
                                    st.write("No diseases in this cluster")
                        else:
                            st.warning(f"⚠ '{selected_gene}' not found in MASH subgraph")
                    else:
                        st.warning("⚠ Knowledge graph data not loaded")

# ============================================================================
# PAGE 2: WGCNA MODULES
# ============================================================================

elif page == "WGCNA Modules":
    st.markdown("## WGCNA Co-expression Modules")
    st.info("WGCNA module analysis coming soon...")

# ============================================================================
# PAGE 3: KNOWLEDGE GRAPH
# ============================================================================

elif page == "Knowledge Graph":
    st.markdown("## Knowledge Graph Browser")
    st.info("Knowledge graph browser coming soon...")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 11px;'><p>Meta Liver v4 - Lollipop plot + Concordance scatter</p></div>", unsafe_allow_html=True)
