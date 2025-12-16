"""
Meta Liver - Interactive Streamlit App for Liver Genomics Analysis
Auto-detects studies and data, no hardcoding
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from robust_data_loader import load_single_omics_studies, load_kg_data
from kg_analysis import get_gene_kg_info, get_cluster_genes, get_cluster_drugs, get_cluster_diseases, interpret_centrality

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_all_data():
    """Load all data"""
    single_omics = load_single_omics_studies()
    kg_data = load_kg_data()
    return single_omics, kg_data


try:
    single_omics_data, kg_data = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    single_omics_data = {}
    kg_data = {}

# ============================================================================
# CONSISTENCY SCORING
# ============================================================================

def compute_consistency_score(gene_name, studies_data):
    """Compute consistency score"""
    
    auc_values = []
    directions = []
    found_count = 0
    
    for study_name, df in studies_data.items():
        # Find gene (case-insensitive)
        if 'Gene' in df.columns:
            gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        elif 'gene' in df.columns:
            gene_match = df[df['gene'].str.contains(gene_name, case=False, na=False)]
        else:
            continue
        
        if gene_match.empty:
            continue
        
        found_count += 1
        row = gene_match.iloc[0]
        
        # Extract AUC
        auc = None
        for col in ['AUC', 'auc', 'AUC_score']:
            if col in row.index:
                try:
                    auc = float(row[col])
                    break
                except:
                    pass
        
        if auc is not None:
            auc_values.append(auc)
        
        # Extract direction
        if 'direction' in row.index:
            directions.append(str(row['direction']).lower())
    
    if found_count == 0:
        return None
    
    # Calculate consistency metrics
    auc_consistency = 1 - (np.std(auc_values) / np.mean(auc_values)) if auc_values else 0
    auc_consistency = max(0, min(1, auc_consistency))
    
    direction_agreement = max(directions.count('mafld'), directions.count('healthy')) / len(directions) if directions else 0
    
    overall_score = (auc_consistency * 0.6 + direction_agreement * 0.4)
    
    # Interpretation
    if overall_score > 0.8:
        interpretation = "Highly consistent signal across studies"
    elif overall_score > 0.6:
        interpretation = "Moderately consistent signal"
    elif overall_score > 0.4:
        interpretation = "Weakly consistent signal"
    else:
        interpretation = "Inconsistent signal across studies"
    
    return {
        'score': overall_score,
        'auc_values': auc_values,
        'auc_median': np.median(auc_values) if auc_values else 0,
        'auc_consistency': auc_consistency,
        'direction_agreement': direction_agreement,
        'interpretation': interpretation,
        'found_count': found_count
    }

# ============================================================================
# LOLLIPOP PLOT
# ============================================================================

def create_lollipop_plot(gene_name, studies_data):
    """Create horizontal lollipop plot with direction cues (triangle markers)"""
    
    plot_data = []
    
    for study_name, df in studies_data.items():
        
        if 'Gene' in df.columns:
            gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        elif 'gene' in df.columns:
            gene_match = df[df['gene'].str.contains(gene_name, case=False, na=False)]
        else:
            continue
        
        if gene_match.empty:
            continue
        
        row = gene_match.iloc[0]
        
        auc = None
        for col in ['AUC', 'auc', 'AUC_score']:
            if col in row.index:
                try:
                    auc = float(row[col])
                    break
                except:
                    pass
        
        lfc = None
        for col in ['avg_logFC', 'avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
            if col in row.index:
                try:
                    lfc = float(row[col])
                    break
                except:
                    pass
        
        if auc is not None:
            # Determine direction and marker symbol
            if 'direction' in row.index:
                dir_val = str(row['direction']).lower()
                if 'nash' in dir_val or 'nafld' in dir_val or 'mafld' in dir_val:
                    direction = 'MAFLD'
                    symbol = 'triangle-up'
                elif 'healthy' in dir_val or 'control' in dir_val or 'chow' in dir_val:
                    direction = 'Healthy'
                    symbol = 'triangle-down'
                else:
                    direction = 'Neutral'
                    symbol = 'circle'
            else:
                if lfc and lfc > 0:
                    direction = 'MAFLD'
                    symbol = 'triangle-up'
                elif lfc and lfc < 0:
                    direction = 'Healthy'
                    symbol = 'triangle-down'
                else:
                    direction = 'Neutral'
                    symbol = 'circle'
            
            # Dot size based on logFC magnitude (subtle)
            size = 10 + abs(lfc if lfc else 0) * 1.5
            size = min(size, 16)  # Cap at 16
            
            plot_data.append({
                'study': study_name,
                'auc': auc,
                'lfc': lfc if lfc else 0,
                'direction': direction,
                'symbol': symbol,
                'size': size
            })
    
    if not plot_data:
        return None
    
    # Sort by AUC for better visualization
    plot_data = sorted(plot_data, key=lambda x: x['auc'])
    
    fig = go.Figure()
    
    # Add lollipop lines (subtle gray)
    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[0.5, item['auc']],
            y=[item['study'], item['study']],
            mode='lines',
            line=dict(color='#cccccc', width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add dots with direction symbols
    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[item['auc']],
            y=[item['study']],
            mode='markers',
            marker=dict(
                size=item['size'],
                symbol=item['symbol'],
                color='#2E86AB' if item['direction'] == 'MAFLD' else '#A23B72' if item['direction'] == 'Healthy' else '#999999',
                line=dict(color='white', width=1)
            ),
            hovertext=f"<b>{item['study']}</b><br>AUC: {item['auc']:.3f}<br>logFC: {item['lfc']:.3f}<br>Direction: {item['direction']}",
            hoverinfo='text',
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text="AUROC Across Studies", font=dict(size=14, color='#000000')),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color='#000000')),
        height=300,
        hovermode='closest',
        xaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color='#000000', size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            tickfont=dict(color='#000000', size=11)
        ),
        showlegend=False,
        plot_bgcolor='#fafafa',
        paper_bgcolor='white'
    )
    
    return fig

# ============================================================================
# AUC vs logFC SCATTER
# ============================================================================

def create_auc_logfc_scatter(gene_name, studies_data):
    """Create AUC vs logFC scatter plot"""
    
    plot_data = []
    
    for study_name, df in studies_data.items():
        
        if 'Gene' in df.columns:
            gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        elif 'gene' in df.columns:
            gene_match = df[df['gene'].str.contains(gene_name, case=False, na=False)]
        else:
            continue
        
        if gene_match.empty:
            continue
        
        row = gene_match.iloc[0]
        
        auc = None
        for col in ['AUC', 'auc', 'AUC_score']:
            if col in row.index:
                try:
                    auc = float(row[col])
                    break
                except:
                    pass
        
        lfc = None
        for col in ['avg_logFC', 'avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
            if col in row.index:
                try:
                    lfc = float(row[col])
                    break
                except:
                    pass
        
        if auc is not None and lfc is not None:
            # Determine direction
            if 'direction' in row.index:
                dir_val = str(row['direction']).lower()
                if 'nash' in dir_val or 'nafld' in dir_val or 'mafld' in dir_val:
                    direction = 'MAFLD'
                    symbol = 'triangle-up'
                elif 'healthy' in dir_val or 'control' in dir_val or 'chow' in dir_val:
                    direction = 'Healthy'
                    symbol = 'triangle-down'
                else:
                    direction = 'Neutral'
                    symbol = 'circle'
            else:
                if lfc > 0:
                    direction = 'MAFLD'
                    symbol = 'triangle-up'
                elif lfc < 0:
                    direction = 'Healthy'
                    symbol = 'triangle-down'
                else:
                    direction = 'Neutral'
                    symbol = 'circle'
            
            plot_data.append({
                'study': study_name,
                'auc': auc,
                'lfc': lfc,
                'direction': direction,
                'symbol': symbol
            })
    
    if len(plot_data) < 2:
        return None
    
    fig = go.Figure()
    
    # Add points with direction symbols
    for item in plot_data:
        fig.add_trace(go.Scatter(
            x=[item['auc']],
            y=[item['lfc']],
            mode='markers',
            marker=dict(
                size=10,
                color='#333333',
                symbol=item['symbol'],
                line=dict(width=0)
            ),
            hovertext=f"<b>{item['study']}</b><br>AUC: {item['auc']:.3f}<br>logFC: {item['lfc']:.3f}<br>Direction: {item['direction']}",
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add reference lines only (no quadrant backgrounds)
    fig.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)
    
    fig.update_layout(
        title=dict(text="Concordance: AUC vs logFC", font=dict(size=14, color='#000000')),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color='#000000')),
        yaxis_title=dict(text="logFC (MAFLD vs Healthy)", font=dict(size=12, color='#000000')),
        height=350,
        hovermode='closest',
        xaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color='#000000', size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            tickfont=dict(color='#000000', size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#f0f0f0'
        ),
        showlegend=False,
        plot_bgcolor='#fafafa',
        paper_bgcolor='white'
    )
    
    return fig

# ============================================================================
# RESULTS TABLE
# ============================================================================

def create_results_table(gene_name, studies_data):
    """Create results table"""
    
    results = []
    
    for study_name, df in studies_data.items():
        
        if 'Gene' in df.columns:
            gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        elif 'gene' in df.columns:
            gene_match = df[df['gene'].str.contains(gene_name, case=False, na=False)]
        else:
            continue
        
        if gene_match.empty:
            continue
        
        row = gene_match.iloc[0]
        
        auc = None
        for col in ['AUC', 'auc', 'AUC_score']:
            if col in row.index:
                try:
                    auc = float(row[col])
                    break
                except:
                    pass
        
        lfc = None
        for col in ['avg_logFC', 'avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
            if col in row.index:
                try:
                    lfc = float(row[col])
                    break
                except:
                    pass
        
        direction = "Unknown"
        if 'direction' in row.index:
            direction = str(row['direction'])
        
        results.append({
            'Study': study_name,
            'AUC': f"{auc:.3f}" if auc else "N/A",
            'logFC': f"{lfc:.3f}" if lfc else "N/A",
            'Direction': direction
        })
    
    if not results:
        return None
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN APP
# ============================================================================

st.sidebar.markdown("## 🔬 Meta Liver")
search_query = st.sidebar.text_input("Search gene:", placeholder="e.g., SAA1, TP53, IL6").strip().upper()

if data_loaded:
    if single_omics_data:
        st.sidebar.success(f"✓ {len(single_omics_data)} studies loaded")
        with st.sidebar.expander("📊 Studies:"):
            for study_name in sorted(single_omics_data.keys()):
                st.write(f"• {study_name}")
    else:
        st.sidebar.warning("⚠ No single-omics studies found")
    
    if kg_data:
        st.sidebar.success(f"✓ Knowledge graph loaded")
    else:
        st.sidebar.warning("⚠ Knowledge graph not available")
else:
    st.sidebar.error("✗ Error loading data")

st.sidebar.markdown("---")

if not search_query:
    st.title("🔬 Meta Liver")
    st.markdown("*Hypothesis Engine for Liver Genomics*")
    
    st.markdown("""
    ## Single-Omics Analysis
    
    Search for a gene to see:
    - **Consistency Score** - How consistent is the signal?
    - **Forest Plot** - AUC across studies
    - **Results Table** - Detailed metrics
    
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
        consistency = compute_consistency_score(search_query, single_omics_data)
        
        if consistency is None:
            st.warning(f"Gene '{search_query}' not found in any study")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Consistency Score", f"{consistency['score']:.2f}")
            
            with col2:
                st.metric("Median AUC", f"{consistency['auc_median']:.3f}")
            
            with col3:
                st.metric("AUC Consistency", f"{consistency['auc_consistency']:.1%}")
            
            with col4:
                st.metric("Studies Found", f"{consistency['found_count']}")
            
            # Interpretation
            st.info(f"✅ **{consistency['interpretation']}**")
            
            # Lollipop plot
            st.markdown("**AUROC Across Studies**")
            fig = create_lollipop_plot(search_query, single_omics_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # AUC vs logFC scatter
            st.markdown("**Concordance: AUC vs logFC**")
            fig_scatter = create_auc_logfc_scatter(search_query, single_omics_data)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough data for concordance plot")
            
            # Results table
            st.markdown("**Detailed Results**")
            results_df = create_results_table(search_query, single_omics_data)
            if results_df is not None:
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Knowledge Graph Section
            st.markdown("---")
            st.markdown("**Knowledge Graph Context**")
            
            if kg_data:
                kg_info = get_gene_kg_info(search_query, kg_data)
                
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
                    st.warning(f"⚠ '{search_query}' not found in MASH subgraph")
            else:
                st.warning("⚠ Knowledge graph data not loaded")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 11px;'><p>Meta Liver - Auto-detecting studies and data</p></div>", unsafe_allow_html=True)
