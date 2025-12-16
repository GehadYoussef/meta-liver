"""
Meta Liver v4 - Simplified with Inline Forest Plot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from kg_analysis import load_kg_data, get_gene_kg_info, get_cluster_nodes, get_top_drugs, interpret_centrality

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the 4 studies
STUDIES = {
    'GSE212837_Human_snRNAseq': 'GSE212837 (Human snRNA)',
    'GSE189600_Human_snRNAseq': 'GSE189600 (Human snRNA)',
    'GSE166504_Mouse_scRNAseq': 'GSE166504 (Mouse scRNA)',
    'GSE210501_Mouse_scRNAseq': 'GSE210501 (Mouse scRNA)',
}

# ============================================================================
# DATA LOADING
# ============================================================================

def find_data_dir():
    """Find data directory"""
    for path in [Path("meta-liver-data"), Path("meta_liver_data"), Path("data")]:
        if path.exists():
            return path
    return None

def find_file(directory, filename_pattern):
    """Find file case-insensitive"""
    if not directory.exists():
        return None
    
    for file in directory.rglob("*"):
        if file.name.lower() == filename_pattern.lower():
            return file
        if filename_pattern.lower() in file.name.lower():
            return file
    return None

def find_subfolder(parent, folder_pattern):
    """Find subfolder case-insensitive"""
    if not parent.exists():
        return None
    
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == folder_pattern.lower():
            return item
    return None

@st.cache_data
def load_studies():
    """Load all single-omics studies"""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}
    
    single_omics_dir = find_subfolder(data_dir, "single_omics")
    if single_omics_dir is None:
        return {}
    
    studies = {}
    
    # Load all parquet and csv files
    for file_path in single_omics_dir.glob("*"):
        if file_path.is_file() and file_path.suffix in ['.csv', '.parquet']:
            try:
                study_name = file_path.stem
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                if not df.empty:
                    studies[study_name] = df
            except Exception as e:
                pass
    
    return studies

# ============================================================================
# CONSISTENCY SCORING
# ============================================================================

def compute_consistency_score(gene_name, studies_data):
    """Compute consistency score"""
    
    auc_values = []
    lfc_values = []
    directions = []
    
    for study_name, df in studies_data.items():
        if study_name not in STUDIES:
            continue
        
        # Find gene
        if 'Gene' in df.columns:
            gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        elif 'gene' in df.columns:
            gene_match = df[df['gene'].str.contains(gene_name, case=False, na=False)]
        else:
            continue
        
        if gene_match.empty:
            continue
        
        row = gene_match.iloc[0]
        
        # Get AUC
        auc = None
        for col in ['AUC', 'auc', 'AUC_score']:
            if col in row.index:
                try:
                    auc = float(row[col])
                    break
                except:
                    pass
        
        # Get logFC
        lfc = None
        for col in ['avg_logFC', 'avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
            if col in row.index:
                try:
                    lfc = float(row[col])
                    break
                except:
                    pass
        
        if auc is not None:
            auc_values.append(auc)
        
        if lfc is not None:
            lfc_values.append(lfc)
            # Use direction column if available, otherwise use sign of lfc
            if 'direction' in row.index:
                dir_val = str(row['direction']).lower()
                if 'nash' in dir_val or 'nafld' in dir_val or 'mafld' in dir_val:
                    directions.append(1)
                elif 'healthy' in dir_val or 'control' in dir_val or 'chow' in dir_val:
                    directions.append(-1)
                else:
                    directions.append(np.sign(lfc))
            else:
                directions.append(np.sign(lfc))
    
    if not auc_values:
        return None
    
    # Compute metrics
    auc_median = np.median(auc_values)
    auc_std = np.std(auc_values) if len(auc_values) > 1 else 0
    
    auc_consistency = 1.0 - (auc_std / 0.5)
    auc_consistency = max(0, min(1, auc_consistency))
    
    direction_agreement = abs(np.mean(directions)) if len(directions) > 0 else 0.0
    
    consistency_score = 0.6 * auc_consistency + 0.4 * direction_agreement
    
    # Interpretation
    if auc_median > 0.7 and direction_agreement > 0.8:
        interpretation = "Highly consistent signal"
    elif auc_median > 0.6 and direction_agreement > 0.6:
        interpretation = "Consistent signal"
    elif auc_median > 0.55:
        interpretation = "Weak but consistent signal"
    elif direction_agreement > 0.8:
        interpretation = "Consistent direction, variable strength"
    else:
        interpretation = "Mixed or inconsistent signal"
    
    return {
        'score': float(consistency_score),
        'auc_median': float(auc_median),
        'auc_consistency': float(auc_consistency),
        'direction_agreement': float(direction_agreement),
        'interpretation': interpretation,
        'auc_values': auc_values,
        'lfc_values': lfc_values,
        'directions': directions
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
            
            # Get study label
            study_label = STUDIES.get(study_name, study_name)
            
            plot_data.append({
                'study': study_label,
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
                color='#333333',
                symbol=item['symbol'],
                line=dict(width=0)
            ),
            hovertext=f"<b>{item['study']}</b><br>AUC: {item['auc']:.3f}<br>logFC: {item['lfc']:.3f}<br>Direction: {item['direction']}",
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add reference lines (subtle)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.7, line_dash="dot", line_color="#aaaaaa", line_width=1)
    
    fig.update_layout(
        title=dict(text=f"AUROC Across Studies: {gene_name}", font=dict(size=14, color='#000000')),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color='#000000')),
        yaxis_title="",
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
            tickfont=dict(color='#000000', size=11),
            showgrid=False
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
    """Create AUC vs logFC scatter plot showing concordance"""
    
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
                if lfc > 0:
                    direction = 'MAFLD'
                    symbol = 'triangle-up'
                else:
                    direction = 'Healthy'
                    symbol = 'triangle-down'
            
            # Get study label
            study_label = STUDIES.get(study_name, study_name)
            
            plot_data.append({
                'study': study_label,
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
        if study_name not in STUDIES:
            continue
        
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
        
        # Determine direction
        if 'direction' in row.index:
            dir_val = str(row['direction']).lower()
            if 'nash' in dir_val or 'nafld' in dir_val or 'mafld' in dir_val:
                direction = '↑ MAFLD'
            elif 'healthy' in dir_val or 'control' in dir_val or 'chow' in dir_val:
                direction = '↓ Healthy'
            else:
                direction = 'Unknown'
        else:
            direction = "↑ MAFLD" if (lfc and lfc > 0) else ("↓ Healthy" if (lfc and lfc < 0) else "Unknown")
        
        results.append({
            'Study': STUDIES[study_name],
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

st.sidebar.title("🔬 Meta Liver")
st.sidebar.markdown("*Hypothesis Engine for Liver Genomics*")
st.sidebar.markdown("---")

search_query = st.sidebar.text_input(
    "🔍 Search Gene",
    placeholder="e.g., SAA1, TP53, IL6...",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Status**")

# Load studies
studies_data = load_studies()

if studies_data:
    st.sidebar.success(f"✓ {len(studies_data)} studies loaded")
    with st.sidebar.expander("📊 Studies:"):
        for study_name in sorted(studies_data.keys()):
            st.write(f"• {study_name}")
else:
    st.sidebar.error("✗ No studies found")

# Load knowledge graph data
data_dir = find_data_dir()
kg_data = load_kg_data(data_dir) if data_dir else {}

if kg_data:
    st.sidebar.success(f"✓ Knowledge graph loaded")
else:
    st.sidebar.warning("⚠ Knowledge graph not available")

st.sidebar.markdown("---")

if not search_query:
    st.title("🔬 Meta Liver v4")
    st.markdown("*Hypothesis Engine for Liver Genomics*")
    
    st.markdown("""
    ## Single-Omics Analysis
    
    Search for a gene to see:
    - **Consistency Score** - How consistent is the signal?
    - **Forest Plot** - AUC across 4 studies
    - **Results Table** - Detailed metrics
    
    ### Try searching for:
    - SAA1
    - TP53
    - IL6
    - TNF
    """)

else:
    st.title(f"🔬 {search_query}")
    
    studies_data = load_studies()
    
    if not studies_data:
        st.error("No studies data found!")
        st.info(f"Found {len(studies_data)} studies")
    else:
        consistency = compute_consistency_score(search_query, studies_data)
        
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
                st.metric("Studies Found", f"{len(consistency['auc_values'])}")
            
            # Interpretation
            st.info(f"✅ **{consistency['interpretation']}**")
            
            # Lollipop plot
            st.markdown("**AUROC Across Studies**")
            fig = create_lollipop_plot(search_query, studies_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # AUC vs logFC scatter
            st.markdown("**Concordance: AUC vs logFC**")
            fig_scatter = create_auc_logfc_scatter(search_query, studies_data)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough data for concordance plot")
            
            # Results table
            st.markdown("**Detailed Results**")
            results_df = create_results_table(search_query, studies_data)
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
                    
                    # Top nodes in same cluster
                    st.markdown("**Top Nodes in Cluster**")
                    cluster_nodes = get_cluster_nodes(kg_info['cluster'], kg_data, top_n=10)
                    if cluster_nodes is not None:
                        st.dataframe(cluster_nodes, use_container_width=True, hide_index=True)
                    else:
                        st.write("No other nodes in this cluster")
                else:
                    st.warning(f"⚠ '{search_query}' not found in MASH subgraph")
                
                # Top drugs in subgraph
                st.markdown("**Top Drugs in MASH Subgraph**")
                top_drugs = get_top_drugs(kg_data, top_n=10)
                if top_drugs is not None:
                    st.dataframe(top_drugs, use_container_width=True, hide_index=True)
                else:
                    st.write("No drug data available")
            else:
                st.warning("⚠ Knowledge graph data not loaded")

    st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 11px;'><p>Meta Liver v4 - Lollipop plot + Concordance scatter</p></div>", unsafe_allow_html=True)
