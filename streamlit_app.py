"""
Meta Liver v4 - Simplified with Inline Forest Plot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the 3 studies
STUDIES = {
    'GSE212837_Human_snRNAseq': 'GSE212837 (Human snRNA)',
    'GSE189600_Human_snRNAseq': 'GSE189600 (Human snRNA)',
    'GSE166504_Mouse_scRNAseq': 'GSE166504 (Mouse scRNA)',
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
    
    for file_path in single_omics_dir.rglob("*"):
        if file_path.suffix in ['.csv', '.parquet']:
            try:
                study_name = file_path.stem
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                studies[study_name] = df
            except:
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
        for col in ['avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
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
# FOREST PLOT
# ============================================================================

def create_forest_plot(gene_name, studies_data):
    """Create forest plot"""
    
    plot_data = []
    
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
        for col in ['avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
            if col in row.index:
                try:
                    lfc = float(row[col])
                    break
                except:
                    pass
        
        if auc is not None:
            plot_data.append({
                'study': STUDIES[study_name],
                'auc': auc,
                'lfc': lfc if lfc else 0,
                'direction': '↑ MAFLD' if (lfc and lfc > 0) else ('↓ Healthy' if (lfc and lfc < 0) else 'Neutral')
            })
    
    if not plot_data:
        return None
    
    fig = go.Figure()
    
    for item in plot_data:
        color = '#1f77b4' if item['lfc'] > 0 else '#ff7f0e' if item['lfc'] < 0 else '#808080'
        
        fig.add_trace(go.Scatter(
            x=[item['auc']],
            y=[item['study']],
            mode='markers',
            marker=dict(size=15, color=color),
            text=f"AUC: {item['auc']:.3f}<br>logFC: {item['lfc']:.3f}<br>{item['direction']}",
            hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
            showlegend=False
        ))
    
    median_auc = np.median([d['auc'] for d in plot_data])
    fig.add_vline(x=median_auc, line_dash="dash", line_color="red", 
                  annotation_text=f"Median: {median_auc:.3f}")
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", 
                  annotation_text="Random (0.5)")
    
    fig.update_layout(
        title=f"AUC Forest Plot: {gene_name}",
        xaxis_title="AUC (Discrimination Ability)",
        yaxis_title="Study",
        height=300,
        hovermode='closest',
        xaxis=dict(range=[0.4, 1.0]),
        showlegend=False
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
        for col in ['avg_LFC', 'logFC', 'log2FC', 'avg_log2FC']:
            if col in row.index:
                try:
                    lfc = float(row[col])
                    break
                except:
                    pass
        
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

if not search_query:
    st.title("🔬 Meta Liver v4")
    st.markdown("*Hypothesis Engine for Liver Genomics*")
    
    st.markdown("""
    ## Single-Omics Analysis
    
    Search for a gene to see:
    - **Consistency Score** - How consistent is the signal?
    - **Forest Plot** - AUC across 3 studies
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
                st.metric("Direction Agreement", f"{consistency['direction_agreement']:.1%}")
            
            # Interpretation
            st.info(f"✅ **{consistency['interpretation']}**")
            
            # Forest plot
            st.markdown("**AUC Across Studies**")
            fig = create_forest_plot(search_query, studies_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.markdown("**Detailed Results**")
            results_df = create_results_table(search_query, studies_data)
            if results_df is not None:
                st.dataframe(results_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 11px;'><p>Meta Liver v4</p></div>", unsafe_allow_html=True)
