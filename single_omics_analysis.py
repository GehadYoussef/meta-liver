"""
Single-Omics Analysis Module
Forest plots, consistency scoring, and cross-study comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional
from robust_data_loader import load_single_omics_studies

# Define the 3 studies
STUDIES = {
    'GSE212837_Human_snRNAseq': 'GSE212837\n(Human snRNA)',
    'GSE189600_Human_snRNAseq': 'GSE189600\n(Human snRNA)',
    'GSE166504_Mouse_scRNAseq': 'GSE166504\n(Mouse scRNA)',
}

# ============================================================================
# CONSISTENCY SCORING
# ============================================================================

def compute_consistency_score(gene_name: str) -> Dict:
    """
    Compute consistency score for a gene across studies
    
    Returns:
        Dict with 'score', 'auc_consistency', 'direction_agreement', 'interpretation'
    """
    
    studies_data = load_single_omics_studies()
    
    # Collect data for this gene
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
        return {
            'score': 0.0,
            'auc_median': np.nan,
            'auc_consistency': 0.0,
            'direction_agreement': 0.0,
            'n_studies': 0,
            'interpretation': 'Gene not found in studies',
            'status': 'not_found'
        }
    
    # Compute AUC consistency
    auc_median = np.median(auc_values)
    auc_std = np.std(auc_values) if len(auc_values) > 1 else 0
    
    # Consistency: how close are values to median?
    # 1.0 = all identical, 0.0 = very spread out
    auc_consistency = 1.0 - (auc_std / 0.5)  # Normalize by typical std
    auc_consistency = max(0, min(1, auc_consistency))
    
    # Direction agreement
    if len(directions) > 0:
        direction_agreement = abs(np.mean(directions))  # 0-1 scale
    else:
        direction_agreement = 0.0
    
    # Overall consistency score
    # 60% AUC consistency, 40% direction agreement
    consistency_score = 0.6 * auc_consistency + 0.4 * direction_agreement
    
    # Interpretation
    if auc_median > 0.7 and direction_agreement > 0.8:
        interpretation = "Highly consistent signal"
        status = "consistent_strong"
    elif auc_median > 0.6 and direction_agreement > 0.6:
        interpretation = "Consistent signal"
        status = "consistent"
    elif auc_median > 0.55:
        interpretation = "Weak but consistent signal"
        status = "weak_consistent"
    elif direction_agreement > 0.8:
        interpretation = "Consistent direction, variable strength"
        status = "direction_consistent"
    else:
        interpretation = "Mixed or inconsistent signal"
        status = "mixed"
    
    return {
        'score': float(consistency_score),
        'auc_median': float(auc_median),
        'auc_consistency': float(auc_consistency),
        'direction_agreement': float(direction_agreement),
        'n_studies': len(auc_values),
        'interpretation': interpretation,
        'status': status,
        'auc_values': auc_values,
        'lfc_values': lfc_values,
        'directions': directions
    }

# ============================================================================
# FOREST PLOT
# ============================================================================

def create_forest_plot(gene_name: str) -> Optional[go.Figure]:
    """Create forest plot of AUC across studies"""
    
    studies_data = load_single_omics_studies()
    
    # Collect data
    plot_data = []
    
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
            plot_data.append({
                'study': STUDIES[study_name],
                'auc': auc,
                'lfc': lfc if lfc else 0,
                'direction': '↑ MAFLD' if (lfc and lfc > 0) else ('↓ Healthy' if (lfc and lfc < 0) else 'Neutral')
            })
    
    if not plot_data:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add AUC points
    for item in plot_data:
        color = '#1f77b4' if item['lfc'] > 0 else '#ff7f0e' if item['lfc'] < 0 else '#808080'
        
        fig.add_trace(go.Scatter(
            x=[item['auc']],
            y=[item['study']],
            mode='markers',
            marker=dict(size=15, color=color),
            name=item['direction'],
            text=f"AUC: {item['auc']:.3f}<br>logFC: {item['lfc']:.3f}<br>{item['direction']}",
            hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
            showlegend=False
        ))
    
    # Add median line
    median_auc = np.median([d['auc'] for d in plot_data])
    fig.add_vline(x=median_auc, line_dash="dash", line_color="red", 
                  annotation_text=f"Median: {median_auc:.3f}")
    
    # Add reference line at 0.5 (random)
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
# DETAILED RESULTS TABLE
# ============================================================================

def create_results_table(gene_name: str) -> Optional[pd.DataFrame]:
    """Create detailed results table"""
    
    studies_data = load_single_omics_studies()
    
    results = []
    
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
        
        # Direction
        if lfc:
            direction = "↑ MAFLD" if lfc > 0 else "↓ Healthy"
        else:
            direction = "Unknown"
        
        results.append({
            'Study': STUDIES[study_name].replace('\n', ' '),
            'AUC': f"{auc:.3f}" if auc else "N/A",
            'logFC': f"{lfc:.3f}" if lfc else "N/A",
            'Direction': direction
        })
    
    if not results:
        return None
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_single_omics_analysis(gene_name: str):
    """Render complete single-omics analysis"""
    
    st.subheader("📊 Single-Omics Analysis")
    
    # Compute consistency score
    consistency = compute_consistency_score(gene_name)
    
    if consistency['status'] == 'not_found':
        st.warning(f"Gene '{gene_name}' not found in single-omics datasets")
        return
    
    # Header with score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Consistency Score", f"{consistency['score']:.2f}", 
                 help="0-1 scale: how consistent is the signal across studies")
    
    with col2:
        st.metric("Median AUC", f"{consistency['auc_median']:.3f}",
                 help="Discrimination ability (0.5=random, 1.0=perfect)")
    
    with col3:
        st.metric("AUC Consistency", f"{consistency['auc_consistency']:.1%}",
                 help="How similar are AUC values across studies")
    
    with col4:
        st.metric("Direction Agreement", f"{consistency['direction_agreement']:.1%}",
                 help="How consistent is the up/down direction")
    
    # Interpretation
    status_colors = {
        'consistent_strong': '✅',
        'consistent': '✅',
        'weak_consistent': '⚠️',
        'direction_consistent': '⚠️',
        'mixed': '❌'
    }
    
    status_icon = status_colors.get(consistency['status'], '❓')
    
    st.info(f"{status_icon} **{consistency['interpretation']}**")
    
    # Forest plot
    st.markdown("**AUC Across Studies**")
    fig = create_forest_plot(gene_name)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not create forest plot")
    
    # Results table
    st.markdown("**Detailed Results**")
    results_df = create_results_table(gene_name)
    
    if results_df is not None:
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No results to display")
    
    # Interpretation guide
    with st.expander("📖 How to interpret these results"):
        st.markdown("""
        **AUC (Area Under Curve):**
        - 0.5 = Random discrimination (no signal)
        - 0.7-0.8 = Good discrimination
        - 0.9+ = Excellent discrimination
        
        **logFC (Log Fold Change):**
        - Positive = Gene enriched in MAFLD hepatocytes
        - Negative = Gene enriched in healthy hepatocytes
        - Magnitude = Strength of effect
        
        **Consistency Score:**
        - Combines AUC consistency (60%) + direction agreement (40%)
        - High score = Reliable, reproducible signal
        - Low score = Inconsistent or study-specific signal
        
        **Direction Agreement:**
        - 100% = All studies agree on up/down
        - 50% = Mixed results (some up, some down)
        - 0% = All studies disagree
        """)
