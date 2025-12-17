"""
Meta Liver - Interactive Streamlit App for Liver Genomics Analysis
Two-tab interface: Single-Omics Evidence | MAFLD Knowledge Graph
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from robust_data_loader import load_single_omics_studies, load_kg_data, load_ppi_data
from kg_analysis import get_gene_kg_info, get_cluster_genes, get_cluster_drugs, get_cluster_diseases, interpret_centrality
from wgcna_ppi_analysis import load_wgcna_module_data, get_gene_module, get_module_genes, get_coexpressed_partners, find_ppi_interactors, get_network_stats

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

# ============================================================================
# ROBUST GENE MATCHING
# ============================================================================

def find_gene_in_study(gene_name, study_df):
    """
    Find gene in study dataframe with robust matching.
    Returns (matched_row, gene_col_name) or (None, None) if not found.
    Exact match preferred over substring match.
    """
    
    # Determine gene column name
    gene_col = None
    if 'Gene' in study_df.columns:
        gene_col = 'Gene'
    elif 'gene' in study_df.columns:
        gene_col = 'gene'
    else:
        return None, None
    
    # Try exact match first (case-insensitive)
    exact_match = study_df[study_df[gene_col].str.lower() == gene_name.lower()]
    if not exact_match.empty:
        return exact_match.iloc[0], gene_col
    
    # Fall back to substring match only if no exact match
    substring_match = study_df[study_df[gene_col].str.contains(gene_name, case=False, na=False)]
    if not substring_match.empty:
        return substring_match.iloc[0], gene_col
    
    return None, None


def extract_metrics_from_row(row):
    """Extract AUC, logFC, and direction from a row"""
    
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
    
    # Determine direction (unified logic)
    direction = None
    if 'direction' in row.index:
        dir_val = str(row['direction']).lower()
        if 'nash' in dir_val or 'nafld' in dir_val or 'mafld' in dir_val:
            direction = 'MAFLD'
        elif 'healthy' in dir_val or 'control' in dir_val or 'chow' in dir_val:
            direction = 'Healthy'
    else:
        # Infer from logFC if direction column missing
        if lfc is not None:
            direction = 'MAFLD' if lfc > 0 else 'Healthy'
    
    return auc, lfc, direction


# ============================================================================
# CONSISTENCY SCORING (UNIFIED)
# ============================================================================

def compute_consistency_score(gene_name, studies_data):
    """
    Compute consistency score with unified direction logic.
    Direction is determined by explicit 'direction' column OR inferred from logFC.
    """
    
    auc_values = []
    directions = []
    found_count = 0
    
    for study_name, df in studies_data.items():
        row, gene_col = find_gene_in_study(gene_name, df)
        
        if row is None:
            continue
        
        found_count += 1
        auc, lfc, direction = extract_metrics_from_row(row)
        
        if auc is not None:
            auc_values.append(auc)
        
        if direction is not None:
            directions.append(direction)
    
    if found_count == 0:
        return None
    
    # AUROC Consistency: How stable are the AUC values?
    auc_consistency = 1 - (np.std(auc_values) / np.mean(auc_values)) if len(auc_values) > 1 else 1.0
    auc_consistency = max(0, min(1, auc_consistency))
    
    # Direction Consistency: How often is the direction the same?
    direction_agreement = max(directions.count('MAFLD'), directions.count('Healthy')) / len(directions) if directions else 0
    
    # Conditional interpretation based on both dimensions
    if auc_consistency > 0.7 and direction_agreement > 0.7:
        interpretation = "Highly consistent: stable AUROC, consistent direction"
    elif auc_consistency > 0.7 and direction_agreement <= 0.7:
        interpretation = "Stable AUROC, mixed direction"
    elif auc_consistency <= 0.7 and direction_agreement > 0.7:
        interpretation = "Variable AUROC, consistent direction"
    else:
        interpretation = "Variable AUROC, mixed direction"
    
    return {
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
        row, gene_col = find_gene_in_study(gene_name, df)
        
        if row is None:
            continue
        
        auc, lfc, direction = extract_metrics_from_row(row)
        
        if auc is None:
            continue
        
        # Determine marker symbol
        if direction == 'MAFLD':
            symbol = 'triangle-up'
            color = '#2E86AB'
        elif direction == 'Healthy':
            symbol = 'triangle-down'
            color = '#A23B72'
        else:
            symbol = 'circle'
            color = '#999999'
        
        # Dot size based on logFC magnitude (subtle)
        size = 10 + abs(lfc if lfc else 0) * 1.5
        size = min(size, 16)  # Cap at 16
        
        plot_data.append({
            'study': study_name,
            'auc': auc,
            'lfc': lfc if lfc else 0,
            'direction': direction,
            'symbol': symbol,
            'color': color,
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
                color=item['color'],
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
        row, gene_col = find_gene_in_study(gene_name, df)
        
        if row is None:
            continue
        
        auc, lfc, direction = extract_metrics_from_row(row)
        
        if auc is None or lfc is None:
            continue
        
        # Determine marker symbol
        if direction == 'MAFLD':
            symbol = 'triangle-up'
        elif direction == 'Healthy':
            symbol = 'triangle-down'
        else:
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
        row, gene_col = find_gene_in_study(gene_name, df)
        
        if row is None:
            continue
        
        auc, lfc, direction = extract_metrics_from_row(row)
        
        results.append({
            'Study': study_name,
            'AUC': f"{auc:.3f}" if auc else "N/A",
            'logFC': f"{lfc:.3f}" if lfc else "N/A",
            'Direction': direction if direction else "Unknown"
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
    
    if wgcna_module_data:
        st.sidebar.success(f"✓ WGCNA modules loaded ({len(wgcna_module_data)} modules)")
    else:
        st.sidebar.warning("⚠ WGCNA modules not available")
    
    if ppi_data:
        st.sidebar.success(f"✓ PPI networks loaded")
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
            # Create three main tabs
            tab_omics, tab_kg, tab_coexpr = st.tabs(["Single-Omics Evidence", "MAFLD Knowledge Graph", "Co-expression and PPI Networks"])
            
            # ================================================================
            # TAB 1: SINGLE-OMICS EVIDENCE
            # ================================================================
            
            with tab_omics:
                st.markdown("""
                This tab summarises gene-level evidence across the single-omics data sets. 
                For the selected gene, we report study-specific AUROC values and differential expression 
                direction (logFC), together with an overall consistency score reflecting how reproducibly 
                the signal is observed across cohorts. Visualisations highlight between-study agreement 
                and potential outliers, and the detailed table provides the underlying per-study statistics 
                used in downstream interpretation.
                """)
                
                st.markdown("---")
                
                # Metrics: Separate AUROC and Direction Consistency
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("AUROC Consistency", f"{consistency['auc_consistency']:.1%}")
                
                with col2:
                    st.metric("Direction Agreement", f"{consistency['direction_agreement']:.1%}")
                
                with col3:
                    st.metric("Median AUC", f"{consistency['auc_median']:.3f}")
                
                with col4:
                    st.metric("Studies Found", f"{consistency['found_count']}")
                
                # Conditional Interpretation
                st.info(f"📊 **{consistency['interpretation']}**")
                
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
            
            # ================================================================
            # TAB 2: MAFLD KNOWLEDGE GRAPH
            # ================================================================
            
            with tab_kg:
                st.markdown("""
                This tab places the selected gene in its network context within the MAFLD MASH subgraph. 
                We report whether the gene is present in the subgraph, its assigned cluster, and centrality 
                metrics (PageRank, betweenness, eigenvector) to indicate whether it behaves as a hub or a 
                peripheral node. The cluster view lists co-clustered genes, drugs, and disease annotations, 
                enabling rapid hypothesis generation about mechanistic neighbours and therapeutically relevant 
                connections.
                """)
                
                st.markdown("---")
                
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
                        
                        # Add Eigenvector centrality
                        col4, col5, col6 = st.columns(3)
                        with col4:
                            st.metric("Eigenvector", f"{kg_info['eigen']:.4f}")
                        with col5:
                            st.write("")  # Spacer
                        with col6:
                            st.write("")  # Spacer
                        
                        st.markdown("---")
                        
                        # Show min/max context for all metrics (whole graph)
                        st.markdown("**Centrality Metrics - Whole Graph Context**")
                        
                        context_col1, context_col2, context_col3 = st.columns(3)
                        
                        with context_col1:
                            st.write(f"**PageRank**")
                            st.write(f"Min: {kg_info.get('pr_min', 0):.4f}")
                            st.write(f"Max: {kg_info.get('pr_max', 0):.4f}")
                            st.write(f"Your node: {kg_info['pagerank']:.4f}")
                            percentile = ((kg_info['pagerank'] - kg_info.get('pr_min', 0)) / 
                                        (kg_info.get('pr_max', 0) - kg_info.get('pr_min', 0) + 1e-10) * 100)
                            st.write(f"Percentile: {percentile:.1f}%")
                        
                        with context_col2:
                            st.write(f"**Betweenness**")
                            st.write(f"Min: {kg_info.get('bet_min', 0):.4f}")
                            st.write(f"Max: {kg_info.get('bet_max', 0):.4f}")
                            st.write(f"Your node: {kg_info['betweenness']:.4f}")
                            percentile = ((kg_info['betweenness'] - kg_info.get('bet_min', 0)) / 
                                        (kg_info.get('bet_max', 0) - kg_info.get('bet_min', 0) + 1e-10) * 100)
                            st.write(f"Percentile: {percentile:.1f}%")
                        
                        with context_col3:
                            st.write(f"**Eigenvector**")
                            st.write(f"Min: {kg_info.get('eigen_min', 0):.4f}")
                            st.write(f"Max: {kg_info.get('eigen_max', 0):.4f}")
                            st.write(f"Your node: {kg_info['eigen']:.4f}")
                            percentile = ((kg_info['eigen'] - kg_info.get('eigen_min', 0)) / 
                                        (kg_info.get('eigen_max', 0) - kg_info.get('eigen_min', 0) + 1e-10) * 100)
                            st.write(f"Percentile: {percentile:.1f}%")
                        
                        st.markdown("---")
                        
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
            
            # ================================================================
            # TAB 3: CO-EXPRESSION AND PPI NETWORKS
            # ================================================================
            
            with tab_coexpr:
                st.markdown("""
                This tab summarises the gene's systems-level context from WGCNA module membership 
                and protein–protein interaction (PPI) networks. We report the WGCNA module assignment 
                (and key module statistics where available) and highlight the most strongly co-expressed 
                partners to indicate shared regulation. We then place the gene in the PPI network to show 
                direct interactors and local network properties, supporting prioritisation of plausible 
                mechanisms and helping to distinguish co-expression structure from physical interaction evidence.
                """)
                
                st.markdown("---")
                
                # WGCNA Section
                st.markdown("**WGCNA Co-expression Module**")
                
                if wgcna_module_data:
                    # Get gene's module assignment
                    gene_module_info = get_gene_module(search_query, wgcna_module_data)
                    
                    if gene_module_info:
                        st.markdown(f"**Module Assignment:** {gene_module_info['module']}")
                        
                        # Get other genes in the same module
                        module_genes = get_module_genes(gene_module_info['module'], wgcna_module_data)
                        
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
                
                # PPI Section
                st.markdown("**Protein-Protein Interaction Network**")
                
                if ppi_data:
                    # Get PPI interactors
                    ppi_df = find_ppi_interactors(search_query, ppi_data)
                    
                    if ppi_df is not None:
                        # Get network stats
                        net_stats = get_network_stats(search_query, ppi_data)
                        
                        if net_stats:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Direct Interactors", net_stats['degree'])
                            with col2:
                                st.write(f"**Network Property:** {net_stats['description']}")
                        
                        st.markdown(f"Direct interaction partners of {search_query}:")
                        st.dataframe(ppi_df, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"⚠ '{search_query}' not found in PPI networks")
                else:
                    st.info("⚠ PPI network data not available")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 11px;'><p>Meta Liver - Three-tab interface: Single-Omics Evidence | MAFLD Knowledge Graph | Co-expression and PPI Networks</p></div>", unsafe_allow_html=True)
