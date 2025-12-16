"""
Meta Liver - Interactive Streamlit App for Liver Genomics Analysis
Integrates WGCNA modules, enrichment analysis, AUC scores, drugs, and PPI networks
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_resource
def load_data():
    """Load all datasets"""
    data_dir = Path("/home/ubuntu/upload")
    
    # Load main data files
    expr_data = pd.read_csv(data_dir / "datExpr_processed.csv", index_col=0)
    traits_data = pd.read_csv(data_dir / "datTraits_processed.csv", index_col=0)
    mes_data = pd.read_csv(data_dir / "MEs_processed.csv", index_col=0)
    mod_trait_cor = pd.read_csv(data_dir / "moduleTraitCor.csv", index_col=0)
    mod_trait_pval = pd.read_csv(data_dir / "moduleTraitPvalue.csv", index_col=0)
    auc_data = pd.read_csv(data_dir / "Coassolo_AUC_scores_target_genes_full_stats.csv")
    ppi_data = pd.read_csv(data_dir / "PPI_network_largest_component.csv")
    drugs_data = pd.read_excel(data_dir / "active_drugs.xlsx", sheet_name="active_drugs_approved", skiprows=3)
    
    # Load enrichment data
    modules = ["black", "brown", "cyan", "darkgreen", "darkorange", "darkturquoise",
               "grey60", "lightcyan", "lightgreen", "lightyellow", "magenta",
               "midnightblue", "orange", "purple", "saddlebrown", "skyblue", "tan", "white", "yellow"]
    
    enrichment_list = {}
    for mod in modules:
        file_path = data_dir / f"{mod}_enrichment.csv"
        if file_path.exists():
            enrichment_list[mod] = pd.read_csv(file_path)
    
    # Load network nodes
    nodes_list = {}
    for mod in modules:
        file_path = data_dir / f"Network-nodes-{mod}.txt"
        if file_path.exists():
            with open(file_path) as f:
                nodes_list[mod] = [line.strip() for line in f.readlines()]
    
    # Load gene mappings
    gene_mapping_list = {}
    for mod in modules:
        file_path = data_dir / f"Nodes-gene-id-mapping-{mod}.csv"
        if file_path.exists():
            gene_mapping_list[mod] = pd.read_csv(file_path)
    
    return {
        'expr_data': expr_data,
        'traits_data': traits_data,
        'mes_data': mes_data,
        'mod_trait_cor': mod_trait_cor,
        'mod_trait_pval': mod_trait_pval,
        'auc_data': auc_data,
        'ppi_data': ppi_data,
        'drugs_data': drugs_data,
        'enrichment_list': enrichment_list,
        'nodes_list': nodes_list,
        'gene_mapping_list': gene_mapping_list,
        'modules': modules
    }

data = load_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🔬 Meta Liver")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Module Explorer", "Gene Analysis", "Drug Discovery", "Enrichment", "PPI Network", "Data Tables"]
)

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

if page == "Dashboard":
    st.title("🔬 Meta Liver - Dashboard")
    st.markdown("Interactive platform for exploring liver genomics data")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", len(data['expr_data']))
    
    with col2:
        st.metric("Genes", len(data['expr_data'].columns))
    
    with col3:
        st.metric("Modules", len(data['modules']))
    
    with col4:
        st.metric("Drugs", len(data['drugs_data']))
    
    st.markdown("---")
    
    # Module-Trait Correlations
    st.subheader("Module-Trait Correlations (Disease Stage)")
    
    plot_data = data['mod_trait_cor'].reset_index()
    plot_data.columns = ['Module', 'Correlation']
    plot_data = plot_data.sort_values('Correlation')
    
    fig = px.bar(
        plot_data,
        x='Correlation',
        y='Module',
        orientation='h',
        color='Correlation',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        title="Module-Trait Correlations"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        overview_text = f"""
        - **Expression Data**: {len(data['expr_data'])} samples × {len(data['expr_data'].columns)} genes
        - **WGCNA Modules**: {len(data['modules'])} co-expression modules
        - **Enrichment Data**: Functional annotations (GO terms, CORUM complexes)
        - **AUC Scores**: {len(data['auc_data'])} genes with discrimination ability
        - **PPI Network**: {len(data['ppi_data'])} protein-protein interactions
        - **Drug Targets**: {len(data['drugs_data'])} approved drugs
        """
        st.markdown(overview_text)
    
    with col2:
        st.subheader("Available Features")
        features_text = """
        1. **Module Explorer**: Browse WGCNA modules and their genes
        2. **Gene Analysis**: Search genes and view expression profiles
        3. **Drug Discovery**: Find drug targets and mechanisms
        4. **Enrichment Analysis**: Functional annotation of modules
        5. **PPI Network**: Explore protein interactions
        6. **Data Tables**: Access all datasets
        """
        st.markdown(features_text)

# ============================================================================
# MODULE EXPLORER PAGE
# ============================================================================

elif page == "Module Explorer":
    st.title("📊 Module Explorer")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_module = st.selectbox("Select Module", data['modules'])
    
    with col2:
        # Module statistics
        n_genes = len(data['nodes_list'].get(selected_module, []))
        mod_name = f"ME{selected_module}"
        corr = data['mod_trait_cor'].loc[mod_name, 'stage'] if mod_name in data['mod_trait_cor'].index else 0
        pval = data['mod_trait_pval'].loc[mod_name, 'stage'] if mod_name in data['mod_trait_pval'].index else 1
        
        st.metric("Genes in Module", n_genes)
        st.metric("Module-Trait Correlation", f"{corr:.4f}", delta=f"p={pval:.4f}")
    
    st.markdown("---")
    
    # Module genes table
    st.subheader("Module Genes")
    
    if selected_module in data['gene_mapping_list'] and len(data['gene_mapping_list'][selected_module]) > 0:
        gene_df = data['gene_mapping_list'][selected_module]
        st.dataframe(gene_df, use_container_width=True, height=400)
    else:
        genes = data['nodes_list'].get(selected_module, [])
        gene_df = pd.DataFrame({'Gene': genes})
        st.dataframe(gene_df, use_container_width=True, height=400)
    
    # Enrichment summary
    if selected_module in data['enrichment_list']:
        st.subheader("Top Enriched Terms")
        enrich_df = data['enrichment_list'][selected_module].head(10)
        st.dataframe(enrich_df, use_container_width=True)

# ============================================================================
# GENE ANALYSIS PAGE
# ============================================================================

elif page == "Gene Analysis":
    st.title("🔍 Gene Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gene_search = st.text_input("Search Gene (ENSG ID or column name)", placeholder="e.g., ENSG00000000003")
    
    with col2:
        search_btn = st.button("Search", type="primary")
    
    if search_btn and gene_search:
        # Find gene in expression data
        found = False
        gene_id = None
        
        if gene_search in data['expr_data'].columns:
            gene_id = gene_search
            found = True
        elif gene_search in data['expr_data'].index:
            gene_id = gene_search
            found = True
        
        if found:
            st.success(f"✓ Gene found: {gene_id}")
            
            # Get expression values
            if gene_id in data['expr_data'].columns:
                expr_vals = data['expr_data'][gene_id].values
            else:
                expr_vals = data['expr_data'].loc[gene_id].values
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gene Information")
                st.write(f"**Gene ID**: {gene_id}")
                
                # Check AUC data
                auc_match = data['auc_data'][data['auc_data']['Gene'].str.contains(gene_search, case=False, na=False)]
                if len(auc_match) > 0:
                    st.write(f"**AUC Score**: {auc_match.iloc[0]['AUC']:.3f}")
                    st.write(f"**% Chow**: {auc_match.iloc[0]['pct_Chow']:.3f}")
                    st.write(f"**% NASH**: {auc_match.iloc[0]['pct_NASH']:.3f}")
            
            with col2:
                st.subheader("Expression Statistics")
                st.write(f"**Mean**: {expr_vals.mean():.3f}")
                st.write(f"**Std Dev**: {expr_vals.std():.3f}")
                st.write(f"**Min**: {expr_vals.min():.3f}")
                st.write(f"**Max**: {expr_vals.max():.3f}")
            
            # Expression plot
            st.subheader("Expression Profile")
            expr_df = pd.DataFrame({
                'Sample': range(len(expr_vals)),
                'Expression': expr_vals
            })
            
            fig = px.scatter(
                expr_df,
                x='Sample',
                y='Expression',
                title=f"Expression Profile: {gene_id}",
                labels={'Sample': 'Sample Index', 'Expression': 'Normalized Expression'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Gene not found in dataset")

# ============================================================================
# DRUG DISCOVERY PAGE
# ============================================================================

elif page == "Drug Discovery":
    st.title("💊 Drug Discovery")
    
    # Clean drug data
    drugs_clean = data['drugs_data'].copy()
    
    # Display drug table
    st.subheader("Approved Drugs with PPI Targets")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        if 'z-score' in drugs_clean.columns:
            z_min = st.slider("Min Z-Score", -5.0, 0.0, -2.0)
            drugs_filtered = drugs_clean[drugs_clean['z-score'] <= z_min]
        else:
            drugs_filtered = drugs_clean
    
    with col2:
        st.write(f"**Showing {len(drugs_filtered)} drugs**")
    
    st.dataframe(drugs_filtered.head(50), use_container_width=True, height=400)
    
    # Drug statistics
    st.markdown("---")
    st.subheader("Drug Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'z-score' in drugs_clean.columns:
            drugs_plot = drugs_clean.dropna(subset=['z-score']).sort_values('z-score').head(15)
            fig = px.bar(
                drugs_plot,
                x='z-score',
                y='Drug Name' if 'Drug Name' in drugs_plot.columns else drugs_plot.columns[1],
                orientation='h',
                title="Top 15 Drugs by Z-Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'distance' in drugs_clean.columns:
            drugs_plot = drugs_clean.dropna(subset=['distance']).sort_values('distance').head(15)
            fig = px.bar(
                drugs_plot,
                x='distance',
                y='Drug Name' if 'Drug Name' in drugs_plot.columns else drugs_plot.columns[1],
                orientation='h',
                title="Top 15 Drugs by Distance",
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ENRICHMENT PAGE
# ============================================================================

elif page == "Enrichment":
    st.title("🏷️ Functional Enrichment Analysis")
    
    selected_module = st.selectbox("Select Module for Enrichment", data['modules'])
    
    if selected_module in data['enrichment_list']:
        enrich_df = data['enrichment_list'][selected_module]
        
        st.subheader(f"Enrichment Results for {selected_module} Module")
        st.write(f"Total enriched terms: {len(enrich_df)}")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            n_show = st.slider("Number of terms to display", 5, 100, 20)
        
        with col2:
            if 'padj' in enrich_df.columns or 'p.adjust' in enrich_df.columns:
                pval_col = 'padj' if 'padj' in enrich_df.columns else 'p.adjust'
                p_threshold = st.number_input("P-value threshold", 0.0, 1.0, 0.05)
                enrich_filtered = enrich_df[enrich_df[pval_col] < p_threshold].head(n_show)
            else:
                enrich_filtered = enrich_df.head(n_show)
        
        st.dataframe(enrich_filtered, use_container_width=True, height=500)
    else:
        st.warning("No enrichment data available for this module")

# ============================================================================
# PPI NETWORK PAGE
# ============================================================================

elif page == "PPI Network":
    st.title("🕸️ Protein-Protein Interaction Network")
    
    st.subheader("Network Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Interactions", len(data['ppi_data']))
    
    with col2:
        unique_proteins = len(set(data['ppi_data']['prot1_hgnc_id'].unique()) | 
                             set(data['ppi_data']['prot2_hgnc_id'].unique()))
        st.metric("Unique Proteins", unique_proteins)
    
    with col3:
        avg_interactions = len(data['ppi_data']) / unique_proteins
        st.metric("Avg Interactions/Protein", f"{avg_interactions:.2f}")
    
    st.markdown("---")
    
    # Top interactions
    st.subheader("Top Protein Interactions")
    
    ppi_summary = data['ppi_data'].groupby('prot_pair').size().reset_index(name='Count')
    ppi_summary = ppi_summary.sort_values('Count', ascending=False).head(20)
    
    fig = px.bar(
        ppi_summary,
        x='Count',
        y='prot_pair',
        orientation='h',
        title="Top 20 Protein Interactions",
        labels={'Count': 'Frequency', 'prot_pair': 'Protein Pair'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Database sources
    st.subheader("Interaction Sources")
    
    if 'source_db' in data['ppi_data'].columns:
        source_counts = data['ppi_data']['source_db'].value_counts()
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Interactions by Database Source"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DATA TABLES PAGE
# ============================================================================

elif page == "Data Tables":
    st.title("📋 Data Tables")
    
    tab1, tab2, tab3, tab4 = st.tabs(["AUC Scores", "Module-Trait Correlations", "Traits", "Drugs"])
    
    with tab1:
        st.subheader("Gene AUC Scores")
        st.dataframe(data['auc_data'], use_container_width=True, height=600)
    
    with tab2:
        st.subheader("Module-Trait Correlations")
        mod_trait_display = data['mod_trait_cor'].reset_index()
        mod_trait_display.columns = ['Module', 'Correlation']
        mod_trait_display['P-Value'] = data['mod_trait_pval'].reset_index()['stage']
        st.dataframe(mod_trait_display, use_container_width=True, height=600)
    
    with tab3:
        st.subheader("Sample Traits")
        traits_display = data['traits_data'].reset_index()
        st.dataframe(traits_display, use_container_width=True, height=600)
    
    with tab4:
        st.subheader("Drug Information")
        st.dataframe(data['drugs_data'], use_container_width=True, height=600)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Meta Liver - Interactive Liver Genomics Analysis Platform</p>
    <p>Built with Streamlit | Data: WGCNA, Enrichment Analysis, PPI Networks, Drug Targets</p>
</div>
""", unsafe_allow_html=True)
