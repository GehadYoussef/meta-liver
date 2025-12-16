"""
Meta Liver v3 - Robust Hypothesis Engine
Simplified version with auto-detecting data loader
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from robust_data_loader import (
    check_data_availability, get_data_summary,
    search_gene_in_studies, search_drug_in_kg,
    load_single_omics_studies, load_kg_data,
    load_wgcna_expr, load_wgcna_mes, load_wgcna_mod_trait_cor
)

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
# SIDEBAR
# ============================================================================

st.sidebar.title("🔬 Meta Liver")
st.sidebar.markdown("*Hypothesis Engine for Liver Genomics*")
st.sidebar.markdown("---")

# Data availability check
avail = check_data_availability()

if not avail['data_dir']:
    st.sidebar.error("⚠️ Data directory not found!")
    st.sidebar.markdown("""
    Make sure `meta-liver-data` folder exists in the same directory as this app.
    """)
else:
    st.sidebar.success("✅ Data directory found")

# Search box
search_query = st.sidebar.text_input(
    "🔍 Search Gene or Drug",
    placeholder="e.g., TP53, APOB, IL6...",
    help="Enter a gene name or drug name"
)

st.sidebar.markdown("---")

# Data availability summary
with st.sidebar.expander("📊 Data Status"):
    st.markdown(get_data_summary())

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not search_query:
    # Landing page
    st.title("🔬 Meta Liver v3")
    st.markdown("*Hypothesis Engine for Liver Genomics Research*")
    
    st.markdown("""
    ## Welcome to Meta Liver
    
    Meta Liver is a research tool that helps you explore your multi-omics liver data.
    
    ### How It Works
    
    1. **Enter a gene or drug name** in the search box (left sidebar)
    2. **Get comprehensive results** showing:
       - Where the gene appears in your data
       - Expression profiles across studies
       - Network context
       - Related genes and drugs
    
    ### Available Data
    """)
    
    # Show data summary
    st.markdown(get_data_summary())
    
    st.markdown("""
    ### Getting Started
    
    Try searching for:
    - **TP53** - Tumor suppressor
    - **APOB** - Apolipoprotein B
    - **IL6** - Inflammatory cytokine
    - **TNF** - Tumor necrosis factor
    
    ---
    
    **👈 Use the search box to get started!**
    """)

else:
    # Search results
    st.title(f"🔬 {search_query}")
    
    # Search in single-omics
    gene_results = search_gene_in_studies(search_query)
    
    # Search in knowledge graphs
    drug_results = search_drug_in_kg(search_query)
    
    if gene_results or drug_results:
        
        # Gene results
        if gene_results:
            st.subheader("📊 Gene Expression Data")
            
            for study_name, df in gene_results.items():
                st.markdown(f"**{study_name}**")
                
                # Show the matching rows
                display_cols = [col for col in df.columns if col not in ['index']]
                st.dataframe(df[display_cols].head(5), use_container_width=True)
                
                # Basic statistics
                if 'AUC' in df.columns:
                    auc_values = pd.to_numeric(df['AUC'], errors='coerce')
                    st.metric("AUC", f"{auc_values.mean():.3f}")
                
                st.markdown("---")
        
        # Drug results
        if drug_results:
            st.subheader("🔗 Knowledge Graph Data")
            
            for kg_name, df in drug_results.items():
                st.markdown(f"**{kg_name}**")
                
                display_cols = [col for col in df.columns if col not in ['index']]
                st.dataframe(df[display_cols].head(5), use_container_width=True)
                
                st.markdown("---")
        
        # WGCNA context (if available)
        if avail['wgcna_expr'] and avail['wgcna_mes']:
            st.subheader("🧬 WGCNA Context")
            
            expr = load_wgcna_expr()
            mes = load_wgcna_mes()
            
            # Find gene in expression matrix
            gene_col = None
            for col in expr.columns:
                if search_query.lower() in col.lower():
                    gene_col = col
                    break
            
            if gene_col:
                st.success(f"Found in WGCNA expression matrix: {gene_col}")
                
                # Compute correlation with module eigenvectors
                gene_expr = expr[gene_col].values
                
                kme_values = {}
                for me_col in mes.columns:
                    module_me = mes[me_col].values
                    corr = np.corrcoef(gene_expr, module_me)[0, 1]
                    kme_values[me_col] = corr
                
                # Find best module
                best_module = max(kme_values, key=lambda x: abs(kme_values[x]))
                best_kme = abs(kme_values[best_module])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Module", best_module.replace('ME', '').upper())
                with col2:
                    st.metric("Module Membership (kME)", f"{best_kme:.3f}")
                with col3:
                    st.metric("Module Type", "Co-expression module")
                
                # Show kME across all modules
                kme_df = pd.DataFrame({
                    'Module': list(kme_values.keys()),
                    'kME': [abs(v) for v in kme_values.values()]
                }).sort_values('kME', ascending=False)
                
                st.markdown("**Module Membership Across All Modules**")
                st.dataframe(kme_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.info(f"Gene '{search_query}' not found in WGCNA expression matrix")
    
    else:
        st.warning(f"No results found for '{search_query}'")
        st.info("""
        The gene/drug might not be in your dataset. Try:
        - Alternative gene names or IDs
        - Checking the data status in the sidebar
        - Searching for a different gene
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 11px;'>
    <p>Meta Liver v3 - Robust Hypothesis Engine for Liver Genomics</p>
    <p>Auto-detecting data loader with graceful error handling</p>
</div>
""", unsafe_allow_html=True)
