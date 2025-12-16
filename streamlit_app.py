"""
Meta Liver - Hypothesis Engine
Query-driven analysis tool for liver genomics research
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import custom modules
from evidence_scoring import compute_evidence_fingerprint, get_intent_description
from gene_dossier import render_gene_dossier
from mechanistic_paths import render_mechanistic_paths, render_ego_network
from data_loaders import get_modules, load_kg_mash_nodes, load_kg_mash_drugs

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
# SIDEBAR: SEARCH & INTENT
# ============================================================================

st.sidebar.title("🔬 Meta Liver")
st.sidebar.markdown("*Hypothesis Engine for Liver Genomics*")
st.sidebar.markdown("---")

# Search box
search_query = st.sidebar.text_input(
    "🔍 Search Gene or Drug",
    placeholder="e.g., TP53, APOB, Metformin...",
    help="Enter a gene name, Ensembl ID, or drug name"
)

# Intent selector
st.sidebar.markdown("### Research Intent")
intent = st.sidebar.radio(
    "What are you looking for?",
    options=["hepatocyte", "fibrosis", "repurposing", "custom"],
    format_func=lambda x: {
        "hepatocyte": "🧬 Hepatocyte Drug Targets",
        "fibrosis": "🔥 Fibrosis/NASH Biology",
        "repurposing": "💊 Mechanistic Repurposing",
        "custom": "⚙️ Custom Weighting"
    }.get(x, x)
)

# Custom weights
if intent == "custom":
    st.sidebar.markdown("### Custom Weights")
    w_hep = st.sidebar.slider("Hepatocyte Score", 0.0, 1.0, 0.5)
    w_microenv = st.sidebar.slider("Microenv Score", 0.0, 1.0, 0.3)
    w_kg = st.sidebar.slider("KG Prior", 0.0, 1.0, 0.2)
    
    # Normalize
    total = w_hep + w_microenv + w_kg
    if total > 0:
        w_hep /= total
        w_microenv /= total
        w_kg /= total
    
    st.sidebar.info(f"""
    **Normalized Weights:**
    - Hepatocyte: {w_hep:.1%}
    - Microenv: {w_microenv:.1%}
    - KG Prior: {w_kg:.1%}
    """)

st.sidebar.markdown("---")

# Intent description
st.sidebar.markdown(f"**{get_intent_description(intent)}**")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not search_query:
    # Landing page
    st.title("🔬 Meta Liver")
    st.markdown("*Hypothesis Engine for Liver Genomics Research*")
    
    st.markdown("""
    ## Welcome to Meta Liver
    
    Meta Liver is a research tool that turns your multi-omics data into actionable hypotheses.
    
    ### How It Works
    
    1. **Enter a gene or drug name** in the search box (left sidebar)
    2. **Choose your research intent** (hepatocyte targets, fibrosis biology, or drug repurposing)
    3. **Get a comprehensive one-page report** with:
       - Evidence fingerprint (hepatocyte + microenvironment signals)
       - WGCNA module context and co-expression
       - Cross-study AUC comparison
       - Mechanistic paths to drugs
       - Knowledge graph neighborhood
    
    ### Key Features
    
    **🧬 Evidence-Based Scoring**
    - Hepatocyte score: Cross-study AUC robustness + direction agreement + logFC strength
    - Microenvironment score: WGCNA hubness + module-trait correlation
    - Knowledge graph prior: Network centrality + disease relevance
    
    **📊 Transparent Reporting**
    - Side-by-side evidence comparison
    - Discordance flags (hepatocyte-high vs microenv-high)
    - Taxonomy labels (hepatocyte driver, microenv hub, broad node, discordant)
    
    **🔗 Mechanistic Explanations**
    - Shortest paths from genes to drugs
    - Plain-language mechanistic routes
    - Network neighborhood exploration
    
    **🎯 Intent-Based Weighting**
    - Hepatocyte targets: Prioritize cell-intrinsic signal
    - Fibrosis/NASH: Prioritize microenvironment signal
    - Repurposing: Balance all signals with KG as guide
    - Custom: Set your own weights
    
    ### Data Sources
    
    - **WGCNA**: 19 co-expression modules from 201 samples × 14,131 genes
    - **Single-Cell Omics**: 4 studies (GSE210501, GSE212837, GSE189600, Coassolo)
    - **Knowledge Graphs**: NASH/hepatic steatosis shortest paths, MASH subgraph (13,544 nodes)
    - **PPI Networks**: Protein interactions + Early MAFLD disease network
    
    ### Getting Started
    
    Try searching for:
    - **TP53** - A well-characterized tumor suppressor
    - **APOB** - Key apolipoprotein in lipid metabolism
    - **IL6** - Inflammatory cytokine
    - **Metformin** - Common NASH therapeutic
    
    ---
    
    **👈 Use the search box to get started!**
    """)

else:
    # Gene/Drug report
    st.title(f"🔬 {search_query}")
    
    # Render dossier
    try:
        fingerprint = render_gene_dossier(search_query, intent=intent)
        
        # Mechanistic paths
        st.markdown("---")
        render_mechanistic_paths(search_query)
        
        # Network neighborhood
        st.markdown("---")
        render_ego_network(search_query)
        
    except Exception as e:
        st.error(f"Error generating report: {e}")
        st.info(f"Gene/drug '{search_query}' may not be in the database. Try a different name.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 11px;'>
    <p>Meta Liver v2.0 - Hypothesis Engine for Liver Genomics</p>
    <p>Built with evidence-based scoring and mechanistic path explanations</p>
</div>
""", unsafe_allow_html=True)
