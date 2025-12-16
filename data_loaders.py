"""
Data loaders for Meta Liver
Loads Parquet files from meta-liver-data folder with caching
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path("meta-liver-data")

# ============================================================================
# WGCNA LOADERS
# ============================================================================

@st.cache_data
def load_wgcna_expr():
    """Load WGCNA expression matrix"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "datExpr_processed.parquet")
    except FileNotFoundError:
        st.error("Expression data not found. Run convert_data.py first.")
        return pd.DataFrame()

@st.cache_data
def load_wgcna_traits():
    """Load WGCNA traits data"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "datTraits_processed.parquet")
    except FileNotFoundError:
        st.error("Traits data not found.")
        return pd.DataFrame()

@st.cache_data
def load_wgcna_mes():
    """Load WGCNA module eigenvectors"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "MEs_processed.parquet")
    except FileNotFoundError:
        st.error("Module eigenvectors not found.")
        return pd.DataFrame()

@st.cache_data
def load_wgcna_mod_trait_cor():
    """Load module-trait correlations"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "moduleTraitCor.parquet")
    except FileNotFoundError:
        st.error("Module-trait correlations not found.")
        return pd.DataFrame()

@st.cache_data
def load_wgcna_mod_trait_pval():
    """Load module-trait p-values"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "moduleTraitPvalue.parquet")
    except FileNotFoundError:
        st.error("Module-trait p-values not found.")
        return pd.DataFrame()

@st.cache_data
def load_pathway_enrichment(module):
    """Load pathway enrichment for a specific module"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "pathway" / f"{module}_enrichment.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_module_genes(module):
    """Load genes in a specific module"""
    try:
        file_path = DATA_DIR / "wgcna" / "modules" / f"Network-nodes-{module}.txt"
        with open(file_path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

@st.cache_data
def load_gene_mapping(module):
    """Load gene ID mapping for a module"""
    try:
        return pd.read_parquet(DATA_DIR / "wgcna" / "modules" / f"Nodes-gene-id-mapping-{module}.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

# ============================================================================
# SINGLE-OMICS LOADERS
# ============================================================================

@st.cache_data
def load_auc_coassolo():
    """Load Coassolo AUC scores"""
    try:
        return pd.read_parquet(DATA_DIR / "single_omics" / "Coassolo_AUC_scores_target_genes_full_stats.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_gse210501():
    """Load GSE210501 (Mouse scRNAseq)"""
    try:
        return pd.read_parquet(DATA_DIR / "single_omics" / "GSE210501_Mouse_scRNAseq.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_gse212837():
    """Load GSE212837 (Human snRNAseq)"""
    try:
        return pd.read_parquet(DATA_DIR / "single_omics" / "GSE212837_Human_snRNAseq.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_gse189600():
    """Load GSE189600 (Human snRNAseq)"""
    try:
        return pd.read_parquet(DATA_DIR / "single_omics" / "GSE189600_Human_snRNAseq.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_active_drugs():
    """Load active drugs"""
    try:
        return pd.read_parquet(DATA_DIR / "single_omics" / "active_drugs.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

# ============================================================================
# KNOWLEDGE GRAPH LOADERS
# ============================================================================

@st.cache_data
def load_kg_nash_paths():
    """Load NASH shortest paths"""
    try:
        return pd.read_parquet(DATA_DIR / "knowledge_graphs" / "NASH_shortest_paths.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_kg_hepatic_paths():
    """Load hepatic steatosis shortest paths"""
    try:
        return pd.read_parquet(DATA_DIR / "knowledge_graphs" / "Hepatic_steatosis_shortest_paths.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_kg_mash_nodes():
    """Load MASH subgraph nodes"""
    try:
        return pd.read_parquet(DATA_DIR / "knowledge_graphs" / "MASH_subgraph_nodes.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_kg_mash_drugs():
    """Load MASH subgraph drugs"""
    try:
        return pd.read_parquet(DATA_DIR / "knowledge_graphs" / "MASH_subgraph_drugs.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

# ============================================================================
# PPI NETWORK LOADERS
# ============================================================================

@st.cache_data
def load_ppi_largest():
    """Load PPI largest component"""
    try:
        return pd.read_parquet(DATA_DIR / "ppi_networks" / "PPI_network_largest_component.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_ppi_early_mafld_centrality():
    """Load Early MAFLD centrality analysis"""
    try:
        return pd.read_parquet(DATA_DIR / "ppi_networks" / "Early_MAFLD_Network" / "Centrality_RWR_result_pvalue.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_ppi_early_mafld_proximity():
    """Load Early MAFLD drug-network proximity"""
    try:
        return pd.read_parquet(DATA_DIR / "ppi_networks" / "Early_MAFLD_Network" / "drug_network_proximity_results.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

def load_key_proteins():
    """Load key proteins in Early MAFLD network"""
    try:
        file_path = DATA_DIR / "ppi_networks" / "Early_MAFLD_Network" / "key_proteins.txt"
        with open(file_path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def load_network_edges():
    """Load network edges for key proteins"""
    try:
        file_path = DATA_DIR / "ppi_networks" / "Early_MAFLD_Network" / "network_edges_key_proteins.txt"
        with open(file_path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_modules():
    """Get list of all WGCNA modules"""
    return ["black", "brown", "cyan", "darkgreen", "darkorange", "darkturquoise",
            "grey60", "lightcyan", "lightgreen", "lightyellow", "magenta",
            "midnightblue", "orange", "purple", "saddlebrown", "skyblue", "tan", "white", "yellow"]

def get_single_omics_studies():
    """Get list of single-omics studies"""
    return {
        "Coassolo": load_auc_coassolo,
        "GSE210501 (Mouse)": load_gse210501,
        "GSE212837 (Human)": load_gse212837,
        "GSE189600 (Human)": load_gse189600,
    }

def find_gene_in_modules(gene_name):
    """Find which modules contain a gene"""
    modules_with_gene = []
    
    for module in get_modules():
        genes = load_module_genes(module)
        if gene_name in genes:
            modules_with_gene.append(module)
    
    return modules_with_gene

def get_gene_auc_across_studies(gene_name):
    """Get AUC scores for a gene across all studies"""
    results = {}
    
    # Coassolo
    df = load_auc_coassolo()
    if not df.empty and 'Gene' in df.columns:
        gene_data = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_data.empty:
            results['Coassolo'] = gene_data.iloc[0].to_dict()
    
    # GSE210501
    df = load_gse210501()
    if not df.empty and 'Gene' in df.columns:
        gene_data = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_data.empty:
            results['GSE210501'] = gene_data.iloc[0].to_dict()
    
    # GSE212837
    df = load_gse212837()
    if not df.empty and 'Gene' in df.columns:
        gene_data = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_data.empty:
            results['GSE212837'] = gene_data.iloc[0].to_dict()
    
    # GSE189600
    df = load_gse189600()
    if not df.empty and 'Gene' in df.columns:
        gene_data = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_data.empty:
            results['GSE189600'] = gene_data.iloc[0].to_dict()
    
    return results

def get_drug_targets(drug_name):
    """Get targets for a drug"""
    df = load_active_drugs()
    if df.empty or 'Name' not in df.columns:
        return {}
    
    drug_data = df[df['Name'].str.contains(drug_name, case=False, na=False)]
    if drug_data.empty:
        return {}
    
    return drug_data.iloc[0].to_dict()

def get_kg_neighbors(node_name, node_type="gene"):
    """Get knowledge graph neighbors for a node"""
    mash_nodes = load_kg_mash_nodes()
    
    if mash_nodes.empty or 'Name' not in mash_nodes.columns:
        return pd.DataFrame()
    
    # Find the node
    node_data = mash_nodes[mash_nodes['Name'].str.contains(node_name, case=False, na=False)]
    
    if node_data.empty:
        return pd.DataFrame()
    
    return node_data
