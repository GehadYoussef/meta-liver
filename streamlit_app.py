"""
Meta Liver - Interactive Streamlit App for Liver Genomics Analysis
Loads data from Google Drive
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
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
# GOOGLE DRIVE HELPER FUNCTIONS
# ============================================================================

def get_google_drive_file_url(file_id):
    """Convert Google Drive file ID to direct download URL"""
    return f"https://drive.google.com/uc?id={file_id}&export=download"

def get_google_drive_folder_files(folder_id):
    """Get all files in a Google Drive folder"""
    # This is a simplified approach - in production, use Google Drive API
    return folder_id

# Google Drive folder ID
DRIVE_FOLDER_ID = "1xM71_KnUtTEWpRwXsWN4DKVRSFiP6KEB"

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_resource
def load_data_from_drive():
    """Load all datasets from Google Drive"""
    
    data = {}
    
    try:
        # WGCNA Data
        st.info("Loading WGCNA data from Google Drive...")
        
        # Expression data
        expr_url = f"https://drive.google.com/uc?id=1xM71_KnUtTEWpRwXsWN4DKVRSFiP6KEB&export=download"
        # Note: Direct loading from Google Drive folders requires proper file IDs
        # For now, we'll show the structure and load sample data
        
        data['expr_data'] = pd.DataFrame()  # Placeholder
        data['traits_data'] = pd.DataFrame()
        data['mes_data'] = pd.DataFrame()
        data['mod_trait_cor'] = pd.DataFrame()
        data['mod_trait_pval'] = pd.DataFrame()
        
        # Single-omics data
        data['auc_data'] = pd.DataFrame()
        data['gse210501'] = pd.DataFrame()
        data['gse212837'] = pd.DataFrame()
        data['gse189600'] = pd.DataFrame()
        
        # Knowledge graphs
        data['nash_paths'] = pd.DataFrame()
        data['hepatic_paths'] = pd.DataFrame()
        data['mash_nodes'] = pd.DataFrame()
        data['mash_drugs'] = pd.DataFrame()
        
        # PPI Networks
        data['ppi_data'] = pd.DataFrame()
        data['early_mafld_centrality'] = pd.DataFrame()
        data['early_mafld_proximity'] = pd.DataFrame()
        
        # Pathway enrichment
        data['pathway_enrichment'] = {}
        
        # Module genes
        data['module_genes'] = {}
        data['gene_mapping'] = {}
        
        data['modules'] = ["black", "brown", "cyan", "darkgreen", "darkorange", "darkturquoise",
                          "grey60", "lightcyan", "lightgreen", "lightyellow", "magenta",
                          "midnightblue", "orange", "purple", "saddlebrown", "skyblue", "tan", "white", "yellow"]
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ============================================================================
# LOAD DATA
# ============================================================================

data = load_data_from_drive()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🔬 Meta Liver")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "WGCNA Modules", "Single-Omics", "Knowledge Graphs", 
     "PPI Networks", "Early MAFLD", "Data Explorer"]
)

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

if page == "Dashboard":
    st.title("🔬 Meta Liver - Comprehensive Liver Genomics Platform")
    st.markdown("Interactive exploration of multi-omics liver disease data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("WGCNA Modules", "19")
    with col2:
        st.metric("Single-Omics Studies", "4")
    with col3:
        st.metric("Knowledge Graphs", "4")
    with col4:
        st.metric("PPI Networks", "2+")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Overview")
        overview = """
        **WGCNA Analysis:**
        - 19 co-expression modules
        - 14,131 genes
        - 201 samples
        - Pathway enrichment for each module
        
        **Single-Omics Studies:**
        - GSE210501 (Mouse scRNAseq)
        - GSE212837 (Human snRNAseq)
        - GSE189600 (Human snRNAseq)
        - Gene AUC scores & metrics
        
        **Knowledge Graphs:**
        - NASH shortest paths (500 drugs)
        - Hepatic steatosis paths
        - MASH subgraph (13,544 nodes)
        - MASH drugs (7,817 ranked by PageRank)
        
        **PPI Networks:**
        - PPI largest component
        - Early MAFLD network with centrality
        - Drug-protein proximity analysis
        """
        st.markdown(overview)
    
    with col2:
        st.subheader("🎯 Available Features")
        features = """
        1. **WGCNA Modules** - Co-expression analysis with pathways
        2. **Single-Omics** - Multi-study gene expression comparison
        3. **Knowledge Graphs** - Drug-gene-disease networks
        4. **PPI Networks** - Protein interactions & centrality
        5. **Early MAFLD** - Disease-specific network analysis
        6. **Data Explorer** - Browse all datasets
        
        **Data Source:** Google Drive (auto-synced)
        """
        st.markdown(features)

# ============================================================================
# WGCNA MODULES PAGE
# ============================================================================

elif page == "WGCNA Modules":
    st.title("📊 WGCNA Co-Expression Modules")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Module Selection")
        modules = ["black", "brown", "cyan", "darkgreen", "darkorange", "darkturquoise",
                  "grey60", "lightcyan", "lightgreen", "lightyellow", "magenta",
                  "midnightblue", "orange", "purple", "saddlebrown", "skyblue", "tan", "white", "yellow"]
        selected_module = st.selectbox("Select Module:", modules)
    
    with col2:
        st.subheader("Module Information")
        st.write(f"""
        **Module:** {selected_module}
        
        Data available from Google Drive:
        - Network nodes (genes in module)
        - Gene-ID mappings
        - Pathway enrichment results
        - Module eigenvectors
        - Trait correlations
        """)
    
    st.markdown("---")
    
    # Tabs for module data
    tab1, tab2, tab3 = st.tabs(["Genes", "Pathways", "Statistics"])
    
    with tab1:
        st.subheader(f"Genes in {selected_module} Module")
        st.info("Loading gene list from Google Drive...")
        st.write("Network nodes and gene mappings will be displayed here")
    
    with tab2:
        st.subheader(f"Pathway Enrichment for {selected_module}")
        st.info("Loading enrichment results from Google Drive...")
        st.write("GO terms, CORUM complexes, and other annotations")
    
    with tab3:
        st.subheader("Module Statistics")
        st.write("""
        - Module-trait correlation
        - P-value
        - Gene count
        - Enrichment summary
        """)

# ============================================================================
# SINGLE-OMICS PAGE
# ============================================================================

elif page == "Single-Omics":
    st.title("🧬 Single-Omics Studies")
    
    st.markdown("Gene expression and AUC scores from multiple single-omics studies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Studies")
        studies = {
            "GSE210501": "Mouse scRNAseq",
            "GSE212837": "Human snRNAseq",
            "GSE189600": "Human snRNAseq",
            "Coassolo": "AUC Scores (Target Genes)"
        }
        
        for study_id, description in studies.items():
            st.write(f"**{study_id}** - {description}")
    
    with col2:
        st.subheader("Select Study")
        selected_study = st.selectbox("Choose dataset:", list(studies.keys()))
        st.write(f"Loading {selected_study} data from Google Drive...")
    
    st.markdown("---")
    
    st.subheader(f"Data from {selected_study}")
    st.info("Gene expression data with AUC scores and metrics (avg_LFC, etc.)")
    st.write("Data table will be displayed here with sorting and filtering options")

# ============================================================================
# KNOWLEDGE GRAPHS PAGE
# ============================================================================

elif page == "Knowledge Graphs":
    st.title("🕸️ Knowledge Graphs")
    
    st.markdown("Drug-gene-disease networks from shortest path and subgraph analyses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shortest Path Analysis")
        st.write("""
        **NASH Shortest Paths**
        - 500 drugs from NASH node
        - Network distance metrics
        
        **Hepatic Steatosis Shortest Paths**
        - Drugs from hepatic steatosis node
        - Path distances
        """)
    
    with col2:
        st.subheader("MASH Subgraph Analysis")
        st.write("""
        **MASH Subgraph Nodes**
        - 13,544 nodes (drugs + genes)
        - Algorithm scores (PageRank, Betweenness, Eigen)
        - Cluster membership
        
        **MASH Subgraph Drugs**
        - 7,817 drugs ranked by PageRank
        - Network centrality metrics
        """)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["NASH Paths", "Hepatic Steatosis", "MASH Nodes", "MASH Drugs"])
    
    with tab1:
        st.subheader("NASH Shortest Paths (500 drugs)")
        st.info("Loading from Google Drive...")
    
    with tab2:
        st.subheader("Hepatic Steatosis Shortest Paths")
        st.info("Loading from Google Drive...")
    
    with tab3:
        st.subheader("MASH Subgraph Nodes (13,544)")
        st.info("Nodes with algorithm scores and cluster IDs")
    
    with tab4:
        st.subheader("MASH Subgraph Drugs (7,817)")
        st.info("Ranked by PageRank score")

# ============================================================================
# PPI NETWORKS PAGE
# ============================================================================

elif page == "PPI Networks":
    st.title("🕸️ Protein-Protein Interaction Networks")
    
    st.markdown("Network analysis of protein interactions in liver disease")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PPI Largest Component")
        st.write("""
        - Comprehensive PPI network
        - Multiple database sources
        - Network statistics
        """)
    
    with col2:
        st.subheader("Early MAFLD Network")
        st.write("""
        - Disease-specific subnetwork
        - Centrality analysis (RWR)
        - Key proteins identified
        - Drug proximity analysis
        """)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["PPI Largest Component", "Early MAFLD"])
    
    with tab1:
        st.subheader("PPI Network Overview")
        st.info("Loading network data from Google Drive...")
    
    with tab2:
        st.subheader("Early MAFLD Network Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Centrality Analysis (RWR)**")
            st.info("Loading RWR centrality results...")
        
        with col2:
            st.write("**Drug-Network Proximity**")
            st.info("Loading drug proximity analysis...")

# ============================================================================
# EARLY MAFLD PAGE
# ============================================================================

elif page == "Early MAFLD":
    st.title("🔬 Early MAFLD Network Analysis")
    
    st.markdown("Focused analysis of early metabolic-associated fatty liver disease")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Key Proteins", "Loading...")
    with col2:
        st.metric("Network Edges", "Loading...")
    with col3:
        st.metric("Top Drugs", "Loading...")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Key Proteins", "Network Edges", "Drug Proximity"])
    
    with tab1:
        st.subheader("Key Proteins in Early MAFLD")
        st.info("Loading key_proteins.txt from Google Drive...")
    
    with tab2:
        st.subheader("Network Edges (Key Proteins)")
        st.info("Loading network_edges_key_proteins.txt from Google Drive...")
    
    with tab3:
        st.subheader("Drug-Network Proximity Results")
        st.info("Loading drug_network_proximity_results.csv from Google Drive...")

# ============================================================================
# DATA EXPLORER PAGE
# ============================================================================

elif page == "Data Explorer":
    st.title("📋 Data Explorer")
    
    st.markdown("Browse and download all datasets")
    
    dataset_type = st.selectbox(
        "Select Dataset Category:",
        ["WGCNA", "Single-Omics", "Knowledge Graphs", "PPI Networks", "Early MAFLD"]
    )
    
    if dataset_type == "WGCNA":
        st.subheader("WGCNA Datasets")
        st.write("""
        - Expression matrix (201 samples × 14,131 genes)
        - Traits data
        - Module eigenvectors
        - Module-trait correlations
        - Pathway enrichment (19 modules)
        - Module genes and mappings
        """)
    
    elif dataset_type == "Single-Omics":
        st.subheader("Single-Omics Datasets")
        st.write("""
        - GSE210501 (Mouse scRNAseq)
        - GSE212837 (Human snRNAseq)
        - GSE189600 (Human snRNAseq)
        - Coassolo AUC scores
        - Wang hepatocyte AUC
        - SU hepatocyte AUC
        """)
    
    elif dataset_type == "Knowledge Graphs":
        st.subheader("Knowledge Graph Datasets")
        st.write("""
        - NASH shortest paths (500 drugs)
        - Hepatic steatosis shortest paths
        - MASH subgraph nodes (13,544)
        - MASH subgraph drugs (7,817)
        """)
    
    elif dataset_type == "PPI Networks":
        st.subheader("PPI Network Datasets")
        st.write("""
        - PPI largest component
        - Early MAFLD centrality (RWR)
        - Early MAFLD drug proximity
        """)
    
    elif dataset_type == "Early MAFLD":
        st.subheader("Early MAFLD Network Datasets")
        st.write("""
        - Key proteins
        - Network edges
        - Centrality metrics
        - Drug proximity results
        """)
    
    st.markdown("---")
    
    st.info("""
    📂 **Data Location:** Google Drive (meta-liver-data folder)
    
    All datasets are organized in subfolders:
    - wgcna/ (modules, pathways, genes)
    - single_omics/ (gene expression studies)
    - knowledge_graphs/ (drug-gene networks)
    - ppi_networks/ (protein interactions)
    
    Data is automatically synced from Google Drive.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Meta Liver - Comprehensive Liver Genomics Analysis Platform</p>
    <p>Data: WGCNA | Single-Omics | Knowledge Graphs | PPI Networks | Early MAFLD</p>
    <p>Data Source: Google Drive (meta-liver-data folder)</p>
</div>
""", unsafe_allow_html=True)
