"""
Google Drive Data Loader for Meta Liver
Handles loading data from Google Drive folders
"""

import pandas as pd
import requests
from io import BytesIO, StringIO
import streamlit as st

# Google Drive folder ID for meta-liver-data
DRIVE_FOLDER_ID = "1xM71_KnUtTEWpRwXsWN4DKVRSFiP6KEB"

# File IDs mapping (you'll need to extract these from your Google Drive)
# Format: "file_name": "file_id"
FILE_IDS = {
    # WGCNA data
    "datExpr_processed.csv": "REPLACE_WITH_FILE_ID",
    "datTraits_processed.csv": "REPLACE_WITH_FILE_ID",
    "MEs_processed.csv": "REPLACE_WITH_FILE_ID",
    "moduleTraitCor.csv": "REPLACE_WITH_FILE_ID",
    "moduleTraitPvalue.csv": "REPLACE_WITH_FILE_ID",
    
    # Pathway enrichment (19 files)
    "black_enrichment.csv": "REPLACE_WITH_FILE_ID",
    "brown_enrichment.csv": "REPLACE_WITH_FILE_ID",
    # ... add other 17 modules
    
    # Module genes
    "Network-nodes-black.txt": "REPLACE_WITH_FILE_ID",
    "Nodes-gene-id-mapping-black.csv": "REPLACE_WITH_FILE_ID",
    # ... add other 18 modules
    
    # Single-omics
    "Coassolo_AUC_scores_target_genes_full_stats.csv": "REPLACE_WITH_FILE_ID",
    "GSE210501_Mouse_scRNAseq.csv": "REPLACE_WITH_FILE_ID",
    "GSE212837_Human_snRNAseq.csv": "REPLACE_WITH_FILE_ID",
    "GSE189600_Human_snRNAseq.csv": "REPLACE_WITH_FILE_ID",
    "active_drugs.xlsx": "REPLACE_WITH_FILE_ID",
    
    # Knowledge graphs
    "NASH_shortest_paths.csv": "REPLACE_WITH_FILE_ID",
    "Hepatic_steatosis_shortest_paths.csv": "REPLACE_WITH_FILE_ID",
    "MASH_subgraph_nodes.csv": "REPLACE_WITH_FILE_ID",
    "MASH_subgraph_drugs.csv": "REPLACE_WITH_FILE_ID",
    
    # PPI networks
    "PPI_network_largest_component.csv": "REPLACE_WITH_FILE_ID",
    "Centrality_RWR_result_pvalue.csv": "REPLACE_WITH_FILE_ID",
    "drug_network_proximity_results.csv": "REPLACE_WITH_FILE_ID",
    "key_proteins.txt": "REPLACE_WITH_FILE_ID",
    "network_edges_key_proteins.txt": "REPLACE_WITH_FILE_ID",
}

def get_file_from_drive(file_id):
    """
    Download a file from Google Drive using file ID
    
    Args:
        file_id (str): Google Drive file ID
        
    Returns:
        BytesIO or StringIO object with file content
    """
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Error downloading file {file_id}: {e}")
        return None

def load_csv_from_drive(file_id):
    """Load CSV file from Google Drive"""
    file_obj = get_file_from_drive(file_id)
    if file_obj:
        return pd.read_csv(file_obj)
    return None

def load_excel_from_drive(file_id):
    """Load Excel file from Google Drive"""
    file_obj = get_file_from_drive(file_id)
    if file_obj:
        return pd.read_excel(file_obj)
    return None

def load_txt_from_drive(file_id):
    """Load text file from Google Drive"""
    file_obj = get_file_from_drive(file_id)
    if file_obj:
        content = file_obj.read().decode('utf-8')
        return [line.strip() for line in content.split('\n') if line.strip()]
    return None

@st.cache_resource
def load_all_data():
    """Load all data from Google Drive"""
    
    data = {}
    
    # Load WGCNA data
    st.info("Loading WGCNA data...")
    data['expr_data'] = load_csv_from_drive(FILE_IDS.get("datExpr_processed.csv"))
    data['traits_data'] = load_csv_from_drive(FILE_IDS.get("datTraits_processed.csv"))
    data['mes_data'] = load_csv_from_drive(FILE_IDS.get("MEs_processed.csv"))
    data['mod_trait_cor'] = load_csv_from_drive(FILE_IDS.get("moduleTraitCor.csv"))
    data['mod_trait_pval'] = load_csv_from_drive(FILE_IDS.get("moduleTraitPvalue.csv"))
    
    # Load single-omics data
    st.info("Loading single-omics data...")
    data['auc_coassolo'] = load_csv_from_drive(FILE_IDS.get("Coassolo_AUC_scores_target_genes_full_stats.csv"))
    data['gse210501'] = load_csv_from_drive(FILE_IDS.get("GSE210501_Mouse_scRNAseq.csv"))
    data['gse212837'] = load_csv_from_drive(FILE_IDS.get("GSE212837_Human_snRNAseq.csv"))
    data['gse189600'] = load_csv_from_drive(FILE_IDS.get("GSE189600_Human_snRNAseq.csv"))
    data['active_drugs'] = load_excel_from_drive(FILE_IDS.get("active_drugs.xlsx"))
    
    # Load knowledge graphs
    st.info("Loading knowledge graphs...")
    data['nash_paths'] = load_csv_from_drive(FILE_IDS.get("NASH_shortest_paths.csv"))
    data['hepatic_paths'] = load_csv_from_drive(FILE_IDS.get("Hepatic_steatosis_shortest_paths.csv"))
    data['mash_nodes'] = load_csv_from_drive(FILE_IDS.get("MASH_subgraph_nodes.csv"))
    data['mash_drugs'] = load_csv_from_drive(FILE_IDS.get("MASH_subgraph_drugs.csv"))
    
    # Load PPI networks
    st.info("Loading PPI networks...")
    data['ppi_largest'] = load_csv_from_drive(FILE_IDS.get("PPI_network_largest_component.csv"))
    data['early_mafld_centrality'] = load_csv_from_drive(FILE_IDS.get("Centrality_RWR_result_pvalue.csv"))
    data['early_mafld_proximity'] = load_csv_from_drive(FILE_IDS.get("drug_network_proximity_results.csv"))
    data['key_proteins'] = load_txt_from_drive(FILE_IDS.get("key_proteins.txt"))
    data['network_edges'] = load_txt_from_drive(FILE_IDS.get("network_edges_key_proteins.txt"))
    
    # Load pathway enrichment (19 modules)
    st.info("Loading pathway enrichment...")
    modules = ["black", "brown", "cyan", "darkgreen", "darkorange", "darkturquoise",
               "grey60", "lightcyan", "lightgreen", "lightyellow", "magenta",
               "midnightblue", "orange", "purple", "saddlebrown", "skyblue", "tan", "white", "yellow"]
    
    data['pathway_enrichment'] = {}
    for module in modules:
        file_key = f"{module}_enrichment.csv"
        data['pathway_enrichment'][module] = load_csv_from_drive(FILE_IDS.get(file_key))
    
    # Load module genes
    st.info("Loading module genes...")
    data['module_genes'] = {}
    data['gene_mapping'] = {}
    
    for module in modules:
        nodes_key = f"Network-nodes-{module}.txt"
        mapping_key = f"Nodes-gene-id-mapping-{module}.csv"
        
        data['module_genes'][module] = load_txt_from_drive(FILE_IDS.get(nodes_key))
        data['gene_mapping'][module] = load_csv_from_drive(FILE_IDS.get(mapping_key))
    
    data['modules'] = modules
    
    return data

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

SETUP_INSTRUCTIONS = """
## How to Get File IDs from Google Drive

1. Open your Google Drive folder: https://drive.google.com/drive/folders/1xM71_KnUtTEWpRwXsWN4DKVRSFiP6KEB

2. For each file/folder, right-click and select "Get link"

3. The URL will look like:
   `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   
4. Extract the FILE_ID part and add it to the FILE_IDS dictionary in this file

5. Repeat for all files in your folder structure

## Alternative: Use Google Drive API

For production deployment, use the official Google Drive API:
1. Create a service account in Google Cloud Console
2. Share your folder with the service account email
3. Use google-auth and google-api-python-client libraries
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
