"""
Robust Data Loader for Meta Liver
Auto-detects files regardless of folder case and handles missing data gracefully
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AUTO-DETECT DATA DIRECTORY
# ============================================================================

def find_data_dir() -> Optional[Path]:
    """Auto-detect data directory (case-insensitive)"""
    
    # Try common locations
    possible_dirs = [
        Path("meta-liver-data"),
        Path("meta_liver_data"),
        Path("data"),
        Path("../meta-liver-data"),
    ]
    
    for dir_path in possible_dirs:
        if dir_path.exists():
            return dir_path.resolve()
    
    return None

def find_file(directory: Path, filename_pattern: str) -> Optional[Path]:
    """Find file in directory (case-insensitive)"""
    
    if not directory.exists():
        return None
    
    # Try exact match first
    exact_path = directory / filename_pattern
    if exact_path.exists():
        return exact_path
    
    # Try case-insensitive search
    for file in directory.rglob("*"):
        if file.name.lower() == filename_pattern.lower():
            return file
        if filename_pattern.lower() in file.name.lower():
            return file
    
    return None

def find_subfolder(parent: Path, folder_pattern: str) -> Optional[Path]:
    """Find subfolder (case-insensitive)"""
    
    if not parent.exists():
        return None
    
    # Try exact match
    exact_path = parent / folder_pattern
    if exact_path.exists():
        return exact_path
    
    # Try case-insensitive
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == folder_pattern.lower():
            return item
    
    return None

# ============================================================================
# CACHED DATA LOADERS
# ============================================================================

DATA_DIR = find_data_dir()

@st.cache_data
def load_wgcna_expr() -> pd.DataFrame:
    """Load WGCNA expression matrix"""
    if DATA_DIR is None:
        return pd.DataFrame()
    
    wgcna_dir = find_subfolder(DATA_DIR, "wgcna")
    if wgcna_dir is None:
        return pd.DataFrame()
    
    # Try different filenames
    for filename in ["datExpr_processed.parquet", "datExpr_processed.csv"]:
        file_path = find_file(wgcna_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    return pd.read_parquet(file_path)
                else:
                    return pd.read_csv(file_path, index_col=0)
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
    
    return pd.DataFrame()

@st.cache_data
def load_wgcna_mes() -> pd.DataFrame:
    """Load WGCNA module eigenvectors"""
    if DATA_DIR is None:
        return pd.DataFrame()
    
    wgcna_dir = find_subfolder(DATA_DIR, "wgcna")
    if wgcna_dir is None:
        return pd.DataFrame()
    
    for filename in ["MEs_processed.parquet", "MEs_processed.csv"]:
        file_path = find_file(wgcna_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    return pd.read_parquet(file_path)
                else:
                    return pd.read_csv(file_path, index_col=0)
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
    
    return pd.DataFrame()

@st.cache_data
def load_wgcna_mod_trait_cor() -> pd.DataFrame:
    """Load module-trait correlations"""
    if DATA_DIR is None:
        return pd.DataFrame()
    
    wgcna_dir = find_subfolder(DATA_DIR, "wgcna")
    if wgcna_dir is None:
        return pd.DataFrame()
    
    for filename in ["moduleTraitCor.parquet", "moduleTraitCor.csv"]:
        file_path = find_file(wgcna_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    return pd.read_parquet(file_path)
                else:
                    return pd.read_csv(file_path, index_col=0)
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
    
    return pd.DataFrame()

@st.cache_data
def load_wgcna_mod_trait_pval() -> pd.DataFrame:
    """Load module-trait p-values"""
    if DATA_DIR is None:
        return pd.DataFrame()
    
    wgcna_dir = find_subfolder(DATA_DIR, "wgcna")
    if wgcna_dir is None:
        return pd.DataFrame()
    
    for filename in ["moduleTraitPvalue.parquet", "moduleTraitPvalue.csv"]:
        file_path = find_file(wgcna_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    return pd.read_parquet(file_path)
                else:
                    return pd.read_csv(file_path, index_col=0)
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
    
    return pd.DataFrame()

@st.cache_data
def load_single_omics_studies() -> Dict[str, pd.DataFrame]:
    """Load all single-omics studies"""
    if DATA_DIR is None:
        return {}
    
    single_omics_dir = find_subfolder(DATA_DIR, "single_omics")
    if single_omics_dir is None:
        return {}
    
    studies = {}
    
    # Look for all CSV and Parquet files
    for file_path in single_omics_dir.rglob("*"):
        if file_path.suffix in ['.csv', '.parquet']:
            try:
                study_name = file_path.stem
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                studies[study_name] = df
            except Exception as e:
                pass  # Skip files that can't be loaded
    
    return studies

@st.cache_data
def load_kg_data() -> Dict[str, pd.DataFrame]:
    """Load all knowledge graph data"""
    if DATA_DIR is None:
        return {}
    
    kg_dir = find_subfolder(DATA_DIR, "knowledge_graphs")
    if kg_dir is None:
        return {}
    
    kg_data = {}
    
    for file_path in kg_dir.rglob("*"):
        if file_path.suffix in ['.csv', '.parquet']:
            try:
                data_name = file_path.stem
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                kg_data[data_name] = df
            except Exception as e:
                pass
    
    return kg_data

@st.cache_data
def load_ppi_data() -> Dict[str, pd.DataFrame]:
    """Load all PPI network data"""
    if DATA_DIR is None:
        return {}
    
    ppi_dir = find_subfolder(DATA_DIR, "ppi_networks")
    if ppi_dir is None:
        return {}
    
    ppi_data = {}
    
    for file_path in ppi_dir.rglob("*"):
        if file_path.suffix in ['.csv', '.parquet']:
            try:
                data_name = file_path.stem
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                ppi_data[data_name] = df
            except Exception as e:
                pass
    
    return ppi_data

# ============================================================================
# DATA AVAILABILITY CHECKER
# ============================================================================

def check_data_availability() -> Dict[str, bool]:
    """Check what data is available"""
    
    availability = {
        'data_dir': DATA_DIR is not None,
        'wgcna_expr': not load_wgcna_expr().empty,
        'wgcna_mes': not load_wgcna_mes().empty,
        'wgcna_mod_trait_cor': not load_wgcna_mod_trait_cor().empty,
        'wgcna_mod_trait_pval': not load_wgcna_mod_trait_pval().empty,
        'single_omics': len(load_single_omics_studies()) > 0,
        'knowledge_graphs': len(load_kg_data()) > 0,
        'ppi_networks': len(load_ppi_data()) > 0,
    }
    
    return availability

def get_data_summary() -> str:
    """Get human-readable data availability summary"""
    
    avail = check_data_availability()
    
    summary = "**Data Availability:**\n\n"
    
    if not avail['data_dir']:
        summary += "❌ Data directory not found\n"
        return summary
    
    summary += "✅ Data directory found\n\n"
    
    if avail['wgcna_expr']:
        expr = load_wgcna_expr()
        summary += f"✅ WGCNA Expression: {expr.shape[0]} samples × {expr.shape[1]} genes\n"
    else:
        summary += "❌ WGCNA Expression: Not found\n"
    
    if avail['wgcna_mes']:
        mes = load_wgcna_mes()
        summary += f"✅ WGCNA Module Eigenvectors: {mes.shape[1]} modules\n"
    else:
        summary += "❌ WGCNA Module Eigenvectors: Not found\n"
    
    if avail['wgcna_mod_trait_cor']:
        summary += "✅ Module-Trait Correlations: Available\n"
    else:
        summary += "❌ Module-Trait Correlations: Not found\n"
    
    if avail['single_omics']:
        studies = load_single_omics_studies()
        summary += f"✅ Single-Omics Studies: {len(studies)} datasets\n"
        for name, df in studies.items():
            summary += f"   - {name}: {len(df)} rows\n"
    else:
        summary += "❌ Single-Omics Studies: Not found\n"
    
    if avail['knowledge_graphs']:
        kg = load_kg_data()
        summary += f"✅ Knowledge Graphs: {len(kg)} datasets\n"
    else:
        summary += "❌ Knowledge Graphs: Not found\n"
    
    if avail['ppi_networks']:
        ppi = load_ppi_data()
        summary += f"✅ PPI Networks: {len(ppi)} datasets\n"
    else:
        summary += "❌ PPI Networks: Not found\n"
    
    return summary

# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def search_gene_in_studies(gene_name: str) -> Dict[str, pd.DataFrame]:
    """Search for a gene across all single-omics studies"""
    
    studies = load_single_omics_studies()
    results = {}
    
    for study_name, df in studies.items():
        # Try to find the gene
        if 'Gene' in df.columns:
            matches = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
            if not matches.empty:
                results[study_name] = matches
        elif 'gene' in df.columns:
            matches = df[df['gene'].str.contains(gene_name, case=False, na=False)]
            if not matches.empty:
                results[study_name] = matches
    
    return results

def search_drug_in_kg(drug_name: str) -> Dict[str, pd.DataFrame]:
    """Search for a drug in knowledge graphs"""
    
    kg_data = load_kg_data()
    results = {}
    
    for kg_name, df in kg_data.items():
        if 'Name' in df.columns:
            matches = df[df['Name'].str.contains(drug_name, case=False, na=False)]
            if not matches.empty:
                results[kg_name] = matches
    
    return results
