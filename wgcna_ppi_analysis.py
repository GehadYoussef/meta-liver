"""
WGCNA and PPI Network Analysis Module
Loads module-specific gene mapping files from wgcna/modules/ folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys


def find_data_dir():
    """Find data directory"""
    possible_dirs = [
        Path("meta-liver-data"),
        Path("meta_liver_data"),
        Path("data"),
        Path("../meta-liver-data"),
        Path.home() / "meta-liver-data",
        Path.home() / "meta_liver_data",
    ]
    
    for path in possible_dirs:
        if path.exists():
            return path
    return None


def find_subfolder(parent: Path, folder_pattern: str):
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


def find_file(directory: Path, filename_pattern: str):
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


def load_wgcna_module_data():
    """
    Load all WGCNA module data by recursively searching for Nodes-gene-id-mapping files.
    Normalizes gene symbol column to 'hgnc_symbol' for consistency.
    Returns dict with module names as keys and gene dataframes as values.
    """
    
    module_data = {}
    
    # Search recursively from current directory and common data locations
    search_roots = [
        Path("."),  # Current directory (repo root on Streamlit Cloud)
        find_data_dir() or Path("."),  # Use same data-dir as other loaders
    ]
    
    for root in search_roots:
        if not root.exists():
            continue
        
        # Recursively search for Nodes-gene-id-mapping files
        for file_path in root.rglob("Nodes-gene-id-mapping-*.csv"):
            try:
                # Extract module name from filename
                match = re.search(r'Nodes-gene-id-mapping-(.+)\.csv', file_path.name)
                if not match:
                    continue
                
                module_name = match.group(1)
                
                # Skip if already loaded
                if module_name in module_data:
                    continue
                
                # Load the gene mapping file
                df = pd.read_csv(file_path)
                
                # Find gene symbol column (flexible naming)
                gene_col = None
                for col in df.columns:
                    if col.lower() in ['hgnc_symbol', 'symbol', 'gene', 'gene_symbol', 'hgnc']:
                        gene_col = col
                        break
                
                if gene_col is None:
                    print(f"DEBUG: No gene symbol column found in {file_path}, skipping", file=sys.stderr)
                    continue
                
                # Normalize: rename the gene column to 'hgnc_symbol' for consistency
                if gene_col != 'hgnc_symbol':
                    df = df.rename(columns={gene_col: 'hgnc_symbol'})
                
                # Store the dataframe
                module_data[module_name] = df
                print(f"DEBUG: Loaded WGCNA module {module_name} from {file_path} (gene col: {gene_col})", file=sys.stderr)
                
            except Exception as e:
                print(f"DEBUG: Error loading {file_path}: {e}", file=sys.stderr)
    
    if module_data:
        print(f"DEBUG: Successfully loaded {len(module_data)} WGCNA modules", file=sys.stderr)
    else:
        print(f"DEBUG: No WGCNA modules found after recursive search", file=sys.stderr)
    
    return module_data


def get_gene_module(gene_name, module_data):
    """
    Find which WGCNA module a gene belongs to.
    Returns module name and gene info.
    """
    
    if not module_data:
        return None
    
    gene_lower = gene_name.lower()
    
    # Search through all modules
    for module_name, gene_df in module_data.items():
        if gene_df.empty or 'hgnc_symbol' not in gene_df.columns:
            continue
        
        # Try exact match (case-insensitive)
        matching = gene_df[gene_df['hgnc_symbol'].str.lower() == gene_lower]
        
        if not matching.empty:
            return {
                'gene': gene_name,
                'module': module_name,
                'data': matching.iloc[0]
            }
    
    return None


def get_module_genes(module_name, module_data):
    """
    Get all genes in a specific WGCNA module.
    Returns dataframe of genes in the module.
    """
    
    if not module_data or module_name not in module_data:
        return None
    
    gene_df = module_data[module_name]
    
    if gene_df.empty:
        return None
    
    # Return the dataframe with relevant columns
    if 'hgnc_symbol' in gene_df.columns:
        display_cols = ['hgnc_symbol']
        if 'ensembl_gene_id' in gene_df.columns:
            display_cols.append('ensembl_gene_id')
        
        result = gene_df[display_cols].copy()
        result.columns = ['Gene', 'Ensembl ID'] if len(display_cols) > 1 else ['Gene']
        return result
    
    return gene_df


def get_all_modules(module_data):
    """
    Get list of all available modules.
    """
    if not module_data:
        return []
    
    return sorted(module_data.keys())


def get_coexpressed_partners(gene_name, expr_df, top_n=15):
    """
    Find genes most strongly co-expressed with the target gene.
    expr_df should be samples x genes (or genes x samples).
    Returns dataframe with top_n genes with highest correlation.
    """
    
    if expr_df.empty:
        return None
    
    # Find gene in expression matrix
    gene_col = None
    gene_lower = gene_name.lower()
    
    # Try columns first
    if isinstance(expr_df.columns, pd.Index):
        matching_cols = [c for c in expr_df.columns if str(c).lower() == gene_lower]
        if matching_cols:
            gene_col = matching_cols[0]
    
    # Try index if not found in columns
    if gene_col is None and isinstance(expr_df.index, pd.Index):
        matching_idx = [i for i in expr_df.index if str(i).lower() == gene_lower]
        if matching_idx:
            # Transpose if gene is in index
            expr_df = expr_df.T
            gene_col = matching_idx[0]
    
    if gene_col is None:
        return None
    
    # Calculate correlations with all other genes
    try:
        gene_expr = expr_df[gene_col]
        correlations = expr_df.corr()[gene_col].drop(gene_col, errors='ignore')
        
        # Get top correlated genes (by absolute correlation)
        top_corr = correlations.abs().nlargest(top_n)
        
        results = []
        for gene, corr_val in top_corr.items():
            actual_corr = correlations[gene]
            results.append({
                'Gene': gene,
                'Correlation': f"{actual_corr:.3f}"
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        return None


def find_ppi_interactors(gene_name, ppi_data):
    """
    Find direct protein-protein interaction partners of a gene.
    ppi_data should be a dict of dataframes, each with 'protein1', 'protein2' columns.
    """
    
    if not ppi_data:
        return None
    
    interactors = set()
    gene_lower = gene_name.lower()
    
    # Search through all PPI datasets
    for ppi_name, ppi_df in ppi_data.items():
        if ppi_df.empty:
            continue
        
        # Identify protein columns (flexible naming)
        protein1_col = None
        protein2_col = None
        
        for col in ppi_df.columns:
            col_lower = col.lower()
            if 'protein1' in col_lower or 'gene1' in col_lower or 'source' in col_lower:
                protein1_col = col
            elif 'protein2' in col_lower or 'gene2' in col_lower or 'target' in col_lower:
                protein2_col = col
        
        if protein1_col is None or protein2_col is None:
            continue
        
        # Find interactions where gene is protein1
        matches1 = ppi_df[ppi_df[protein1_col].astype(str).str.lower() == gene_lower]
        interactors.update(matches1[protein2_col].tolist())
        
        # Find interactions where gene is protein2
        matches2 = ppi_df[ppi_df[protein2_col].astype(str).str.lower() == gene_lower]
        interactors.update(matches2[protein1_col].tolist())
    
    if not interactors:
        return None
    
    results = []
    for interactor in sorted(interactors):
        results.append({'Interactor': interactor})
    
    return pd.DataFrame(results)


def get_network_stats(gene_name, ppi_data):
    """
    Get local network statistics for a gene in PPI network.
    Returns degree, clustering coefficient, etc.
    """
    
    if not ppi_data:
        return None
    
    # Count direct interactions
    degree = 0
    gene_lower = gene_name.lower()
    
    for ppi_name, ppi_df in ppi_data.items():
        if ppi_df.empty:
            continue
        
        # Identify protein columns
        protein1_col = None
        protein2_col = None
        
        for col in ppi_df.columns:
            col_lower = col.lower()
            if 'protein1' in col_lower or 'gene1' in col_lower or 'source' in col_lower:
                protein1_col = col
            elif 'protein2' in col_lower or 'gene2' in col_lower or 'target' in col_lower:
                protein2_col = col
        
        if protein1_col is None or protein2_col is None:
            continue
        
        degree += len(ppi_df[ppi_df[protein1_col].astype(str).str.lower() == gene_lower])
        degree += len(ppi_df[ppi_df[protein2_col].astype(str).str.lower() == gene_lower])
    
    if degree == 0:
        return None
    
    return {
        'degree': degree,
        'description': f"Direct interactors in PPI network"
    }
