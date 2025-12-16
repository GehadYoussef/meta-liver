"""
WGCNA and PPI Network Analysis Module
Properly handles WGCNA folder structure including modules subfolder
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
    Load WGCNA module data from wgcna/modules subfolder.
    Looks for files like: geneModuleMembership.csv, moduleEigenvectors.csv, etc.
    """
    data_dir = find_data_dir()
    if data_dir is None:
        return {}
    
    wgcna_dir = find_subfolder(data_dir, "wgcna")
    if wgcna_dir is None:
        return {}
    
    modules_dir = find_subfolder(wgcna_dir, "modules")
    if modules_dir is None:
        return {}
    
    module_data = {}
    
    # Load gene-module membership
    for filename in ["geneModuleMembership.csv", "geneModuleMembership.parquet", 
                     "gene_module_membership.csv", "gene_module_membership.parquet"]:
        file_path = find_file(modules_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    module_data['membership'] = pd.read_parquet(file_path)
                else:
                    module_data['membership'] = pd.read_csv(file_path, index_col=0)
                break
            except Exception as e:
                pass
    
    # Load module eigenvectors
    for filename in ["moduleEigenvectors.csv", "moduleEigenvectors.parquet",
                     "module_eigenvectors.csv", "module_eigenvectors.parquet"]:
        file_path = find_file(modules_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    module_data['eigenvectors'] = pd.read_parquet(file_path)
                else:
                    module_data['eigenvectors'] = pd.read_csv(file_path, index_col=0)
                break
            except Exception as e:
                pass
    
    # Load module trait correlations
    for filename in ["moduleTraitCor.csv", "moduleTraitCor.parquet",
                     "module_trait_cor.csv", "module_trait_cor.parquet"]:
        file_path = find_file(modules_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    module_data['trait_cor'] = pd.read_parquet(file_path)
                else:
                    module_data['trait_cor'] = pd.read_csv(file_path, index_col=0)
                break
            except Exception as e:
                pass
    
    return module_data


def get_gene_module(gene_name, module_data):
    """
    Find which WGCNA module a gene belongs to.
    Returns module name and membership data.
    """
    
    if not module_data or 'membership' not in module_data:
        return None
    
    membership_df = module_data['membership']
    
    # Try exact match first (case-insensitive)
    gene_lower = gene_name.lower()
    
    if isinstance(membership_df.index, pd.Index):
        matching = [idx for idx in membership_df.index if str(idx).lower() == gene_lower]
        if matching:
            gene_idx = matching[0]
            row = membership_df.loc[gene_idx]
            
            # Find module column (usually first column or named 'Module', 'module', etc.)
            module_col = None
            for col in membership_df.columns:
                if 'module' in col.lower():
                    module_col = col
                    break
            
            if module_col is None and len(membership_df.columns) > 0:
                module_col = membership_df.columns[0]
            
            if module_col:
                return {
                    'gene': gene_idx,
                    'module': row[module_col],
                    'membership_score': float(row[module_col]) if pd.api.types.is_numeric_dtype(row[module_col]) else None,
                    'data': row
                }
    
    return None


def get_module_genes(module_name, module_data):
    """
    Get all genes in a specific WGCNA module.
    Returns dataframe of genes in the module.
    """
    
    if not module_data or 'membership' not in module_data:
        return None
    
    membership_df = module_data['membership']
    
    # Find module column
    module_col = None
    for col in membership_df.columns:
        if 'module' in col.lower():
            module_col = col
            break
    
    if module_col is None and len(membership_df.columns) > 0:
        module_col = membership_df.columns[0]
    
    if module_col is None:
        return None
    
    # Filter to genes in this module
    module_mask = membership_df[module_col].astype(str).str.lower() == str(module_name).lower()
    genes_in_module = membership_df[module_mask]
    
    if genes_in_module.empty:
        return None
    
    # Return top genes by membership score if available
    if len(genes_in_module.columns) > 1:
        # Sort by second column (usually membership score)
        genes_in_module = genes_in_module.sort_values(genes_in_module.columns[1], ascending=False)
    
    return genes_in_module.head(20)


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
