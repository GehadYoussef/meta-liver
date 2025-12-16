"""
WGCNA and PPI Network Analysis Module
"""

import pandas as pd
import numpy as np


def find_gene_module(gene_name, expr_df):
    """
    Find which WGCNA module a gene belongs to.
    expr_df should have genes as index/columns and modules as values.
    """
    
    if expr_df.empty:
        return None
    
    # Try to find gene in index (genes as rows)
    if gene_name.lower() in [g.lower() for g in expr_df.index]:
        matching_idx = [i for i, g in enumerate(expr_df.index) if g.lower() == gene_name.lower()]
        if matching_idx:
            return expr_df.iloc[matching_idx[0]]
    
    # Try to find gene in columns (genes as columns)
    if gene_name.lower() in [g.lower() for g in expr_df.columns]:
        matching_cols = [c for c in expr_df.columns if c.lower() == gene_name.lower()]
        if matching_cols:
            return expr_df[matching_cols[0]]
    
    return None


def get_module_genes(module_name, expr_df):
    """
    Get all genes in a specific WGCNA module.
    Returns a list of gene names.
    """
    
    if expr_df.empty:
        return []
    
    genes = []
    
    # Check if expr_df contains module assignments
    # Could be structured as: genes x samples with module names as values
    # Or genes x 1 with module assignment
    
    for col in expr_df.columns:
        module_vals = expr_df[col]
        if isinstance(module_vals.iloc[0], str):
            # This column contains module assignments
            matching_genes = module_vals[module_vals.str.lower() == module_name.lower()].index.tolist()
            genes.extend(matching_genes)
    
    # Also check index
    if hasattr(expr_df.index, 'name') and expr_df.index.name and 'module' in expr_df.index.name.lower():
        matching_genes = expr_df.index[expr_df.index.str.lower() == module_name.lower()].tolist()
        genes.extend(matching_genes)
    
    return list(set(genes))  # Remove duplicates


def get_coexpressed_partners(gene_name, expr_df, top_n=10):
    """
    Find genes most strongly co-expressed with the target gene.
    Returns top_n genes with highest correlation.
    """
    
    if expr_df.empty:
        return None
    
    # Find gene in expression matrix
    gene_col = None
    if gene_name.lower() in [g.lower() for g in expr_df.columns]:
        matching_cols = [c for c in expr_df.columns if c.lower() == gene_name.lower()]
        if matching_cols:
            gene_col = matching_cols[0]
    
    if gene_col is None:
        return None
    
    # Calculate correlations with all other genes
    gene_expr = expr_df[gene_col]
    correlations = expr_df.corr()[gene_col].drop(gene_col)
    
    # Get top correlated genes (by absolute correlation)
    top_corr = correlations.abs().nlargest(top_n)
    
    results = []
    for gene, corr_val in top_corr.items():
        actual_corr = correlations[gene]
        results.append({
            'Gene': gene,
            'Correlation': f"{actual_corr:.3f}",
            'Abs_Correlation': abs(actual_corr)
        })
    
    return pd.DataFrame(results).drop('Abs_Correlation', axis=1)


def find_ppi_interactors(gene_name, ppi_data):
    """
    Find direct protein-protein interaction partners of a gene.
    ppi_data should be a dict of dataframes, each with 'protein1', 'protein2' columns.
    """
    
    if not ppi_data or not ppi_data:
        return None
    
    interactors = set()
    
    # Search through all PPI datasets
    for ppi_name, ppi_df in ppi_data.items():
        if ppi_df.empty:
            continue
        
        # Look for gene in either protein1 or protein2 columns
        for col in ['protein1', 'Protein1', 'gene1', 'Gene1', 'source', 'Source']:
            if col in ppi_df.columns:
                matches = ppi_df[ppi_df[col].str.lower() == gene_name.lower()]
                
                # Get partner column
                partner_col = None
                for pcol in ['protein2', 'Protein2', 'gene2', 'Gene2', 'target', 'Target']:
                    if pcol in ppi_df.columns:
                        partner_col = pcol
                        break
                
                if partner_col:
                    interactors.update(matches[partner_col].tolist())
        
        for col in ['protein2', 'Protein2', 'gene2', 'Gene2', 'target', 'Target']:
            if col in ppi_df.columns:
                matches = ppi_df[ppi_df[col].str.lower() == gene_name.lower()]
                
                # Get partner column
                partner_col = None
                for pcol in ['protein1', 'Protein1', 'gene1', 'Gene1', 'source', 'Source']:
                    if pcol in ppi_df.columns:
                        partner_col = pcol
                        break
                
                if partner_col:
                    interactors.update(matches[partner_col].tolist())
    
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
    
    for ppi_name, ppi_df in ppi_data.items():
        if ppi_df.empty:
            continue
        
        for col in ['protein1', 'Protein1', 'gene1', 'Gene1', 'source', 'Source']:
            if col in ppi_df.columns:
                degree += len(ppi_df[ppi_df[col].str.lower() == gene_name.lower()])
        
        for col in ['protein2', 'Protein2', 'gene2', 'Gene2', 'target', 'Target']:
            if col in ppi_df.columns:
                degree += len(ppi_df[ppi_df[col].str.lower() == gene_name.lower()])
    
    if degree == 0:
        return None
    
    return {
        'degree': degree,
        'description': f"Direct interactors in PPI network"
    }
