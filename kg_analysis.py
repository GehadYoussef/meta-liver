"""
Knowledge Graph Analysis Module
Analyzes gene position in MASH subgraph with robust filtering
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


def load_kg_data_from_dict(kg_dict):
    """
    Convert kg_dict (keyed by filestem) to standard format (keyed by 'nodes', 'drugs')
    This handles the format returned by robust_data_loader.load_kg_data()
    """
    formatted = {}
    
    # Look for nodes dataframe
    for key in ['MASH_subgraph_nodes', 'mash_subgraph_nodes', 'nodes']:
        if key in kg_dict:
            formatted['nodes'] = kg_dict[key]
            break
    
    # Look for drugs dataframe
    for key in ['MASH_subgraph_drugs', 'mash_subgraph_drugs', 'drugs']:
        if key in kg_dict:
            formatted['drugs'] = kg_dict[key]
            break
    
    return formatted


def get_gene_kg_info(gene_name, kg_data):
    """Get knowledge graph information for a gene, including empirical percentile ranks"""
    
    # Handle both formats: dict keyed by 'nodes' or dict keyed by filestem
    if isinstance(kg_data, dict):
        if 'nodes' not in kg_data and 'MASH_subgraph_nodes' in kg_data:
            kg_data = load_kg_data_from_dict(kg_data)
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Exact match first (case-insensitive)
    gene_match = nodes_df[nodes_df['Name'].str.lower() == gene_name.lower()]
    
    # If no exact match, try substring (but warn about it)
    if gene_match.empty:
        gene_match = nodes_df[nodes_df['Name'].str.contains(f"^{gene_name}$", case=False, na=False, regex=True)]
    
    if gene_match.empty:
        return {
            'found': False,
            'message': f"'{gene_name}' not found in MASH subgraph"
        }
    
    # If multiple matches, take the first (but this shouldn't happen with exact match)
    gene_row = gene_match.iloc[0]
    
    # Get node values
    pr_val = float(gene_row.get('PageRank Score', 0))
    bet_val = float(gene_row.get('Betweenness Score', 0))
    eigen_val = float(gene_row.get('Eigen Score', 0))
    
    # Compute empirical percentile ranks (0-100)
    # Percentile = (number of values <= this value) / (total values) * 100
    pr_percentile = 0.0
    bet_percentile = 0.0
    eigen_percentile = 0.0
    
    if 'PageRank Score' in nodes_df.columns:
        pr_percentile = (nodes_df['PageRank Score'] <= pr_val).sum() / len(nodes_df) * 100
    
    if 'Betweenness Score' in nodes_df.columns:
        bet_percentile = (nodes_df['Betweenness Score'] <= bet_val).sum() / len(nodes_df) * 100
    
    if 'Eigen Score' in nodes_df.columns:
        eigen_percentile = (nodes_df['Eigen Score'] <= eigen_val).sum() / len(nodes_df) * 100
    
    # Get min/max for display
    pr_min = float(nodes_df['PageRank Score'].min()) if 'PageRank Score' in nodes_df.columns else 0
    pr_max = float(nodes_df['PageRank Score'].max()) if 'PageRank Score' in nodes_df.columns else 0
    
    bet_min = float(nodes_df['Betweenness Score'].min()) if 'Betweenness Score' in nodes_df.columns else 0
    bet_max = float(nodes_df['Betweenness Score'].max()) if 'Betweenness Score' in nodes_df.columns else 0
    
    eigen_min = float(nodes_df['Eigen Score'].min()) if 'Eigen Score' in nodes_df.columns else 0
    eigen_max = float(nodes_df['Eigen Score'].max()) if 'Eigen Score' in nodes_df.columns else 0
    
    # Extract metrics
    info = {
        'found': True,
        'name': gene_row['Name'],
        'type': gene_row.get('Type', 'Unknown'),
        'cluster': gene_row.get('Cluster', 'Unknown'),
        'pagerank': pr_val,
        'betweenness': bet_val,
        'eigen': eigen_val,
        'pr_min': pr_min,
        'pr_max': pr_max,
        'pr_percentile': pr_percentile,
        'bet_min': bet_min,
        'bet_max': bet_max,
        'bet_percentile': bet_percentile,
        'eigen_min': eigen_min,
        'eigen_max': eigen_max,
        'eigen_percentile': eigen_percentile
    }
    
    return info


def get_cluster_genes(cluster_id, kg_data):
    """Get all genes/proteins in cluster, sorted by PageRank"""
    
    # Handle both formats
    if isinstance(kg_data, dict):
        if 'nodes' not in kg_data and 'MASH_subgraph_nodes' in kg_data:
            kg_data = load_kg_data_from_dict(kg_data)
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Robust cluster matching (handle string/int mismatch)
    cluster_id_str = str(cluster_id).lower()
    
    # Filter to cluster
    cluster_mask = nodes_df['Cluster'].astype(str).str.lower() == cluster_id_str
    genes = nodes_df[cluster_mask].copy()
    
    # Filter to genes/proteins (case-insensitive, flexible matching)
    if 'Type' in genes.columns:
        type_mask = genes['Type'].astype(str).str.lower().str.contains('gene|protein', na=False)
        genes = genes[type_mask]
    
    if genes.empty:
        return None
    
    # Sort by PageRank (descending)
    genes = genes.sort_values('PageRank Score', ascending=False)
    
    # Format for display
    results = []
    for _, row in genes.iterrows():
        results.append({
            'Name': row['Name'],
            'PageRank': f"{float(row['PageRank Score']):.4f}",
            'Betweenness': f"{float(row['Betweenness Score']):.4f}",
            'Eigen': f"{float(row['Eigen Score']):.4f}"
        })
    
    return pd.DataFrame(results)


def get_cluster_drugs(cluster_id, kg_data):
    """Get all drugs in cluster, sorted by PageRank"""
    
    # Handle both formats
    if isinstance(kg_data, dict):
        if 'nodes' not in kg_data and 'MASH_subgraph_nodes' in kg_data:
            kg_data = load_kg_data_from_dict(kg_data)
    
    drugs = None
    
    # Try to use dedicated drugs dataframe first
    if 'drugs' in kg_data and not kg_data['drugs'].empty:
        drugs_df = kg_data['drugs']
        cluster_id_str = str(cluster_id).lower()
        cluster_mask = drugs_df['Cluster'].astype(str).str.lower() == cluster_id_str
        drugs = drugs_df[cluster_mask].copy()
    
    # Fall back to filtering nodes table
    if drugs is None or drugs.empty:
        if 'nodes' not in kg_data:
            return None
        
        nodes_df = kg_data['nodes']
        cluster_id_str = str(cluster_id).lower()
        
        # Filter to cluster
        cluster_mask = nodes_df['Cluster'].astype(str).str.lower() == cluster_id_str
        drugs = nodes_df[cluster_mask].copy()
        
        # Filter to drugs (case-insensitive, exact match)
        if 'Type' in drugs.columns:
            type_mask = drugs['Type'].astype(str).str.lower() == 'drug'
            drugs = drugs[type_mask]
    
    if drugs.empty:
        return None
    
    # Sort by PageRank (descending)
    drugs = drugs.sort_values('PageRank Score', ascending=False)
    
    # Format for display
    results = []
    for _, row in drugs.iterrows():
        results.append({
            'Name': row['Name'],
            'PageRank': f"{float(row['PageRank Score']):.4f}",
            'Betweenness': f"{float(row['Betweenness Score']):.4f}",
            'Eigen': f"{float(row['Eigen Score']):.4f}"
        })
    
    return pd.DataFrame(results)


def get_cluster_diseases(cluster_id, kg_data):
    """Get all diseases in cluster, sorted by PageRank"""
    
    # Handle both formats
    if isinstance(kg_data, dict):
        if 'nodes' not in kg_data and 'MASH_subgraph_nodes' in kg_data:
            kg_data = load_kg_data_from_dict(kg_data)
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    cluster_id_str = str(cluster_id).lower()
    
    # Filter to cluster
    cluster_mask = nodes_df['Cluster'].astype(str).str.lower() == cluster_id_str
    diseases = nodes_df[cluster_mask].copy()
    
    # Filter to diseases (case-insensitive, exact match)
    if 'Type' in diseases.columns:
        type_mask = diseases['Type'].astype(str).str.lower() == 'disease'
        diseases = diseases[type_mask]
    
    if diseases.empty:
        return None
    
    # Sort by PageRank (descending)
    diseases = diseases.sort_values('PageRank Score', ascending=False)
    
    # Format for display
    results = []
    for _, row in diseases.iterrows():
        results.append({
            'Name': row['Name'],
            'PageRank': f"{float(row['PageRank Score']):.4f}",
            'Betweenness': f"{float(row['Betweenness Score']):.4f}",
            'Eigen': f"{float(row['Eigen Score']):.4f}"
        })
    
    return pd.DataFrame(results)


def interpret_centrality(pagerank, betweenness, eigen):
    """Interpret centrality scores"""
    
    interpretations = []
    
    if pagerank > 0.01:
        interpretations.append("High PageRank (central in network)")
    elif pagerank > 0.001:
        interpretations.append("Moderate PageRank")
    else:
        interpretations.append("Low PageRank (peripheral)")
    
    if betweenness > 0.01:
        interpretations.append("High Betweenness (bridges clusters)")
    elif betweenness > 0.001:
        interpretations.append("Moderate Betweenness")
    else:
        interpretations.append("Low Betweenness")
    
    if eigen > 0.01:
        interpretations.append("High Eigen centrality (connected to hubs)")
    elif eigen > 0.001:
        interpretations.append("Moderate Eigen centrality")
    else:
        interpretations.append("Low Eigen centrality")
    
    return " • ".join(interpretations)
