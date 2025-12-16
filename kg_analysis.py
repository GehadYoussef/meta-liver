"""
Knowledge Graph Analysis Module
Analyzes gene position in MASH subgraph
"""

import pandas as pd
import numpy as np
from pathlib import Path


def find_data_dir():
    """Find data directory"""
    for path in [Path("meta-liver-data"), Path("meta_liver_data"), Path("data")]:
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


def load_kg_data(data_dir=None):
    """Load knowledge graph data files"""
    if data_dir is None:
        data_dir = find_data_dir()
    
    if data_dir is None:
        return {}
    
    kg_dir = find_subfolder(data_dir, "knowledge_graphs")
    if kg_dir is None:
        return {}
    
    kg_data = {}
    
    # Load MASH subgraph nodes
    for filename in ["MASH_subgraph_nodes.parquet", "MASH_subgraph_nodes.csv"]:
        file_path = find_file(kg_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    kg_data['nodes'] = pd.read_parquet(file_path)
                else:
                    kg_data['nodes'] = pd.read_csv(file_path)
                break
            except Exception as e:
                pass
    
    # Load MASH subgraph drugs
    for filename in ["MASH_subgraph_drugs.parquet", "MASH_subgraph_drugs.csv"]:
        file_path = find_file(kg_dir, filename)
        if file_path:
            try:
                if file_path.suffix == '.parquet':
                    kg_data['drugs'] = pd.read_parquet(file_path)
                else:
                    kg_data['drugs'] = pd.read_csv(file_path)
                break
            except Exception as e:
                pass
    
    return kg_data


def get_gene_kg_info(gene_name, kg_data):
    """Get knowledge graph information for a gene"""
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Search for gene (case-insensitive)
    gene_match = nodes_df[nodes_df['Name'].str.contains(gene_name, case=False, na=False)]
    
    if gene_match.empty:
        return {
            'found': False,
            'message': f"'{gene_name}' not found in MASH subgraph"
        }
    
    gene_row = gene_match.iloc[0]
    
    # Extract metrics
    info = {
        'found': True,
        'name': gene_row['Name'],
        'type': gene_row.get('Type', 'Unknown'),
        'cluster': gene_row.get('Cluster', 'Unknown'),
        'pagerank': float(gene_row.get('PageRank Score', 0)),
        'betweenness': float(gene_row.get('Betweenness Score', 0)),
        'eigen': float(gene_row.get('Eigen Score', 0))
    }
    
    return info


def get_cluster_genes(cluster_id, kg_data):
    """Get all genes/proteins in cluster, sorted by PageRank"""
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Filter to cluster and genes/proteins
    genes = nodes_df[
        (nodes_df['Cluster'] == cluster_id) &
        (nodes_df['Type'].str.contains('gene|protein', case=False, na=False))
    ].copy()
    
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
    
    # Try to use dedicated drugs dataframe first
    if 'drugs' in kg_data and not kg_data['drugs'].empty:
        drugs = kg_data['drugs'][
            kg_data['drugs']['Cluster'] == cluster_id
        ].copy()
    else:
        # Fall back to filtering nodes table
        if 'nodes' not in kg_data:
            return None
        
        nodes_df = kg_data['nodes']
        drugs = nodes_df[
            (nodes_df['Cluster'] == cluster_id) &
            (nodes_df['Type'].str.lower() == 'drug')
        ].copy()
    
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
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Filter to cluster and diseases
    diseases = nodes_df[
        (nodes_df['Cluster'] == cluster_id) &
        (nodes_df['Type'].str.lower() == 'disease')
    ].copy()
    
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
