"""
Knowledge Graph Analysis Module
Analyzes gene position in MASH subgraph without edge information
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_kg_data(data_dir):
    """Load knowledge graph data files"""
    kg_dir = data_dir / "knowledge_graphs"
    
    kg_data = {}
    
    # Load MASH subgraph nodes
    nodes_file = kg_dir / "MASH_subgraph_nodes.parquet"
    if nodes_file.exists():
        kg_data['nodes'] = pd.read_parquet(nodes_file)
    
    # Load MASH subgraph drugs
    drugs_file = kg_dir / "MASH_subgraph_drugs.parquet"
    if drugs_file.exists():
        kg_data['drugs'] = pd.read_parquet(drugs_file)
    
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


def get_cluster_nodes(cluster_id, kg_data, top_n=10):
    """Get top nodes in the same cluster"""
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Filter to cluster
    cluster_nodes = nodes_df[nodes_df['Cluster'] == cluster_id].copy()
    
    if cluster_nodes.empty:
        return None
    
    # Sort by PageRank (descending)
    cluster_nodes = cluster_nodes.sort_values('PageRank Score', ascending=False)
    
    # Get top N
    top_nodes = cluster_nodes.head(top_n)
    
    # Format for display
    results = []
    for _, row in top_nodes.iterrows():
        results.append({
            'Name': row['Name'],
            'Type': row.get('Type', 'Unknown'),
            'PageRank': f"{float(row['PageRank Score']):.4f}",
            'Betweenness': f"{float(row['Betweenness Score']):.4f}",
            'Eigen': f"{float(row['Eigen Score']):.4f}"
        })
    
    return pd.DataFrame(results)


def get_top_drugs(kg_data, top_n=10):
    """Get top drugs in the subgraph by PageRank"""
    
    if 'drugs' not in kg_data:
        return None
    
    drugs_df = kg_data['drugs'].copy()
    
    # Sort by PageRank (descending)
    drugs_df = drugs_df.sort_values('PageRank Score', ascending=False)
    
    # Get top N
    top_drugs = drugs_df.head(top_n)
    
    # Format for display
    results = []
    for _, row in top_drugs.iterrows():
        results.append({
            'Drug': row['Name'],
            'PageRank': f"{float(row['PageRank Score']):.4f}",
            'Betweenness': f"{float(row['Betweenness Score']):.4f}",
            'Eigen': f"{float(row['Eigen Score']):.4f}",
            'Cluster': row.get('Cluster', 'Unknown')
        })
    
    return pd.DataFrame(results)


def interpret_centrality(pagerank, betweenness, eigen):
    """Interpret centrality scores"""
    
    # Normalize scores to 0-1 range for interpretation
    # (assuming they're already normalized or we use relative ranking)
    
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
