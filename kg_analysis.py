"""
Knowledge Graph Analysis Module
Analyzes gene position in MASH subgraph
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
    
    if 'nodes' not in kg_data:
        return None
    
    nodes_df = kg_data['nodes']
    
    # Filter to cluster and drugs
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
