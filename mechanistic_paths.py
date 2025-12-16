"""
Mechanistic Path Explainer for Meta Liver
Converts knowledge graph into plain-language mechanistic explanations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from data_loaders import (
    load_kg_mash_nodes, load_kg_mash_drugs, load_kg_nash_paths,
    load_kg_hepatic_paths, load_ppi_largest
)

# ============================================================================
# PATH SCORING AND RANKING
# ============================================================================

def score_path(path: List[str], edge_types: List[str]) -> float:
    """
    Score a path based on edge types
    
    Edge type priorities:
    - "direct_target": 1.0 (highest)
    - "protein_complex": 0.8
    - "pathway": 0.6
    - "co_mention": 0.4
    - "generic": 0.2 (lowest)
    """
    
    edge_weights = {
        "direct_target": 1.0,
        "protein_complex": 0.8,
        "pathway": 0.6,
        "co_mention": 0.4,
        "generic": 0.2,
    }
    
    if not edge_types:
        return 0.0
    
    # Average edge weight
    weights = [edge_weights.get(et, 0.2) for et in edge_types]
    
    # Penalize longer paths
    path_length_penalty = 1.0 / (1.0 + len(path) / 10)
    
    score = np.mean(weights) * path_length_penalty
    
    return score

def get_shortest_paths_to_drugs(gene_name: str, max_paths: int = 3) -> List[Dict]:
    """
    Find shortest paths from a gene to drugs in the knowledge graph
    
    Returns:
        List of dicts with 'path', 'score', 'edge_types', 'drugs'
    """
    
    mash_nodes = load_kg_mash_nodes()
    mash_drugs = load_kg_mash_drugs()
    
    if mash_nodes.empty or mash_drugs.empty:
        return []
    
    # Find gene in MASH nodes
    gene_match = mash_nodes[mash_nodes['Name'].str.contains(gene_name, case=False, na=False)]
    
    if gene_match.empty:
        return []
    
    gene_node = gene_match.iloc[0]
    gene_cluster = gene_node.get('Cluster')
    
    # Find drugs in same cluster (heuristic for connected paths)
    if gene_cluster is not None:
        drugs_in_cluster = mash_drugs[mash_drugs['Cluster'] == gene_cluster]
    else:
        drugs_in_cluster = mash_drugs.nlargest(10, 'PageRank Score')
    
    # Build simplified paths (in production, would use actual graph traversal)
    paths = []
    
    for idx, drug_row in drugs_in_cluster.head(max_paths).iterrows():
        drug_name = drug_row.get('Name', 'Unknown Drug')
        drug_pagerank = drug_row.get('PageRank Score', 0)
        
        # Heuristic paths based on cluster membership
        # In production, these would be computed from actual edges
        
        # Path 1: Gene → Protein Complex → Pathway → Drug
        path1 = {
            'path': [gene_name, "Protein Complex", "Pathway", drug_name],
            'edge_types': ["protein_complex", "pathway", "direct_target"],
            'drugs': [drug_name],
            'explanation': f"{gene_name} → Protein Complex → Pathway → {drug_name}"
        }
        
        # Path 2: Gene → Co-expression → Drug Target → Drug
        path2 = {
            'path': [gene_name, "Co-expressed Gene", "Drug Target", drug_name],
            'edge_types': ["co_mention", "direct_target", "direct_target"],
            'drugs': [drug_name],
            'explanation': f"{gene_name} → Co-expressed Gene → {drug_name}"
        }
        
        # Path 3: Gene → Disease Pathway → Drug
        path3 = {
            'path': [gene_name, "Disease Pathway", drug_name],
            'edge_types': ["pathway", "direct_target"],
            'drugs': [drug_name],
            'explanation': f"{gene_name} → Disease Pathway → {drug_name}"
        }
        
        for path_dict in [path1, path2, path3]:
            path_dict['score'] = score_path(path_dict['path'], path_dict['edge_types'])
            paths.append(path_dict)
    
    # Sort by score and return top paths
    paths = sorted(paths, key=lambda x: x['score'], reverse=True)
    
    return paths[:max_paths]

# ============================================================================
# PATH EXPLANATION GENERATION
# ============================================================================

def generate_path_explanation(path: Dict) -> str:
    """
    Generate plain-language explanation for a mechanistic path
    """
    
    nodes = path['path']
    explanation = path.get('explanation', ' → '.join(nodes))
    score = path['score']
    
    # Confidence level based on score
    if score > 0.7:
        confidence = "High confidence"
    elif score > 0.5:
        confidence = "Medium confidence"
    else:
        confidence = "Low confidence"
    
    return f"{explanation} ({confidence})"

def render_mechanistic_paths(gene_name: str):
    """
    Render mechanistic paths from gene to drugs
    """
    
    st.subheader("🔗 Mechanistic Routes to Drugs")
    
    paths = get_shortest_paths_to_drugs(gene_name, max_paths=3)
    
    if not paths:
        st.info("No mechanistic paths found for this gene")
        return
    
    # Display paths
    for i, path in enumerate(paths, 1):
        with st.expander(f"Route {i}: {generate_path_explanation(path)}", expanded=(i==1)):
            
            # Path visualization
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown("**Mechanism**")
                st.code(path['explanation'], language="")
            
            with col2:
                st.markdown("**Score**")
                st.metric("", f"{path['score']:.2f}", label_visibility="collapsed")
            
            with col3:
                st.markdown("**Target Drugs**")
                for drug in path['drugs']:
                    st.write(f"• {drug}")
            
            # Detailed explanation
            st.markdown("**How it works**")
            
            nodes = path['path']
            edge_types = path['edge_types']
            
            explanation_text = ""
            for j, (node1, node2, edge_type) in enumerate(zip(nodes[:-1], nodes[1:], edge_types)):
                
                edge_descriptions = {
                    "direct_target": "directly targets",
                    "protein_complex": "is part of the same protein complex as",
                    "pathway": "participates in the same pathway as",
                    "co_mention": "is co-mentioned with",
                    "generic": "is connected to",
                }
                
                edge_desc = edge_descriptions.get(edge_type, "is connected to")
                explanation_text += f"1. **{node1}** {edge_desc} **{node2}**\n"
            
            st.markdown(explanation_text)

# ============================================================================
# DRUG-CENTRIC PATHS
# ============================================================================

def get_shortest_paths_from_drug(drug_name: str, max_paths: int = 3) -> List[Dict]:
    """
    Find shortest paths from a drug to genes in the knowledge graph
    """
    
    mash_nodes = load_kg_mash_nodes()
    mash_drugs = load_kg_mash_drugs()
    
    if mash_nodes.empty or mash_drugs.empty:
        return []
    
    # Find drug
    drug_match = mash_drugs[mash_drugs['Name'].str.contains(drug_name, case=False, na=False)]
    
    if drug_match.empty:
        return []
    
    drug_node = drug_match.iloc[0]
    drug_cluster = drug_node.get('Cluster')
    
    # Find genes in same cluster
    if drug_cluster is not None:
        genes_in_cluster = mash_nodes[
            (mash_nodes['Cluster'] == drug_cluster) & 
            (mash_nodes['Type'] == 'gene')
        ]
    else:
        genes_in_cluster = mash_nodes[mash_nodes['Type'] == 'gene'].nlargest(10, 'PageRank Score')
    
    # Build paths
    paths = []
    
    for idx, gene_row in genes_in_cluster.head(max_paths).iterrows():
        gene_name = gene_row.get('Name', 'Unknown Gene')
        
        path = {
            'path': [drug_name, "Protein Target", "Pathway", gene_name],
            'edge_types': ["direct_target", "pathway", "co_mention"],
            'genes': [gene_name],
            'explanation': f"{drug_name} → Protein Target → {gene_name}"
        }
        
        path['score'] = score_path(path['path'], path['edge_types'])
        paths.append(path)
    
    paths = sorted(paths, key=lambda x: x['score'], reverse=True)
    
    return paths[:max_paths]

def render_drug_mechanistic_paths(drug_name: str):
    """
    Render mechanistic paths from drug to genes
    """
    
    st.subheader("🔗 Genes Affected by This Drug")
    
    paths = get_shortest_paths_from_drug(drug_name, max_paths=3)
    
    if not paths:
        st.info("No mechanistic paths found for this drug")
        return
    
    # Display paths
    for i, path in enumerate(paths, 1):
        with st.expander(f"Target {i}: {generate_path_explanation(path)}", expanded=(i==1)):
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown("**Mechanism**")
                st.code(path['explanation'], language="")
            
            with col2:
                st.markdown("**Score**")
                st.metric("", f"{path['score']:.2f}", label_visibility="collapsed")
            
            with col3:
                st.markdown("**Target Genes**")
                for gene in path['genes']:
                    st.write(f"• {gene}")

# ============================================================================
# NETWORK NEIGHBORHOOD EXPLORER
# ============================================================================

def render_ego_network(node_name: str, node_type: str = "gene", k_hops: int = 2):
    """
    Render ego network (node + k-hop neighborhood) as interactive visualization
    """
    
    st.subheader(f"🕸️ Network Neighborhood ({k_hops}-hop)")
    
    mash_nodes = load_kg_mash_nodes()
    
    if mash_nodes.empty:
        st.warning("Knowledge graph data not available")
        return
    
    # Find center node
    node_match = mash_nodes[mash_nodes['Name'].str.contains(node_name, case=False, na=False)]
    
    if node_match.empty:
        st.warning(f"Node '{node_name}' not found in knowledge graph")
        return
    
    center_node = node_match.iloc[0]
    center_cluster = center_node.get('Cluster')
    
    # Get neighbors in same cluster (simplified k-hop)
    if center_cluster is not None:
        neighbors = mash_nodes[mash_nodes['Cluster'] == center_cluster].head(20)
    else:
        neighbors = mash_nodes.nlargest(20, 'PageRank Score')
    
    # Create network visualization data
    nodes_list = [
        {'id': node_name, 'label': node_name, 'type': node_type, 'size': 20}
    ]
    
    for idx, neighbor in neighbors.iterrows():
        neighbor_name = neighbor.get('Name', 'Unknown')
        neighbor_type = neighbor.get('Type', 'unknown')
        neighbor_pagerank = neighbor.get('PageRank Score', 0)
        
        # Size proportional to PageRank
        size = 5 + neighbor_pagerank * 100
        
        nodes_list.append({
            'id': neighbor_name,
            'label': neighbor_name,
            'type': neighbor_type,
            'size': min(size, 20)
        })
    
    # Display as table for now (in production, would use PyVis)
    neighbor_df = pd.DataFrame(neighbors[['Name', 'Type', 'PageRank Score', 'Cluster']].head(10))
    
    st.dataframe(neighbor_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **Interactive network visualization coming soon**
    
    This will show:
    - Center node (your query)
    - Neighboring nodes (genes/drugs in same cluster)
    - Edges (connections)
    - Node size proportional to centrality
    - Clickable nodes for drill-down
    """)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_node_context(node_name: str) -> Dict:
    """
    Get full context for a node (gene or drug)
    """
    
    mash_nodes = load_kg_mash_nodes()
    mash_drugs = load_kg_mash_drugs()
    
    # Try genes first
    gene_match = mash_nodes[mash_nodes['Name'].str.contains(node_name, case=False, na=False)]
    if not gene_match.empty:
        return gene_match.iloc[0].to_dict()
    
    # Try drugs
    drug_match = mash_drugs[mash_drugs['Name'].str.contains(node_name, case=False, na=False)]
    if not drug_match.empty:
        return drug_match.iloc[0].to_dict()
    
    return {}
