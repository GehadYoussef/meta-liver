"""
Evidence Scoring Engine for Meta Liver
Computes hepatocyte, microenvironment, and knowledge-graph scores
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import streamlit as st
from data_loaders import (
    load_gse210501, load_gse212837, load_gse189600, load_auc_coassolo,
    load_wgcna_expr, load_wgcna_mod_trait_cor, load_kg_mash_nodes,
    load_wgcna_mes, get_modules, load_gene_mapping
)

# ============================================================================
# HEPATOCYTE SCORE
# ============================================================================

def compute_hepatocyte_score(gene_name: str) -> Dict:
    """
    Compute hepatocyte-intrinsic evidence score
    
    Components:
    - Median AUC across single-cell studies (dispersion-penalized)
    - Direction agreement (sign of logFC)
    - logFC magnitude (winsorized)
    
    Returns:
        Dict with 'score', 'auc_median', 'direction_agreement', 'logfc_strength', 'studies'
    """
    
    studies_data = {}
    auc_values = []
    logfc_values = []
    directions = []
    
    # GSE210501 (Mouse scRNAseq)
    df = load_gse210501()
    if not df.empty and 'Gene' in df.columns:
        gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_match.empty:
            row = gene_match.iloc[0]
            auc = row.get('AUC', np.nan)
            logfc = row.get('avg_LFC', row.get('logFC', np.nan))
            
            if pd.notna(auc):
                auc_values.append(auc)
                studies_data['GSE210501'] = {'auc': auc, 'logfc': logfc}
                if pd.notna(logfc):
                    logfc_values.append(logfc)
                    directions.append(np.sign(logfc))
    
    # GSE212837 (Human snRNAseq)
    df = load_gse212837()
    if not df.empty and 'Gene' in df.columns:
        gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_match.empty:
            row = gene_match.iloc[0]
            auc = row.get('AUC', np.nan)
            logfc = row.get('avg_LFC', row.get('logFC', np.nan))
            
            if pd.notna(auc):
                auc_values.append(auc)
                studies_data['GSE212837'] = {'auc': auc, 'logfc': logfc}
                if pd.notna(logfc):
                    logfc_values.append(logfc)
                    directions.append(np.sign(logfc))
    
    # GSE189600 (Human snRNAseq)
    df = load_gse189600()
    if not df.empty and 'Gene' in df.columns:
        gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_match.empty:
            row = gene_match.iloc[0]
            auc = row.get('AUC', np.nan)
            logfc = row.get('avg_LFC', row.get('logFC', np.nan))
            
            if pd.notna(auc):
                auc_values.append(auc)
                studies_data['GSE189600'] = {'auc': auc, 'logfc': logfc}
                if pd.notna(logfc):
                    logfc_values.append(logfc)
                    directions.append(np.sign(logfc))
    
    # Coassolo AUC
    df = load_auc_coassolo()
    if not df.empty and 'Gene' in df.columns:
        gene_match = df[df['Gene'].str.contains(gene_name, case=False, na=False)]
        if not gene_match.empty:
            row = gene_match.iloc[0]
            auc = row.get('AUC', np.nan)
            if pd.notna(auc):
                auc_values.append(auc)
                studies_data['Coassolo'] = {'auc': auc}
    
    # Compute components
    if not auc_values:
        return {
            'score': 0.0,
            'auc_median': np.nan,
            'auc_dispersion': np.nan,
            'direction_agreement': 0.0,
            'logfc_strength': 0.0,
            'n_studies': 0,
            'studies': studies_data
        }
    
    # AUC median with dispersion penalty
    auc_median = np.median(auc_values)
    auc_std = np.std(auc_values) if len(auc_values) > 1 else 0
    auc_dispersion = 1.0 - (auc_std / 0.5)  # Normalize by typical std
    auc_dispersion = max(0, auc_dispersion)  # Floor at 0
    
    # Direction agreement (allow one neutral)
    if len(directions) > 0:
        direction_agreement = abs(np.mean(directions))  # 0-1 scale
    else:
        direction_agreement = 0.0
    
    # logFC strength (winsorized)
    if logfc_values:
        logfc_winsorized = np.clip(logfc_values, -2, 2)  # Winsorize at ±2
        logfc_strength = np.mean(np.abs(logfc_winsorized)) / 2.0  # Normalize to 0-1
    else:
        logfc_strength = 0.0
    
    # Composite score: 50% AUC robustness, 30% direction, 20% magnitude
    hepatocyte_score = (
        0.5 * (auc_median * auc_dispersion) +
        0.3 * direction_agreement +
        0.2 * logfc_strength
    )
    
    return {
        'score': float(hepatocyte_score),
        'auc_median': float(auc_median),
        'auc_dispersion': float(auc_dispersion),
        'direction_agreement': float(direction_agreement),
        'logfc_strength': float(logfc_strength),
        'n_studies': len(auc_values),
        'studies': studies_data
    }

# ============================================================================
# MICROENVIRONMENT SCORE
# ============================================================================

def compute_microenvironment_score(gene_name: str) -> Dict:
    """
    Compute microenvironment (fibrosis/inflammation) evidence score
    
    Components:
    - WGCNA module hubness (kME)
    - Module-trait correlation (especially fibrosis/inflammation traits)
    
    Returns:
        Dict with 'score', 'kme', 'module', 'module_trait_cor', 'module_annotation'
    """
    
    expr_data = load_wgcna_expr()
    mes_data = load_wgcna_mes()
    mod_trait_cor = load_wgcna_mod_trait_cor()
    
    if expr_data.empty or mes_data.empty:
        return {
            'score': 0.0,
            'kme': np.nan,
            'module': None,
            'module_trait_cor': np.nan,
            'module_annotation': 'unknown',
            'studies': {}
        }
    
    # Find gene in expression matrix
    gene_col = None
    for col in expr_data.columns:
        if gene_name.lower() in col.lower():
            gene_col = col
            break
    
    if gene_col is None:
        return {
            'score': 0.0,
            'kme': np.nan,
            'module': None,
            'module_trait_cor': np.nan,
            'module_annotation': 'not_found',
            'studies': {}
        }
    
    # Get gene expression vector
    gene_expr = expr_data[gene_col].values
    
    # Compute kME (correlation with module eigenvectors)
    kme_values = {}
    for col in mes_data.columns:
        module_me = mes_data[col].values
        kme = np.corrcoef(gene_expr, module_me)[0, 1]
        kme_values[col] = kme
    
    # Find best module
    best_module = max(kme_values, key=lambda x: abs(kme_values[x]))
    best_kme = abs(kme_values[best_module])
    
    # Get module-trait correlation
    module_trait_cor = 0.0
    if best_module in mod_trait_cor.columns:
        # Average absolute correlation across traits
        trait_cors = mod_trait_cor[best_module].values
        module_trait_cor = np.nanmean(np.abs(trait_cors))
    
    # Annotate module (heuristic: modules with high inflammatory/fibrosis traits)
    inflammatory_modules = ['brown', 'orange', 'purple', 'yellow']  # Example
    fibroblast_modules = ['blue', 'turquoise', 'green']  # Example
    
    if best_module in inflammatory_modules:
        annotation = 'inflammatory'
    elif best_module in fibroblast_modules:
        annotation = 'fibroblast'
    else:
        annotation = 'mixed'
    
    # Composite score: 60% kME, 40% module-trait correlation
    microenv_score = 0.6 * best_kme + 0.4 * module_trait_cor
    
    return {
        'score': float(microenv_score),
        'kme': float(best_kme),
        'module': best_module,
        'module_trait_cor': float(module_trait_cor),
        'module_annotation': annotation,
        'kme_all': {k: float(v) for k, v in kme_values.items()}
    }

# ============================================================================
# KNOWLEDGE GRAPH PRIOR
# ============================================================================

def compute_kg_prior(gene_name: str) -> Dict:
    """
    Compute knowledge graph relevance prior
    
    Components:
    - Shortest path distance to known drugs
    - MASH subgraph centrality (PageRank, betweenness)
    - Cluster membership
    
    Returns:
        Dict with 'score', 'centrality', 'shortest_path_to_drug', 'cluster'
    """
    
    mash_nodes = load_kg_mash_nodes()
    
    if mash_nodes.empty or 'Name' not in mash_nodes.columns:
        return {
            'score': 0.0,
            'centrality': np.nan,
            'shortest_path_to_drug': np.inf,
            'cluster': None,
            'node_type': 'unknown'
        }
    
    # Find gene in MASH subgraph
    gene_match = mash_nodes[mash_nodes['Name'].str.contains(gene_name, case=False, na=False)]
    
    if gene_match.empty:
        return {
            'score': 0.0,
            'centrality': np.nan,
            'shortest_path_to_drug': np.inf,
            'cluster': None,
            'node_type': 'not_found'
        }
    
    node_data = gene_match.iloc[0]
    
    # Extract centrality metrics
    pagerank = node_data.get('PageRank Score', np.nan)
    betweenness = node_data.get('Betweenness Sco', np.nan)  # Note: column name might be truncated
    eigen = node_data.get('Eigen Score', np.nan)
    
    # Normalize centrality to 0-1
    centrality = 0.0
    if pd.notna(pagerank):
        centrality += 0.5 * (pagerank / 0.2)  # Normalize by typical max
    if pd.notna(betweenness):
        centrality += 0.3 * (betweenness / 0.2)
    if pd.notna(eigen):
        centrality += 0.2 * (eigen / 1e-11)
    
    centrality = min(1.0, centrality)  # Cap at 1.0
    
    # Get cluster
    cluster = node_data.get('Cluster', None)
    node_type = node_data.get('Type', 'unknown')
    
    # Shortest path to drug (would need to compute from edges)
    # For now, use a heuristic based on cluster
    shortest_path = 2 if cluster is not None else 3
    
    # KG prior score: 50% centrality, 50% path distance
    kg_score = 0.5 * centrality + 0.5 * (1.0 / (1.0 + shortest_path))
    
    return {
        'score': float(kg_score),
        'centrality': float(centrality),
        'pagerank': float(pagerank) if pd.notna(pagerank) else None,
        'betweenness': float(betweenness) if pd.notna(betweenness) else None,
        'eigen': float(eigen) if pd.notna(eigen) else None,
        'shortest_path_to_drug': shortest_path,
        'cluster': cluster,
        'node_type': node_type
    }

# ============================================================================
# COMPOSITE SCORING WITH INTENT WEIGHTING
# ============================================================================

def compute_evidence_fingerprint(gene_name: str, intent: str = "hepatocyte") -> Dict:
    """
    Compute full evidence fingerprint with intent-based weighting
    
    Intent options:
    - "hepatocyte": Weight hepatocyte 70%, microenv 20%, KG 10%
    - "fibrosis": Weight microenv 70%, hepatocyte 20%, KG 10%
    - "repurposing": Weight KG 50%, both scores 25% each
    - "custom": Use custom weights
    
    Returns:
        Dict with hepatocyte, microenv, kg scores + composite + discordance flags
    """
    
    # Compute component scores
    hep_score = compute_hepatocyte_score(gene_name)
    microenv_score = compute_microenvironment_score(gene_name)
    kg_prior = compute_kg_prior(gene_name)
    
    # Intent-based weighting
    weights = {
        "hepatocyte": {"hep": 0.70, "microenv": 0.20, "kg": 0.10},
        "fibrosis": {"hep": 0.20, "microenv": 0.70, "kg": 0.10},
        "repurposing": {"hep": 0.25, "microenv": 0.25, "kg": 0.50},
    }
    
    w = weights.get(intent, weights["hepatocyte"])
    
    # Composite score
    composite_score = (
        w["hep"] * hep_score['score'] +
        w["microenv"] * microenv_score['score'] +
        w["kg"] * kg_prior['score']
    )
    
    # Discordance detection
    hep_high = hep_score['score'] > 0.6
    microenv_high = microenv_score['score'] > 0.6
    kg_high = kg_prior['score'] > 0.6
    
    if hep_high and not microenv_high:
        discordance = "hepatocyte-high / microenv-low"
        taxonomy = "hepatocyte driver"
    elif microenv_high and not hep_high:
        discordance = "microenv-high / hepatocyte-low"
        taxonomy = "microenvironment hub"
    elif hep_high and microenv_high:
        discordance = "concordant"
        taxonomy = "broad disease node"
    else:
        discordance = "low signal"
        taxonomy = "context-specific"
    
    return {
        'gene': gene_name,
        'hepatocyte_score': hep_score,
        'microenv_score': microenv_score,
        'kg_prior': kg_prior,
        'composite_score': float(composite_score),
        'intent': intent,
        'weights': w,
        'discordance': discordance,
        'taxonomy': taxonomy,
        'heterogeneity': 'consistent' if hep_score['n_studies'] >= 2 else 'single_study'
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_intent_description(intent: str) -> str:
    """Get human-readable description of intent"""
    descriptions = {
        "hepatocyte": "Find hepatocyte drug targets (prioritize cell-intrinsic signal)",
        "fibrosis": "Understand fibrosis/NASH biology (prioritize microenvironment)",
        "repurposing": "Mechanistic repurposing (balance all signals, use KG as guide)",
    }
    return descriptions.get(intent, "Custom weighting")

def get_taxonomy_color(taxonomy: str) -> str:
    """Get color for taxonomy label"""
    colors = {
        "hepatocyte driver": "#1f77b4",  # Blue
        "microenvironment hub": "#ff7f0e",  # Orange
        "broad disease node": "#2ca02c",  # Green
        "context-specific": "#d62728",  # Red
    }
    return colors.get(taxonomy, "#808080")  # Gray default
