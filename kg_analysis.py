"""
Knowledge Graph Analysis Module

Analyzes a gene's position in the MAFLD/MASH subgraph and provides:
- Robust matching of gene/drug/disease names
- Centrality metrics + empirically-derived percentile ranks
- Cluster neighbours (genes, drugs, diseases) with percentiles for speed
- Human-readable centrality interpretation

Designed to be imported by streamlit_app.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import re
import numpy as np
import pandas as pd


# =============================================================================
# Optional legacy filesystem helpers (kept for backwards compatibility)
# =============================================================================

def find_data_dir() -> Optional[Path]:
    """Best-effort locate a local data directory."""
    possible_dirs = [
        Path("meta-liver-data"),
        Path("meta_liver_data"),
        Path("data"),
        Path("../data"),
        Path("../meta-liver-data"),
        Path("../meta_liver_data"),
    ]
    for d in possible_dirs:
        if d.exists():
            return d
    return None


def find_subfolder(parent: Path, folder_pattern: str) -> Optional[Path]:
    """Find a subfolder with case-insensitive matching."""
    if parent is None or not parent.exists():
        return None
    exact_path = parent / folder_pattern
    if exact_path.exists():
        return exact_path
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == folder_pattern.lower():
            return item
    return None


def find_file(directory: Path, filename_pattern: str) -> Optional[Path]:
    """Find a file in a directory (case-insensitive)."""
    if directory is None or not directory.exists():
        return None
    exact_path = directory / filename_pattern
    if exact_path.exists():
        return exact_path
    for item in directory.iterdir():
        if item.is_file() and item.name.lower() == filename_pattern.lower():
            return item
    return None


# =============================================================================
# Internal utilities
# =============================================================================

_NAME_COL_CANDIDATES = ["Name", "Gene", "gene", "Symbol", "symbol", "node", "Node"]
_CLUSTER_COL_CANDIDATES = ["Cluster", "cluster", "community", "Community"]

_PAGERANK_COL_CANDIDATES = ["PageRank Score", "PageRank", "pagerank", "page_rank"]
_BETWEENNESS_COL_CANDIDATES = ["Betweenness Score", "Betweenness", "betweenness"]
_EIGEN_COL_CANDIDATES = ["Eigen Score", "Eigen", "eigen", "eigenvector", "Eigenvector"]

_TYPE_COL_CANDIDATES = ["Type", "type", "node_type"]


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def _to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalise_name(x: Any) -> str:
    """Normalise a node name for matching (gene symbols, drug names, diseases)."""
    if x is None:
        return ""
    s = str(x).strip().upper()

    # Trim common adornments: "TP53 (HUMAN)" -> "TP53"
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)

    # Keep left side of "ENSG000001234.5" -> "ENSG000001234"
    if "." in s:
        left, right = s.rsplit(".", 1)
        if right.isdigit() and len(left) >= 6:
            s = left

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _normalise_name_alnum(x: Any) -> str:
    """Stricter normalisation used as a fallback (removes punctuation)."""
    s = _normalise_name(x)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _pct_rank(v: pd.Series) -> pd.Series:
    """
    Percentile rank in [0,100]. NaNs stay NaN.
    If all values are NaN, returns all NaN.
    """
    if v is None:
        return pd.Series(np.nan, index=None)
    if not isinstance(v, pd.Series):
        v = pd.Series(v)
    if not v.notna().any():
        return pd.Series(np.nan, index=v.index)
    return v.rank(pct=True, method="average") * 100.0


def _weighted_geomean_of_percentiles(
    pr_pct: pd.Series,
    bet_pct: pd.Series,
    eig_pct: pd.Series,
    w_pr: float = 0.5,
    w_bet: float = 0.25,
    w_eig: float = 0.25,
) -> pd.Series:
    """
    Composite centrality computed as a weighted geometric mean of available percentiles.

    Key fix vs prior version:
      - Missing percentiles do NOT get coerced to 0.
      - We renormalise weights per-row over available components.
      - If all components missing -> composite is NaN.

    Returns composite in [0,1] (not a percentile). You can then percentile-rank it.
    """
    eps = 1e-12

    prp = (pr_pct / 100.0)
    betp = (bet_pct / 100.0)
    eigp = (eig_pct / 100.0)

    # valid masks
    v_pr = prp.notna()
    v_bet = betp.notna()
    v_eig = eigp.notna()

    # weight sum per row (renormalise to available components)
    wsum = (w_pr * v_pr.astype(float)) + (w_bet * v_bet.astype(float)) + (w_eig * v_eig.astype(float))

    # log-space mean to avoid underflow
    log_pr = np.log(np.maximum(prp.fillna(1.0).astype(float), eps))
    log_bet = np.log(np.maximum(betp.fillna(1.0).astype(float), eps))
    log_eig = np.log(np.maximum(eigp.fillna(1.0).astype(float), eps))

    num = (w_pr * v_pr.astype(float) * log_pr) + (w_bet * v_bet.astype(float) * log_bet) + (w_eig * v_eig.astype(float) * log_eig)

    comp = np.exp(np.where(wsum.values > 0, (num / wsum).values, np.nan))
    return pd.Series(comp, index=pr_pct.index, dtype=float)


def _ensure_precomputed_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cached numeric + percentile columns for centrality and a composite percentile.
    This runs once per nodes dataframe (idempotent).

    Fixes:
      - Preserve missing metrics as NaN (do NOT coerce to 0.0)
      - Composite is computed from available percentiles with per-row weight renormalisation
    """
    if nodes_df is None or nodes_df.empty:
        return nodes_df

    if "__kg_precomputed__" in nodes_df.attrs and nodes_df.attrs["__kg_precomputed__"]:
        return nodes_df

    name_col = _pick_col(nodes_df, _NAME_COL_CANDIDATES) or "Name"
    if name_col not in nodes_df.columns:
        nodes_df[name_col] = ""

    nodes_df["__name_norm__"] = nodes_df[name_col].map(_normalise_name)
    nodes_df["__name_norm_alnum__"] = nodes_df[name_col].map(_normalise_name_alnum)

    pr_col = _pick_col(nodes_df, _PAGERANK_COL_CANDIDATES)
    bet_col = _pick_col(nodes_df, _BETWEENNESS_COL_CANDIDATES)
    eig_col = _pick_col(nodes_df, _EIGEN_COL_CANDIDATES)

    # Numeric versions (NaN if missing)
    nodes_df["__pagerank__"] = _to_num_series(nodes_df[pr_col]) if pr_col else pd.Series(np.nan, index=nodes_df.index)
    nodes_df["__betweenness__"] = _to_num_series(nodes_df[bet_col]) if bet_col else pd.Series(np.nan, index=nodes_df.index)
    nodes_df["__eigen__"] = _to_num_series(nodes_df[eig_col]) if eig_col else pd.Series(np.nan, index=nodes_df.index)

    # Percentiles computed on available values (NaN stays NaN)
    nodes_df["__pagerank_pct__"] = _pct_rank(nodes_df["__pagerank__"])
    nodes_df["__betweenness_pct__"] = _pct_rank(nodes_df["__betweenness__"])
    nodes_df["__eigen_pct__"] = _pct_rank(nodes_df["__eigen__"])

    # Composite based on percentile ranks (scale-invariant, robust to missing)
    comp = _weighted_geomean_of_percentiles(
        nodes_df["__pagerank_pct__"],
        nodes_df["__betweenness_pct__"],
        nodes_df["__eigen_pct__"],
        w_pr=0.5,
        w_bet=0.25,
        w_eig=0.25,
    )
    nodes_df["__composite__"] = comp
    nodes_df["__composite_pct__"] = _pct_rank(nodes_df["__composite__"])

    nodes_df.attrs["__kg_precomputed__"] = True
    nodes_df.attrs["__kg_name_col__"] = name_col
    return nodes_df


def _standardise_kg_dict(kg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      - {'nodes': df, 'drugs': df, ...}
      - {'MASH_subgraph_nodes': df, 'MASH_subgraph_drugs': df, ...}
    and returns a normalised dict.
    """
    if not isinstance(kg_dict, dict):
        return {}

    if "nodes" in kg_dict:
        return kg_dict

    lower_map = {str(k).lower(): k for k in kg_dict.keys()}

    def _get(key_variants: List[str]) -> Optional[Any]:
        for kv in key_variants:
            if kv.lower() in lower_map:
                return kg_dict[lower_map[kv.lower()]]
        return None

    nodes = _get(["mash_subgraph_nodes", "MASH_subgraph_nodes", "subgraph_nodes", "nodes"])
    drugs = _get(["mash_subgraph_drugs", "MASH_subgraph_drugs", "subgraph_drugs", "drugs"])
    diseases = _get(["mash_subgraph_diseases", "MASH_subgraph_diseases", "subgraph_diseases", "diseases"])

    out: Dict[str, Any] = {}
    if isinstance(nodes, pd.DataFrame):
        out["nodes"] = nodes
    if isinstance(drugs, pd.DataFrame):
        out["drugs"] = drugs
    if isinstance(diseases, pd.DataFrame):
        out["diseases"] = diseases

    return out


def load_kg_data_from_dict(kg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Public helper to normalise KG dict keys to {'nodes','drugs','diseases'}."""
    return _standardise_kg_dict(kg_dict)


def _match_single_node(nodes_df: pd.DataFrame, query: str) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """
    Return best match row for query. Matching strategy:
      1) exact match on normalised name
      2) exact match on strict alnum normalisation
      3) contains match on normalised name (guarded)
    """
    meta: Dict[str, Any] = {"match_type": None, "n_matches": 0, "matched_name": None}

    q1 = _normalise_name(query)
    q2 = _normalise_name_alnum(query)

    if q1 == "" and q2 == "":
        return None, meta

    df = nodes_df

    # 1) Exact on __name_norm__
    m = df[df["__name_norm__"] == q1]
    if not m.empty:
        row = m.iloc[0]
        meta.update(
            match_type="exact",
            n_matches=int(len(m)),
            matched_name=str(row.get(df.attrs.get("__kg_name_col__", "Name"), row.get("Name", ""))),
        )
        return row, meta

    # 2) Exact on __name_norm_alnum__
    m = df[df["__name_norm_alnum__"] == q2]
    if not m.empty:
        row = m.iloc[0]
        meta.update(
            match_type="exact_alnum",
            n_matches=int(len(m)),
            matched_name=str(row.get(df.attrs.get("__kg_name_col__", "Name"), row.get("Name", ""))),
        )
        return row, meta

    # 3) Guarded contains (avoid very short queries)
    if len(q1) >= 3:
        m = df[df["__name_norm__"].str.contains(re.escape(q1), na=False)]
        if not m.empty:
            name_col = df.attrs.get("__kg_name_col__", "Name")
            m = m.assign(__len__=m[name_col].astype(str).str.len())
            m = m.sort_values(["__len__"], ascending=True)
            row = m.iloc[0]
            meta.update(
                match_type="contains",
                n_matches=int(len(m)),
                matched_name=str(row.get(name_col, row.get("Name", ""))),
            )
            return row, meta

    return None, meta


# =============================================================================
# Public API
# =============================================================================

def get_gene_kg_info(gene_name: str, kg_data: Any) -> Optional[Dict[str, Any]]:
    """Get KG info for a gene, including centrality metrics and empirical percentiles."""
    if not isinstance(kg_data, dict):
        return None

    kg_data = _standardise_kg_dict(kg_data)
    nodes_df = kg_data.get("nodes")
    if nodes_df is None or nodes_df.empty:
        return None

    nodes_df = _ensure_precomputed_nodes(nodes_df)

    gene_row, match_meta = _match_single_node(nodes_df, gene_name)
    if gene_row is None:
        return None

    name_col = nodes_df.attrs.get("__kg_name_col__", "Name")
    cluster_col = _pick_col(nodes_df, _CLUSTER_COL_CANDIDATES) or "Cluster"
    type_col = _pick_col(nodes_df, _TYPE_COL_CANDIDATES)

    # Preserve missingness as NaN (UI can render N/A)
    pr = float(gene_row["__pagerank__"]) if pd.notna(gene_row["__pagerank__"]) else np.nan
    bet = float(gene_row["__betweenness__"]) if pd.notna(gene_row["__betweenness__"]) else np.nan
    eig = float(gene_row["__eigen__"]) if pd.notna(gene_row["__eigen__"]) else np.nan

    pr_pct = float(gene_row["__pagerank_pct__"]) if pd.notna(gene_row["__pagerank_pct__"]) else np.nan
    bet_pct = float(gene_row["__betweenness_pct__"]) if pd.notna(gene_row["__betweenness_pct__"]) else np.nan
    eig_pct = float(gene_row["__eigen_pct__"]) if pd.notna(gene_row["__eigen_pct__"]) else np.nan
    comp = float(gene_row["__composite__"]) if pd.notna(gene_row["__composite__"]) else np.nan
    comp_pct = float(gene_row["__composite_pct__"]) if pd.notna(gene_row["__composite_pct__"]) else np.nan

    # Min/max context (ignore NaNs; if no values, report NaN)
    pr_min = float(nodes_df["__pagerank__"].min(skipna=True)) if nodes_df["__pagerank__"].notna().any() else np.nan
    pr_max = float(nodes_df["__pagerank__"].max(skipna=True)) if nodes_df["__pagerank__"].notna().any() else np.nan
    bet_min = float(nodes_df["__betweenness__"].min(skipna=True)) if nodes_df["__betweenness__"].notna().any() else np.nan
    bet_max = float(nodes_df["__betweenness__"].max(skipna=True)) if nodes_df["__betweenness__"].notna().any() else np.nan
    eig_min = float(nodes_df["__eigen__"].min(skipna=True)) if nodes_df["__eigen__"].notna().any() else np.nan
    eig_max = float(nodes_df["__eigen__"].max(skipna=True)) if nodes_df["__eigen__"].notna().any() else np.nan
    comp_min = float(nodes_df["__composite__"].min(skipna=True)) if nodes_df["__composite__"].notna().any() else np.nan
    comp_max = float(nodes_df["__composite__"].max(skipna=True)) if nodes_df["__composite__"].notna().any() else np.nan

    info: Dict[str, Any] = {
        "name": str(gene_row.get(name_col, gene_name)),
        "type": str(gene_row.get(type_col, "")) if type_col else str(gene_row.get("Type", "")),
        "cluster": gene_row.get(cluster_col, None),

        "pagerank": pr,
        "pagerank_min": pr_min,
        "pagerank_max": pr_max,
        "pagerank_percentile": pr_pct,

        "betweenness": bet,
        "bet_min": bet_min,
        "bet_max": bet_max,
        "bet_percentile": bet_pct,

        "eigen": eig,
        "eigen_min": eig_min,
        "eigen_max": eig_max,
        "eigen_percentile": eig_pct,

        "composite": comp,
        "composite_min": comp_min,
        "composite_max": comp_max,
        "composite_percentile": comp_pct,
    }

    info.update(match_meta)
    return info


def _cluster_filter(df: pd.DataFrame, cluster_id: Any) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cluster_col = _pick_col(df, _CLUSTER_COL_CANDIDATES) or "Cluster"
    if cluster_col not in df.columns:
        return df.iloc[0:0]
    cid = str(cluster_id).strip().lower()
    return df[df[cluster_col].astype(str).str.strip().str.lower() == cid]


def _fmt_num_for_table(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        v = float(x)
        if abs(v) >= 1e6:
            return f"{v:,.2f}"
        if abs(v) >= 1e4:
            return f"{v:,.3f}"
        return f"{v:.4f}"
    except Exception:
        return "N/A"


def _fmt_pct_for_table(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "N/A"


def get_cluster_genes(cluster_id: Any, kg_data: Any) -> Optional[pd.DataFrame]:
    """Get genes/proteins in a cluster, sorted by PageRank (desc), with percentiles."""
    if not isinstance(kg_data, dict):
        return None
    kg_data = _standardise_kg_dict(kg_data)
    nodes_df = kg_data.get("nodes")
    if nodes_df is None or nodes_df.empty:
        return None
    nodes_df = _ensure_precomputed_nodes(nodes_df)

    genes = _cluster_filter(nodes_df, cluster_id).copy()
    if genes.empty:
        return None

    type_col = _pick_col(genes, _TYPE_COL_CANDIDATES)
    if type_col and type_col in genes.columns:
        genes = genes[genes[type_col].astype(str).str.lower().str.contains(r"gene|protein", na=False)]
    if genes.empty:
        return None

    name_col = nodes_df.attrs.get("__kg_name_col__", "Name")
    genes = genes.sort_values("__pagerank__", ascending=False)

    out = pd.DataFrame({
        "Name": genes[name_col].astype(str),
        "PageRank": genes["__pagerank__"].map(_fmt_num_for_table),
        "PR %ile": genes["__pagerank_pct__"].map(_fmt_pct_for_table),
        "Betweenness": genes["__betweenness__"].map(_fmt_num_for_table),
        "Bet %ile": genes["__betweenness_pct__"].map(_fmt_pct_for_table),
        "Eigen": genes["__eigen__"].map(_fmt_num_for_table),
        "Eigen %ile": genes["__eigen_pct__"].map(_fmt_pct_for_table),
        "Composite %ile": genes["__composite_pct__"].map(_fmt_pct_for_table),
    })

    return out


def get_cluster_drugs(cluster_id: Any, kg_data: Any) -> Optional[pd.DataFrame]:
    """Get drugs in a cluster, sorted by PageRank (desc), with percentiles (relative to nodes where possible)."""
    if not isinstance(kg_data, dict):
        return None
    kg_data = _standardise_kg_dict(kg_data)

    nodes_df = kg_data.get("nodes")
    if isinstance(nodes_df, pd.DataFrame) and not nodes_df.empty:
        nodes_df = _ensure_precomputed_nodes(nodes_df)
    else:
        nodes_df = None

    drugs_df = kg_data.get("drugs")
    if isinstance(drugs_df, pd.DataFrame) and not drugs_df.empty:
        drugs = _cluster_filter(drugs_df, cluster_id).copy()
    else:
        drugs = None

    # Fallback to nodes table if no dedicated drugs DF
    if drugs is None or drugs.empty:
        if nodes_df is None:
            return None
        drugs = _cluster_filter(nodes_df, cluster_id).copy()
        type_col = _pick_col(drugs, _TYPE_COL_CANDIDATES)
        if type_col and type_col in drugs.columns:
            drugs = drugs[drugs[type_col].astype(str).str.lower().str.contains("drug", na=False)]
        if drugs.empty:
            return None

    name_col = _pick_col(drugs, _NAME_COL_CANDIDATES) or "Name"
    if name_col not in drugs.columns:
        return None

    drugs["__name_norm__"] = drugs[name_col].map(_normalise_name)

    pr_col = _pick_col(drugs, _PAGERANK_COL_CANDIDATES)
    bet_col = _pick_col(drugs, _BETWEENNESS_COL_CANDIDATES)
    eig_col = _pick_col(drugs, _EIGEN_COL_CANDIDATES)

    if pr_col:
        drugs["__pagerank__"] = _to_num_series(drugs[pr_col])
    else:
        drugs["__pagerank__"] = pd.Series(np.nan, index=drugs.index)

    if bet_col:
        drugs["__betweenness__"] = _to_num_series(drugs[bet_col])
    else:
        drugs["__betweenness__"] = pd.Series(np.nan, index=drugs.index)

    if eig_col:
        drugs["__eigen__"] = _to_num_series(drugs[eig_col])
    else:
        drugs["__eigen__"] = pd.Series(np.nan, index=drugs.index)

    # Percentiles: map from nodes if possible; otherwise within-drugs percentiles
    if nodes_df is not None:
        prp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__pagerank_pct__"]))
        betp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__betweenness_pct__"]))
        eigp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__eigen_pct__"]))
        comp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__composite_pct__"]))

        drugs["__pagerank_pct__"] = drugs["__name_norm__"].map(prp_map)
        drugs["__betweenness_pct__"] = drugs["__name_norm__"].map(betp_map)
        drugs["__eigen_pct__"] = drugs["__name_norm__"].map(eigp_map)
        drugs["__composite_pct__"] = drugs["__name_norm__"].map(comp_map)
    else:
        drugs["__pagerank_pct__"] = _pct_rank(drugs["__pagerank__"])
        drugs["__betweenness_pct__"] = _pct_rank(drugs["__betweenness__"])
        drugs["__eigen_pct__"] = _pct_rank(drugs["__eigen__"])
        drugs["__composite_pct__"] = pd.Series(np.nan, index=drugs.index)

    drugs = drugs.sort_values("__pagerank__", ascending=False)

    out = pd.DataFrame({
        "Name": drugs[name_col].astype(str),
        "PageRank": drugs["__pagerank__"].map(_fmt_num_for_table),
        "PR %ile": drugs["__pagerank_pct__"].map(_fmt_pct_for_table),
        "Betweenness": drugs["__betweenness__"].map(_fmt_num_for_table),
        "Bet %ile": drugs["__betweenness_pct__"].map(_fmt_pct_for_table),
        "Eigen": drugs["__eigen__"].map(_fmt_num_for_table),
        "Eigen %ile": drugs["__eigen_pct__"].map(_fmt_pct_for_table),
        "Composite %ile": drugs["__composite_pct__"].map(_fmt_pct_for_table),
    })
    return out


def get_cluster_diseases(cluster_id: Any, kg_data: Any) -> Optional[pd.DataFrame]:
    """Get diseases in a cluster, sorted by PageRank (desc), with percentiles (relative to nodes where possible)."""
    if not isinstance(kg_data, dict):
        return None
    kg_data = _standardise_kg_dict(kg_data)

    nodes_df = kg_data.get("nodes")
    if isinstance(nodes_df, pd.DataFrame) and not nodes_df.empty:
        nodes_df = _ensure_precomputed_nodes(nodes_df)
    else:
        nodes_df = None

    diseases_df = kg_data.get("diseases")
    if isinstance(diseases_df, pd.DataFrame) and not diseases_df.empty:
        dis = _cluster_filter(diseases_df, cluster_id).copy()
    else:
        dis = None

    if dis is None or dis.empty:
        if nodes_df is None:
            return None
        dis = _cluster_filter(nodes_df, cluster_id).copy()
        type_col = _pick_col(dis, _TYPE_COL_CANDIDATES)
        if type_col and type_col in dis.columns:
            dis = dis[dis[type_col].astype(str).str.lower().str.contains("disease", na=False)]
        if dis.empty:
            return None

    name_col = _pick_col(dis, _NAME_COL_CANDIDATES) or "Name"
    if name_col not in dis.columns:
        return None

    dis["__name_norm__"] = dis[name_col].map(_normalise_name)

    pr_col = _pick_col(dis, _PAGERANK_COL_CANDIDATES)
    bet_col = _pick_col(dis, _BETWEENNESS_COL_CANDIDATES)
    eig_col = _pick_col(dis, _EIGEN_COL_CANDIDATES)

    dis["__pagerank__"] = _to_num_series(dis[pr_col]) if pr_col else pd.Series(np.nan, index=dis.index)
    dis["__betweenness__"] = _to_num_series(dis[bet_col]) if bet_col else pd.Series(np.nan, index=dis.index)
    dis["__eigen__"] = _to_num_series(dis[eig_col]) if eig_col else pd.Series(np.nan, index=dis.index)

    if nodes_df is not None:
        prp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__pagerank_pct__"]))
        betp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__betweenness_pct__"]))
        eigp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__eigen_pct__"]))
        comp_map = dict(zip(nodes_df["__name_norm__"], nodes_df["__composite_pct__"]))

        dis["__pagerank_pct__"] = dis["__name_norm__"].map(prp_map)
        dis["__betweenness_pct__"] = dis["__name_norm__"].map(betp_map)
        dis["__eigen_pct__"] = dis["__name_norm__"].map(eigp_map)
        dis["__composite_pct__"] = dis["__name_norm__"].map(comp_map)
    else:
        dis["__pagerank_pct__"] = _pct_rank(dis["__pagerank__"])
        dis["__betweenness_pct__"] = _pct_rank(dis["__betweenness__"])
        dis["__eigen_pct__"] = _pct_rank(dis["__eigen__"])
        dis["__composite_pct__"] = pd.Series(np.nan, index=dis.index)

    dis = dis.sort_values("__pagerank__", ascending=False)

    out = pd.DataFrame({
        "Name": dis[name_col].astype(str),
        "PageRank": dis["__pagerank__"].map(_fmt_num_for_table),
        "PR %ile": dis["__pagerank_pct__"].map(_fmt_pct_for_table),
        "Betweenness": dis["__betweenness__"].map(_fmt_num_for_table),
        "Bet %ile": dis["__betweenness_pct__"].map(_fmt_pct_for_table),
        "Eigen": dis["__eigen__"].map(_fmt_num_for_table),
        "Eigen %ile": dis["__eigen_pct__"].map(_fmt_pct_for_table),
        "Composite %ile": dis["__composite_pct__"].map(_fmt_pct_for_table),
    })
    return out


def interpret_centrality(
    pagerank: float,
    betweenness: float,
    eigen: float,
    pagerank_pct: Optional[float] = None,
    betweenness_pct: Optional[float] = None,
    eigen_pct: Optional[float] = None,
    composite_pct: Optional[float] = None,
) -> str:
    """
    Interpret centrality.

    If percentiles are provided, interpretation is percentile-based (recommended).
    If not, falls back to conservative raw-threshold heuristics.
    """

    def _is_nanlike(x: Any) -> bool:
        return x is None or (isinstance(x, float) and np.isnan(x))

    def _bucket(p: Optional[float]) -> Optional[str]:
        if _is_nanlike(p):
            return None
        p = float(p)
        if p >= 95:
            return "top 5%"
        if p >= 90:
            return "top 10%"
        if p >= 75:
            return "top 25%"
        if p >= 50:
            return "top half"
        if p >= 25:
            return "bottom half"
        return "bottom quartile"

    if not _is_nanlike(pagerank_pct) or not _is_nanlike(betweenness_pct) or not _is_nanlike(eigen_pct) or not _is_nanlike(composite_pct):
        pr_b = _bucket(pagerank_pct)
        bet_b = _bucket(betweenness_pct)
        eig_b = _bucket(eigen_pct)
        comp_b = _bucket(composite_pct)

        parts: List[str] = []
        if comp_b:
            parts.append(f"Composite centrality: {comp_b}")
        if pr_b:
            parts.append(f"PageRank: {pr_b}")
        if bet_b:
            parts.append(f"Betweenness: {bet_b}")
        if eig_b:
            parts.append(f"Eigenvector: {eig_b}")

        if parts:
            return " · ".join(parts)
        return "Centrality percentiles unavailable for interpretation."

    # Raw fallback (less stable across graph size/normalisation)
    hub_like = (not _is_nanlike(pagerank) and pagerank > 0.01) or (not _is_nanlike(eigen) and eigen > 0.01)
    bridge_like = (not _is_nanlike(betweenness) and betweenness > 0.05)

    if hub_like and bridge_like:
        return "Likely a hub-bridge: influential and also connects parts of the network."
    if hub_like:
        return "Likely a hub: relatively influential within the network."
    if bridge_like:
        return "Likely a bridge: may sit on paths connecting network regions."
    return "Likely peripheral: lower influence and fewer connecting paths."
