"""
Meta Liver - Interactive Streamlit App for Liver Genomics Analysis

Tabs:
- Single-Omics Evidence
- MAFLD Knowledge Graph
- WGCNA Fibrosis Stage Networks
- In vitro MASLD model (iHeps)
- Bulk Omics (tissue)
"""

from __future__ import annotations

import sys
import importlib
import inspect
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List, Set

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# IMPORTANT: force imports from THIS app folder first (Streamlit Cloud safety)
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# =============================================================================
# MERGED-IN: robust_data_loader.py (inlined)
# IMPORTANT: do NOT add another "from __future__ import annotations" here
# =============================================================================

from pathlib import Path as _RDL_Path
from typing import Optional as _RDL_Optional, Dict as _RDL_Dict, Tuple as _RDL_Tuple, List as _RDL_List
import warnings as _RDL_warnings
import re as _RDL_re
import os as _RDL_os

import pandas as _RDL_pd


# =============================================================================
# AUTO-DETECT DATA DIRECTORY (RUNTIME)
# =============================================================================

def find_data_dir() -> _RDL_Optional[_RDL_Path]:
    """
    Auto-detect data directory (computed at runtime).

    Resolution order:
    1) META_LIVER_DATA_DIR env var (if set)
    2) common relative folders (repo layout)
    3) common home folders
    """
    env = _RDL_os.environ.get("META_LIVER_DATA_DIR", "").strip()
    if env:
        p = _RDL_Path(env).expanduser()
        if p.exists() and p.is_dir():
            return p.resolve()

    here = _RDL_Path(__file__).resolve().parent

    possible_dirs = [
        _RDL_Path("meta-liver-data"),
        _RDL_Path("meta_liver_data"),
        _RDL_Path("data"),
        _RDL_Path("../meta-liver-data"),
        here / "meta-liver-data",
        here / "meta_liver_data",
        here / "data",
        _RDL_Path.home() / "meta-liver-data",
        _RDL_Path.home() / "meta_liver_data",
    ]
    for dir_path in possible_dirs:
        try:
            if dir_path.exists() and dir_path.is_dir():
                return dir_path.resolve()
        except Exception:
            continue
    return None


def _name_key(s: str) -> str:
    """Normalise a name for tolerant comparisons (case-insensitive, ignore punctuation/spaces)."""
    return _RDL_re.sub(r"[^a-z0-9]+", "", str(s).lower()).strip()


def find_subfolder(parent: _RDL_Path, folder_pattern: str, *, max_depth: int = 4) -> _RDL_Optional[_RDL_Path]:
    """
    Find a subfolder (case-insensitive, tolerant to underscores/dashes/spaces).
    Prefers immediate child matches; then does a shallow search up to max_depth.
    """
    if parent is None or not parent.exists():
        return None

    exact_path = parent / folder_pattern
    if exact_path.exists() and exact_path.is_dir():
        return exact_path

    target_key = _name_key(folder_pattern)

    # 1) immediate children first (fast)
    try:
        for item in parent.iterdir():
            if item.is_dir() and (_name_key(item.name) == target_key):
                return item
    except Exception:
        pass

    # 2) shallow walk with depth control (avoids expensive full rglob on big trees)
    parent = parent.resolve()
    try:
        for root, dirs, _files in _RDL_os.walk(parent):
            root_p = _RDL_Path(root)
            rel = root_p.relative_to(parent)
            depth = 0 if str(rel) == "." else len(rel.parts)
            if depth > max_depth:
                dirs[:] = []
                continue

            for d in dirs:
                if _name_key(d) == target_key:
                    return (root_p / d)
    except Exception:
        pass

    return None


def find_file(directory: _RDL_Path, filename_pattern: str, *, max_depth: int = 6) -> _RDL_Optional[_RDL_Path]:
    """
    Find a file under directory.

    Matching priority:
    1) exact path directory/filename_pattern
    2) case-insensitive exact filename match
    3) tolerant name-key match (ignoring punctuation/spaces)
    4) substring match (last resort)
    """
    if directory is None or not directory.exists():
        return None

    exact_path = directory / filename_pattern
    if exact_path.exists() and exact_path.is_file():
        return exact_path

    target_lower = filename_pattern.lower()
    target_key = _name_key(filename_pattern)

    directory = directory.resolve()

    # shallow walk to avoid huge scans
    try:
        for root, _dirs, files in _RDL_os.walk(directory):
            root_p = _RDL_Path(root)
            rel = root_p.relative_to(directory)
            depth = 0 if str(rel) == "." else len(rel.parts)
            if depth > max_depth:
                continue

            # exact case-insensitive match first
            for fn in files:
                if fn.lower() == target_lower:
                    return root_p / fn

            # tolerant key match
            for fn in files:
                if _name_key(fn) == target_key:
                    return root_p / fn

            # substring fallback
            for fn in files:
                if target_lower in fn.lower():
                    return root_p / fn
    except Exception:
        pass

    return None


def _find_wgcna_dir(data_dir: _RDL_Path) -> _RDL_Optional[_RDL_Path]:
    """Prefer 'wgcna' (current), but still accept legacy 'wcgna'."""
    if data_dir is None:
        return None
    return find_subfolder(data_dir, "wgcna") or find_subfolder(data_dir, "wcgna")


def _find_bulk_omics_dir(data_dir: _RDL_Path) -> _RDL_Optional[_RDL_Path]:
    """
    Find Bulk Omics directory. Accepts names like:
    Bulk_Omics, bulk_omics, Bulk Omics, bulk-omics, etc.
    """
    if data_dir is None:
        return None
    return find_subfolder(data_dir, "bulk_omics")


# =============================================================================
# SAFE READERS
# =============================================================================

def _read_table(file_path: _RDL_Path, *, index_col: _RDL_Optional[int] = None) -> _RDL_pd.DataFrame:
    """Read TSV/CSV/Parquet defensively; returns empty DF on failure."""
    if file_path is None or not file_path.exists():
        return _RDL_pd.DataFrame()

    try:
        suf = file_path.suffix.lower()
        if suf == ".parquet":
            return _RDL_pd.read_parquet(file_path)
        if suf == ".csv":
            return _RDL_pd.read_csv(file_path, index_col=index_col)
        if suf in (".tsv", ".txt"):
            try:
                return _RDL_pd.read_csv(file_path, sep="\t", index_col=index_col)
            except Exception:
                return _RDL_pd.read_csv(file_path, index_col=index_col)
        return _RDL_pd.DataFrame()
    except Exception as e:
        _RDL_warnings.warn(f"Failed to read {file_path}: {e}")
        return _RDL_pd.DataFrame()


def _load_first_existing(directory: _RDL_Path, candidates: _RDL_List[str], *, index_col: _RDL_Optional[int] = None) -> _RDL_pd.DataFrame:
    """Try a list of filenames (case-insensitive); return the first that loads."""
    if directory is None or not directory.exists():
        return _RDL_pd.DataFrame()

    for name in candidates:
        fp = find_file(directory, name)
        if fp is not None:
            df = _read_table(fp, index_col=index_col)
            if not df.empty:
                return df
    return _RDL_pd.DataFrame()


# =============================================================================
# WGCNA LOADERS (supports wgcna/ and wcgna/)
# =============================================================================

def load_wgcna_expr() -> _RDL_pd.DataFrame:
    """Load WGCNA expression matrix."""
    data_dir = find_data_dir()
    if data_dir is None:
        return _RDL_pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return _RDL_pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["datExpr_processed.parquet", "datExpr_processed.csv"],
        index_col=0,
    )


def load_wgcna_mes() -> _RDL_pd.DataFrame:
    """Load WGCNA module eigengenes."""
    data_dir = find_data_dir()
    if data_dir is None:
        return _RDL_pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return _RDL_pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["MEs_processed.parquet", "MEs_processed.csv"],
        index_col=0,
    )


def load_wgcna_mod_trait_cor() -> _RDL_pd.DataFrame:
    """Load module-trait correlations."""
    data_dir = find_data_dir()
    if data_dir is None:
        return _RDL_pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return _RDL_pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["moduleTraitCor.parquet", "moduleTraitCor.csv"],
        index_col=0,
    )


def load_wgcna_mod_trait_pval() -> _RDL_pd.DataFrame:
    """Load module-trait p-values."""
    data_dir = find_data_dir()
    if data_dir is None:
        return _RDL_pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return _RDL_pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["moduleTraitPvalue.parquet", "moduleTraitPvalue.csv"],
        index_col=0,
    )


def load_wgcna_pathways() -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """
    Load all pathway/enrichment tables under <wgcna_dir>/pathways/.

    Returns dict keyed by module (lowercase), e.g. 'black' -> dataframe.
    Module key is inferred from filename: first token split on _ - space.
    Example: 'black_enrichment.csv' -> 'black'.
    """
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return {}

    pathways_dir = find_subfolder(wgcna_dir, "pathways")
    if pathways_dir is None or not pathways_dir.exists():
        return {}

    out: _RDL_Dict[str, _RDL_pd.DataFrame] = {}
    for file_path in pathways_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in (".csv", ".tsv", ".txt", ".parquet"):
            continue

        df = _read_table(file_path, index_col=None)
        if df.empty:
            continue

        # remove common saved-index artifact
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        stem = file_path.stem.strip()
        if not stem:
            continue

        tok = _RDL_re.split(r"[_\-\s]+", stem)[0].strip()
        module_key = (tok if tok else stem).lower()

        if module_key not in out:
            out[module_key] = df

    return out


def load_wgcna_module_trait_heatmap_pdf_path() -> _RDL_Optional[_RDL_Path]:
    """Return a Path to the module-trait heatmap PDF if present, else None."""
    data_dir = find_data_dir()
    if data_dir is None:
        return None

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return None

    candidates = [
        "module-trait-relationships-heatmap.pdf",
        "module_trait_relationships_heatmap.pdf",
        "moduleTraitRelationshipsHeatmap.pdf",
        "moduleTraitRelationships_heatmap.pdf",
        "heatmap.pdf",
    ]
    for name in candidates:
        fp = find_file(wgcna_dir, name)
        if fp is not None and fp.exists():
            return fp
    return None


# =============================================================================
# SINGLE-OMICS / KG / PPI LOADERS
# =============================================================================

def load_single_omics_studies() -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """Load all single-omics studies (all CSV/Parquet files under single_omics)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    single_omics_dir = find_subfolder(data_dir, "single_omics")
    if single_omics_dir is None:
        return {}

    studies: _RDL_Dict[str, _RDL_pd.DataFrame] = {}
    for file_path in single_omics_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in (".csv", ".parquet"):
            continue

        study_name = file_path.stem
        df = _read_table(file_path, index_col=None)
        if not df.empty:
            studies[study_name] = df

    return studies


def load_kg_data() -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """Load all knowledge graph data (all CSV/Parquet files under knowledge_graphs)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    kg_dir = find_subfolder(data_dir, "knowledge_graphs")
    if kg_dir is None:
        return {}

    kg_data: _RDL_Dict[str, _RDL_pd.DataFrame] = {}
    for file_path in kg_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in (".csv", ".parquet"):
            continue

        name = file_path.stem
        df = _read_table(file_path, index_col=None)
        if not df.empty:
            kg_data[name] = df

    return kg_data


def load_ppi_data() -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """Load all PPI network data (all CSV/Parquet files under ppi_networks)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    ppi_dir = find_subfolder(data_dir, "ppi_networks")
    if ppi_dir is None:
        return {}

    ppi_data: _RDL_Dict[str, _RDL_pd.DataFrame] = {}
    for file_path in ppi_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in (".csv", ".parquet"):
            continue

        name = file_path.stem
        df = _read_table(file_path, index_col=None)
        if not df.empty:
            ppi_data[name] = df

    return ppi_data


# =============================================================================
# BULK OMICS LOADERS
# =============================================================================

_BULK_SYMBOL_CANDS = ["Symbol", "symbol", "gene", "Gene", "gene_symbol", "GeneSymbol"]
_BULK_LOGFC_CANDS = ["log2FoldChange", "logFC", "log2FC", "log2_fc", "log2foldchange"]
_BULK_PVAL_CANDS = ["pvalue", "pval", "p_value", "PValue", "P.Value"]
_BULK_PADJ_CANDS = ["padj", "FDR", "adj_pval", "adj_pvalue", "qvalue", "q_value"]


def _pick_col_ci(cols: _RDL_List[str], cands: _RDL_List[str]) -> _RDL_Optional[str]:
    m = {str(c).lower(): c for c in cols}
    for cand in cands:
        if cand in cols:
            return cand
        cl = cand.lower()
        if cl in m:
            return m[cl]
    return None


def _normalise_symbol(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("", "NAN", "NONE"):
        return ""
    return s


def _maybe_promote_index_to_symbol(df: _RDL_pd.DataFrame) -> _RDL_pd.DataFrame:
    """
    Recover gene identifiers when they are not in a 'Symbol' column.

    Common cases:
    - DESeq2 rownames saved into an unnamed first column ('Unnamed: 0')
    - Identifiers stored as the DataFrame index
    """
    if df is None or df.empty:
        return df

    cols = list(df.columns)

    # Unnamed first column -> Symbol
    unnamed0 = None
    for c in cols:
        c0 = str(c).strip().lower()
        if c0 in ("unnamed: 0", "unnamed:0", ""):
            unnamed0 = c
            break
    if unnamed0 is not None:
        tmp = df.copy()
        tmp = tmp.rename(columns={unnamed0: "Symbol"})
        return tmp

    # non-default index -> Symbol
    if not isinstance(df.index, _RDL_pd.RangeIndex):
        try:
            idx = df.index.astype(str)
            if idx.notna().any():
                tmp = df.reset_index().rename(columns={"index": "Symbol"})
                return tmp
        except Exception:
            pass

    return df


def normalise_bulk_deg_table(df: _RDL_pd.DataFrame) -> _RDL_pd.DataFrame:
    """
    Normalise a bulk DEG table to ensure:
    - Symbol column present (including recovery from unnamed/index cases)
    - log2FoldChange / pvalue / padj standardised (if present)
    - duplicate Symbols resolved deterministically
    """
    if df is None or df.empty:
        return _RDL_pd.DataFrame()

    out = _maybe_promote_index_to_symbol(df.copy())

    sym_col = _pick_col_ci(list(out.columns), _BULK_SYMBOL_CANDS)
    lfc_col = _pick_col_ci(list(out.columns), _BULK_LOGFC_CANDS)
    p_col = _pick_col_ci(list(out.columns), _BULK_PVAL_CANDS)
    q_col = _pick_col_ci(list(out.columns), _BULK_PADJ_CANDS)

    if sym_col is None:
        return _RDL_pd.DataFrame()

    ren: _RDL_Dict[str, str] = {sym_col: "Symbol"}
    if lfc_col is not None:
        ren[lfc_col] = "log2FoldChange"
    if p_col is not None:
        ren[p_col] = "pvalue"
    if q_col is not None:
        ren[q_col] = "padj"

    out = out.rename(columns=ren)

    out["Symbol"] = out["Symbol"].astype(str).map(_normalise_symbol)
    out = out.loc[out["Symbol"] != ""].copy()

    if "log2FoldChange" in out.columns:
        out["log2FoldChange"] = _RDL_pd.to_numeric(out["log2FoldChange"], errors="coerce")
    if "pvalue" in out.columns:
        out["pvalue"] = _RDL_pd.to_numeric(out["pvalue"], errors="coerce")
    if "padj" in out.columns:
        out["padj"] = _RDL_pd.to_numeric(out["padj"], errors="coerce")

    # Resolve duplicates by best significance then largest absolute effect
    if out["Symbol"].duplicated().any():
        def _rank_row(r) -> _RDL_Tuple[float, float]:
            q = r.get("padj", float("nan"))
            p = r.get("pvalue", float("nan"))
            lfc = r.get("log2FoldChange", float("nan"))
            best_sig = q if _RDL_pd.notna(q) else (p if _RDL_pd.notna(p) else float("inf"))
            abs_lfc = abs(lfc) if _RDL_pd.notna(lfc) else 0.0
            return (best_sig, -abs_lfc)

        out["__rk__"] = out.apply(_rank_row, axis=1)
        out = out.sort_values("__rk__", ascending=True)
        out = out.drop_duplicates(subset=["Symbol"], keep="first").drop(columns=["__rk__"])

    return out


def load_bulk_omics_tables() -> _RDL_Dict[str, _RDL_Dict[str, _RDL_pd.DataFrame]]:
    """
    Load Bulk Omics DEG tables organised as:
      meta-liver-data/Bulk_Omics/<contrast_folder>/*.(tsv|csv|txt|parquet)

    Returns:
      {contrast_folder_name: {study_file_stem: normalised_df}}
    """
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    bulk_dir = _find_bulk_omics_dir(data_dir)
    if bulk_dir is None or not bulk_dir.exists():
        return {}

    out: _RDL_Dict[str, _RDL_Dict[str, _RDL_pd.DataFrame]] = {}

    for contrast_dir in sorted([p for p in bulk_dir.iterdir() if p.is_dir()]):
        contrast_name = contrast_dir.name
        studies: _RDL_Dict[str, _RDL_pd.DataFrame] = {}

        for fp in sorted(contrast_dir.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in (".tsv", ".csv", ".txt", ".parquet"):
                continue

            df = _read_table(fp, index_col=None)
            if df.empty:
                continue

            df = normalise_bulk_deg_table(df)
            if df.empty:
                continue

            studies[fp.stem] = df

        if studies:
            out[contrast_name] = studies

    return out


def load_bulk_omics_studies() -> _RDL_Dict[str, _RDL_Dict[str, _RDL_pd.DataFrame]]:
    return load_bulk_omics_tables()


def search_gene_in_bulk_omics(gene_symbol: str) -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """
    Search a gene across all Bulk Omics contrasts.
    Returns {contrast_name: dataframe_of_per-study_hits}
    """
    g = _normalise_symbol(gene_symbol)
    if not g:
        return {}

    data = load_bulk_omics_tables()
    if not data:
        return {}

    out: _RDL_Dict[str, _RDL_pd.DataFrame] = {}
    for contrast, studies in data.items():
        rows = []
        for study, df in (studies or {}).items():
            if df is None or df.empty or "Symbol" not in df.columns:
                continue
            hit = df.loc[df["Symbol"] == g]
            if hit.empty:
                continue
            r = hit.iloc[0].to_dict()
            r["Study"] = study
            rows.append(r)

        if rows:
            tmp = _RDL_pd.DataFrame(rows)
            front = [c for c in ["Study", "Symbol", "log2FoldChange", "padj", "pvalue"] if c in tmp.columns]
            rest = [c for c in tmp.columns if c not in front]
            out[contrast] = tmp[front + rest]

    return out


# =============================================================================
# DATA AVAILABILITY / SUMMARY
# =============================================================================

def check_data_availability() -> _RDL_Dict[str, bool]:
    """Check what data is available (pure; upstream Streamlit should cache if desired)."""
    data_dir_ok = find_data_dir() is not None

    expr_ok = False
    mes_ok = False
    cor_ok = False
    pval_ok = False
    pathways_ok = False
    heatmap_ok = False
    single_ok = False
    kg_ok = False
    ppi_ok = False
    bulk_ok = False

    if data_dir_ok:
        expr_ok = not load_wgcna_expr().empty
        mes_ok = not load_wgcna_mes().empty
        cor_ok = not load_wgcna_mod_trait_cor().empty
        pval_ok = not load_wgcna_mod_trait_pval().empty
        pathways_ok = len(load_wgcna_pathways()) > 0
        heatmap_ok = load_wgcna_module_trait_heatmap_pdf_path() is not None
        single_ok = len(load_single_omics_studies()) > 0
        kg_ok = len(load_kg_data()) > 0
        ppi_ok = len(load_ppi_data()) > 0
        bulk_ok = len(load_bulk_omics_tables()) > 0

    return {
        "data_dir": data_dir_ok,
        "wgcna_expr": expr_ok,
        "wgcna_mes": mes_ok,
        "wgcna_mod_trait_cor": cor_ok,
        "wgcna_mod_trait_pval": pval_ok,
        "wgcna_pathways": pathways_ok,
        "wgcna_heatmap_pdf": heatmap_ok,
        "single_omics": single_ok,
        "bulk_omics": bulk_ok,
        "knowledge_graphs": kg_ok,
        "ppi_networks": ppi_ok,
    }


def get_data_summary() -> str:
    """Human-readable data availability summary (no Streamlit formatting dependencies)."""
    avail = check_data_availability()

    if not avail["data_dir"]:
        return "Data Availability:\n\n✗ Data directory not found"

    lines: _RDL_List[str] = ["Data Availability:\n", "✓ Data directory found\n"]

    if avail["wgcna_expr"]:
        expr = load_wgcna_expr()
        lines.append(f"✓ WGCNA Expression: {expr.shape[0]} samples × {expr.shape[1]} genes")
    else:
        lines.append("✗ WGCNA Expression: Not found")

    if avail["wgcna_mes"]:
        mes = load_wgcna_mes()
        lines.append(f"✓ WGCNA Module Eigengenes: {mes.shape[1]} modules")
    else:
        lines.append("✗ WGCNA Module Eigengenes: Not found")

    lines.append("✓ Module-Trait Correlations: Available" if avail["wgcna_mod_trait_cor"] else "✗ Module-Trait Correlations: Not found")
    lines.append("✓ Module-Trait P-values: Available" if avail["wgcna_mod_trait_pval"] else "✗ Module-Trait P-values: Not found")

    lines.append("✓ Pathways/Enrichment: Available" if avail["wgcna_pathways"] else "✗ Pathways/Enrichment: Not found")
    lines.append("✓ Module–trait heatmap PDF: Available" if avail["wgcna_heatmap_pdf"] else "✗ Module–trait heatmap PDF: Not found")

    if avail["single_omics"]:
        studies = load_single_omics_studies()
        lines.append(f"✓ Single-Omics Studies: {len(studies)} datasets")
    else:
        lines.append("✗ Single-Omics Studies: Not found")

    if avail["bulk_omics"]:
        bulk = load_bulk_omics_tables()
        n_contrasts = len(bulk)
        n_studies = sum(len(v) for v in bulk.values())
        lines.append(f"✓ Bulk Omics: {n_contrasts} contrasts, {n_studies} study tables")
    else:
        lines.append("✗ Bulk Omics: Not found")

    lines.append(f"✓ Knowledge Graphs: {len(load_kg_data())} datasets" if avail["knowledge_graphs"] else "✗ Knowledge Graphs: Not found")
    lines.append(f"✓ PPI Networks: {len(load_ppi_data())} datasets" if avail["ppi_networks"] else "✗ PPI Networks: Not found")

    return "\n".join(lines)


# =============================================================================
# SEARCH HELPERS
# =============================================================================

def search_gene_in_studies(gene_name: str) -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """Search for a gene across all single-omics studies (substring match)."""
    studies = load_single_omics_studies()
    results: _RDL_Dict[str, _RDL_pd.DataFrame] = {}

    for study_name, df in studies.items():
        if df is None or df.empty:
            continue

        if "Gene" in df.columns:
            col = df["Gene"].astype(str)
        elif "gene" in df.columns:
            col = df["gene"].astype(str)
        else:
            continue

        matches = df[col.str.contains(gene_name, case=False, na=False)]
        if not matches.empty:
            results[study_name] = matches

    return results


def search_drug_in_kg(drug_name: str) -> _RDL_Dict[str, _RDL_pd.DataFrame]:
    """Search for a drug in knowledge graph tables (substring match on Name column)."""
    kg_data = load_kg_data()
    results: _RDL_Dict[str, _RDL_pd.DataFrame] = {}

    for kg_name, df in kg_data.items():
        if df is None or df.empty:
            continue
        if "Name" not in df.columns:
            continue

        col = df["Name"].astype(str)
        matches = df[col.str.contains(drug_name, case=False, na=False)]
        if not matches.empty:
            results[kg_name] = matches

    return results


# -----------------------------------------------------------------------------
# Expose the inlined loader as an importable module named "robust_data_loader"
# -----------------------------------------------------------------------------
import types as _types

_robust = _types.ModuleType("robust_data_loader")
for _name in [
    "find_data_dir",
    "find_subfolder",
    "find_file",
    "load_wgcna_expr",
    "load_wgcna_mes",
    "load_wgcna_mod_trait_cor",
    "load_wgcna_mod_trait_pval",
    "load_wgcna_pathways",
    "load_wgcna_module_trait_heatmap_pdf_path",
    "load_single_omics_studies",
    "load_kg_data",
    "load_ppi_data",
    "normalise_bulk_deg_table",
    "load_bulk_omics_tables",
    "load_bulk_omics_studies",
    "search_gene_in_bulk_omics",
    "check_data_availability",
    "get_data_summary",
    "search_gene_in_studies",
    "search_drug_in_kg",
]:
    if _name in globals():
        setattr(_robust, _name, globals()[_name])

sys.modules["robust_data_loader"] = _robust
# =============================================================================
# END MERGED-IN robust_data_loader.py
# =============================================================================


# -----------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Meta Liver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DOI = "10.1101/2024.10.10.617610"
APP_DOI_URL = "https://doi.org/10.1101/2024.10.10.617610"

APP_CITATION = (
    "Weihs J, Baldo F, Cardinali A, Youssef G, Ludwik K, Haep N, Tang P, Kumar P, "
    "Engelmann C, Quach S, Meindl M, Kucklick M, Engelmann S, Chillian B, Rothe M, "
    "Meierhofer D, Lurje I, Hammerich L, Ramachandran P, Kendall TJ, Fallowfield JA, "
    "Stachelscheid H, Sauer I, Tacke F, Bufler P, Hudert C, Han N, Rezvani M. "
    "Combined stem cell and predictive models reveal flavin cofactors as targets in metabolic liver dysfunction. "
    "bioRxiv 2024.10.10.617610. doi: 10.1101/2024.10.10.617610"
)

APP_TEAM = (
    "Computational biology: [Professor Namshik Han](https://www.linkedin.com/in/namshik/) (University of Cambridge) and team; "
    "[Dr Gehad Youssef](https://www.linkedin.com/in/dr-gehad-youssef) led the single-omics analysis, "
    "[Dr Fatima Baldo](https://www.linkedin.com/in/fatima-baldo/) led the knowledge graph work, "
    "and [Dr Alessandra Cardinali](https://www.linkedin.com/in/cardinali-alessandra/) led the WGCNA analyses. "
    "Experimental models: [Dr Milad (Milo) Rezvani](https://www.linkedin.com/in/dr-milad-milo-rezvani-aa6a5286/) (Charité) and team; "
    "Julian Weihs led the MAFLD in vitro model."
)


# -----------------------------------------------------------------------------
# STREAMLIT COMPAT SHIMS (avoid use_container_width deprecation warnings)
# -----------------------------------------------------------------------------
def _st_df(df: pd.DataFrame, *, hide_index: bool = True):
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def _st_plotly(fig: go.Figure):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# SAFE IMPORT / RELOAD HELPERS
# -----------------------------------------------------------------------------
def _safe_import_reload(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        mod = importlib.reload(mod)
        return mod, None
    except Exception as e:
        return None, e


# -----------------------------------------------------------------------------
# CORE LOADERS
# -----------------------------------------------------------------------------
loader_err = None
try:
    from robust_data_loader import (
        load_single_omics_studies,
        load_kg_data,
        load_ppi_data,
        find_data_dir as _find_data_dir_global,
        find_subfolder as _find_subfolder_global,
        normalise_bulk_deg_table as _norm_bulk_global,
    )
except Exception as e:
    loader_err = e
    load_single_omics_studies = None  # type: ignore
    load_kg_data = None  # type: ignore
    load_ppi_data = None  # type: ignore
    _find_data_dir_global = None  # type: ignore
    _find_subfolder_global = None  # type: ignore
    _norm_bulk_global = None  # type: ignore

kg_mod, kg_err = _safe_import_reload("kg_analysis")
wgcna_mod, wgcna_err = _safe_import_reload("wgcna_ppi_analysis")
iva, iva_err = _safe_import_reload("invitro_analysis")
soa, soa_err = _safe_import_reload("single_omics_analysis")
bo, bo_err = _safe_import_reload("bulk_omics")


# =============================================================================
# GENERAL FORMAT HELPERS
# =============================================================================
def _is_nanlike(x: object) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except Exception:
        return x is None


def _fmt_pct01(x: object, decimals: int = 1) -> str:
    """x is expected to be in [0,1]."""
    if _is_nanlike(x):
        return "missing"
    try:
        return f"{float(x):.{decimals}%}"
    except Exception:
        return "missing"


def _fmt_auc(x: object) -> str:
    if _is_nanlike(x):
        return "missing"
    try:
        v = float(x)
        if not (0.0 <= v <= 1.0):
            return "missing"
        return f"{v:.3f}"
    except Exception:
        return "missing"


def _fmt_num(x: object, decimals: int = 4, sci_if_small: bool = False) -> str:
    if _is_nanlike(x):
        return "N/A"
    try:
        v = float(x)
        if sci_if_small and 0 < abs(v) < 10 ** (-(decimals)):
            return f"<{10 ** (-(decimals)):.0e}"
        return f"{v:.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_num_commas(x: object, decimals: int = 2) -> str:
    if _is_nanlike(x):
        return "N/A"
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return "N/A"


def _normalise_gene(g: object) -> str:
    s = str(g).strip().upper()
    if s in ("", "NAN", "NONE"):
        return ""
    return s


# =============================================================================
# SINGLE-OMICS PLOT HELPERS
# =============================================================================
def _collect_gene_metrics(gene_name: str, studies_data: dict) -> list[dict]:
    """
    Returns per-study dicts: study, auc_raw, auc_disc, auc_oriented, lfc, direction.
    Uses soa.find_gene_in_study + soa.extract_metrics_from_row for consistency.
    """
    if soa is None:
        return []

    out = []
    for study_name, df in (studies_data or {}).items():
        try:
            row, _ = soa.find_gene_in_study(gene_name, df)
        except Exception:
            row = None

        if row is None:
            continue

        try:
            auc, lfc, direction = soa.extract_metrics_from_row(row)
        except Exception:
            continue

        auc_raw = None
        try:
            if auc is not None and not np.isnan(auc) and 0.0 <= float(auc) <= 1.0:
                auc_raw = float(auc)
        except Exception:
            auc_raw = None

        lfc_val = None
        try:
            if lfc is not None and not np.isnan(lfc):
                lfc_val = float(lfc)
        except Exception:
            lfc_val = None

        auc_disc = float(max(auc_raw, 1.0 - auc_raw)) if auc_raw is not None else None

        auc_oriented = None
        if auc_raw is not None:
            auc_oriented = float(1.0 - auc_raw) if direction == "Healthy" else float(auc_raw)

        out.append(
            {
                "study": study_name,
                "auc_raw": auc_raw,
                "auc_disc": auc_disc,
                "auc_oriented": auc_oriented,
                "lfc": lfc_val,
                "direction": direction,
            }
        )

    return out


def _marker_style(direction: str):
    if direction == "MAFLD":
        return dict(symbol="triangle-up", color="#2E86AB")
    if direction == "Healthy":
        return dict(symbol="triangle-down", color="#A23B72")
    return dict(symbol="circle", color="#777777")


def make_lollipop(metrics: list[dict], auc_key: str, title: str, subtitle: str | None = None) -> go.Figure | None:
    vals = [m for m in metrics if m.get(auc_key) is not None]
    if not vals:
        return None

    vals = sorted(vals, key=lambda x: x[auc_key])
    fig = go.Figure()

    for m in vals:
        fig.add_trace(
            go.Scatter(
                x=[0.5, m[auc_key]],
                y=[m["study"], m["study"]],
                mode="lines",
                line=dict(color="#cccccc", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for m in vals:
        style = _marker_style(m.get("direction"))
        lfc = m.get("lfc")
        size = 10 + abs(lfc if lfc is not None else 0.0) * 1.5
        size = float(min(size, 16))

        hover = f"<b>{m['study']}</b><br>{auc_key}: {m[auc_key]:.3f}"
        if m.get("auc_raw") is not None:
            hover += f"<br>AUC_raw: {m['auc_raw']:.3f}"
        if m.get("auc_disc") is not None:
            hover += f"<br>AUC_disc: {m['auc_disc']:.3f}"
        if m.get("auc_oriented") is not None:
            hover += f"<br>AUC_oriented: {m['auc_oriented']:.3f}"
        if m.get("lfc") is not None:
            hover += f"<br>logFC: {m['lfc']:.3f}"
        if m.get("direction") is not None:
            hover += f"<br>Direction: {m['direction']}"

        fig.add_trace(
            go.Scatter(
                x=[m[auc_key]],
                y=[m["study"]],
                mode="markers",
                marker=dict(
                    size=size,
                    symbol=style["symbol"],
                    color=style["color"],
                    line=dict(color="white", width=1),
                ),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    title_text = title if subtitle is None else f"{title}<br><span style='font-size:11px;color:#666'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#000000")),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color="#000000")),
        height=320,
        hovermode="closest",
        xaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(tickfont=dict(color="#000000", size=11)),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
    )
    return fig


def make_scatter_auc_logfc(metrics: list[dict], auc_key: str, title: str, subtitle: str | None = None) -> go.Figure | None:
    pts = [m for m in metrics if m.get(auc_key) is not None and m.get("lfc") is not None]
    if len(pts) < 2:
        return None

    fig = go.Figure()
    for m in pts:
        style = _marker_style(m.get("direction"))
        fig.add_trace(
            go.Scatter(
                x=[m[auc_key]],
                y=[m["lfc"]],
                mode="markers",
                marker=dict(size=10, symbol=style["symbol"], color="#333333", line=dict(width=0)),
                hovertext=(
                    f"<b>{m['study']}</b>"
                    f"<br>{auc_key}: {m[auc_key]:.3f}"
                    f"<br>logFC: {m['lfc']:.3f}"
                    f"<br>Direction: {m.get('direction', 'Unknown')}"
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", line_width=1.5)

    title_text = title if subtitle is None else f"{title}<br><span style='font-size:11px;color:#666'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#000000")),
        xaxis_title=dict(text="AUROC", font=dict(size=12, color="#000000")),
        yaxis_title=dict(text="logFC (MAFLD vs Healthy)", font=dict(size=12, color="#000000")),
        height=340,
        hovermode="closest",
        xaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0",
        ),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
    )
    return fig


def make_auc_disc_distribution(metrics: list[dict]) -> go.Figure | None:
    vals = [m["auc_disc"] for m in metrics if m.get("auc_disc") is not None]
    if not vals:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=vals,
            boxpoints="all",
            jitter=0.25,
            pointpos=0,
            name="AUC-disc",
            marker=dict(size=8),
            line=dict(width=1),
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#999999", line_width=1.5)
    fig.update_layout(
        title=dict(text="AUC-disc distribution (stability view)", font=dict(size=14, color="#000000")),
        yaxis_title=dict(text="AUC-disc = max(AUC, 1−AUC)", font=dict(size=12, color="#000000")),
        height=320,
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        yaxis=dict(
            range=[0.45, 1.0],
            tickfont=dict(color="#000000", size=11),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#f0f0f0",
        ),
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000", size=11)),
    )
    return fig


# =============================================================================
# KG DISPLAY HELPERS
# =============================================================================
def _fmt_pct(p: object) -> str:
    if _is_nanlike(p):
        return "N/A"
    try:
        return f"{float(p):.1f}%"
    except Exception:
        return "N/A"


def _pct_str_to_float(s: object) -> float:
    if s is None:
        return np.nan
    try:
        t = str(s).strip().replace("%", "")
        return float(t)
    except Exception:
        return np.nan


def _num_str_to_float(s: object) -> float:
    if s is None:
        return np.nan
    try:
        t = str(s).strip().replace(",", "")
        return float(t)
    except Exception:
        return np.nan


def _prepare_cluster_table(df: pd.DataFrame, sort_key: str, top_n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    tmp = df.copy()

    pct_cols = ["Composite %ile", "PR %ile", "Bet %ile", "Eigen %ile"]
    num_cols = ["PageRank", "Betweenness", "Eigen"]
    for c in pct_cols:
        if c in tmp.columns:
            tmp[f"__{c}_num__"] = tmp[c].map(_pct_str_to_float)
    for c in num_cols:
        if c in tmp.columns:
            tmp[f"__{c}_num__"] = tmp[c].map(_num_str_to_float)

    sort_map = {
        "Composite %ile": "__Composite %ile_num__",
        "PR %ile": "__PR %ile_num__",
        "Bet %ile": "__Bet %ile_num__",
        "Eigen %ile": "__Eigen %ile_num__",
        "PageRank (raw)": "__PageRank_num__",
        "Betweenness (raw)": "__Betweenness_num__",
        "Eigen (raw)": "__Eigen_num__",
        "Name": "Name",
    }

    col = sort_map.get(sort_key, sort_key)
    if col in tmp.columns:
        asc = True if sort_key == "Name" else False
        tmp = tmp.sort_values(col, ascending=asc, na_position="last")

    tmp = tmp.head(int(top_n)).copy()
    tmp = tmp[[c for c in tmp.columns if not c.startswith("__")]]
    return tmp


# =============================================================================
# WGCNA DISPLAY HELPERS
# =============================================================================
def _match_module_row(df: pd.DataFrame, module: str) -> str | None:
    if df is None or df.empty:
        return None
    m = str(module).strip()
    candidates = [m, m.lower(), m.upper(), f"ME{m}", f"ME{m.lower()}", f"ME{m.upper()}"]
    for c in candidates:
        if c in df.index:
            return c
    idx_lower = {str(i).lower(): i for i in df.index}
    for c in candidates:
        key = str(c).lower()
        if key in idx_lower:
            return idx_lower[key]
    return None


def _module_trait_table(module_name: str, cor_df: pd.DataFrame, pval_df: pd.DataFrame) -> pd.DataFrame | None:
    if cor_df is None or cor_df.empty:
        return None

    row_key = _match_module_row(cor_df, module_name)
    if row_key is None:
        return None

    cor_row = cor_df.loc[row_key].copy()
    out = cor_row.reset_index()
    out.columns = ["Trait", "Correlation"]

    if pval_df is not None and not pval_df.empty:
        p_key = _match_module_row(pval_df, module_name)
        if p_key is not None:
            p_row = pval_df.loc[p_key].reset_index()
            p_row.columns = ["Trait", "P-value"]
            out = out.merge(p_row, on="Trait", how="left")

    out["__abs_corr__"] = pd.to_numeric(out["Correlation"], errors="coerce").abs()
    out = out.sort_values("__abs_corr__", ascending=False).drop(columns=["__abs_corr__"])
    return out


def _annotate_genes_with_drugs(gene_df: pd.DataFrame, gene_to_drugs_map: dict, max_drugs_per_gene: int) -> pd.DataFrame:
    if gene_df is None or gene_df.empty:
        return gene_df
    if gene_to_drugs_map is None or len(gene_to_drugs_map) == 0:
        return gene_df
    if "Gene" not in gene_df.columns:
        return gene_df

    out = gene_df.copy()
    if "Ensembl ID" in out.columns:
        out = out.drop(columns=["Ensembl ID"])

    def _fmt_drugs(g: str) -> str:
        g0 = str(g).strip().upper()
        recs = gene_to_drugs_map.get(g0, [])
        if not recs:
            return ""
        parts = []
        for r in recs[: int(max_drugs_per_gene)]:
            nm = str(r.get("Drug Name", "")).strip()
            acc = str(r.get("DrugBank_Accession", "")).strip()
            if nm and acc and acc.lower() != "nan":
                parts.append(f"{nm} ({acc})")
            elif nm:
                parts.append(nm)
            elif acc and acc.lower() != "nan":
                parts.append(acc)
        return "; ".join(parts)

    def _n_drugs(g: str) -> int:
        g0 = str(g).strip().upper()
        return int(len(gene_to_drugs_map.get(g0, [])))

    out.insert(1, "n_drugs", out["Gene"].map(_n_drugs))
    out.insert(2, "Drugs", out["Gene"].map(_fmt_drugs))
    return out


# =============================================================================
# DATA LOADING (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_all_data_cached():
    single_omics = {}
    kg_data0 = {}
    ppi_data0 = {}
    wgcna_module_data0 = {}
    wgcna_cor0 = pd.DataFrame()
    wgcna_pval0 = pd.DataFrame()
    wgcna_pathways0 = {}
    active_drugs = pd.DataFrame()
    gene_to_drugs0 = {}
    bulk_omics_data0 = {}

    if load_single_omics_studies is not None:
        single_omics = load_single_omics_studies() or {}
    if load_kg_data is not None:
        kg_data0 = load_kg_data() or {}
    if load_ppi_data is not None:
        ppi_data0 = load_ppi_data() or {}

    if wgcna_mod is not None:
        try:
            wgcna_module_data0 = wgcna_mod.load_wgcna_module_data() or {}
        except Exception:
            wgcna_module_data0 = {}

        try:
            wgcna_cor0 = wgcna_mod.load_wgcna_mod_trait_cor()
        except Exception:
            wgcna_cor0 = pd.DataFrame()

        try:
            wgcna_pval0 = wgcna_mod.load_wgcna_mod_trait_pval()
        except Exception:
            wgcna_pval0 = pd.DataFrame()

        try:
            wgcna_pathways0 = wgcna_mod.load_wgcna_pathways() or {}
        except Exception:
            wgcna_pathways0 = {}

        try:
            active_drugs = wgcna_mod.load_wgcna_active_drugs()
            if active_drugs is None:
                active_drugs = pd.DataFrame()
        except Exception:
            active_drugs = pd.DataFrame()

        try:
            if isinstance(active_drugs, pd.DataFrame) and not active_drugs.empty:
                gene_to_drugs0 = wgcna_mod.build_gene_to_drugs_index(active_drugs) or {}
        except Exception:
            gene_to_drugs0 = {}

    if bo is not None:
        try:
            bulk_omics_data0 = bo.load_bulk_omics() or {}
        except Exception:
            bulk_omics_data0 = {}

    return (
        single_omics,
        kg_data0,
        wgcna_module_data0,
        ppi_data0,
        wgcna_cor0,
        wgcna_pval0,
        wgcna_pathways0,
        active_drugs,
        gene_to_drugs0,
        bulk_omics_data0,
    )


data_loaded = True
load_error = None
try:
    (
        single_omics_data,
        kg_data,
        wgcna_module_data,
        ppi_data,
        wgcna_cor,
        wgcna_pval,
        wgcna_pathways,
        active_drugs_df,
        gene_to_drugs,
        bulk_omics_data,
    ) = load_all_data_cached()
except Exception as e:
    data_loaded = False
    load_error = e
    single_omics_data = {}
    kg_data = {}
    wgcna_module_data = {}
    ppi_data = {}
    wgcna_cor = pd.DataFrame()
    wgcna_pval = pd.DataFrame()
    wgcna_pathways = {}
    active_drugs_df = pd.DataFrame()
    gene_to_drugs = {}
    bulk_omics_data = {}


# =============================================================================
# GENE SCREENER: DATA EXTRACTORS / CACHES
# =============================================================================
def _ensure_session_cache(name: str) -> dict:
    if name not in st.session_state or not isinstance(st.session_state.get(name), dict):
        st.session_state[name] = {}
    return st.session_state[name]


def _get_single_omics_gene_universe() -> List[str]:
    cache = st.session_state.get("_single_omics_universe", None)
    if isinstance(cache, list) and cache:
        return cache
    genes: Set[str] = set()
    for _nm, df in (single_omics_data or {}).items():
        if df is None or df.empty:
            continue
        gcol = None
        for c in ["Gene", "gene", "Symbol", "symbol", "gene_symbol", "GeneSymbol"]:
            if c in df.columns:
                gcol = c
                break
        if gcol is None:
            continue
        try:
            vals = df[gcol].astype(str).map(_normalise_gene)
            for v in vals:
                if v:
                    genes.add(v)
        except Exception:
            continue
    out = sorted(genes)
    st.session_state["_single_omics_universe"] = out
    return out


def _looks_like_gene_symbol(x: str) -> bool:
    s = str(x).strip()
    if not s:
        return False
    if " " in s or "/" in s or "\\" in s:
        return False
    if len(s) < 2 or len(s) > 20:
        return False
    for ch in s:
        if not (ch.isalnum() or ch == "-"):
            return False
    return s.upper() == s


def _get_kg_gene_universe() -> List[str]:
    cache = st.session_state.get("_kg_universe", None)
    if isinstance(cache, list) and cache:
        return cache

    genes: Set[str] = set()
    for _nm, df in (kg_data or {}).items():
        if df is None or df.empty:
            continue
        if "Name" not in df.columns:
            continue

        type_col = None
        for c in ["Type", "type", "NodeType", "node_type", "Node Type", "nodeType", "kind", "Kind"]:
            if c in df.columns:
                type_col = c
                break

        try:
            if type_col is not None:
                t = df[type_col].astype(str).str.lower()
                mask = t.str.contains("gene") | t.str.contains("protein")
                names = df.loc[mask, "Name"].astype(str)
            else:
                names = df["Name"].astype(str)
        except Exception:
            continue

        for v in names:
            s = _normalise_gene(v)
            if s and _looks_like_gene_symbol(s):
                genes.add(s)

    out = sorted(genes)
    st.session_state["_kg_universe"] = out
    return out


def _normalise_module_name(m: object) -> str:
    s = str(m).strip()
    if s.lower().startswith("me"):
        s = s[2:]
    return s.strip().lower()


def _wgcna_modules_available(wgcna_mod_trait_cor: pd.DataFrame, wgcna_module_data_in: dict) -> List[str]:
    cache = st.session_state.get("_wgcna_modules_ui", None)
    if isinstance(cache, list) and cache:
        return cache

    mods: Set[str] = set()
    if isinstance(wgcna_mod_trait_cor, pd.DataFrame) and not wgcna_mod_trait_cor.empty:
        for idx in wgcna_mod_trait_cor.index:
            mods.add(_normalise_module_name(idx))
    if isinstance(wgcna_module_data_in, dict):
        for k, v in wgcna_module_data_in.items():
            if isinstance(k, str) and k and isinstance(v, pd.DataFrame) and ("Gene" in v.columns or "gene" in v.columns):
                mods.add(_normalise_module_name(k))

    out = sorted({m for m in mods if m})
    st.session_state["_wgcna_modules_ui"] = out
    return out


def _get_module_trait_value(module_name: str, trait: str, cor_df: pd.DataFrame, pval_df: pd.DataFrame) -> Tuple[float, float]:
    if cor_df is None or cor_df.empty or trait is None or trait == "":
        return np.nan, np.nan
    row_key = _match_module_row(cor_df, module_name)
    if row_key is None:
        return np.nan, np.nan
    try:
        corr = float(pd.to_numeric(cor_df.loc[row_key, trait], errors="coerce"))
    except Exception:
        corr = np.nan

    p = np.nan
    if pval_df is not None and not pval_df.empty:
        p_key = _match_module_row(pval_df, module_name)
        if p_key is not None:
            try:
                p = float(pd.to_numeric(pval_df.loc[p_key, trait], errors="coerce"))
            except Exception:
                p = np.nan
    return corr, p


def _infer_gene_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ["Gene", "gene", "Symbol", "symbol", "Gene name", "gene_name", "GeneName", "gene_name_symbol", "Gene Symbol"]:
        if c in df.columns:
            return c
    return None


def _infer_lfc_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ["log2FoldChange", "logFC", "log2FC", "log2_fc", "lfc", "LFC"]:
        if c in df.columns:
            return c
    return None


def _infer_padj_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ["padj", "FDR", "qvalue", "q_value", "adj_pval", "adj_pvalue"]:
        if c in df.columns:
            return c
    for c in ["pvalue", "pval", "p_value", "P.Value"]:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def _load_invitro_all_tables_cached() -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Loads all invitro DEG tables under meta-liver-data/stem_cell_model/.
    Returns {(line, contrast): df_idx}, where df_idx is indexed by Gene and has:
      log2FoldChange (float), padj (float, may be NaN)
    """
    out: Dict[Tuple[str, str], pd.DataFrame] = {}

    if _find_data_dir_global is None or _find_subfolder_global is None:
        return out

    data_dir = _find_data_dir_global()
    if data_dir is None:
        return out

    stem_dir = _find_subfolder_global(data_dir, "stem_cell_model") or _find_subfolder_global(data_dir, "stem_cell")
    if stem_dir is None or not stem_dir.exists():
        return out

    try:
        files = [p for p in stem_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".parquet", ".csv", ".tsv", ".txt")]
    except Exception:
        files = []

    def _parse_line_contrast(stem: str) -> Tuple[Optional[str], Optional[str]]:
        s = str(stem).strip()
        if s.lower().startswith("processed_degs_"):
            s2 = s[len("processed_degs_") :]
            parts = s2.split("_", 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        parts = s.rsplit("_", 1)
        if len(parts) == 2:
            return parts[1].strip(), parts[0].strip()
        return None, None

    for fp in files:
        line, contrast = _parse_line_contrast(fp.stem)
        if not line or not contrast:
            continue

        df = _read_table(fp, index_col=None)
        if df is None or df.empty:
            continue

        gcol = _infer_gene_col(df)
        lfc_col = _infer_lfc_col(df)
        padj_col = _infer_padj_col(df)

        if gcol is None:
            tmp = _maybe_promote_index_to_symbol(df.copy()).copy()
            if "Symbol" in tmp.columns:
                df = tmp
                gcol = "Symbol"

        if gcol is None or lfc_col is None:
            continue

        tmp = df.copy()
        tmp = tmp.rename(columns={gcol: "Gene", lfc_col: "log2FoldChange"})
        tmp["Gene"] = tmp["Gene"].astype(str).map(_normalise_gene)
        tmp = tmp.loc[tmp["Gene"] != ""].copy()
        tmp["log2FoldChange"] = pd.to_numeric(tmp["log2FoldChange"], errors="coerce")

        if padj_col is not None and padj_col in tmp.columns:
            tmp = tmp.rename(columns={padj_col: "padj"})
            tmp["padj"] = pd.to_numeric(tmp["padj"], errors="coerce")
        else:
            tmp["padj"] = np.nan

        # de-dupe by best padj then largest abs lfc
        if tmp["Gene"].duplicated().any():
            tmp["__rk__"] = list(zip(tmp["padj"].fillna(np.inf), -tmp["log2FoldChange"].abs().fillna(0.0)))
            tmp = tmp.sort_values("__rk__", ascending=True).drop_duplicates("Gene", keep="first").drop(columns=["__rk__"])

        tmp = tmp[["Gene", "log2FoldChange", "padj"]].copy()
        tmp = tmp.set_index("Gene", drop=True)
        out[(str(line).strip(), str(contrast).strip())] = tmp

    return out


def _invitro_contrasts_available(invitro_tables: Dict[Tuple[str, str], pd.DataFrame]) -> List[str]:
    return sorted({contrast for (_line, contrast) in invitro_tables.keys()})


def _bulk_groups_available(bulk_data: dict) -> List[str]:
    try:
        return sorted(list((bulk_data or {}).keys()))
    except Exception:
        return []


def _bulk_group_gene_set(group: str) -> Set[str]:
    cache = _ensure_session_cache("_cache_bulk_group_genes")
    g0 = str(group)
    if g0 in cache and isinstance(cache[g0], set):
        return cache[g0]

    genes: Set[str] = set()
    studies = (bulk_omics_data or {}).get(g0, {})
    if isinstance(studies, dict):
        for _study, df in studies.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            dfn = df
            if "Symbol" not in dfn.columns and _norm_bulk_global is not None:
                try:
                    dfn = _norm_bulk_global(dfn)
                except Exception:
                    dfn = df
            if isinstance(dfn, pd.DataFrame) and "Symbol" in dfn.columns:
                try:
                    for v in dfn["Symbol"].astype(str).map(_normalise_gene):
                        if v:
                            genes.add(v)
                except Exception:
                    pass
    cache[g0] = genes
    return genes


def _bulk_study_index_df(group: str, study: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tiny, indexed DF for fast lookups:
      index=Symbol, columns=[log2FoldChange, padj]
    Cached per (group, study) in session_state.
    """
    cache = _ensure_session_cache("_cache_bulk_study_idxdf")
    key = (str(group), str(study))
    if key in cache and isinstance(cache[key], pd.DataFrame):
        return cache[key]

    dfn = df
    if isinstance(dfn, pd.DataFrame) and not dfn.empty and "Symbol" not in dfn.columns and _norm_bulk_global is not None:
        try:
            dfn = _norm_bulk_global(dfn)
        except Exception:
            dfn = df

    if not isinstance(dfn, pd.DataFrame) or dfn.empty or "Symbol" not in dfn.columns:
        out = pd.DataFrame(columns=["log2FoldChange", "padj"])
        cache[key] = out
        return out

    cols = ["Symbol"]
    if "log2FoldChange" in dfn.columns:
        cols.append("log2FoldChange")
    if "padj" in dfn.columns:
        cols.append("padj")

    tmp = dfn[cols].copy()
    tmp["Symbol"] = tmp["Symbol"].astype(str).map(_normalise_gene)
    tmp = tmp.loc[tmp["Symbol"] != ""].copy()

    if "log2FoldChange" in tmp.columns:
        tmp["log2FoldChange"] = pd.to_numeric(tmp["log2FoldChange"], errors="coerce")
    else:
        tmp["log2FoldChange"] = np.nan

    if "padj" in tmp.columns:
        tmp["padj"] = pd.to_numeric(tmp["padj"], errors="coerce")
    else:
        tmp["padj"] = np.nan

    if tmp["Symbol"].duplicated().any():
        tmp["__rk__"] = list(zip(tmp["padj"].fillna(np.inf), -tmp["log2FoldChange"].abs().fillna(0.0)))
        tmp = tmp.sort_values("__rk__", ascending=True).drop_duplicates("Symbol", keep="first").drop(columns=["__rk__"])

    tmp = tmp.set_index("Symbol", drop=True)
    cache[key] = tmp
    return tmp


# =============================================================================
# GENE SCREENER: PER-GENE EVIDENCE (cached in session_state)
# =============================================================================
def _get_single_omics_summary(gene: str, studies: dict) -> dict:
    cache = _ensure_session_cache("_cache_singleomics")
    g = _normalise_gene(gene)
    if not g:
        return {}
    if g in cache:
        return cache[g]

    out = {}
    if soa is not None and studies:
        try:
            cs = soa.compute_consistency_score(g, studies)
        except Exception:
            cs = None
        if isinstance(cs, dict):
            out = {
                "evidence_score": cs.get("evidence_score", np.nan),
                "direction_agreement": cs.get("direction_agreement", np.nan),
                "auc_median_discriminative": cs.get("auc_median_discriminative", np.nan),
                "found_count": cs.get("found_count", np.nan),
                "n_auc": cs.get("n_auc", np.nan),
            }
    cache[g] = out
    return out


def _get_kg_summary(gene: str, kg_data_in: dict) -> dict:
    cache = _ensure_session_cache("_cache_kg")
    g = _normalise_gene(gene)
    if not g:
        return {}
    if g in cache:
        return cache[g]

    out = {}
    if kg_mod is not None and kg_data_in:
        try:
            info = kg_mod.get_gene_kg_info(g, kg_data_in)
        except Exception:
            info = None
        if isinstance(info, dict) and info:
            out = {
                "cluster": info.get("cluster", None),
                "composite_percentile": info.get("composite_percentile", np.nan),
                "pagerank_percentile": info.get("pagerank_percentile", np.nan),
                "bet_percentile": info.get("bet_percentile", np.nan),
                "eigen_percentile": info.get("eigen_percentile", np.nan),
                "composite": info.get("composite", np.nan),
                "pagerank": info.get("pagerank", np.nan),
                "betweenness": info.get("betweenness", np.nan),
                "eigen": info.get("eigen", np.nan),
            }
    cache[g] = out
    return out


def _get_wgcna_summary(gene: str, wgcna_module_data_in: dict, trait: str) -> dict:
    cache = _ensure_session_cache("_cache_wgcna")
    g = _normalise_gene(gene)
    if not g:
        return {}
    key = (g, str(trait or ""))
    if key in cache:
        return cache[key]

    module = None
    if wgcna_mod is not None and wgcna_module_data_in:
        try:
            mi = wgcna_mod.get_gene_module(g, wgcna_module_data_in)
        except Exception:
            mi = None
        if isinstance(mi, dict) and "module" in mi:
            module = mi.get("module", None)

    if module is not None:
        corr, p = _get_module_trait_value(str(module), str(trait), wgcna_cor, wgcna_pval)
    else:
        corr, p = np.nan, np.nan

    n_drugs = 0
    has_drug = False
    try:
        recs = (gene_to_drugs or {}).get(g, [])
        n_drugs = int(len(recs))
        has_drug = n_drugs > 0
    except Exception:
        pass

    out = {
        "module": module,
        "trait": trait,
        "trait_corr": corr,
        "trait_p": p,
        "has_drug_target": has_drug,
        "n_drugs": n_drugs,
    }
    cache[key] = out
    return out


def _get_invitro_summary(gene: str, invitro_tables: Dict[Tuple[str, str], pd.DataFrame], padj_thr: float, require_both_lines: bool) -> dict:
    cache = _ensure_session_cache("_cache_invitro")
    g = _normalise_gene(gene)
    if not g:
        return {}
    key = (g, float(padj_thr), bool(require_both_lines))
    if key in cache:
        return cache[key]

    per_contrast: Dict[str, dict] = {}
    if invitro_tables:
        contrasts = sorted({c for (_l, c) in invitro_tables.keys()})
        for c in contrasts:
            rows = []
            for (line, cc), df in invitro_tables.items():
                if cc != c or df is None or df.empty:
                    continue
                if g not in df.index:
                    continue
                r = df.loc[g]
                lfc = float(r.get("log2FoldChange", np.nan)) if pd.notna(r.get("log2FoldChange", np.nan)) else np.nan
                padj = float(r.get("padj", np.nan)) if pd.notna(r.get("padj", np.nan)) else np.nan
                sig = True if np.isnan(padj) else (padj <= padj_thr)

                if np.isnan(lfc):
                    direction = "missing"
                elif lfc > 0:
                    direction = "up"
                elif lfc < 0:
                    direction = "down"
                else:
                    direction = "zero"

                rows.append({"line": str(line), "lfc": lfc, "padj": padj, "sig": sig, "dir": direction})

            if not rows:
                continue

            sig_rows = [r for r in rows if r["sig"] and r["dir"] in ("up", "down")]
            if require_both_lines:
                ok = False
                cons_dir = "mixed"
                if len(sig_rows) >= 2:
                    dirs = {r["dir"] for r in sig_rows}
                    if len(dirs) == 1:
                        ok = True
                        cons_dir = list(dirs)[0]
                per_contrast[c] = {
                    "ok": ok,
                    "direction": cons_dir,
                    "n_lines_hit": len(rows),
                    "n_lines_sig": len(sig_rows),
                    "min_padj": np.nanmin([r["padj"] for r in rows]) if any(pd.notna(r["padj"]) for r in rows) else np.nan,
                    "mean_lfc": float(np.nanmean([r["lfc"] for r in rows])),
                }
            else:
                ok = len(sig_rows) >= 1
                cons_dir = "mixed"
                if len(sig_rows) == 1:
                    cons_dir = sig_rows[0]["dir"]
                elif len(sig_rows) > 1:
                    dirs = {r["dir"] for r in sig_rows}
                    cons_dir = list(dirs)[0] if len(dirs) == 1 else "mixed"
                per_contrast[c] = {
                    "ok": ok,
                    "direction": cons_dir,
                    "n_lines_hit": len(rows),
                    "n_lines_sig": len(sig_rows),
                    "min_padj": np.nanmin([r["padj"] for r in rows]) if any(pd.notna(r["padj"]) for r in rows) else np.nan,
                    "mean_lfc": float(np.nanmean([r["lfc"] for r in rows])),
                }

    ok_contrasts = [c for c, d in per_contrast.items() if d.get("ok") is True]
    out = {
        "ok_contrasts": ok_contrasts,
        "n_ok": len(ok_contrasts),
        "per_contrast": per_contrast,
    }
    cache[key] = out
    return out


def _get_bulk_summary(
    gene: str,
    bulk_data_in: dict,
    groups: List[str],
    padj_thr: float,
    min_n_studies: int,
    require_consistent_direction: bool,
    require_sig_all_found: bool,
) -> dict:
    cache = _ensure_session_cache("_cache_bulk")
    g = _normalise_gene(gene)
    if not g:
        return {}
    key = (
        g,
        tuple(sorted(groups or [])),
        float(padj_thr),
        int(min_n_studies),
        bool(require_consistent_direction),
        bool(require_sig_all_found),
    )
    if key in cache:
        return cache[key]

    per_group: Dict[str, dict] = {}

    for grp in (groups or []):
        studies = (bulk_data_in or {}).get(grp, {})
        if not isinstance(studies, dict) or not studies:
            continue

        found = 0
        pos = 0
        neg = 0
        sig_found = 0
        any_padj_present = False
        signs: List[int] = []

        for study, df in studies.items():
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue
            if not isinstance(df, pd.DataFrame):
                continue

            idxdf = _bulk_study_index_df(grp, study, df)
            if idxdf is None or idxdf.empty:
                continue
            if g not in idxdf.index:
                continue

            r = idxdf.loc[g]
            lfc = pd.to_numeric(r.get("log2FoldChange", np.nan), errors="coerce")
            padj = pd.to_numeric(r.get("padj", np.nan), errors="coerce")

            found += 1

            if pd.notna(padj):
                any_padj_present = True
                if float(padj) <= padj_thr:
                    sig_found += 1

            if pd.notna(lfc):
                if float(lfc) > 0:
                    pos += 1
                    signs.append(1)
                elif float(lfc) < 0:
                    neg += 1
                    signs.append(-1)

        if found == 0:
            continue

        if require_sig_all_found:
            sig_ok = (sig_found == found) if any_padj_present else True
        else:
            sig_ok = (sig_found >= 1) if any_padj_present else True

        dir_ok = True
        direction = "mixed"
        if require_consistent_direction:
            if len(signs) == 0:
                dir_ok = False
                direction = "missing"
            else:
                if all(s == 1 for s in signs):
                    direction = "up"
                elif all(s == -1 for s in signs):
                    direction = "down"
                else:
                    direction = "mixed"
                    dir_ok = False
        else:
            if len(signs) > 0:
                direction = "up" if pos >= neg else "down"

        ok = (found >= int(min_n_studies)) and dir_ok and sig_ok

        per_group[grp] = {
            "ok": ok,
            "n_found": found,
            "n_sig": sig_found,
            "direction": direction,
            "pos": pos,
            "neg": neg,
        }

    ok_groups = [g0 for g0, d in per_group.items() if d.get("ok") is True]
    out = {
        "ok_groups": ok_groups,
        "n_ok": len(ok_groups),
        "per_group": per_group,
    }
    cache[key] = out
    return out


# =============================================================================
# GENE SCREENER: MAIN DRIVER (UPDATED)
#   - removed "Gene contains"
#   - removed "Max genes to scan" (always scans full candidate universe)
#   - invitro + bulk: support OR vs AND selection mode
#   - returns top max_return AFTER scanning, not "first N hits"
# =============================================================================
def run_gene_screener(
    *,
    max_return: int,
    # single-omics
    use_single: bool,
    min_agreement: float,
    min_auc_disc: float,
    min_evidence: float,
    # KG
    use_kg: bool,
    require_cluster: bool,
    min_composite_pctile: float,
    # WGCNA
    use_wgcna: bool,
    wgcna_modules: List[str],
    wgcna_trait: str,
    wgcna_corr_dir: str,
    wgcna_min_abs_corr: float,
    wgcna_p_thr: float,
    require_drug_target: bool,
    # in vitro
    use_invitro: bool,
    invitro_contrasts: List[str],
    invitro_match_mode: str,  # "Any" or "All"
    invitro_padj_thr: float,
    invitro_require_both_lines: bool,
    # bulk
    use_bulk: bool,
    bulk_groups: List[str],
    bulk_match_mode: str,  # "Any" or "All"
    bulk_padj_thr: float,
    bulk_min_n_studies: int,
    bulk_require_consistent_dir: bool,
    bulk_require_sig_all_found: bool,
) -> pd.DataFrame:
    invitro_tables = _load_invitro_all_tables_cached()

    candidates: Set[str] = set()

    # Prefer restrictive / cheap seed sets first
    if use_invitro and invitro_contrasts:
        for (_line, contrast), df in invitro_tables.items():
            if contrast not in invitro_contrasts:
                continue
            if df is None or df.empty:
                continue
            candidates.update(set(df.index.astype(str)))
    elif use_bulk and bulk_groups:
        for grp in bulk_groups:
            candidates.update(_bulk_group_gene_set(grp))
    elif use_single:
        candidates.update(_get_single_omics_gene_universe())
    elif use_kg:
        candidates.update(_get_kg_gene_universe())
    else:
        # if nothing selected, still use single-omics if present, else KG
        u1 = _get_single_omics_gene_universe()
        if u1:
            candidates.update(u1)
        else:
            candidates.update(_get_kg_gene_universe())

    cand_list = sorted({_normalise_gene(g) for g in candidates if _normalise_gene(g)})

    rows: List[dict] = []

    want_modules = {_normalise_module_name(x) for x in (wgcna_modules or [])}
    invitro_mode_any = str(invitro_match_mode or "Any").lower().startswith("any")
    bulk_mode_any = str(bulk_match_mode or "Any").lower().startswith("any")

    for g in cand_list:
        # KG filter
        kg_sum = _get_kg_summary(g, kg_data) if use_kg else {}
        if use_kg:
            if not kg_sum:
                continue
            if require_cluster:
                cl = kg_sum.get("cluster", None)
                if cl is None or str(cl).strip() == "" or str(cl).lower() == "nan":
                    continue
            comp = kg_sum.get("composite_percentile", np.nan)
            compf = _pct_str_to_float(comp) if isinstance(comp, str) else (float(comp) if pd.notna(comp) else np.nan)
            if pd.notna(min_composite_pctile):
                if pd.isna(compf) or float(compf) < float(min_composite_pctile):
                    continue

        # WGCNA filter
        w_sum = _get_wgcna_summary(g, wgcna_module_data, wgcna_trait) if use_wgcna else {}
        if use_wgcna:
            if not w_sum:
                continue
            mod = w_sum.get("module", None)
            if want_modules:
                mod_key = _normalise_module_name(mod) if mod is not None else ""
                if mod_key not in want_modules:
                    continue

            corr = w_sum.get("trait_corr", np.nan)
            p = w_sum.get("trait_p", np.nan)
            if pd.isna(corr):
                continue

            if pd.notna(wgcna_min_abs_corr) and abs(float(corr)) < float(wgcna_min_abs_corr):
                continue
            if wgcna_corr_dir == "Positive" and not (float(corr) > 0):
                continue
            if wgcna_corr_dir == "Negative" and not (float(corr) < 0):
                continue
            if pd.notna(wgcna_p_thr) and not pd.isna(p) and float(p) > float(wgcna_p_thr):
                continue
            if require_drug_target and not bool(w_sum.get("has_drug_target", False)):
                continue

        # in vitro filter (OR/AND across selected contrasts)
        inv_sum = _get_invitro_summary(g, invitro_tables, invitro_padj_thr, invitro_require_both_lines) if use_invitro else {}
        if use_invitro:
            if not inv_sum:
                continue
            ok_contr = set(inv_sum.get("ok_contrasts", []))
            want_contr = set(invitro_contrasts or [])
            if want_contr:
                if invitro_mode_any:
                    if len(want_contr.intersection(ok_contr)) == 0:
                        continue
                else:
                    if not want_contr.issubset(ok_contr):
                        continue

        # bulk filter (OR/AND across selected groups)
        bulk_sum = _get_bulk_summary(
            g,
            bulk_omics_data,
            bulk_groups,
            bulk_padj_thr,
            bulk_min_n_studies,
            bulk_require_consistent_dir,
            bulk_require_sig_all_found,
        ) if use_bulk else {}
        if use_bulk:
            if not bulk_sum:
                continue
            ok_groups = set(bulk_sum.get("ok_groups", []))
            want_groups = set(bulk_groups or [])
            if want_groups:
                if bulk_mode_any:
                    if len(want_groups.intersection(ok_groups)) == 0:
                        continue
                else:
                    if not want_groups.issubset(ok_groups):
                        continue

        # single-omics filter (last)
        so_sum = _get_single_omics_summary(g, single_omics_data) if use_single else {}
        if use_single:
            if not so_sum:
                continue
            da = so_sum.get("direction_agreement", np.nan)
            aucd = so_sum.get("auc_median_discriminative", np.nan)
            ev = so_sum.get("evidence_score", np.nan)

            if pd.isna(da) or float(da) < float(min_agreement):
                continue
            if pd.isna(aucd) or float(aucd) < float(min_auc_disc):
                continue
            if pd.isna(ev) or float(ev) < float(min_evidence):
                continue

        # build evidence row (always include all tabs)
        so_all = _get_single_omics_summary(g, single_omics_data) if single_omics_data else {}
        kg_all = _get_kg_summary(g, kg_data) if kg_data else {}
        w_all = _get_wgcna_summary(g, wgcna_module_data, wgcna_trait) if (wgcna_module_data and wgcna_trait) else {}
        inv_all = _get_invitro_summary(g, invitro_tables, invitro_padj_thr, invitro_require_both_lines) if invitro_tables else {}
        bulk_all = _get_bulk_summary(
            g,
            bulk_omics_data,
            bulk_groups if bulk_groups else [],
            bulk_padj_thr,
            bulk_min_n_studies,
            bulk_require_consistent_dir,
            bulk_require_sig_all_found,
        ) if bulk_omics_data else {}

        rows.append(
            {
                "Gene": g,
                "SingleOmics_evidence": so_all.get("evidence_score", np.nan),
                "SingleOmics_dir_agreement": so_all.get("direction_agreement", np.nan),
                "SingleOmics_auc_disc_median": so_all.get("auc_median_discriminative", np.nan),
                "SingleOmics_found": so_all.get("found_count", np.nan),
                "KG_cluster": kg_all.get("cluster", None),
                "KG_composite_%ile": kg_all.get("composite_percentile", np.nan),
                "WGCNA_module": w_all.get("module", None),
                f"WGCNA_{wgcna_trait}_corr": w_all.get("trait_corr", np.nan),
                f"WGCNA_{wgcna_trait}_p": w_all.get("trait_p", np.nan),
                "WGCNA_has_drug_target": w_all.get("has_drug_target", False),
                "WGCNA_n_drugs": w_all.get("n_drugs", 0),
                "InVitro_ok_contrasts": "; ".join(inv_all.get("ok_contrasts", [])) if inv_all else "",
                "Bulk_ok_groups": "; ".join(bulk_all.get("ok_groups", [])) if bulk_all else "",
            }
        )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    # sort: single-omics desc then agreement desc then auc desc (if present)
    for c in ["SingleOmics_evidence", "SingleOmics_dir_agreement", "SingleOmics_auc_disc_median"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    sort_cols = [c for c in ["SingleOmics_evidence", "SingleOmics_dir_agreement", "SingleOmics_auc_disc_median"] if c in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(by=sort_cols, ascending=False, na_position="last")

    if max_return is not None and int(max_return) > 0:
        df_out = df_out.head(int(max_return)).copy()

    return df_out


def _render_gene_screener_results_tab(df: pd.DataFrame, trait_for_wgcna: str):
    st.markdown("This tab is generated by the Gene Screener in the sidebar.")
    st.markdown("---")

    if df is None or df.empty:
        st.info("No genes matched the current screener criteria.")
        return

    st.subheader(f"Selected genes ({len(df)})")
    _st_df(df, hide_index=True)

    st.markdown("---")
    st.subheader("Open one of the selected genes in the explorer")
    gene_pick = st.selectbox("Pick a gene", [""] + df["Gene"].tolist(), key="screener_jump_select")
    if gene_pick:
        st.session_state["gene_search"] = gene_pick
        st.rerun()

    st.markdown("---")
    st.subheader("Per-gene evidence (expand)")
    invitro_tables = _load_invitro_all_tables_cached()
    for g in df["Gene"].head(50).tolist():
        with st.expander(g, expanded=False):
            so = _get_single_omics_summary(g, single_omics_data)
            kg = _get_kg_summary(g, kg_data)
            wg = _get_wgcna_summary(g, wgcna_module_data, trait_for_wgcna) if trait_for_wgcna else {}
            inv = _get_invitro_summary(
                g,
                invitro_tables,
                float(st.session_state.get("screener_invitro_padj_thr", 0.05)),
                bool(st.session_state.get("screener_invitro_both_lines", True)),
            )
            bul = _get_bulk_summary(
                g,
                bulk_omics_data,
                st.session_state.get("screener_bulk_groups", []) or [],
                float(st.session_state.get("screener_bulk_padj_thr", 0.05)),
                int(st.session_state.get("screener_bulk_min_n", 2)),
                bool(st.session_state.get("screener_bulk_consistent_dir", True)),
                bool(st.session_state.get("screener_bulk_sig_all", False)),
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Single-omics Evidence", _fmt_pct01(so.get("evidence_score", np.nan)))
                st.caption(f"Agreement: {_fmt_pct01(so.get('direction_agreement', np.nan))}")
            with c2:
                st.metric("Median AUC-disc", _fmt_auc(so.get("auc_median_discriminative", np.nan)))
                st.caption(f"Studies found: {so.get('found_count', 'N/A')}")
            with c3:
                st.metric("KG cluster", "N/A" if kg.get("cluster", None) is None else str(kg.get("cluster")))
                st.caption(f"Composite %ile: {_fmt_pct(kg.get('composite_percentile', np.nan))}")
            with c4:
                st.metric("WGCNA module", "N/A" if wg.get("module", None) is None else str(wg.get("module")))
                st.caption(f"{trait_for_wgcna} corr: {_fmt_num(wg.get('trait_corr', np.nan), decimals=3)}")

            st.markdown("---")
            left, right = st.columns(2)
            with left:
                st.markdown("**In vitro (summary)**")
                if not inv:
                    st.write("No invitro evidence found for this gene (or invitro tables not loaded).")
                else:
                    st.write(f"Contrasts OK: {inv.get('n_ok', 0)}")
                    pc = inv.get("per_contrast", {})
                    if isinstance(pc, dict) and pc:
                        view = []
                        for c, d in pc.items():
                            view.append(
                                {
                                    "Contrast": c,
                                    "OK": d.get("ok", False),
                                    "Direction": d.get("direction", "missing"),
                                    "n_lines_hit": d.get("n_lines_hit", 0),
                                    "n_lines_sig": d.get("n_lines_sig", 0),
                                    "min_padj": d.get("min_padj", np.nan),
                                    "mean_lfc": d.get("mean_lfc", np.nan),
                                }
                            )
                        _st_df(pd.DataFrame(view), hide_index=True)

            with right:
                st.markdown("**Bulk omics (summary)**")
                if not bul or not isinstance(bul.get("per_group", None), dict) or not bul["per_group"]:
                    st.write("No bulk summary computed for this gene under the current settings.")
                else:
                    view = []
                    for grp, d in bul["per_group"].items():
                        view.append(
                            {
                                "Group": grp,
                                "OK": d.get("ok", False),
                                "Direction": d.get("direction", "mixed"),
                                "n_found": d.get("n_found", 0),
                                "n_sig": d.get("n_sig", 0),
                                "pos": d.get("pos", 0),
                                "neg": d.get("neg", 0),
                            }
                        )
                    _st_df(pd.DataFrame(view), hide_index=True)


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## 🔬 Meta Liver")

search_query = st.sidebar.text_input(
    "Search gene:",
    placeholder="e.g., SAA1, TP53, IL6",
    key="gene_search",
).strip().upper()

st.sidebar.caption(f"Citation: doi:{APP_DOI}")
st.sidebar.markdown("---")

if loader_err is not None:
    st.sidebar.error(f"Loader import failed: {loader_err}")
if not data_loaded:
    st.sidebar.error("Error loading data")
    if load_error is not None:
        st.sidebar.caption(str(load_error))

with st.sidebar.expander("Gene screener", expanded=True):
    invitro_tables_for_ui = _load_invitro_all_tables_cached()
    invitro_contrasts_ui = _invitro_contrasts_available(invitro_tables_for_ui)

    bulk_groups_ui = _bulk_groups_available(bulk_omics_data)
    wgcna_modules_ui = _wgcna_modules_available(wgcna_cor, wgcna_module_data)

    trait_cols = []
    if isinstance(wgcna_cor, pd.DataFrame) and not wgcna_cor.empty:
        trait_cols = list(wgcna_cor.columns)
    fibrosis_like = [t for t in trait_cols if "fibros" in str(t).lower()]
    trait_default = (fibrosis_like[0] if fibrosis_like else (trait_cols[0] if trait_cols else ""))

    with st.form("gene_screener_form", clear_on_submit=False):
        max_return = st.number_input(
            "Max genes to return",
            min_value=10,
            max_value=5000,
            value=int(st.session_state.get("screener_max_return", 250)),
            step=10,
            key="screener_max_return",
        )

        st.markdown("#### Single-omics criteria")
        use_single = st.checkbox("Enable single-omics filter", value=bool(st.session_state.get("screener_use_single", False)), key="screener_use_single")
        if use_single:
            c1, c2, c3 = st.columns(3)
            with c1:
                min_agree = st.slider("Min direction agreement", 0.0, 1.0, float(st.session_state.get("screener_min_agree", 0.70)), 0.05, key="screener_min_agree")
            with c2:
                min_auc = st.slider("Min median AUC-disc", 0.50, 1.0, float(st.session_state.get("screener_min_auc", 0.65)), 0.01, key="screener_min_auc")
            with c3:
                min_ev = st.slider("Min evidence score", 0.0, 1.0, float(st.session_state.get("screener_min_ev", 0.30)), 0.05, key="screener_min_ev")
        else:
            min_agree, min_auc, min_ev = 0.0, 0.5, 0.0

        st.markdown("#### Knowledge graph criteria")
        use_kg = st.checkbox("Enable knowledge graph filter", value=bool(st.session_state.get("screener_use_kg", False)), key="screener_use_kg")
        if use_kg:
            c1, c2 = st.columns(2)
            with c1:
                require_cluster = st.checkbox("Gene must have a cluster", value=bool(st.session_state.get("screener_kg_require_cluster", True)), key="screener_kg_require_cluster")
            with c2:
                min_comp = st.slider("Min composite centrality %ile", 0.0, 100.0, float(st.session_state.get("screener_kg_min_comp", 80.0)), 1.0, key="screener_kg_min_comp")
        else:
            require_cluster, min_comp = True, 0.0

        st.markdown("#### WGCNA criteria")
        use_wgcna = st.checkbox("Enable WGCNA filter", value=bool(st.session_state.get("screener_use_wgcna", False)), key="screener_use_wgcna")
        if use_wgcna:
            modules_sel = st.multiselect(
                "Module(s) (optional)",
                options=wgcna_modules_ui,
                default=st.session_state.get("screener_wgcna_modules", []),
                key="screener_wgcna_modules",
            )

            trait_sel = st.selectbox(
                "Fibrosis trait column",
                options=(fibrosis_like if fibrosis_like else trait_cols) if trait_cols else [""],
                index=0,
                key="screener_wgcna_trait",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                corr_dir = st.selectbox("Direction", ["Either", "Positive", "Negative"], index=1, key="screener_wgcna_corr_dir")
            with c2:
                min_abs_corr = st.slider("Min |correlation|", 0.0, 1.0, float(st.session_state.get("screener_wgcna_min_abs_corr", 0.20)), 0.05, key="screener_wgcna_min_abs_corr")
            with c3:
                p_thr = st.slider("Max p-value (if available)", 0.0, 1.0, float(st.session_state.get("screener_wgcna_p_thr", 0.05)), 0.01, key="screener_wgcna_p_thr")

            require_drug = st.checkbox("Gene must have a drug target", value=bool(st.session_state.get("screener_wgcna_require_drug", False)), key="screener_wgcna_require_drug")
        else:
            modules_sel, trait_sel, corr_dir, min_abs_corr, p_thr, require_drug = [], trait_default, "Either", 0.0, 1.0, False

        st.markdown("#### In vitro (iHeps) criteria")
        use_invitro = st.checkbox("Enable in vitro filter", value=bool(st.session_state.get("screener_use_invitro", False)), key="screener_use_invitro")
        if use_invitro:
            invitro_contrasts_sel = st.multiselect(
                "Select contrasts",
                options=invitro_contrasts_ui,
                default=st.session_state.get("screener_invitro_contrasts", invitro_contrasts_ui[:1] if invitro_contrasts_ui else []),
                key="screener_invitro_contrasts",
            )
            invitro_match_mode = st.radio(
                "Match mode across selected contrasts",
                ["Any (OR)", "All (AND)"],
                index=0,
                horizontal=True,
                key="screener_invitro_match_mode",
            )
            c1, c2 = st.columns(2)
            with c1:
                invitro_padj_thr = st.slider("padj/p-value threshold", 0.0, 1.0, float(st.session_state.get("screener_invitro_padj_thr", 0.05)), 0.01, key="screener_invitro_padj_thr")
            with c2:
                invitro_both_lines = st.checkbox("Require both iHeps lines", value=bool(st.session_state.get("screener_invitro_both_lines", True)), key="screener_invitro_both_lines")
        else:
            invitro_contrasts_sel, invitro_match_mode, invitro_padj_thr, invitro_both_lines = [], "Any (OR)", 0.05, True

        st.markdown("#### Bulk omics (tissue) criteria")
        use_bulk = st.checkbox("Enable bulk-omics filter", value=bool(st.session_state.get("screener_use_bulk", False)), key="screener_use_bulk")
        if use_bulk:
            bulk_groups_sel = st.multiselect(
                "Select bulk groups",
                options=bulk_groups_ui,
                default=st.session_state.get("screener_bulk_groups", bulk_groups_ui[:1] if bulk_groups_ui else []),
                key="screener_bulk_groups",
            )
            bulk_match_mode = st.radio(
                "Match mode across selected groups",
                ["Any (OR)", "All (AND)"],
                index=0,
                horizontal=True,
                key="screener_bulk_match_mode",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                bulk_padj_thr = st.slider("padj threshold", 0.0, 1.0, float(st.session_state.get("screener_bulk_padj_thr", 0.05)), 0.01, key="screener_bulk_padj_thr")
            with c2:
                bulk_min_n = st.number_input("Min studies with gene per group", min_value=1, max_value=1000, value=int(st.session_state.get("screener_bulk_min_n", 2)), step=1, key="screener_bulk_min_n")
            with c3:
                bulk_consistent_dir = st.checkbox("Require consistent direction", value=bool(st.session_state.get("screener_bulk_consistent_dir", True)), key="screener_bulk_consistent_dir")
            bulk_sig_all = st.checkbox("Require significant in all found studies (strict)", value=bool(st.session_state.get("screener_bulk_sig_all", False)), key="screener_bulk_sig_all")
        else:
            bulk_groups_sel, bulk_match_mode, bulk_padj_thr, bulk_min_n, bulk_consistent_dir, bulk_sig_all = [], "Any (OR)", 0.05, 2, True, False

        run_btn = st.form_submit_button("Run screener")

    if st.button("Clear screener results", use_container_width=True):
        st.session_state.pop("screener_results_df", None)
        st.session_state.pop("screener_trait_used", None)
        st.rerun()

    if run_btn:
        with st.spinner("Screening genes..."):
            df_res = run_gene_screener(
                max_return=int(max_return),
                use_single=bool(use_single),
                min_agreement=float(min_agree),
                min_auc_disc=float(min_auc),
                min_evidence=float(min_ev),
                use_kg=bool(use_kg),
                require_cluster=bool(require_cluster),
                min_composite_pctile=float(min_comp),
                use_wgcna=bool(use_wgcna),
                wgcna_modules=[str(x) for x in (modules_sel or [])],
                wgcna_trait=str(trait_sel or ""),
                wgcna_corr_dir=str(corr_dir),
                wgcna_min_abs_corr=float(min_abs_corr),
                wgcna_p_thr=float(p_thr),
                require_drug_target=bool(require_drug),
                use_invitro=bool(use_invitro),
                invitro_contrasts=[str(x) for x in (invitro_contrasts_sel or [])],
                invitro_match_mode=str(invitro_match_mode),
                invitro_padj_thr=float(invitro_padj_thr),
                invitro_require_both_lines=bool(invitro_both_lines),
                use_bulk=bool(use_bulk),
                bulk_groups=[str(x) for x in (bulk_groups_sel or [])],
                bulk_match_mode=str(bulk_match_mode),
                bulk_padj_thr=float(bulk_padj_thr),
                bulk_min_n_studies=int(bulk_min_n),
                bulk_require_consistent_dir=bool(bulk_consistent_dir),
                bulk_require_sig_all_found=bool(bulk_sig_all),
            )
        st.session_state["screener_results_df"] = df_res
        st.session_state["screener_trait_used"] = str(trait_sel or "")
        st.success(f"Done. {0 if df_res is None else len(df_res)} genes selected.")


with st.sidebar.expander("Diagnostics", expanded=False):
    if soa is not None:
        st.caption(f"single_omics_analysis loaded from: {getattr(soa, '__file__', 'unknown')}")
    else:
        st.caption("single_omics_analysis not available")
    if kg_err is not None:
        st.caption(f"kg_analysis import error: {kg_err}")
    if wgcna_err is not None:
        st.caption(f"wgcna_ppi_analysis import error: {wgcna_err}")
    if iva_err is not None:
        st.caption(f"invitro_analysis import error: {iva_err}")
    if bo_err is not None:
        st.caption(f"bulk_omics import error: {bo_err}")


# =============================================================================
# MAIN PAGE
# =============================================================================
screener_df = st.session_state.get("screener_results_df", None)
has_screener = isinstance(screener_df, pd.DataFrame) and not screener_df.empty
screener_trait_used = str(st.session_state.get("screener_trait_used", ""))

if not search_query:
    if not has_screener:
        st.title("🔬 Meta Liver")
        st.markdown("*Hypothesis Engine for Liver Genomics in Metabolic Liver Dysfunction*")
        st.markdown(
            f"""
Meta Liver is an interactive companion to the study cited below. It enables gene-centric exploration of single-omics evidence (signal strength and cross-study consistency), network context within a MAFLD/MASH knowledge graph, WGCNA-derived co-expression modules, in vitro MASLD model DEGs, and bulk tissue differential expression contrasts.

Use the sidebar to either explore a single gene, or run the Gene Screener to select genes across criteria.

If you use this app, please cite:  
{APP_CITATION}  
doi: [{APP_DOI}]({APP_DOI_URL})

{APP_TEAM}
"""
        )
    else:
        tab_overview, tab_screen = st.tabs(["Overview", "Gene Screener Results"])
        with tab_overview:
            st.title("🔬 Meta Liver")
            st.markdown("*Hypothesis Engine for Liver Genomics in Metabolic Liver Dysfunction*")
            st.markdown(
                f"""
Meta Liver is an interactive companion to the study cited below. It enables gene-centric exploration of single-omics evidence (signal strength and cross-study consistency), network context within a MAFLD/MASH knowledge graph, WGCNA-derived co-expression modules, in vitro MASLD model DEGs, and bulk tissue differential expression contrasts.

If you use this app, please cite:  
{APP_CITATION}  
doi: [{APP_DOI}]({APP_DOI_URL})

{APP_TEAM}
"""
            )
        with tab_screen:
            _render_gene_screener_results_tab(screener_df, screener_trait_used)
else:
    tab_names = [
        "Single-Omics Evidence",
        "MAFLD Knowledge Graph",
        "WGCNA Fibrosis Stage Networks",
        "In vitro MASLD model",
        "Bulk Omics (tissue)",
    ]
    if has_screener:
        tab_names.append("Gene Screener Results")

    tabs = st.tabs(tab_names)
    tab_omics, tab_kg, tab_wgcna, tab_invitro, tab_bulk = tabs[:5]
    tab_screen = tabs[5] if has_screener and len(tabs) > 5 else None

    st.title(f"🔬 {search_query}")

    if soa is None:
        st.warning("single_omics_analysis module is not available, so the Single-Omics tab cannot run.")
    if kg_mod is None:
        st.warning("kg_analysis module is not available, so the Knowledge Graph tab may be limited.")
    if wgcna_mod is None:
        st.warning("wgcna_ppi_analysis module is not available, so the WGCNA tab may be limited.")
    if iva is None:
        st.warning("invitro_analysis module is not available, so the In vitro tab may be limited.")
    if bo is None:
        st.warning("bulk_omics module is not available, so the Bulk Omics tab may be limited.")

    with tab_omics:
        st.markdown(
            """
This tab summarises gene-level evidence across the single-omics datasets. AUROC reflects per-study discriminative performance, logFC indicates direction (MAFLD vs Healthy), and the Evidence Score summarises strength, stability, direction agreement, and study support.
"""
        )
        st.markdown("---")

        if not single_omics_data:
            st.warning("No single-omics studies found.")
        elif soa is None:
            st.warning("single_omics_analysis not available.")
        else:
            consistency = None
            try:
                consistency = soa.compute_consistency_score(search_query, single_omics_data)
            except Exception as e:
                st.error(f"Single-omics scoring failed: {e}")
                consistency = None

            if consistency is None:
                st.info(f"Gene '{search_query}' not found in any single-omics study (or scoring failed).")
            else:
                help_text = {
                    "Evidence Score": "Overall evidence across studies (Strength × Stability × Direction Agreement × Study Weight).",
                    "Direction Agreement": "Fraction of studies where the gene’s direction (MAFLD vs Healthy) matches the majority.",
                    "Median AUC (disc)": "Median discriminative AUC across studies: AUC-disc = max(AUC, 1−AUC).",
                    "Studies Found": "Number of studies where the gene is present (even if AUROC is missing).",
                    "Strength": "How far the median AUC-disc is above 0.5 (0=no signal; 1=perfect).",
                    "Stability": "Cross-study consistency of AUC-disc (1=very consistent; 0=very variable).",
                    "Study Weight": "Downweights scores supported by very few AUROC values (increases with n_auc).",
                    "Valid AUROC (n_auc)": "Number of studies with a usable AUROC value for this gene.",
                    "Median AUC (raw)": "Median of the raw AUROC values as stored in the study tables (diagnostic).",
                    "Median AUC (oriented)": "Median AUROC after aligning direction so MAFLD is treated as ‘positive’ (diagnostic).",
                    "AUC-disc IQR": "Interquartile range of AUC-disc across studies (lower = more stable).",
                }

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Evidence Score", _fmt_pct01(consistency.get("evidence_score")))
                    st.caption(help_text["Evidence Score"])
                with col2:
                    st.metric("Direction Agreement", _fmt_pct01(consistency.get("direction_agreement")))
                    st.caption(help_text["Direction Agreement"])
                with col3:
                    st.metric("Median AUC (disc)", _fmt_auc(consistency.get("auc_median_discriminative")))
                    st.caption(help_text["Median AUC (disc)"])
                with col4:
                    st.metric("Studies Found", f"{consistency.get('found_count', 0)}")
                    st.caption(help_text["Studies Found"])

                interp = consistency.get("interpretation", "")
                if interp:
                    st.info(f"📊 **{interp}**")
                st.markdown("---")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Strength", f"{consistency.get('strength', np.nan):.3f}")
                    st.caption(help_text["Strength"])
                with c2:
                    st.metric("Stability", f"{consistency.get('stability', np.nan):.3f}")
                    st.caption(help_text["Stability"])
                with c3:
                    st.metric("Study Weight", f"{consistency.get('n_weight', np.nan):.3f}")
                    st.caption(help_text["Study Weight"])
                with c4:
                    st.metric("Valid AUROC (n_auc)", f"{consistency.get('n_auc', 'N/A')}")
                    st.caption(help_text["Valid AUROC (n_auc)"])

                st.markdown("---")
                d1, d2, d3 = st.columns(3)

                auc_disc_vals = []
                try:
                    auc_disc_vals = [
                        max(a, 1.0 - a)
                        for a in consistency.get("auc_values", [])
                        if a is not None and not np.isnan(a)
                    ]
                except Exception:
                    auc_disc_vals = []

                disc_iqr = np.nan
                if len(auc_disc_vals) > 1:
                    disc_iqr = float(np.subtract(*np.percentile(auc_disc_vals, [75, 25])))

                with d1:
                    st.metric("Median AUC (raw)", f"{consistency.get('auc_median', np.nan):.3f}")
                    st.caption(help_text["Median AUC (raw)"])
                with d2:
                    st.metric("Median AUC (oriented)", f"{consistency.get('auc_median_oriented', np.nan):.3f}")
                    st.caption(help_text["Median AUC (oriented)"])
                with d3:
                    st.metric("AUC-disc IQR", "NA" if np.isnan(disc_iqr) else f"{disc_iqr:.3f}")
                    st.caption(help_text["AUC-disc IQR"])

                st.markdown("---")

                auc_view = st.radio(
                    "AUROC view (plots)",
                    ["Discriminative (AUC-disc)", "Raw (as reported)", "Oriented (MAFLD-positive)"],
                    index=0,
                    horizontal=True,
                    key="omics_auc_view_toggle",
                )

                auc_key_map = {
                    "Discriminative (AUC-disc)": "auc_disc",
                    "Raw (as reported)": "auc_raw",
                    "Oriented (MAFLD-positive)": "auc_oriented",
                }
                auc_key = auc_key_map[auc_view]

                metrics = _collect_gene_metrics(search_query, single_omics_data)

                fig = make_lollipop(
                    metrics,
                    auc_key=auc_key,
                    title=f"AUROC Across Studies ({auc_view})",
                    subtitle=(
                        "AUC-disc = max(AUC, 1−AUC)."
                        if auc_key == "auc_disc"
                        else "Raw AUROC values as stored in each study table."
                        if auc_key == "auc_raw"
                        else "AUROC aligned so MAFLD is treated as ‘positive’."
                    ),
                )
                if fig:
                    _st_plotly(fig)
                else:
                    st.info("No AUROC values available for this gene under the selected AUROC view.")

                st.markdown("---")
                st.markdown("**Concordance: AUROC vs logFC**")
                fig_scatter = make_scatter_auc_logfc(
                    metrics,
                    auc_key=auc_key,
                    title=f"Concordance: {auc_view} vs logFC",
                    subtitle="Checks whether discriminative signal aligns with up/down regulation.",
                )
                if fig_scatter:
                    _st_plotly(fig_scatter)
                else:
                    st.info("Not enough data for concordance plot (need AUROC + logFC in ≥2 studies).")

                st.markdown("---")
                st.markdown("**AUC-disc distribution (stability view)**")
                fig_dist = make_auc_disc_distribution(metrics)
                if fig_dist:
                    _st_plotly(fig_dist)
                else:
                    st.info("Not enough AUROC values to show a distribution.")

                st.markdown("---")
                st.markdown("**Detailed Results**")
                try:
                    results_df = soa.create_results_table(search_query, single_omics_data)
                except Exception as e:
                    results_df = None
                    st.error(f"Failed to build results table: {e}")
                if results_df is not None:
                    _st_df(results_df, hide_index=True)
                else:
                    st.info("No per-study rows found for this gene.")

    with tab_kg:
        if kg_mod is None:
            st.warning("⚠ Knowledge graph module not available.")
        else:
            st.markdown(
                """
This tab places the selected gene in its network context within the MAFLD/MASH subgraph. It reports whether the gene is present, its assigned cluster, and centrality metrics (PageRank, betweenness, eigenvector). The cluster view lists co-clustered genes, drugs, and disease annotations.
"""
            )
            st.markdown("---")

            if kg_data:
                kg_info = kg_mod.get_gene_kg_info(search_query, kg_data)

                if kg_info:
                    cluster_id = kg_info.get("cluster", None)

                    h1, h2, h3, h4 = st.columns(4)
                    with h1:
                        st.metric("Cluster", "N/A" if cluster_id is None else str(cluster_id))
                    with h2:
                        st.metric("Composite centrality", _fmt_pct(kg_info.get("composite_percentile")))
                        st.caption(f"raw: {_fmt_num(kg_info.get('composite'), decimals=6)}")
                    with h3:
                        st.metric("Betweenness", _fmt_pct(kg_info.get("bet_percentile")))
                        st.caption(f"raw: {_fmt_num_commas(kg_info.get('betweenness'), decimals=2)}")
                    with h4:
                        st.metric("PageRank", _fmt_pct(kg_info.get("pagerank_percentile")))
                        st.caption(f"raw: {_fmt_num(kg_info.get('pagerank'), decimals=4)}")

                    h5, h6, _, _ = st.columns(4)
                    with h5:
                        st.metric("Eigenvector", _fmt_pct(kg_info.get("eigen_percentile")))
                        st.caption(f"raw: {_fmt_num(kg_info.get('eigen'), decimals=6, sci_if_small=True)}")
                    with h6:
                        st.caption("Percentiles computed across the MASH subgraph nodes table.")

                    st.markdown("---")

                    interpretation = kg_mod.interpret_centrality(
                        kg_info.get("pagerank", np.nan),
                        kg_info.get("betweenness", np.nan),
                        kg_info.get("eigen", np.nan),
                        pagerank_pct=kg_info.get("pagerank_percentile"),
                        betweenness_pct=kg_info.get("bet_percentile"),
                        eigen_pct=kg_info.get("eigen_percentile"),
                        composite_pct=kg_info.get("composite_percentile"),
                    )
                    st.info(f"📍 {interpretation}")

                    if cluster_id is None or str(cluster_id).strip() == "" or str(cluster_id).lower() == "nan":
                        st.warning("Cluster ID missing for this node; cannot display cluster neighbours.")
                    else:
                        st.markdown("**Nodes in Cluster**")

                        ctl1, ctl2 = st.columns(2)
                        with ctl1:
                            top_n = st.slider("Show top N per table", 10, 300, 50, step=10, key="kg_top_n")
                        with ctl2:
                            sort_key = st.selectbox(
                                "Sort by",
                                [
                                    "Composite %ile",
                                    "PR %ile",
                                    "Bet %ile",
                                    "Eigen %ile",
                                    "PageRank (raw)",
                                    "Betweenness (raw)",
                                    "Eigen (raw)",
                                    "Name",
                                ],
                                index=0,
                                key="kg_sort_key",
                            )

                        tab_genes, tab_drugs, tab_diseases = st.tabs(["Genes/Proteins", "Drugs", "Diseases"])

                        with tab_genes:
                            genes_df = kg_mod.get_cluster_genes(cluster_id, kg_data)
                            if genes_df is not None and not genes_df.empty:
                                _st_df(_prepare_cluster_table(genes_df, sort_key, top_n), hide_index=True)
                            else:
                                st.write("No genes/proteins in this cluster")

                        with tab_drugs:
                            drugs_df = kg_mod.get_cluster_drugs(cluster_id, kg_data)
                            if drugs_df is not None and not drugs_df.empty:
                                _st_df(_prepare_cluster_table(drugs_df, sort_key, top_n), hide_index=True)
                            else:
                                st.write("No drugs in this cluster")

                        with tab_diseases:
                            diseases_df = kg_mod.get_cluster_diseases(cluster_id, kg_data)
                            if diseases_df is not None and not diseases_df.empty:
                                _st_df(_prepare_cluster_table(diseases_df, sort_key, top_n), hide_index=True)
                            else:
                                st.write("No diseases in this cluster")
                else:
                    st.warning(f"⚠ '{search_query}' not found in MASH subgraph")
            else:
                st.warning("⚠ Knowledge graph data not loaded")

    with tab_wgcna:
        if wgcna_mod is None:
            st.warning("⚠ WGCNA module not available.")
        else:
            st.markdown(
                """
This tab focuses on WGCNA-derived co-expression context (module assignment, module–trait relationships, enrichment tables) and shows local PPI evidence where available.
"""
            )
            st.markdown("---")

            st.markdown("**WGCNA Co-expression Module**")
            if not wgcna_module_data:
                st.warning("No WGCNA module data loaded.")
            else:
                gene_module_info = None
                try:
                    gene_module_info = wgcna_mod.get_gene_module(search_query, wgcna_module_data)
                except Exception:
                    gene_module_info = None

                if not gene_module_info or "module" not in gene_module_info:
                    st.info("No WGCNA module assignment found for this gene.")
                else:
                    module_name = gene_module_info["module"]
                    module_key = _normalise_module_name(module_name)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Module", str(module_name))
                    with c2:
                        trait_for_view = st.selectbox(
                            "Trait column (for module–trait table)",
                            options=(fibrosis_like if fibrosis_like else trait_cols) if trait_cols else [""],
                            index=0,
                            key="wgcna_trait_view",
                        )
                    with c3:
                        st.caption("Stage-stratified section removed as requested.")

                    st.markdown("---")
                    st.markdown("**Module–Trait Relationships (sorted by |corr|)**")
                    mt = _module_trait_table(str(module_name), wgcna_cor, wgcna_pval)
                    if mt is not None and not mt.empty:
                        _st_df(mt, hide_index=True)
                    else:
                        st.info("Module–trait table not available.")

                    st.markdown("---")
                    st.markdown("**Module enrichment / pathways (if available)**")
                    pw = None
                    if isinstance(wgcna_pathways, dict):
                        pw = wgcna_pathways.get(module_key) or wgcna_pathways.get(str(module_name).lower()) or wgcna_pathways.get(str(module_name))
                    if isinstance(pw, pd.DataFrame) and not pw.empty:
                        show_pw = st.slider("Show top N enrichment rows", 5, 200, 25, step=5, key="wgcna_pw_topn")
                        _st_df(pw.head(int(show_pw)), hide_index=True)
                    else:
                        st.info("No enrichment table found for this module.")

                    st.markdown("---")
                    st.markdown("**Top genes in module**")
                    module_genes = None
                    try:
                        module_genes = wgcna_mod.get_module_genes(module_name, wgcna_module_data)
                    except Exception:
                        module_genes = None

                    if isinstance(module_genes, pd.DataFrame) and not module_genes.empty:
                        cA, cB = st.columns(2)
                        with cA:
                            show_top_genes = st.slider("Show top N genes", min_value=5, max_value=300, value=25, step=5, key="wgcna_top_n_genes")
                        with cB:
                            max_drugs_per_gene = st.slider("Show up to N drugs per gene", min_value=1, max_value=25, value=5, step=1, key="wgcna_max_drugs_per_gene")

                        view_df = module_genes.copy()
                        if "gene" in view_df.columns and "Gene" not in view_df.columns:
                            view_df = view_df.rename(columns={"gene": "Gene"})
                        if "Gene" in view_df.columns:
                            view_df["Gene"] = view_df["Gene"].astype(str).map(_normalise_gene)

                        view_df = _annotate_genes_with_drugs(view_df, gene_to_drugs, int(max_drugs_per_gene))
                        _st_df(view_df.head(int(show_top_genes)), hide_index=True)
                    else:
                        st.info("Module gene table not available from wgcna_ppi_analysis.")

                    st.markdown("---")
                    st.markdown("**PPI neighbours (best-effort extraction)**")
                    ppi_hits = []
                    if isinstance(ppi_data, dict) and ppi_data:
                        for nm, df in ppi_data.items():
                            if not isinstance(df, pd.DataFrame) or df.empty:
                                continue

                            cols = [str(c) for c in df.columns]
                            # try common edge column patterns
                            pairs = None
                            for a, b in [
                                ("source", "target"),
                                ("Source", "Target"),
                                ("protein1", "protein2"),
                                ("Protein1", "Protein2"),
                                ("node1", "node2"),
                                ("Node1", "Node2"),
                                ("from", "to"),
                                ("From", "To"),
                            ]:
                                if a in df.columns and b in df.columns:
                                    pairs = (a, b)
                                    break
                            if pairs is None:
                                continue

                            a, b = pairs
                            try:
                                a_ser = df[a].astype(str).map(_normalise_gene)
                                b_ser = df[b].astype(str).map(_normalise_gene)
                            except Exception:
                                continue

                            mask = (a_ser == search_query) | (b_ser == search_query)
                            sub = df.loc[mask].copy()
                            if sub.empty:
                                continue

                            try:
                                sub["__A__"] = a_ser[mask].values
                                sub["__B__"] = b_ser[mask].values
                            except Exception:
                                pass

                            # neighbour is the opposite end
                            try:
                                sub["Neighbour"] = np.where(sub["__A__"] == search_query, sub["__B__"], sub["__A__"])
                            except Exception:
                                sub["Neighbour"] = ""

                            sub["SourceTable"] = nm
                            keep_cols = ["SourceTable", "Neighbour"]
                            for c in ["score", "combined_score", "weight", "confidence", "Confidence"]:
                                if c in sub.columns:
                                    keep_cols.append(c)
                            keep_cols = [c for c in keep_cols if c in sub.columns]
                            ppi_hits.append(sub[keep_cols])

                    if ppi_hits:
                        ppi_df = pd.concat(ppi_hits, ignore_index=True)
                        if "Neighbour" in ppi_df.columns:
                            ppi_df["Neighbour"] = ppi_df["Neighbour"].astype(str).map(_normalise_gene)
                            ppi_df = ppi_df.loc[ppi_df["Neighbour"] != ""].copy()
                        topn = st.slider("Show top N PPI edges", 10, 500, 50, step=10, key="ppi_topn")
                        _st_df(ppi_df.head(int(topn)), hide_index=True)
                    else:
                        st.info("No PPI edges found (or PPI tables not in a recognised edge format).")

    with tab_invitro:
        if iva is None:
            st.warning("⚠ In vitro module not available.")
        else:
            if hasattr(iva, "render_invitro_tab"):
                try:
                    iva.render_invitro_tab(search_query)
                except Exception as e:
                    st.error(f"In vitro tab failed: {e}")
            else:
                st.info("invitro_analysis.render_invitro_tab not found.")

    with tab_bulk:
        if bo is None:
            st.warning("⚠ Bulk omics module not available.")
        else:
            if hasattr(bo, "render_bulk_omics_tab"):
                try:
                    bo.render_bulk_omics_tab(search_query, bulk_data=bulk_omics_data)
                except Exception as e:
                    st.error(f"Bulk omics tab failed: {e}")
            else:
                st.info("bulk_omics.render_bulk_omics_tab not found.")

    if tab_screen is not None:
        with tab_screen:
            _render_gene_screener_results_tab(screener_df, screener_trait_used)
