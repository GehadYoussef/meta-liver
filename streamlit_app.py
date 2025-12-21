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
from typing import Optional, Tuple, Any

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

    This name is what streamlit_app.py imports.
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


# Backwards-compatible alias (if other modules import the old name)
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
# so all existing imports across your app keep working unchanged.
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

# 2) UPDATE APP_TEAM to:
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
# SAFE IMPORT / RELOAD HELPERS (prevents one broken module killing the whole app)
# -----------------------------------------------------------------------------
def _safe_import_reload(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        mod = importlib.reload(mod)
        return mod, None
    except Exception as e:
        return None, e


# -----------------------------------------------------------------------------
# CORE LOADERS (these should be robust_data_loader + analysis modules)
# -----------------------------------------------------------------------------
loader_err = None
try:
    from robust_data_loader import (
        load_single_omics_studies,
        load_kg_data,
        load_ppi_data,
    )
except Exception as e:
    loader_err = e
    load_single_omics_studies = None  # type: ignore
    load_kg_data = None  # type: ignore
    load_ppi_data = None  # type: ignore

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


# =============================================================================
# SINGLE-OMICS PLOT HELPERS (uses soa methods for consistency)
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
# KG DISPLAY HELPERS (UI ONLY)
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
# DATA LOADING (cached) – bulk uses bo.load_bulk_omics to avoid loader-name mismatch
# =============================================================================
@st.cache_data(show_spinner=False)
def load_all_data_cached():
    single_omics = {}
    kg_data = {}
    ppi_data = {}
    wgcna_module_data = {}
    wgcna_cor = pd.DataFrame()
    wgcna_pval = pd.DataFrame()
    wgcna_pathways = {}
    active_drugs = pd.DataFrame()
    gene_to_drugs = {}
    bulk_omics_data = {}

    # core loaders
    if load_single_omics_studies is not None:
        single_omics = load_single_omics_studies() or {}
    if load_kg_data is not None:
        kg_data = load_kg_data() or {}
    if load_ppi_data is not None:
        ppi_data = load_ppi_data() or {}

    # WGCNA module
    if wgcna_mod is not None:
        try:
            wgcna_module_data = wgcna_mod.load_wgcna_module_data() or {}
        except Exception:
            wgcna_module_data = {}

        try:
            wgcna_cor = wgcna_mod.load_wgcna_mod_trait_cor()
        except Exception:
            wgcna_cor = pd.DataFrame()

        try:
            wgcna_pval = wgcna_mod.load_wgcna_mod_trait_pval()
        except Exception:
            wgcna_pval = pd.DataFrame()

        try:
            wgcna_pathways = wgcna_mod.load_wgcna_pathways() or {}
        except Exception:
            wgcna_pathways = {}

        try:
            active_drugs = wgcna_mod.load_wgcna_active_drugs()
            if active_drugs is None:
                active_drugs = pd.DataFrame()
        except Exception:
            active_drugs = pd.DataFrame()

        try:
            if isinstance(active_drugs, pd.DataFrame) and not active_drugs.empty:
                gene_to_drugs = wgcna_mod.build_gene_to_drugs_index(active_drugs) or {}
        except Exception:
            gene_to_drugs = {}

    # Bulk omics
    if bo is not None:
        try:
            bulk_omics_data = bo.load_bulk_omics() or {}
        except Exception:
            bulk_omics_data = {}

    return (
        single_omics,
        kg_data,
        wgcna_module_data,
        ppi_data,
        wgcna_cor,
        wgcna_pval,
        wgcna_pathways,
        active_drugs,
        gene_to_drugs,
        bulk_omics_data,
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
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## 🔬 Meta Liver")
search_query = st.sidebar.text_input("Search gene:", placeholder="e.g., SAA1, TP53, IL6").strip().upper()

# diagnostics without crashing the app
if loader_err is not None:
    st.sidebar.error(f"Loader import failed: {loader_err}")

if soa is not None:
    st.sidebar.caption(f"single_omics_analysis loaded from: {getattr(soa, '__file__', 'unknown')}")
else:
    st.sidebar.caption("single_omics_analysis not available")

st.sidebar.caption(f"Citation: doi:{APP_DOI}")
st.sidebar.markdown("---")

if not data_loaded:
    st.sidebar.error("✗ Error loading data")
    if load_error is not None:
        st.sidebar.caption(str(load_error))
else:
    if single_omics_data:
        st.sidebar.success(f"✓ {len(single_omics_data)} studies loaded")
        with st.sidebar.expander("📊 Studies:"):
            for study_name in sorted(single_omics_data.keys()):
                st.write(f"• {study_name}")
    else:
        st.sidebar.warning("⚠ No single-omics studies found")

    st.sidebar.success("✓ Knowledge graph loaded" if kg_data else "⚠ Knowledge graph not available")
    st.sidebar.success(
        f"✓ WGCNA modules loaded ({len(wgcna_module_data)} modules)" if wgcna_module_data else "⚠ WGCNA modules not available"
    )
    st.sidebar.success(
        "✓ WGCNA moduleTraitCor loaded" if isinstance(wgcna_cor, pd.DataFrame) and not wgcna_cor.empty else "⚠ moduleTraitCor not available"
    )
    st.sidebar.success(
        "✓ WGCNA moduleTraitPvalue loaded" if isinstance(wgcna_pval, pd.DataFrame) and not wgcna_pval.empty else "⚠ moduleTraitPvalue not available"
    )
    st.sidebar.success(
        f"✓ WGCNA pathways loaded ({len(wgcna_pathways)} modules)" if isinstance(wgcna_pathways, dict) and len(wgcna_pathways) > 0 else "⚠ WGCNA pathways not available"
    )
    st.sidebar.success(
        "✓ Active drugs loaded" if isinstance(active_drugs_df, pd.DataFrame) and not active_drugs_df.empty else "⚠ Active drugs not available"
    )
    st.sidebar.success("✓ PPI networks loaded" if ppi_data else "⚠ PPI networks not available")

    if iva is not None:
        try:
            invitro_files = iva.discover_invitro_deg_files()
            st.sidebar.success(
                f"✓ In vitro model DEGs found ({len(invitro_files)})" if invitro_files else "⚠ In vitro model DEGs not found (expected: stem_cell_model/*.parquet)"
            )
        except Exception as e:
            st.sidebar.warning(f"⚠ In vitro model check failed: {e}")
    else:
        st.sidebar.warning("⚠ In vitro module not available")

    if isinstance(bulk_omics_data, dict) and len(bulk_omics_data) > 0:
        n_files = int(sum(len(v) for v in bulk_omics_data.values()))
        st.sidebar.success(f"✓ Bulk-omics loaded ({n_files} files / {len(bulk_omics_data)} groups)")
    else:
        st.sidebar.warning("⚠ Bulk-omics not available (expected: Bulk_Omics/<group>/*.tsv|.csv|.parquet)")


# =============================================================================
# MAIN PAGE
# =============================================================================
if not search_query:
    st.title("🔬 Meta Liver")
    st.markdown("*Hypothesis Engine for Liver Genomics in Metabolic Liver Dysfunction*")
    st.markdown(
        f"""
Meta Liver is an interactive companion to the study cited below. It enables gene-centric exploration of single-omics evidence (signal strength and cross-study consistency), network context within a MAFLD/MASH knowledge graph, WGCNA-derived co-expression modules (including fibrosis stage–stratified analyses where available), in vitro MASLD model DEGs, and bulk tissue differential expression contrasts.

Enter a gene symbol in the sidebar to open the analysis tabs for that gene.

If you use this app, please cite:  
{APP_CITATION}  
doi: [{APP_DOI}]({APP_DOI_URL})

{APP_TEAM}
"""
    )
else:
    st.title(f"🔬 {search_query}")

    if soa is None:
        st.error("single_omics_analysis module is not available, so the Single-Omics tab cannot run.")
    if kg_mod is None:
        st.warning("kg_analysis module is not available, so the Knowledge Graph tab may be limited.")
    if wgcna_mod is None:
        st.warning("wgcna_ppi_analysis module is not available, so the WGCNA tab may be limited.")
    if iva is None:
        st.warning("invitro_analysis module is not available, so the In vitro tab may be limited.")
    if bo is None:
        st.warning("bulk_omics module is not available, so the Bulk Omics tab may be limited.")

    if not single_omics_data:
        st.error("No single-omics studies found!")
    else:
        consistency = None
        if soa is not None:
            try:
                consistency = soa.compute_consistency_score(search_query, single_omics_data)
            except Exception as e:
                st.error(f"Single-omics scoring failed: {e}")
                consistency = None

        if consistency is None:
            st.warning(f"Gene '{search_query}' not found in any single-omics study (or scoring failed).")
        else:
            tab_omics, tab_kg, tab_wgcna, tab_invitro, tab_bulk = st.tabs(
                [
                    "Single-Omics Evidence",
                    "MAFLD Knowledge Graph",
                    "WGCNA Fibrosis Stage Networks",
                    "In vitro MASLD model",
                    "Bulk Omics (tissue)",
                ]
            )

            with tab_omics:
                st.markdown(
                    """
This tab summarises gene-level evidence across the single-omics datasets. AUROC reflects per-study discriminative performance, logFC indicates direction (MAFLD vs Healthy), and the Evidence Score summarises strength, stability, direction agreement, and study support.
"""
                )
                st.markdown("---")

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
                if soa is not None:
                    try:
                        results_df = soa.create_results_table(search_query, single_omics_data)
                    except Exception as e:
                        results_df = None
                        st.error(f"Failed to build results table: {e}")
                    if results_df is not None:
                        _st_df(results_df, hide_index=True)
                    else:
                        st.info("No per-study rows found for this gene.")
                else:
                    st.info("single_omics_analysis not available.")

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
This tab focuses on WGCNA-derived co-expression context. It reports WGCNA module assignment, module–trait relationships, module enrichment tables, then shows direct PPI interactors and local network statistics for the selected gene.
"""
                    )
                    st.markdown("---")

                    st.markdown("**WGCNA Co-expression Module**")
                    if wgcna_module_data:
                        gene_module_info = wgcna_mod.get_gene_module(search_query, wgcna_module_data)
                        if gene_module_info:
                            module_name = gene_module_info["module"]
                            st.markdown(f"**Module Assignment:** {module_name}")

                            module_genes = wgcna_mod.get_module_genes(module_name, wgcna_module_data)
                            if module_genes is not None:
                                st.markdown(f"Top genes in module {module_name}:")

                                show_top_genes = st.slider(
                                    "Show top N genes",
                                    min_value=5,
                                    max_value=200,
                                    value=15,
                                    step=5,
                                    key="wgcna_top_n_genes",
                                )

                                max_drugs_per_gene = st.slider(
                                    "Show up to N drugs per gene",
                                    min_value=1,
                                    max_value=25,
                                    value=5,
                                    step=1,
                                    key="wgcna_max_drugs_per_gene",
                                )

                                view_df = module_genes.head(int(show_top_genes)).copy()
                                view_df = _annotate_genes_with_drugs(view_df, gene_to_drugs, max_drugs_per_gene)
                                _st_df(view_df, hide_index=True)
                            else:
                                st.info(f"No other genes found in module {module_name}")

                            st.markdown("---")
                            st.markdown("**Module–trait relationships (WGCNA)**")
                            mt = _module_trait_table(module_name, wgcna_cor, wgcna_pval)
                            if mt is None or mt.empty:
                                st.info("Module–trait tables not available for this module (check module index names).")
                            else:
                                _st_df(mt, hide_index=True)

                            st.markdown("---")
                            st.markdown("**Pathways / enrichment (module)**")
                            top_n_pathways = st.slider(
                                "Show top N pathways",
                                min_value=10,
                                max_value=300,
                                value=50,
                                step=10,
                                key="wgcna_top_n_pathways",
                            )
                            key = str(module_name).strip().lower()
                            dfp = (wgcna_pathways or {}).get(key)
                            if dfp is None or dfp.empty:
                                st.info(f"No enrichment table found for module '{module_name}' under (wgcna|wcgna)/pathways/.")
                            else:
                                _st_df(dfp.head(int(top_n_pathways)), hide_index=True)
                        else:
                            st.info(f"⚠ '{search_query}' not found in WGCNA module assignments")
                    else:
                        st.info("⚠ WGCNA module data not available")

                    st.markdown("---")
                    st.markdown("**Protein–Protein Interaction Network**")
                    if ppi_data and hasattr(wgcna_mod, "find_ppi_interactors") and hasattr(wgcna_mod, "get_network_stats"):
                        ppi_df = wgcna_mod.find_ppi_interactors(search_query, ppi_data)
                        if ppi_df is not None:
                            net_stats = wgcna_mod.get_network_stats(search_query, ppi_data)
                            if net_stats:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Direct Interactors", net_stats.get("degree", "N/A"))
                                with col2:
                                    st.write(f"**Network Property:** {net_stats.get('description', 'N/A')}")
                            st.markdown(f"Direct interaction partners of {search_query}:")
                            _st_df(ppi_df, hide_index=True)
                        else:
                            st.info(f"⚠ '{search_query}' not found in PPI networks")
                    else:
                        st.info("⚠ PPI network data not available")

            with tab_invitro:
                if iva is None:
                    st.warning("⚠ In vitro module not available.")
                else:
                    st.markdown(
                        """
This tab summarises differential expression from a human stem cell-derived MASLD model using induced hepatocytes (iHeps).

Healthy controls are labelled **HCM**. Disease modelling conditions include **OA+PA**, **OA+PA + Resistin/Myostatin**, and **OA+PA + Resistin/Myostatin + PBMC co-culture** (immune cells were not sequenced; iHeps RNA-seq only). Two iHeps lines are supported: **1b** and **5a**.
"""
                    )
                    st.markdown("---")
                    try:
                        iva.render_invitro_tab(search_query)
                    except Exception as e:
                        st.error(f"In vitro tab failed: {e}")

            with tab_bulk:
                if bo is None:
                    st.warning("⚠ Bulk omics module not available.")
                else:
                    st.markdown(
                        "In Bulk Omics contrasts, **log2FoldChange > 0 means the gene is enriched in the first-named group of the contrast folder** "
                        "(for example ‘MASLD vs Control’: positive log2FC implies enriched in MASLD)."
                    )
                    st.markdown("---")

                    sig = None
                    try:
                        sig = inspect.signature(bo.render_bulk_omics_tab)
                    except Exception:
                        sig = None

                    try:
                        if sig is not None and "bulk_data" in sig.parameters:
                            bo.render_bulk_omics_tab(search_query, bulk_data=bulk_omics_data)
                        else:
                            bo.render_bulk_omics_tab(search_query)
                    except Exception as e:
                        st.error(f"Bulk Omics tab failed: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 11px;'>"
    "<p>Meta Liver - Single-Omics Evidence | MAFLD Knowledge Graph | WGCNA Networks | In vitro MASLD model | Bulk Omics</p>"
    f"<p>doi: <a href='{APP_DOI_URL}' target='_blank'>{APP_DOI}</a></p>"
    "</div>",
    unsafe_allow_html=True,
)
