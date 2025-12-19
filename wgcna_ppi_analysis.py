"""
WGCNA and PPI Network Analysis Module

This module currently supports:
1) WGCNA module gene mapping files from <data_dir>/wcgna/modules/
2) Module–trait matrices from <data_dir>/wcgna/ (moduleTraitCor / moduleTraitPvalue)
3) Enrichment/pathway tables from <data_dir>/wcgna/pathways/
4) Basic co-expression (correlation) helper on an expression matrix
5) Simple PPI neighbour lookup and degree stats across one or more PPI tables
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# PATH HELPERS
# -----------------------------------------------------------------------------

def find_data_dir() -> Optional[Path]:
    """Find data directory."""
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


def find_subfolder(parent: Path, folder_pattern: str) -> Optional[Path]:
    """Find subfolder (case-insensitive)."""
    if not parent.exists():
        return None

    exact_path = parent / folder_pattern
    if exact_path.exists():
        return exact_path

    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == folder_pattern.lower():
            return item

    return None


def find_file(directory: Path, filename_pattern: str) -> Optional[Path]:
    """Find file in directory (case-insensitive; substring match allowed)."""
    if not directory.exists():
        return None

    exact_path = directory / filename_pattern
    if exact_path.exists():
        return exact_path

    pat = filename_pattern.lower()
    for fp in directory.rglob("*"):
        if not fp.is_file():
            continue
        name = fp.name.lower()
        if name == pat:
            return fp
        if pat in name:
            return fp

    return None


# -----------------------------------------------------------------------------
# WGCNA MODULE GENE MAPPINGS
# -----------------------------------------------------------------------------

def load_wgcna_module_data() -> Dict[str, pd.DataFrame]:
    """
    Load per-module gene mapping files from <data_dir>/wcgna/modules/.

    Accepts .parquet and .csv with flexible gene-symbol column names.
    Normalises the gene symbol column to 'hgnc_symbol' (upper-case).

    Returns:
        dict[module_name -> dataframe]
    """
    module_data: Dict[str, pd.DataFrame] = {}

    data_dir = find_data_dir()
    if data_dir is None:
        print("DEBUG: data_dir not found", file=sys.stderr)
        return {}

    wcgna_dir = find_subfolder(data_dir, "wcgna")
    if wcgna_dir is None:
        print(f"DEBUG: wcgna folder not found under {data_dir}", file=sys.stderr)
        return {}

    modules_dir = wcgna_dir / "modules"
    if not modules_dir.exists():
        print(f"DEBUG: modules folder not found: {modules_dir}", file=sys.stderr)
        return {}

    candidates = []
    for p in modules_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".csv", ".parquet"]:
            name = p.name.lower()
            if "mapping" in name or ("gene" in name and "id" in name):
                candidates.append(p)

    if not candidates:
        candidates = [p for p in modules_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".csv", ".parquet"]]

    for file_path in candidates:
        try:
            df = pd.read_parquet(file_path) if file_path.suffix.lower() == ".parquet" else pd.read_csv(file_path)

            if df is None or df.empty:
                continue

            gene_col = None
            for col in df.columns:
                if str(col).lower() in ["hgnc_symbol", "symbol", "gene", "gene_symbol", "hgnc", "genesymbol"]:
                    gene_col = col
                    break

            if gene_col is None:
                continue

            if gene_col != "hgnc_symbol":
                df = df.rename(columns={gene_col: "hgnc_symbol"})
            df["hgnc_symbol"] = df["hgnc_symbol"].astype(str).str.strip().str.upper()

            stem = file_path.stem
            stem = re.sub(r"(?i)nodes[-_ ]gene[-_ ]id[-_ ]mapping[-_ ]?", "", stem)
            stem = re.sub(r"(?i)gene[-_ ]id[-_ ]mapping[-_ ]?", "", stem)
            stem = re.sub(r"(?i)module[-_ ]?", "", stem)
            module_name = re.split(r"[-_ ]+", stem.strip())[-1]

            if module_name and module_name not in module_data:
                module_data[module_name] = df
                print(f"DEBUG: Loaded WGCNA module '{module_name}' from {file_path.name}", file=sys.stderr)

        except Exception as e:
            print(f"DEBUG: Error loading {file_path}: {e}", file=sys.stderr)

    print(f"DEBUG: Loaded {len(module_data)} WGCNA modules from {modules_dir}", file=sys.stderr)
    return module_data


def get_gene_module(gene_name: str, module_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Find which WGCNA module a gene belongs to.
    Returns module name and first matching row.
    """
    if not module_data:
        return None

    gene_lower = str(gene_name).strip().lower()

    for module_name, gene_df in module_data.items():
        if gene_df is None or gene_df.empty or "hgnc_symbol" not in gene_df.columns:
            continue

        matching = gene_df[gene_df["hgnc_symbol"].astype(str).str.lower() == gene_lower]
        if not matching.empty:
            return {"gene": gene_name, "module": module_name, "data": matching.iloc[0]}

    return None


def get_module_genes(module_name: str, module_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Get all genes in a specific WGCNA module.
    Returns a dataframe with Gene (+ optional Ensembl ID).
    """
    if not module_data or module_name not in module_data:
        return None

    gene_df = module_data[module_name]
    if gene_df is None or gene_df.empty:
        return None

    if "hgnc_symbol" in gene_df.columns:
        display_cols = ["hgnc_symbol"]
        if "ensembl_gene_id" in gene_df.columns:
            display_cols.append("ensembl_gene_id")

        result = gene_df[display_cols].copy()
        result.columns = ["Gene", "Ensembl ID"] if len(display_cols) > 1 else ["Gene"]
        return result

    return gene_df


def get_all_modules(module_data: Dict[str, pd.DataFrame]) -> list[str]:
    """Get list of all available modules."""
    if not module_data:
        return []
    return sorted(module_data.keys())


# -----------------------------------------------------------------------------
# MODULE–TRAIT MATRICES (moduleTraitCor / moduleTraitPvalue)
# -----------------------------------------------------------------------------

def _standardise_module_trait_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure module is in the index and traits are columns.
    Handles common patterns from CSV/Parquet exports.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "module" in out.columns:
        out["module"] = out["module"].astype(str).str.strip()
        out = out.set_index("module")

    if "Unnamed: 0" in out.columns:
        maybe_idx = out["Unnamed: 0"].astype(str)
        if maybe_idx.nunique() == len(out):
            out = out.drop(columns=["Unnamed: 0"])
            out.index = maybe_idx.values

    out.index = out.index.astype(str).str.strip()
    return out


def load_wcgna_mod_trait_cor() -> pd.DataFrame:
    """Load <data_dir>/wcgna/moduleTraitCor.(parquet|csv)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wcgna_dir = find_subfolder(data_dir, "wcgna")
    if wcgna_dir is None:
        return pd.DataFrame()

    fp = find_file(wcgna_dir, "moduleTraitCor.parquet") or find_file(wcgna_dir, "moduleTraitCor.csv")
    if fp is None:
        return pd.DataFrame()

    df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
    return _standardise_module_trait_matrix(df)


def load_wcgna_mod_trait_pval() -> pd.DataFrame:
    """Load <data_dir>/wcgna/moduleTraitPvalue.(parquet|csv)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wcgna_dir = find_subfolder(data_dir, "wcgna")
    if wcgna_dir is None:
        return pd.DataFrame()

    fp = find_file(wcgna_dir, "moduleTraitPvalue.parquet") or find_file(wcgna_dir, "moduleTraitPvalue.csv")
    if fp is None:
        return pd.DataFrame()

    df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
    return _standardise_module_trait_matrix(df)


# -----------------------------------------------------------------------------
# PATHWAYS / ENRICHMENT TABLES
# -----------------------------------------------------------------------------

def load_wcgna_pathways() -> Dict[str, pd.DataFrame]:
    """
    Load enrichment/pathway outputs from <data_dir>/wcgna/pathways/.

    Behaviour:
    - Reads .csv or .parquet
    - Infers module from filename stem: first token before '_' or '-' (e.g. 'black_enrichment' -> 'black')
    - Returns dict keyed by module (lowercase)
    """
    out: Dict[str, pd.DataFrame] = {}

    data_dir = find_data_dir()
    if data_dir is None:
        return out

    wcgna_dir = find_subfolder(data_dir, "wcgna")
    if wcgna_dir is None:
        return out

    pathways_dir = wcgna_dir / "pathways"
    if not pathways_dir.exists():
        return out

    for fp in pathways_dir.rglob("*"):
        if not fp.is_file() or fp.suffix.lower() not in [".csv", ".parquet"]:
            continue

        try:
            df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
            if df is None or df.empty:
                continue

            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            stem = fp.stem.strip()
            if not stem:
                continue

            module = re.split(r"[_\-\s]+", stem)[0].strip().lower()
            if not module:
                continue

            if module not in out:
                out[module] = df

        except Exception:
            continue

    return out


# -----------------------------------------------------------------------------
# CO-EXPRESSION HELPERS
# -----------------------------------------------------------------------------

def get_coexpressed_partners(gene_name: str, expr_df: pd.DataFrame, top_n: int = 15) -> Optional[pd.DataFrame]:
    """
    Find genes most strongly co-expressed with the target gene.
    expr_df should be samples x genes (or genes x samples).

    Returns dataframe of top_n genes by absolute correlation.
    """
    if expr_df is None or expr_df.empty:
        return None

    gene_col = None
    gene_lower = str(gene_name).strip().lower()

    if isinstance(expr_df.columns, pd.Index):
        matching_cols = [c for c in expr_df.columns if str(c).lower() == gene_lower]
        if matching_cols:
            gene_col = matching_cols[0]

    if gene_col is None and isinstance(expr_df.index, pd.Index):
        matching_idx = [i for i in expr_df.index if str(i).lower() == gene_lower]
        if matching_idx:
            expr_df = expr_df.T
            gene_col = matching_idx[0]

    if gene_col is None:
        return None

    try:
        correlations = expr_df.corr(numeric_only=True)[gene_col].drop(gene_col, errors="ignore")
        top_corr = correlations.abs().nlargest(top_n)

        results = []
        for gene, _abs_val in top_corr.items():
            actual_corr = float(correlations[gene])
            results.append({"Gene": gene, "Correlation": f"{actual_corr:.3f}"})

        return pd.DataFrame(results)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# PPI HELPERS
# -----------------------------------------------------------------------------

def find_ppi_interactors(gene_name: str, ppi_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Find direct PPI partners of a gene across one or more PPI dataframes.
    Each df should have columns like protein1/protein2 or gene1/gene2 or source/target.
    """
    if not ppi_data:
        return None

    interactors = set()
    gene_lower = str(gene_name).strip().lower()

    for _ppi_name, ppi_df in ppi_data.items():
        if ppi_df is None or ppi_df.empty:
            continue

        protein1_col = None
        protein2_col = None

        for col in ppi_df.columns:
            col_lower = str(col).lower()
            if protein1_col is None and ("protein1" in col_lower or "gene1" in col_lower or "source" in col_lower):
                protein1_col = col
            if protein2_col is None and ("protein2" in col_lower or "gene2" in col_lower or "target" in col_lower):
                protein2_col = col

        if protein1_col is None or protein2_col is None:
            continue

        matches1 = ppi_df[ppi_df[protein1_col].astype(str).str.lower() == gene_lower]
        interactors.update(matches1[protein2_col].astype(str).tolist())

        matches2 = ppi_df[ppi_df[protein2_col].astype(str).str.lower() == gene_lower]
        interactors.update(matches2[protein1_col].astype(str).tolist())

    interactors = {i for i in interactors if str(i).strip() and str(i).lower() != gene_lower}

    if not interactors:
        return None

    return pd.DataFrame([{"Interactor": x} for x in sorted(interactors)])


def get_network_stats(gene_name: str, ppi_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Get a minimal local network statistic: degree (count of direct edges).
    """
    if not ppi_data:
        return None

    degree = 0
    gene_lower = str(gene_name).strip().lower()

    for _ppi_name, ppi_df in ppi_data.items():
        if ppi_df is None or ppi_df.empty:
            continue

        protein1_col = None
        protein2_col = None

        for col in ppi_df.columns:
            col_lower = str(col).lower()
            if protein1_col is None and ("protein1" in col_lower or "gene1" in col_lower or "source" in col_lower):
                protein1_col = col
            if protein2_col is None and ("protein2" in col_lower or "gene2" in col_lower or "target" in col_lower):
                protein2_col = col

        if protein1_col is None or protein2_col is None:
            continue

        degree += int((ppi_df[protein1_col].astype(str).str.lower() == gene_lower).sum())
        degree += int((ppi_df[protein2_col].astype(str).str.lower() == gene_lower).sum())

    if degree == 0:
        return None

    return {"degree": degree, "description": "Direct interactors in PPI network"}
