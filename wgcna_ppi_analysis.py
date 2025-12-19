"""
WGCNA and PPI Network Analysis Module

Supports:
1) WGCNA module gene mapping files from <data_dir>/(wgcna|wcgna)/modules/
2) Module–trait matrices from <data_dir>/(wgcna|wcgna)/ (moduleTraitCor / moduleTraitPvalue)
3) Enrichment/pathway tables from <data_dir>/(wgcna|wcgna)/pathways/
4) Active drugs table from <data_dir>/(wgcna|wcgna)/active_drugs.(parquet|xlsx|csv)
5) Basic co-expression (correlation) helper on an expression matrix
6) Simple PPI neighbour lookup and degree stats across one or more PPI tables
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List

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
            return path.resolve()
    return None


def find_subfolder(parent: Path, folder_pattern: str) -> Optional[Path]:
    """Find subfolder (case-insensitive)."""
    if parent is None or not parent.exists():
        return None

    exact_path = parent / folder_pattern
    if exact_path.exists() and exact_path.is_dir():
        return exact_path

    pat = folder_pattern.lower()
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == pat:
            return item

    return None


def find_file(directory: Path, filename_pattern: str) -> Optional[Path]:
    """Find file in directory (case-insensitive; substring match allowed)."""
    if directory is None or not directory.exists():
        return None

    exact_path = directory / filename_pattern
    if exact_path.exists() and exact_path.is_file():
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


def _find_wgcna_dir(data_dir: Path) -> Optional[Path]:
    """Prefer wgcna/, but accept legacy wcgna/."""
    if data_dir is None:
        return None
    return find_subfolder(data_dir, "wgcna") or find_subfolder(data_dir, "wcgna")


# -----------------------------------------------------------------------------
# ACTIVE DRUGS (targets -> genes)
# -----------------------------------------------------------------------------

def _read_active_drugs_excel(fp: Path) -> pd.DataFrame:
    """
    Robust reader for Excel exports where the real header row is embedded a few rows down.
    Looks for a row containing 'DrugBank_Accession' and uses that as the header.
    """
    raw = pd.read_excel(fp, header=None)

    header_idx = None
    for i in range(min(50, len(raw))):
        first_cell = str(raw.iloc[i, 0]).strip()
        if first_cell == "DrugBank_Accession":
            header_idx = i
            break

    if header_idx is None:
        return pd.read_excel(fp)

    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = [str(x).strip() for x in raw.iloc[header_idx].tolist()]
    df = df.reset_index(drop=True)
    return df


def _standardise_active_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise key column names and types for the active drugs table."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    col_map = {}
    lower_cols = {str(c).lower(): c for c in out.columns}

    def _pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c.lower() in lower_cols:
                return lower_cols[c.lower()]
        return None

    acc = _pick("DrugBank_Accession", "drugbank_accession", "drugbank id", "drugbank_id")
    name = _pick("Drug Name", "drug name", "name", "drug")
    targets = _pick("Drug Targets", "drug targets", "targets", "target_genes", "target genes")
    dist = _pick("distance", "dist")
    z = _pick("z-score", "zscore", "z_score")

    if acc and acc != "DrugBank_Accession":
        col_map[acc] = "DrugBank_Accession"
    if name and name != "Drug Name":
        col_map[name] = "Drug Name"
    if targets and targets != "Drug Targets":
        col_map[targets] = "Drug Targets"
    if dist and dist != "distance":
        col_map[dist] = "distance"
    if z and z != "z-score":
        col_map[z] = "z-score"

    if col_map:
        out = out.rename(columns=col_map)

    for c in ["DrugBank_Accession", "Drug Name", "Drug Targets"]:
        if c not in out.columns:
            out[c] = np.nan

    out["DrugBank_Accession"] = out["DrugBank_Accession"].astype(str).str.strip()
    out["Drug Name"] = out["Drug Name"].astype(str).str.strip()
    out["Drug Targets"] = out["Drug Targets"].astype(str).str.strip()

    if "distance" in out.columns:
        out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    if "z-score" in out.columns:
        out["z-score"] = pd.to_numeric(out["z-score"], errors="coerce")

    out = out.dropna(how="all")
    return out


def load_wgcna_active_drugs() -> pd.DataFrame:
    """
    Load <data_dir>/(wgcna|wcgna)/active_drugs.(parquet|xlsx|csv).

    Returns:
        dataframe with at least: DrugBank_Accession, Drug Name, Drug Targets
    """
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    fp = (
        find_file(wgcna_dir, "active_drugs.parquet")
        or find_file(wgcna_dir, "active_drugs.xlsx")
        or find_file(wgcna_dir, "active_drugs.csv")
    )
    if fp is None:
        return pd.DataFrame()

    try:
        if fp.suffix.lower() == ".parquet":
            df = pd.read_parquet(fp)
        elif fp.suffix.lower() in [".xlsx", ".xls"]:
            df = _read_active_drugs_excel(fp)
        else:
            df = pd.read_csv(fp)

        return _standardise_active_drugs(df)
    except Exception as e:
        print(f"DEBUG: Error loading active drugs from {fp}: {e}", file=sys.stderr)
        return pd.DataFrame()


def build_gene_to_drugs_index(active_drugs_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build an index: GENE (upper) -> list of drug records where that gene appears in Drug Targets.

    Records include: Drug Name, DrugBank_Accession, distance, z-score, Indication (if present).
    Sorting: best (most negative) z-score first, then smaller distance.
    """
    if active_drugs_df is None or active_drugs_df.empty:
        return {}

    df = active_drugs_df.copy()

    targets_col = "Drug Targets" if "Drug Targets" in df.columns else None
    if targets_col is None:
        return {}

    has_ind = "Indication" in df.columns
    has_moa = "Mechanism of Action" in df.columns

    gene_to_drugs: Dict[str, List[Dict[str, Any]]] = {}

    splitter = re.compile(r"[;,|]\s*|\s*,\s*")

    for _, row in df.iterrows():
        targets_raw = row.get(targets_col, "")
        if targets_raw is None:
            continue

        targets_str = str(targets_raw).strip()
        if not targets_str or targets_str.lower() in ["nan", "none"]:
            continue

        targets = [t.strip().upper() for t in splitter.split(targets_str) if t and str(t).strip()]
        if not targets:
            continue

        rec = {
            "Drug Name": str(row.get("Drug Name", "")).strip(),
            "DrugBank_Accession": str(row.get("DrugBank_Accession", "")).strip(),
            "distance": row.get("distance", np.nan),
            "z-score": row.get("z-score", np.nan),
        }
        if has_ind:
            rec["Indication"] = row.get("Indication", np.nan)
        if has_moa:
            rec["Mechanism of Action"] = row.get("Mechanism of Action", np.nan)

        for g in targets:
            gene_to_drugs.setdefault(g, []).append(rec)

    def _sort_key(r: Dict[str, Any]):
        z = r.get("z-score", np.nan)
        d = r.get("distance", np.nan)
        z_key = float(z) if pd.notna(z) else float("inf")
        d_key = float(d) if pd.notna(d) else float("inf")
        return (z_key, d_key, str(r.get("Drug Name", "")))

    for g in list(gene_to_drugs.keys()):
        gene_to_drugs[g] = sorted(gene_to_drugs[g], key=_sort_key)

    return gene_to_drugs


# -----------------------------------------------------------------------------
# WGCNA MODULE GENE MAPPINGS
# -----------------------------------------------------------------------------

def load_wgcna_module_data() -> Dict[str, pd.DataFrame]:
    """
    Load per-module gene mapping files from <data_dir>/(wgcna|wcgna)/modules/.

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

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        print(f"DEBUG: wgcna/wcgna folder not found under {data_dir}", file=sys.stderr)
        return {}

    modules_dir = wgcna_dir / "modules"
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
    """Find which WGCNA module a gene belongs to; returns module name and first matching row."""
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
    """Get all genes in a specific WGCNA module; returns dataframe with Gene (+ optional Ensembl ID)."""
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
    """Ensure module is in the index and traits are columns; handles common export patterns."""
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


def load_wgcna_mod_trait_cor() -> pd.DataFrame:
    """Load <data_dir>/(wgcna|wcgna)/moduleTraitCor.(parquet|csv)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    fp = find_file(wgcna_dir, "moduleTraitCor.parquet") or find_file(wgcna_dir, "moduleTraitCor.csv")
    if fp is None:
        return pd.DataFrame()

    df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
    return _standardise_module_trait_matrix(df)


def load_wgcna_mod_trait_pval() -> pd.DataFrame:
    """Load <data_dir>/(wgcna|wcgna)/moduleTraitPvalue.(parquet|csv)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    fp = find_file(wgcna_dir, "moduleTraitPvalue.parquet") or find_file(wgcna_dir, "moduleTraitPvalue.csv")
    if fp is None:
        return pd.DataFrame()

    df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
    return _standardise_module_trait_matrix(df)


def load_wcgna_mod_trait_cor() -> pd.DataFrame:
    return load_wgcna_mod_trait_cor()


def load_wcgna_mod_trait_pval() -> pd.DataFrame:
    return load_wgcna_mod_trait_pval()


# -----------------------------------------------------------------------------
# PATHWAYS / ENRICHMENT TABLES
# -----------------------------------------------------------------------------

def load_wgcna_pathways() -> Dict[str, pd.DataFrame]:
    """
    Load enrichment/pathway outputs from <data_dir>/(wgcna|wcgna)/pathways/.

    Behaviour:
    - Reads .csv or .parquet
    - Infers module from filename stem: first token before '_' or '-' or space (e.g. 'black_enrichment' -> 'black')
    - Returns dict keyed by module (lowercase)
    """
    out: Dict[str, pd.DataFrame] = {}

    data_dir = find_data_dir()
    if data_dir is None:
        return out

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return out

    pathways_dir = wgcna_dir / "pathways"
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


def load_wcgna_pathways() -> Dict[str, pd.DataFrame]:
    return load_wgcna_pathways()


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
    """Get a minimal local network statistic: degree (count of direct edges)."""
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
