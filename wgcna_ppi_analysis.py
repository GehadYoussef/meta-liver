"""
WGCNA and PPI Network Analysis Module

Supports:
1) WGCNA module gene mapping files from <data_dir>/(wgcna|wcgna)/modules/
2) Module–trait matrices from <data_dir>/(wgcna|wcgna)/ (moduleTraitCor / moduleTraitPvalue)
3) Enrichment/pathway tables from <data_dir>/(wgcna|wcgna)/pathways/
4) Active drugs table from <data_dir>/(wgcna|wcgna)/active_drugs.(parquet|csv|xlsx)
5) Basic co-expression (correlation) helper on an expression matrix
6) Simple PPI neighbour lookup and degree stats across one or more PPI tables

Notes on active_drugs:
- Expected columns include:
  - "Drug Targets" (comma-separated gene symbols)
  - "Drug Name", "DrugBank_Accession", and optionally "distance", "z-score"
- Matching is case-insensitive and uses whole-token matching after splitting targets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Iterable, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# WGCNA MODULE GENE MAPPINGS
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# ACTIVE DRUGS (gene -> drugs via "Drug Targets" column)
# ---------------------------------------------------------------------

def load_wgcna_active_drugs() -> pd.DataFrame:
    """
    Load <data_dir>/(wgcna|wcgna)/active_drugs.(parquet|csv|xlsx|xls).

    Returns empty DataFrame if not found.
    """
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    fp = (
        find_file(wgcna_dir, "active_drugs.parquet")
        or find_file(wgcna_dir, "active_drugs.csv")
        or find_file(wgcna_dir, "active_drugs.xlsx")
        or find_file(wgcna_dir, "active_drugs.xls")
    )
    if fp is None:
        return pd.DataFrame()

    try:
        if fp.suffix.lower() == ".parquet":
            df = pd.read_parquet(fp)
        elif fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
        else:
            df = pd.read_excel(fp)
    except Exception as e:
        print(f"DEBUG: Failed to read active_drugs from {fp}: {e}", file=sys.stderr)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalise column names a bit (keep originals too)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _infer_targets_col(active_drugs: pd.DataFrame) -> Optional[str]:
    """
    Prefer exact 'Drug Targets', otherwise pick the first column containing 'target'.
    """
    if active_drugs is None or active_drugs.empty:
        return None

    exact = _pick_col(active_drugs, ["Drug Targets", "drug_targets", "targets", "Target", "Targets"])
    if exact is not None:
        return exact

    for c in active_drugs.columns:
        if "target" in str(c).lower():
            return c

    return None


def _split_targets(val: Any) -> set[str]:
    """
    Split a targets string into tokens (gene symbols), tolerant to separators.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()

    s = str(val).strip()
    if not s:
        return set()

    # Common separators: comma, semicolon, pipe, slash, newline
    parts = re.split(r"[,\n;\|/]+", s)
    out = set()
    for p in parts:
        t = p.strip().upper()
        # Keep conservative token cleaning
        t = re.sub(r"\s+", "", t)
        if t:
            out.add(t)
    return out


def build_gene_to_drugs_index(active_drugs: pd.DataFrame) -> Dict[str, list[int]]:
    """
    Build index: gene_symbol -> list of row indices in active_drugs where gene appears in targets.

    This is designed to be built once (e.g., at app load) and reused.
    """
    idx: Dict[str, list[int]] = {}

    if active_drugs is None or active_drugs.empty:
        return idx

    targets_col = _infer_targets_col(active_drugs)
    if targets_col is None:
        return idx

    for i, val in active_drugs[targets_col].items():
        genes = _split_targets(val)
        for g in genes:
            idx.setdefault(g, []).append(i)

    return idx


def summarise_drugs_for_genes(
    genes: Iterable[str],
    active_drugs: pd.DataFrame,
    gene_to_rows: Optional[Dict[str, list[int]]] = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    For each gene, return a 1-row summary with drug columns:
      Gene, n_drugs, Top drugs, Top DrugBank IDs, Best distance, Best z-score
    """
    if active_drugs is None or active_drugs.empty:
        return pd.DataFrame({"Gene": [str(g).strip().upper() for g in genes], "n_drugs": 0})

    if gene_to_rows is None:
        gene_to_rows = build_gene_to_drugs_index(active_drugs)

    col_name = _pick_col(active_drugs, ["Drug Name", "drug name", "name", "Drug"])
    col_dbid = _pick_col(active_drugs, ["DrugBank_Accession", "drugbank_accession", "DrugBank", "DrugBank ID"])
    col_dist = _pick_col(active_drugs, ["distance", "Distance"])
    col_z = _pick_col(active_drugs, ["z-score", "zscore", "Z-score", "Z Score", "z score"])

    rows = []
    for g in genes:
        gene = str(g).strip().upper()
        hit_rows = gene_to_rows.get(gene, []) if gene_to_rows else []

        if not hit_rows:
            rows.append({
                "Gene": gene,
                "n_drugs": 0,
                "Top drugs": "",
                "Top DrugBank IDs": "",
                "Best distance": np.nan,
                "Best z-score": np.nan,
            })
            continue

        sub = active_drugs.loc[hit_rows].copy()

        # Try to sort by distance (ascending), then z-score (ascending; more negative usually "better")
        sort_cols = []
        if col_dist is not None:
            sub[col_dist] = pd.to_numeric(sub[col_dist], errors="coerce")
            sort_cols.append(col_dist)
        if col_z is not None:
            sub[col_z] = pd.to_numeric(sub[col_z], errors="coerce")
            sort_cols.append(col_z)

        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=[True] * len(sort_cols), na_position="last")

        top = sub.head(int(top_n))

        top_names = []
        if col_name is not None:
            top_names = [str(x) for x in top[col_name].fillna("").tolist() if str(x).strip()]

        top_ids = []
        if col_dbid is not None:
            top_ids = [str(x) for x in top[col_dbid].fillna("").tolist() if str(x).strip()]

        best_dist = float(top[col_dist].dropna().iloc[0]) if col_dist is not None and top[col_dist].notna().any() else np.nan
        best_z = float(top[col_z].dropna().iloc[0]) if col_z is not None and top[col_z].notna().any() else np.nan

        rows.append({
            "Gene": gene,
            "n_drugs": int(len(hit_rows)),
            "Top drugs": ", ".join(top_names[:int(top_n)]),
            "Top DrugBank IDs": ", ".join(top_ids[:int(top_n)]),
            "Best distance": best_dist,
            "Best z-score": best_z,
        })

    return pd.DataFrame(rows)


def get_module_genes(
    module_name: str,
    module_data: Dict[str, pd.DataFrame],
    active_drugs: Optional[pd.DataFrame] = None,
    gene_to_drugs_index: Optional[Dict[str, list[int]]] = None,
    top_n_genes: Optional[int] = None,
    drugs_top_n: int = 5,
    replace_ensembl_with_drugs_for_module: str = "brown"
) -> Optional[pd.DataFrame]:
    """
    Get genes in a WGCNA module.

    Default behaviour:
      - returns Gene (+ optional Ensembl ID) if available.

    If active_drugs is provided AND module_name matches replace_ensembl_with_drugs_for_module (default 'brown'):
      - returns Gene + drug summary columns instead of Ensembl ID.
    """
    if not module_data or module_name not in module_data:
        return None

    gene_df = module_data[module_name]
    if gene_df is None or gene_df.empty:
        return None

    if "hgnc_symbol" in gene_df.columns:
        base = gene_df.copy()
        base["hgnc_symbol"] = base["hgnc_symbol"].astype(str).str.strip().str.upper()
        genes = base["hgnc_symbol"].tolist()

        if top_n_genes is not None:
            genes = genes[:int(top_n_genes)]

        # Replace Ensembl with drug columns for the requested module (e.g. brown)
        if (
            active_drugs is not None
            and not active_drugs.empty
            and str(module_name).strip().lower() == str(replace_ensembl_with_drugs_for_module).strip().lower()
        ):
            drug_tbl = summarise_drugs_for_genes(
                genes=genes,
                active_drugs=active_drugs,
                gene_to_rows=gene_to_drugs_index,
                top_n=drugs_top_n
            )
            return drug_tbl

        # Otherwise show Gene (+ Ensembl if present)
        display_cols = ["hgnc_symbol"]
        if "ensembl_gene_id" in base.columns:
            display_cols.append("ensembl_gene_id")

        result = base[display_cols].copy()
        result.columns = ["Gene", "Ensembl ID"] if len(display_cols) > 1 else ["Gene"]

        if top_n_genes is not None:
            result = result.head(int(top_n_genes))

        return result

    return gene_df


def get_all_modules(module_data: Dict[str, pd.DataFrame]) -> list[str]:
    """Get list of all available modules."""
    if not module_data:
        return []
    return sorted(module_data.keys())


# ---------------------------------------------------------------------
# MODULE–TRAIT MATRICES (moduleTraitCor / moduleTraitPvalue)
# ---------------------------------------------------------------------

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


# Backwards-compatible aliases (your streamlit_app.py may still import these names)
def load_wcgna_mod_trait_cor() -> pd.DataFrame:
    return load_wgcna_mod_trait_cor()


def load_wcgna_mod_trait_pval() -> pd.DataFrame:
    return load_wgcna_mod_trait_pval()


# ---------------------------------------------------------------------
# PATHWAYS / ENRICHMENT TABLES
# ---------------------------------------------------------------------

def load_wgcna_pathways() -> Dict[str, pd.DataFrame]:
    """
    Load enrichment/pathway outputs from <data_dir>/(wgcna|wcgna)/pathways/.

    Behaviour:
    - Reads .csv or .parquet
    - Infers module from filename stem: first token before '_' or '-' or space
      (e.g. 'black_enrichment' -> 'black')
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


# Backwards-compatible alias
def load_wcgna_pathways() -> Dict[str, pd.DataFrame]:
    return load_wgcna_pathways()


# ---------------------------------------------------------------------
# CO-EXPRESSION HELPERS
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# PPI HELPERS
# ---------------------------------------------------------------------

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
