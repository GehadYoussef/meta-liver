"""
Robust Data Loader for Meta Liver

Purpose
- Auto-detects the data directory at runtime
- Finds folders/files case-insensitively
- Loads WGCNA (supports both 'wgcna' and 'wcgna'), single-omics, knowledge graph, and PPI tables
- Stays "pure": no Streamlit imports, no st.cache_data, no UI side-effects

Usage (in Streamlit)
- Wrap calls in @st.cache_data in streamlit_app.py (recommended)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import warnings
import re
import pandas as pd


# ============================================================================
# AUTO-DETECT DATA DIRECTORY (RUNTIME)
# ============================================================================

def find_data_dir() -> Optional[Path]:
    """Auto-detect data directory (case-insensitive) - computed at runtime."""
    possible_dirs = [
        Path("meta-liver-data"),
        Path("meta_liver_data"),
        Path("data"),
        Path("../meta-liver-data"),
        Path.home() / "meta-liver-data",
        Path.home() / "meta_liver_data",
    ]
    for dir_path in possible_dirs:
        if dir_path.exists():
            return dir_path.resolve()
    return None


def find_subfolder(parent: Path, folder_pattern: str) -> Optional[Path]:
    """Find immediate subfolder (case-insensitive)."""
    if parent is None or not parent.exists():
        return None

    exact_path = parent / folder_pattern
    if exact_path.exists() and exact_path.is_dir():
        return exact_path

    pat = folder_pattern.lower()
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == pat:
            return item

    # Fall back to deeper search (optional but useful if structure varies)
    for item in parent.rglob("*"):
        if item.is_dir() and item.name.lower() == pat:
            return item

    return None


def find_file(directory: Path, filename_pattern: str) -> Optional[Path]:
    """Find file in directory (case-insensitive)."""
    if directory is None or not directory.exists():
        return None

    exact_path = directory / filename_pattern
    if exact_path.exists() and exact_path.is_file():
        return exact_path

    target = filename_pattern.lower()
    for file in directory.rglob("*"):
        if not file.is_file():
            continue
        name = file.name.lower()
        if name == target or target in name:
            return file

    return None


def _find_wgcna_dir(data_dir: Path) -> Optional[Path]:
    """
    Prefer 'wgcna' (you renamed to this), but still accept legacy 'wcgna'.
    """
    if data_dir is None:
        return None
    return find_subfolder(data_dir, "wgcna") or find_subfolder(data_dir, "wcgna")


# ============================================================================
# SAFE READERS
# ============================================================================

def _read_table(file_path: Path, *, index_col: Optional[int] = None) -> pd.DataFrame:
    """Read CSV/Parquet defensively; returns empty DF on failure."""
    if file_path is None or not file_path.exists():
        return pd.DataFrame()

    try:
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path, index_col=index_col)
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"Failed to read {file_path.name}: {e}")
        return pd.DataFrame()


def _load_first_existing(directory: Path, candidates: list[str], *, index_col: Optional[int] = None) -> pd.DataFrame:
    """Try a list of filenames (case-insensitive); return the first that loads."""
    if directory is None or not directory.exists():
        return pd.DataFrame()

    for name in candidates:
        fp = find_file(directory, name)
        if fp is not None:
            df = _read_table(fp, index_col=index_col)
            if not df.empty:
                return df
    return pd.DataFrame()


# ============================================================================
# WGCNA LOADERS (supports wgcna/ and wcgna/)
# ============================================================================

def load_wgcna_expr() -> pd.DataFrame:
    """Load WGCNA expression matrix."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["datExpr_processed.parquet", "datExpr_processed.csv"],
        index_col=0
    )


def load_wgcna_mes() -> pd.DataFrame:
    """Load WGCNA module eigengenes."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["MEs_processed.parquet", "MEs_processed.csv"],
        index_col=0
    )


def load_wgcna_mod_trait_cor() -> pd.DataFrame:
    """Load module-trait correlations."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["moduleTraitCor.parquet", "moduleTraitCor.csv"],
        index_col=0
    )


def load_wgcna_mod_trait_pval() -> pd.DataFrame:
    """Load module-trait p-values."""
    data_dir = find_data_dir()
    if data_dir is None:
        return pd.DataFrame()

    wgcna_dir = _find_wgcna_dir(data_dir)
    if wgcna_dir is None:
        return pd.DataFrame()

    return _load_first_existing(
        wgcna_dir,
        ["moduleTraitPvalue.parquet", "moduleTraitPvalue.csv"],
        index_col=0
    )


def load_wgcna_pathways() -> Dict[str, pd.DataFrame]:
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

    out: Dict[str, pd.DataFrame] = {}
    for file_path in pathways_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".csv", ".parquet"]:
            continue

        df = _read_table(file_path, index_col=None)
        if df.empty:
            continue

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        stem = file_path.stem.strip()
        if not stem:
            continue

        tok = re.split(r"[_\-\s]+", stem)[0].strip()
        module_key = (tok if tok else stem).lower()

        if module_key not in out:
            out[module_key] = df

    return out


def load_wgcna_module_trait_heatmap_pdf_path() -> Optional[Path]:
    """
    Return a Path to the module-trait heatmap PDF if present, else None.
    """
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


# ============================================================================
# SINGLE-OMICS / KG / PPI LOADERS
# ============================================================================

def load_single_omics_studies() -> Dict[str, pd.DataFrame]:
    """Load all single-omics studies (all CSV/Parquet files under single_omics)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    single_omics_dir = find_subfolder(data_dir, "single_omics")
    if single_omics_dir is None:
        return {}

    studies: Dict[str, pd.DataFrame] = {}
    for file_path in single_omics_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".csv", ".parquet"]:
            continue

        study_name = file_path.stem
        df = _read_table(file_path, index_col=None)
        if not df.empty:
            studies[study_name] = df

    return studies


def load_kg_data() -> Dict[str, pd.DataFrame]:
    """Load all knowledge graph data (all CSV/Parquet files under knowledge_graphs)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    kg_dir = find_subfolder(data_dir, "knowledge_graphs")
    if kg_dir is None:
        return {}

    kg_data: Dict[str, pd.DataFrame] = {}
    for file_path in kg_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".csv", ".parquet"]:
            continue

        name = file_path.stem
        df = _read_table(file_path, index_col=None)
        if not df.empty:
            kg_data[name] = df

    return kg_data


def load_ppi_data() -> Dict[str, pd.DataFrame]:
    """Load all PPI network data (all CSV/Parquet files under ppi_networks)."""
    data_dir = find_data_dir()
    if data_dir is None:
        return {}

    ppi_dir = find_subfolder(data_dir, "ppi_networks")
    if ppi_dir is None:
        return {}

    ppi_data: Dict[str, pd.DataFrame] = {}
    for file_path in ppi_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".csv", ".parquet"]:
            continue

        name = file_path.stem
        df = _read_table(file_path, index_col=None)
        if not df.empty:
            ppi_data[name] = df

    return ppi_data


# ============================================================================
# DATA AVAILABILITY / SUMMARY
# ============================================================================

def check_data_availability() -> Dict[str, bool]:
    """Check what data is available (loads are cheap if Streamlit caches upstream)."""
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

    return {
        "data_dir": data_dir_ok,
        "wgcna_expr": expr_ok,
        "wgcna_mes": mes_ok,
        "wgcna_mod_trait_cor": cor_ok,
        "wgcna_mod_trait_pval": pval_ok,
        "wgcna_pathways": pathways_ok,
        "wgcna_heatmap_pdf": heatmap_ok,
        "single_omics": single_ok,
        "knowledge_graphs": kg_ok,
        "ppi_networks": ppi_ok,
    }


def get_data_summary() -> str:
    """Human-readable data availability summary (no Streamlit formatting dependencies)."""
    avail = check_data_availability()

    if not avail["data_dir"]:
        return "Data Availability:\n\n✗ Data directory not found"

    lines = ["Data Availability:\n", "✓ Data directory found\n"]

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
        for name, df in studies.items():
            lines.append(f"  - {name}: {len(df)} rows")
    else:
        lines.append("✗ Single-Omics Studies: Not found")

    lines.append(f"✓ Knowledge Graphs: {len(load_kg_data())} datasets" if avail["knowledge_graphs"] else "✗ Knowledge Graphs: Not found")
    lines.append(f"✓ PPI Networks: {len(load_ppi_data())} datasets" if avail["ppi_networks"] else "✗ PPI Networks: Not found")

    return "\n".join(lines)


# ============================================================================
# SEARCH HELPERS
# ============================================================================

def search_gene_in_studies(gene_name: str) -> Dict[str, pd.DataFrame]:
    """Search for a gene across all single-omics studies (substring match)."""
    studies = load_single_omics_studies()
    results: Dict[str, pd.DataFrame] = {}

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


def search_drug_in_kg(drug_name: str) -> Dict[str, pd.DataFrame]:
    """Search for a drug in knowledge graph tables (substring match on Name column)."""
    kg_data = load_kg_data()
    results: Dict[str, pd.DataFrame] = {}

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
