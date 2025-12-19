"""
In vitro (human stem cell-derived iHeps) MASLD model analysis for Meta Liver.

This module backs a dedicated Streamlit tab that summarises DEGs from two iHeps lines
(1b and 5a) under three conditions compared with healthy controls (HCM):
  - OA+PA vs HCM
  - OA+PA + Resistin/Myostatin vs HCM
  - OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM
(PBMCs were not included in RNA-seq; the DEGs reflect iHeps only.)

Expected location (inside the app data directory):
  meta-liver-data/stem_cell_model/

Accepted DEG filenames (CSV or Parquet):
  1) processed_degs_<LINE>_<CONTRAST>.(csv|parquet)
     e.g. processed_degs_1b_OAPAvsHCM.parquet
  2) <CONTRAST>_<LINE>.(csv|parquet)
     e.g. OAPAvsHCM_1b.parquet

Accepted contrasts:
  OAPAvsHCM
  OAPAResMyovsHCM
  OAPAResMyoPBMCsvsHCM

Gene identifiers:
- DEG tables may store Ensembl IDs, gene symbols, or both.
- If a user searches by gene symbol and the symbol is not present, we try to map
  symbol -> Ensembl using:
    (a) gene_mapping.csv (if present), and/or
    (b) embedded columns inside the DEG tables (if present).
- If mapping fails, we fall back to using the entered identifier as-is.

Parquet note:
Reading Parquet requires pyarrow or fastparquet. If missing, the UI will show
a clear install hint rather than crashing the whole app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from robust_data_loader import find_data_dir


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CONTRAST_TOKENS = [
    "OAPAvsHCM",
    "OAPAResMyovsHCM",
    "OAPAResMyoPBMCsvsHCM",
]

CONTRAST_LABELS = {
    "OAPAvsHCM": "OA+PA vs HCM",
    "OAPAResMyovsHCM": "OA+PA + Resistin/Myostatin vs HCM",
    "OAPAResMyoPBMCsvsHCM": "OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM",
}

CONTRAST_HELP = {
    "OA+PA vs HCM": "Fatty-acid overload model (oleic + palmitic acid) compared with healthy controls.",
    "OA+PA + Resistin/Myostatin vs HCM": "Fatty acids plus adipose/muscle-derived signalling molecules.",
    "OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM": "As above plus PBMC co-culture; PBMCs not sequenced.",
}

LINE_TOKENS = ["1b", "5a"]

GENE_MAPPING_FILENAME = "gene_mapping.csv"

_RE_PROCESSED = re.compile(r"^processed_degs_(?P<line>[^_]+)_(?P<contrast>.+)$", flags=re.IGNORECASE)
_RE_CONTRAST_LINE = re.compile(r"^(?P<contrast>.+)_(?P<line>[^_]+)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class InVitroKey:
    line: str        # "1b" or "5a"
    contrast: str    # canonical token e.g. "OAPAvsHCM"


# -----------------------------------------------------------------------------
# Folder discovery
# -----------------------------------------------------------------------------

def _find_stem_cell_model_dir() -> Optional[Path]:
    data_dir = find_data_dir()
    if data_dir is None:
        return None

    direct = data_dir / "stem_cell_model"
    if direct.exists() and direct.is_dir():
        return direct

    for p in data_dir.iterdir():
        if p.is_dir() and p.name.lower() == "stem_cell_model":
            return p

    for p in data_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == "stem_cell_model":
            return p

    return None


# -----------------------------------------------------------------------------
# File discovery + parsing
# -----------------------------------------------------------------------------

def _canon_contrast(token: str) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    for k in CONTRAST_TOKENS:
        if k.lower() == t.lower():
            return k
    return None


def _canon_line(line: str) -> Optional[str]:
    if line is None:
        return None
    t = str(line).strip()
    for k in LINE_TOKENS:
        if k.lower() == t.lower():
            return k
    return None


def _parse_deg_stem(stem: str) -> Optional[InVitroKey]:
    """
    Supports:
      processed_degs_<LINE>_<CONTRAST>
      <CONTRAST>_<LINE>
    """
    if not stem:
        return None

    s = stem.strip()

    m = _RE_PROCESSED.match(s)
    if m:
        line = _canon_line(m.group("line"))
        contrast = _canon_contrast(m.group("contrast"))
        if line and contrast:
            return InVitroKey(line=line, contrast=contrast)
        return None

    m = _RE_CONTRAST_LINE.match(s)
    if m:
        contrast = _canon_contrast(m.group("contrast"))
        line = _canon_line(m.group("line"))
        if line and contrast:
            return InVitroKey(line=line, contrast=contrast)
        return None

    return None


def discover_invitro_deg_files() -> Dict[InVitroKey, Path]:
    """
    Discovers DEG files in stem_cell_model/. Returns {key -> path}.
    Searches recursively so subfolders like stem_cell_model/parquet/ work.
    Ignores gene_mapping.csv.
    If both CSV and Parquet exist for the same key, prefers Parquet.
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return {}

    out: Dict[InVitroKey, Path] = {}

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.lower() == GENE_MAPPING_FILENAME.lower():
            continue

        ext = p.suffix.lower()
        if ext not in (".csv", ".parquet"):
            continue

        key = _parse_deg_stem(p.stem)
        if key is None:
            continue

        if key in out:
            # prefer parquet
            if out[key].suffix.lower() == ".csv" and p.suffix.lower() == ".parquet":
                out[key] = p
        else:
            out[key] = p

    return out


# -----------------------------------------------------------------------------
# Gene mapping
# -----------------------------------------------------------------------------

def _load_gene_mapping_csv() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Loads gene_mapping.csv if present. Returns (symbol_to_ensg, ensg_to_symbol).

    Robust column detection: accepts common variants like:
      Ensembl: 'Gene stable ID', 'ensembl_gene_id', 'Ensembl ID'
      Symbol:  'Gene name', 'symbol', 'gene_symbol', 'Gene'
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return {}, {}

    fp = root / GENE_MAPPING_FILENAME
    if not fp.exists():
        return {}, {}

    try:
        gm = pd.read_csv(fp)
    except Exception:
        return {}, {}

    cols = {c.lower(): c for c in gm.columns}

    ensg_col = (
        cols.get("gene stable id")
        or cols.get("ensembl id")
        or cols.get("ensembl_gene_id")
        or cols.get("ensembl")
        or cols.get("ensg")
    )
    sym_col = (
        cols.get("gene name")
        or cols.get("gene_symbol")
        or cols.get("symbol")
        or cols.get("gene")
        or cols.get("hgnc_symbol")
    )

    if ensg_col is None or sym_col is None:
        return {}, {}

    tmp = gm[[ensg_col, sym_col]].copy()
    tmp[ensg_col] = tmp[ensg_col].astype(str).str.strip().str.upper()
    tmp[sym_col] = tmp[sym_col].astype(str).str.strip().str.upper()

    symbol_to_ensg: Dict[str, str] = {}
    ensg_to_symbol: Dict[str, str] = {}

    for _, r in tmp.iterrows():
        ensg = str(r[ensg_col]).strip().upper()
        sym = str(r[sym_col]).strip().upper()
        if not ensg or ensg == "NAN" or not sym or sym == "NAN":
            continue
        # strip version suffix if any
        ensg = ensg.split(".")[0]
        symbol_to_ensg.setdefault(sym, ensg)
        ensg_to_symbol.setdefault(ensg, sym)

    return symbol_to_ensg, ensg_to_symbol


def _resolve_query_to_gene_id(query: str, symbol_to_ensg: Dict[str, str]) -> Tuple[str, Optional[str]]:
    """
    Returns (gene_id_to_search, resolved_symbol_if_any)
    - If query is ENSG-like -> return ENSG
    - Else try symbol_to_ensg
    - Else return query as-is
    """
    q = str(query).strip().upper()
    if not q:
        return "", None

    if q.startswith("ENSG") or q.startswith("ENSMUSG"):
        return q.split(".")[0], None

    ensg = symbol_to_ensg.get(q)
    if ensg:
        return ensg.split(".")[0], q

    return q, q


# -----------------------------------------------------------------------------
# Robust table readers + normalisation
# -----------------------------------------------------------------------------

# IMPORTANT: split candidates into Ensembl vs Symbol (do NOT treat 'Gene' as Ensembl by default)
_ENS_COL_CANDIDATES = [
    "Ensembl ID", "Gene stable ID", "Gene stable id",
    "ensembl_gene_id", "ensembl id", "ensembl_id", "ensembl",
    "ensg", "gene_id", "GeneID"
]
_SYMBOL_COL_CANDIDATES = [
    "Gene", "gene", "symbol", "gene_symbol", "hgnc_symbol",
    "Gene name", "gene name", "external_gene_name"
]

_LOGFC_COL_CANDIDATES = ["log2FoldChange", "logFC", "log2FC", "log2_fc", "log2foldchange"]
_PVAL_COL_CANDIDATES = ["pvalue", "pval", "PValue", "p_value"]
_PADJ_COL_CANDIDATES = ["padj", "FDR", "adj_pval", "adj_pvalue", "qvalue", "q_value"]
_STAT_COL_CANDIDATES = ["stat", "WaldStatistic", "wald_stat", "t", "t_stat"]
_BASEMEAN_COL_CANDIDATES = ["baseMean", "base_mean", "mean", "avg_expression"]


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _read_deg_file_safe(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            return df, None
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            return df, None
        return None, f"Unsupported file type: {path.name}"
    except Exception as e:
        if path.suffix.lower() == ".parquet":
            msg = (
                f"Could not read parquet file: {path.name}\n\n"
                f"Underlying error: {type(e).__name__}: {e}\n\n"
                "Fix: add a parquet engine to your environment, for example include this in requirements.txt:\n"
                "  pyarrow\n"
                "Then redeploy/restart the app."
            )
            return None, msg
        return None, f"Could not read {path.name}: {type(e).__name__}: {e}"


def _looks_ensembl(x: str) -> bool:
    if not x:
        return False
    u = str(x).strip().upper()
    u = u.split(".")[0]
    return u.startswith("ENSG") or u.startswith("ENSMUSG")


def normalise_deg_table(
    df: pd.DataFrame,
    ensg_to_symbol: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Standardises key columns to:
      Gene (symbol if available, else Ensembl), Ensembl ID (if available),
      logFC, pval, padj, stat, baseMean
    Leaves other columns intact.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # If the index looks like Ensembl IDs, bring it back even if index.name is None
    try:
        idx_vals = out.index.astype(str)
        if (idx_vals.str.upper().str.startswith(("ENSG", "ENSMUSG")).mean() > 0.5) and ("Ensembl ID" not in out.columns):
            out = out.reset_index().rename(columns={"index": "Ensembl ID"})
    except Exception:
        pass

    cols = list(out.columns)

    ens_col = _pick_col(cols, _ENS_COL_CANDIDATES)
    sym_col = _pick_col(cols, _SYMBOL_COL_CANDIDATES)

    logfc_col = _pick_col(cols, _LOGFC_COL_CANDIDATES)
    pval_col = _pick_col(cols, _PVAL_COL_CANDIDATES)
    padj_col = _pick_col(cols, _PADJ_COL_CANDIDATES)
    stat_col = _pick_col(cols, _STAT_COL_CANDIDATES)
    base_col = _pick_col(cols, _BASEMEAN_COL_CANDIDATES)

    ren: Dict[str, str] = {}
    if ens_col is not None:
        ren[ens_col] = "Ensembl ID"
    if sym_col is not None and sym_col != ens_col:
        ren[sym_col] = "Gene"
    if logfc_col is not None:
        ren[logfc_col] = "logFC"
    if pval_col is not None:
        ren[pval_col] = "pval"
    if padj_col is not None:
        ren[padj_col] = "padj"
    if stat_col is not None:
        ren[stat_col] = "stat"
    if base_col is not None:
        ren[base_col] = "baseMean"

    out = out.rename(columns=ren)

    # Clean Ensembl IDs
    if "Ensembl ID" in out.columns:
        out["Ensembl ID"] = (
            out["Ensembl ID"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )
        out.loc[out["Ensembl ID"].notna(), "Ensembl ID"] = out.loc[out["Ensembl ID"].notna(), "Ensembl ID"].astype(str).str.split(".").str[0]

    # Clean Gene symbols if present
    if "Gene" in out.columns:
        out["Gene"] = (
            out["Gene"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )

    # If "Gene" exists but is actually Ensembl IDs (common if column is called "Gene"), move it
    if "Gene" in out.columns and "Ensembl ID" not in out.columns:
        try:
            frac_ens = out["Gene"].dropna().astype(str).apply(_looks_ensembl).mean() if out["Gene"].notna().any() else 0.0
        except Exception:
            frac_ens = 0.0
        if frac_ens > 0.7:
            out["Ensembl ID"] = out["Gene"].astype(str).str.split(".").str[0]
            out["Gene"] = np.nan  # will fill from mapping below if possible

    # If no Gene symbol, but we have Ensembl and mapping, fill it
    if "Gene" not in out.columns or out["Gene"].isna().all():
        out["Gene"] = np.nan
        if ensg_to_symbol and "Ensembl ID" in out.columns:
            out["Gene"] = out["Ensembl ID"].map(lambda x: ensg_to_symbol.get(str(x).upper(), np.nan))

    # If we have Gene but no Ensembl, try to infer Ensembl from embedded mapping columns (if any)
    # (This is optional; we mainly need reliable lookup columns.)
    # Numeric coercions
    for c in ["logFC", "pval", "padj", "stat", "baseMean"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Prefer unique rows by Ensembl if present, else by Gene
    if "Ensembl ID" in out.columns and out["Ensembl ID"].notna().any():
        out = out.drop_duplicates(subset=["Ensembl ID"], keep="first")
    elif "Gene" in out.columns:
        out = out.drop_duplicates(subset=["Gene"], keep="first")

    return out


def _build_mapping_from_tables(tables: Dict[InVitroKey, pd.DataFrame]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build symbol<->ensg mapping from loaded tables if they contain both columns.
    """
    symbol_to_ensg: Dict[str, str] = {}
    ensg_to_symbol: Dict[str, str] = {}

    for df in tables.values():
        if df is None or df.empty:
            continue
        if "Ensembl ID" not in df.columns or "Gene" not in df.columns:
            continue
        tmp = df[["Ensembl ID", "Gene"]].dropna()
        if tmp.empty:
            continue
        for _, r in tmp.iterrows():
            ensg = str(r["Ensembl ID"]).strip().upper()
            sym = str(r["Gene"]).strip().upper()
            if not ensg or ensg == "NAN" or not sym or sym == "NAN":
                continue
            ensg = ensg.split(".")[0]
            symbol_to_ensg.setdefault(sym, ensg)
            ensg_to_symbol.setdefault(ensg, sym)

    return symbol_to_ensg, ensg_to_symbol


def load_all_invitro_deg_tables() -> Tuple[Dict[InVitroKey, pd.DataFrame], List[str], Dict[str, str], Dict[str, str]]:
    """
    Loads all discovered DEG tables.
    Returns (tables, errors, symbol_to_ensg, ensg_to_symbol).
    """
    files = discover_invitro_deg_files()
    tables: Dict[InVitroKey, pd.DataFrame] = {}
    errors: List[str] = []

    # Start with mapping from gene_mapping.csv (if exists)
    symbol_to_ensg, ensg_to_symbol = _load_gene_mapping_csv()

    # Load tables (normalise using whatever mapping we have so far)
    for key, fp in files.items():
        df, err = _read_deg_file_safe(fp)
        if err is not None:
            errors.append(err)
            continue
        tables[key] = normalise_deg_table(df, ensg_to_symbol=ensg_to_symbol)

    # If mapping was absent/weak, enrich mapping from embedded columns in the loaded tables
    built_sym2ensg, built_ensg2sym = _build_mapping_from_tables(tables)
    if built_sym2ensg:
        # merge (CSV mapping takes precedence)
        for sym, ensg in built_sym2ensg.items():
            symbol_to_ensg.setdefault(sym, ensg)
        for ensg, sym in built_ensg2sym.items():
            ensg_to_symbol.setdefault(ensg, sym)

    # Re-normalise with the enriched mapping so Gene symbols fill properly where possible
    for key, df in list(tables.items()):
        # We need the raw again to re-normalise; easiest is to leave as-is if Gene exists.
        # But if Gene is all-missing and Ensembl exists, fill Gene now.
        if df is None or df.empty:
            continue
        if "Gene" in df.columns and df["Gene"].notna().any():
            continue
        if "Ensembl ID" in df.columns and ensg_to_symbol:
            df = df.copy()
            df["Gene"] = df["Ensembl ID"].map(lambda x: ensg_to_symbol.get(str(x).upper(), np.nan))
            tables[key] = df

    return tables, errors, symbol_to_ensg, ensg_to_symbol


# -----------------------------------------------------------------------------
# Gene-centric summaries + direction consensus between lines
# -----------------------------------------------------------------------------

def _get_gene_row(df: pd.DataFrame, gene_id_or_symbol: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    q = str(gene_id_or_symbol).strip().upper()
    q = q.split(".")[0]

    # If query looks like Ensembl, match Ensembl ID when available
    if (q.startswith("ENSG") or q.startswith("ENSMUSG")) and "Ensembl ID" in df.columns:
        hit = df.loc[df["Ensembl ID"].astype(str).str.upper().str.split(".").str[0] == q]
        if not hit.empty:
            return hit.iloc[0]

    # Otherwise match Gene symbol
    if "Gene" in df.columns:
        hit = df.loc[df["Gene"].astype(str).str.upper() == q]
        if not hit.empty:
            return hit.iloc[0]

    # As last resort, if there's a column literally called "Gene" but it holds Ensembl IDs and wasn't normalised
    if "Gene" in df.columns and (q.startswith("ENSG") or q.startswith("ENSMUSG")):
        hit = df.loc[df["Gene"].astype(str).str.upper().str.split(".").str[0] == q]
        if not hit.empty:
            return hit.iloc[0]

    return None


def gene_summary_table(
    tables: Dict[InVitroKey, pd.DataFrame],
    query: str,
    symbol_to_ensg: Dict[str, str],
    ensg_to_symbol: Dict[str, str],
) -> pd.DataFrame:
    """
    Per-dataset summary for the queried gene.
    Query can be a symbol or Ensembl ID.
    """
    gene_id, resolved_symbol = _resolve_query_to_gene_id(query, symbol_to_ensg)
    if not gene_id:
        return pd.DataFrame()

    rows = []
    for k, df in tables.items():
        # Try Ensembl (if we resolved it)
        r = _get_gene_row(df, gene_id)

        # If missing, also try the symbol directly (covers tables storing symbols only)
        if r is None and resolved_symbol:
            r = _get_gene_row(df, resolved_symbol)

        # If still missing, try the raw query
        if r is None:
            r = _get_gene_row(df, str(query).strip().upper())

        label = CONTRAST_LABELS.get(k.contrast, k.contrast)

        if r is None:
            rows.append({
                "iHeps line": k.line,
                "Contrast": label,
                "Gene ID": gene_id,
                "Gene symbol": (resolved_symbol or ensg_to_symbol.get(gene_id)),
                "logFC": np.nan,
                "padj": np.nan,
                "pval": np.nan,
                "Direction": "missing",
            })
            continue

        logfc = float(r["logFC"]) if ("logFC" in r and pd.notna(r["logFC"])) else np.nan
        padj = float(r["padj"]) if ("padj" in r and pd.notna(r["padj"])) else np.nan
        pval = float(r["pval"]) if ("pval" in r and pd.notna(r["pval"])) else np.nan

        direction = (
            "Up in model" if pd.notna(logfc) and logfc > 0
            else "Down in model" if pd.notna(logfc) and logfc < 0
            else "missing"
        )

        # Determine display symbol
        display_sym = resolved_symbol or ensg_to_symbol.get(gene_id)
        if not display_sym and "Gene" in r and pd.notna(r["Gene"]):
            display_sym = str(r["Gene"]).strip().upper()

        rows.append({
            "iHeps line": k.line,
            "Contrast": label,
            "Gene ID": gene_id,
            "Gene symbol": display_sym,
            "logFC": logfc,
            "padj": padj,
            "pval": pval,
            "Direction": direction,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    contrast_order = [CONTRAST_LABELS[c] for c in CONTRAST_TOKENS]
    out["__crank__"] = out["Contrast"].map(lambda x: contrast_order.index(x) if x in contrast_order else 999)
    out = out.sort_values(["iHeps line", "__crank__"], ascending=[True, True]).drop(columns=["__crank__"])
    return out


def direction_consensus_by_contrast(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each contrast, compare direction between 1b and 5a (even if non-significant).
    Returns: Contrast | Direction 1b | Direction 5a | Consensus
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    need = {"iHeps line", "Contrast", "logFC"}
    if not need.issubset(set(summary_df.columns)):
        return pd.DataFrame()

    df = summary_df.copy()

    def _dir_from_logfc(x):
        if pd.isna(x):
            return "missing"
        if float(x) > 0:
            return "Up"
        if float(x) < 0:
            return "Down"
        return "0"

    df["__dir__"] = df["logFC"].apply(_dir_from_logfc)

    piv = df.pivot_table(index="Contrast", columns="iHeps line", values="__dir__", aggfunc="first").reset_index()

    if "1b" not in piv.columns:
        piv["1b"] = "missing"
    if "5a" not in piv.columns:
        piv["5a"] = "missing"

    def _cons(r):
        d1 = r.get("1b", "missing")
        d2 = r.get("5a", "missing")
        if d1 == "missing" or d2 == "missing":
            return "missing"
        if d1 == d2:
            return f"Agree ({d1})"
        return "Disagree"

    piv["Consensus"] = piv.apply(_cons, axis=1)
    piv = piv.rename(columns={"1b": "Direction 1b", "5a": "Direction 5a"})

    contrast_order = [CONTRAST_LABELS[c] for c in CONTRAST_TOKENS]
    piv["__crank__"] = piv["Contrast"].map(lambda x: contrast_order.index(x) if x in contrast_order else 999)
    piv = piv.sort_values("__crank__").drop(columns=["__crank__"])
    return piv


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_gene_logfc_heatmap(summary_df: pd.DataFrame, label: str) -> Optional[go.Figure]:
    if summary_df is None or summary_df.empty or "logFC" not in summary_df.columns:
        return None

    mat = summary_df.pivot(index="iHeps line", columns="Contrast", values="logFC")
    if mat.empty:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=mat.values.astype(float),
        x=list(mat.columns),
        y=list(mat.index),
        colorbar=dict(title="logFC"),
        hovertemplate="Line: %{y}<br>Contrast: %{x}<br>logFC: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{label} logFC across iHeps lines and contrasts", font=dict(size=14)),
        height=280,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_gene_dotplot(summary_df: pd.DataFrame, label: str) -> Optional[go.Figure]:
    if summary_df is None or summary_df.empty:
        return None
    if "padj" not in summary_df.columns or "logFC" not in summary_df.columns:
        return None

    df = summary_df.copy()

    def neglog10(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)) or float(x) <= 0:
                return np.nan
            return -math.log10(float(x))
        except Exception:
            return np.nan

    df["neglog10_padj"] = df["padj"].apply(neglog10)

    fig = go.Figure()
    for _, r in df.iterrows():
        sz = r["neglog10_padj"]
        size = 8 if pd.isna(sz) else float(min(20, 6 + 3.0 * sz))
        fig.add_trace(go.Scatter(
            x=[r["Contrast"]],
            y=[r["logFC"]],
            mode="markers",
            marker=dict(size=size, line=dict(width=1, color="white")),
            hovertemplate=(
                f"Line: {r['iHeps line']}<br>"
                f"Contrast: {r['Contrast']}<br>"
                f"logFC: {r['logFC'] if pd.notna(r['logFC']) else 'missing'}<br>"
                f"padj: {r['padj'] if pd.notna(r['padj']) else 'missing'}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.update_layout(
        title=dict(text=f"{label} effect size (logFC) with significance (dot size)", font=dict(size=14)),
        xaxis_title="Contrast",
        yaxis_title="logFC (model vs HCM)",
        height=360,
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


def make_volcano(
    df: pd.DataFrame,
    title: str,
    highlight_gene_id: Optional[str] = None,
    highlight_label: Optional[str] = None,
    ensg_to_symbol: Optional[Dict[str, str]] = None,
    fdr_thresh: float = 0.05,
    abs_logfc_thresh: float = 1.0,
) -> Optional[go.Figure]:
    if df is None or df.empty or "logFC" not in df.columns:
        return None

    tmp = df.copy()

    # Choose p column
    if "padj" in tmp.columns:
        p = tmp["padj"].astype(float)
        y_label = "-log10(FDR)"
    elif "pval" in tmp.columns:
        p = tmp["pval"].astype(float)
        y_label = "-log10(p)"
    else:
        p = pd.Series(np.nan, index=tmp.index)
        y_label = "-log10(p)"

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp["neglog10p"] = -np.log10(p)

    # Hover label text: prefer symbol if we can map; else show Gene/Ensembl
    if "Gene" in tmp.columns:
        base_text = tmp["Gene"].astype(str)
    elif "Ensembl ID" in tmp.columns:
        base_text = tmp["Ensembl ID"].astype(str)
    else:
        base_text = pd.Series([""] * len(tmp), index=tmp.index)

    if ensg_to_symbol and "Ensembl ID" in tmp.columns:
        sym = tmp["Ensembl ID"].map(lambda g: ensg_to_symbol.get(str(g).upper().split(".")[0], ""))
        text = sym.where(sym.astype(str) != "", base_text)
    else:
        text = base_text

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=tmp["logFC"],
        y=tmp["neglog10p"],
        mode="markers",
        marker=dict(size=4),
        text=text,
        hovertemplate=(
            "Gene: %{text}<br>"
            "logFC: %{x:.3f}<br>"
            f"{y_label}: %{{y:.3f}}<extra></extra>"   # <-- FIX: escape braces for Plotly
        ),
        showlegend=False,
    ))

    fig.add_vline(x=abs_logfc_thresh, line_dash="dash", line_width=1)
    fig.add_vline(x=-abs_logfc_thresh, line_dash="dash", line_width=1)
    if fdr_thresh and fdr_thresh > 0:
        fig.add_hline(y=-math.log10(fdr_thresh), line_dash="dash", line_width=1)

    # Highlight point
    if highlight_gene_id:
        g = str(highlight_gene_id).strip().upper().split(".")[0]

        hit = pd.DataFrame()
        if "Ensembl ID" in tmp.columns and (g.startswith("ENSG") or g.startswith("ENSMUSG")):
            hit = tmp.loc[tmp["Ensembl ID"].astype(str).str.upper().str.split(".").str[0] == g]
        if hit.empty and "Gene" in tmp.columns:
            hit = tmp.loc[tmp["Gene"].astype(str).str.upper() == g]
        if hit.empty and highlight_label and "Gene" in tmp.columns:
            hit = tmp.loc[tmp["Gene"].astype(str).str.upper() == str(highlight_label).strip().upper()]

        if not hit.empty:
            label = highlight_label
            if not label:
                if ensg_to_symbol and (g.startswith("ENSG") or g.startswith("ENSMUSG")):
                    label = ensg_to_symbol.get(g, g)
                else:
                    label = g

            fig.add_trace(go.Scatter(
                x=hit["logFC"],
                y=hit["neglog10p"],
                mode="markers+text",
                text=[label],
                textposition="top center",
                marker=dict(size=10),
                hovertemplate=(
                    "Gene: %{text}<br>"
                    "logFC: %{x:.3f}<br>"
                    f"{y_label}: %{{y:.3f}}<extra></extra>"  # <-- FIX here too
                ),
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="logFC (model vs HCM)",
        yaxis_title=y_label,
        height=420,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def top_deg_tables(
    df: pd.DataFrame,
    ensg_to_symbol: Optional[Dict[str, str]] = None,
    n: int = 25,
    padj_thresh: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or "logFC" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    tmp = df.copy()
    if "padj" in tmp.columns:
        tmp = tmp.loc[tmp["padj"].isna() | (tmp["padj"] <= float(padj_thresh))].copy()

    if ensg_to_symbol and "Ensembl ID" in tmp.columns:
        tmp = tmp.copy()
        tmp.insert(1, "Gene symbol", tmp["Ensembl ID"].map(lambda g: ensg_to_symbol.get(str(g).upper().split(".")[0], "")))
    elif "Gene" in tmp.columns:
        tmp = tmp.copy()
        tmp.insert(1, "Gene symbol", tmp["Gene"].astype(str))

    up = tmp.sort_values("logFC", ascending=False).head(int(n)).copy()
    down = tmp.sort_values("logFC", ascending=True).head(int(n)).copy()

    keep = [c for c in ["Ensembl ID", "Gene", "Gene symbol", "logFC", "padj", "pval", "stat", "baseMean"] if c in tmp.columns]
    return (up[keep] if keep else up), (down[keep] if keep else down)


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def _st_dataframe(st, df: pd.DataFrame, hide_index: bool = True) -> None:
    """
    Streamlit compatibility wrapper for the upcoming 'width' API replacing use_container_width.
    """
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def _st_plotly(st, fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def render_invitro_tab(query: str) -> None:
    """
    Streamlit rendering for the in vitro model tab.
    """
    import streamlit as st

    root = _find_stem_cell_model_dir()
    if root is None:
        st.warning("No in vitro data folder found. Expected: meta-liver-data/stem_cell_model/")
        return

    files = discover_invitro_deg_files()
    if not files:
        st.warning(
            "No in vitro DEG files were found in stem_cell_model/.\n\n"
            "Accepted names include:\n"
            "  processed_degs_1b_OAPAvsHCM.parquet\n"
            "  OAPAvsHCM_1b.parquet\n"
            "and the same with .csv."
        )
        st.caption(f"Looking in: {root}")
        try:
            present = sorted([p.name for p in root.rglob("*") if p.is_file()])
            if present:
                st.caption("Files detected under stem_cell_model/ (recursive):")
                st.code("\n".join(present[:400]))
                if len(present) > 400:
                    st.caption(f"... and {len(present) - 400} more")
        except Exception:
            pass
        return

    tables, errors, symbol_to_ensg, ensg_to_symbol = load_all_invitro_deg_tables()
    if errors:
        st.error(errors[0])
        if len(errors) > 1:
            st.info("More read errors were encountered for other files as well.")
        return

    if not tables:
        st.warning("In vitro DEG files were found, but none could be loaded.")
        return

    # Availability panel
    st.markdown("### Dataset availability")
    avail_rows = []
    for c in CONTRAST_TOKENS:
        for l in LINE_TOKENS:
            key = InVitroKey(line=l, contrast=c)
            avail_rows.append({
                "iHeps line": l,
                "Contrast": CONTRAST_LABELS[c],
                "File": files[key].name if key in files else "missing",
            })
    _st_dataframe(st, pd.DataFrame(avail_rows), hide_index=True)

    st.markdown("---")

    view = st.radio("Choose view", ["Gene summary", "Volcano explorer"], index=0, horizontal=True)

    if view == "Gene summary":
        st.markdown("### Gene summary")
        st.caption("Direction consensus is computed from logFC sign between 1b and 5a for the same contrast (significance not required).")

        summ = gene_summary_table(tables, query, symbol_to_ensg, ensg_to_symbol)
        if summ is None or summ.empty:
            st.warning("No rows could be generated for this query. Tip: try an ENSG ID directly to validate matching.")
            return

        gene_id, resolved_symbol = _resolve_query_to_gene_id(query, symbol_to_ensg)
        label = resolved_symbol or ensg_to_symbol.get(gene_id) or str(query).strip().upper()

        _st_dataframe(st, summ, hide_index=True)

        cons = direction_consensus_by_contrast(summ)
        if cons is not None and not cons.empty:
            st.markdown("### Direction consensus between lines (1b vs 5a)")
            _st_dataframe(st, cons, hide_index=True)

        fig_hm = make_gene_logfc_heatmap(summ, label)
        if fig_hm is not None:
            _st_plotly(st, fig_hm)

        fig_dot = make_gene_dotplot(summ, label)
        if fig_dot is not None:
            _st_plotly(st, fig_dot)

        st.markdown("### Contrast notes")
        for lbl, expl in CONTRAST_HELP.items():
            st.caption(f"{lbl}: {expl}")

        return

    # Volcano explorer
    st.markdown("### Volcano explorer")

    lines = sorted({k.line for k in tables.keys()})
    contrasts = [c for c in CONTRAST_TOKENS if any(k.contrast == c for k in tables.keys())]

    c1, c2 = st.columns(2)
    with c1:
        line_sel = st.selectbox("iHeps line", options=lines, index=0)
    with c2:
        contrast_sel = st.selectbox("Contrast", options=[CONTRAST_LABELS[c] for c in contrasts], index=0)

    contrast_token = next((c for c in contrasts if CONTRAST_LABELS[c] == contrast_sel), contrasts[0])

    key = InVitroKey(line=line_sel, contrast=contrast_token)
    if key not in tables:
        st.warning("Selected dataset is missing.")
        return

    df = tables.get(key)
    if df is None or df.empty:
        st.warning("Selected dataset is empty.")
        return

    gene_id, resolved_symbol = _resolve_query_to_gene_id(query, symbol_to_ensg)
    label = resolved_symbol or ensg_to_symbol.get(gene_id) or str(query).strip().upper()

    t1, t2, t3 = st.columns(3)
    with t1:
        fdr = st.number_input("FDR threshold", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")
    with t2:
        lfc_thr = st.number_input("|logFC| threshold", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    with t3:
        topn = st.number_input("Top N genes (tables)", min_value=5, max_value=200, value=25, step=5)

    title = f"{line_sel} — {CONTRAST_LABELS.get(contrast_token, contrast_token)}"
    fig = make_volcano(
        df,
        title=title,
        highlight_gene_id=gene_id,
        highlight_label=label,
        ensg_to_symbol=ensg_to_symbol,
        fdr_thresh=float(fdr),
        abs_logfc_thresh=float(lfc_thr),
    )
    if fig is not None:
        _st_plotly(st, fig)

    up, down = top_deg_tables(df, ensg_to_symbol=ensg_to_symbol, n=int(topn), padj_thresh=float(fdr))
    st.markdown("### Top genes (FDR-filtered where available)")
    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("Upregulated")
        _st_dataframe(st, up, hide_index=True)
    with c_down:
        st.markdown("Downregulated")
        _st_dataframe(st, down, hide_index=True)
