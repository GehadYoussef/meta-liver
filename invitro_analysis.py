"""
In vitro (human stem cell-derived) MASLD model analysis for Meta Liver.

This module backs a dedicated Streamlit tab that summarises DEGs from two iHeps lines
(1b, 5a) under three conditions compared with healthy controls (HCM):
- OA+PA vs HCM
- OA+PA + Resistin/Myostatin vs HCM
- OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM
(PBMCs were not included in RNA-seq; the DEGs reflect iHeps only.)

Expected data location (inside the app data directory):
  stem_cell_model/

Expected DEG filenames (new scheme):
  OAPAvsHCM_1b.(parquet|csv)
  OAPAvsHCM_5a.(parquet|csv)
  OAPAResMyovsHCM_1b.(parquet|csv)
  OAPAResMyovsHCM_5a.(parquet|csv)
  OAPAResMyoPBMCsvsHCM_1b.(parquet|csv)
  OAPAResMyoPBMCsvsHCM_5a.(parquet|csv)

Also supported (legacy scheme):
  processed_degs_<LINE>_<CONTRAST>.parquet
  processed_degs_<LINE>_<CONTRAST>.csv

Gene IDs:
- DEG tables typically use Ensembl stable IDs in the 'Gene' column.
- If a mapping file is available (e.g., stem_cell_model/gene_mapping.csv with
  columns like "Gene stable ID" and "Gene name"), the tab will display gene symbols
  but will always keep Ensembl IDs for matching.
- If a gene symbol is not found in the mapping, the Ensembl ID is displayed/used.

Parquet reading:
- Requires an optional engine (pyarrow or fastparquet). If missing, the module
  will fall back to CSV (if present) or show a clear install hint.
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
# Labels / ordering
# -----------------------------------------------------------------------------

_CONTRAST_LABELS = {
    "OAPAvsHCM": "OA+PA vs HCM",
    "OAPAResMyovsHCM": "OA+PA + Resistin/Myostatin vs HCM",
    "OAPAResMyoPBMCsvsHCM": "OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM",
}

_CONTRAST_HELP = {
    "OA+PA vs HCM": "Fatty-acid overload model (oleic + palmitic acid) compared with healthy controls.",
    "OA+PA + Resistin/Myostatin vs HCM": "Fatty acids plus adipose/muscle-derived signalling molecules.",
    "OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM": "As above plus PBMC co-culture; PBMCs not sequenced.",
}

_CONTRAST_ORDER = ["OAPAvsHCM", "OAPAResMyovsHCM", "OAPAResMyoPBMCsvsHCM"]
_LINE_ORDER = ["1b", "5a"]


# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------

# New naming: <contrast>_<line>.(parquet|csv)
_FILE_RE_NEW = re.compile(
    r"^(?P<contrast>OAPAvsHCM|OAPAResMyovsHCM|OAPAResMyoPBMCsvsHCM)_(?P<line>1b|5a)\.(?P<ext>parquet|csv)$",
    flags=re.IGNORECASE,
)

# Legacy naming: processed_degs_<line>_<contrast>.(parquet|csv)
_FILE_RE_LEGACY = re.compile(
    r"^processed_degs_(?P<line>[^_]+)_(?P<contrast>.+)\.(?P<ext>parquet|csv)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class InVitroKey:
    line: str        # "1b" or "5a"
    contrast: str    # canonical contrast token e.g. "OAPAvsHCM"


def _find_stem_cell_model_dir() -> Optional[Path]:
    data_dir = find_data_dir()
    if data_dir is None:
        return None

    # direct candidates
    candidates = [
        data_dir / "stem_cell_model",
        data_dir / "stem-cell-model",
        data_dir / "stemcell_model",
        data_dir / "stemcell",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    # fallback: scan 1 level
    for p in data_dir.iterdir():
        if not p.is_dir():
            continue
        nm = p.name.lower().replace("-", "_")
        if nm == "stem_cell_model" or ("stem" in nm and "cell" in nm):
            return p

    return None


def _canonical_contrast(token: str) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    # match against known tokens case-insensitively
    for k in _CONTRAST_LABELS.keys():
        if k.lower() == t.lower():
            return k
    return None


def _canonical_line(token: str) -> str:
    if token is None:
        return ""
    return str(token).strip()


def discover_invitro_deg_files() -> Dict[InVitroKey, Dict[str, Path]]:
    """
    Discover DEG files under stem_cell_model/.

    Returns a dict:
      { InVitroKey(line, contrast) : { "parquet": path?, "csv": path? } }

    This does not attempt to read parquet/csv.
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return {}

    out: Dict[InVitroKey, Dict[str, Path]] = {}

    for p in root.iterdir():
        if not p.is_file():
            continue

        name = p.name

        # ignore mapping files
        if name.lower().startswith("gene_mapping"):
            continue

        m = _FILE_RE_NEW.match(name)
        if m:
            contrast = _canonical_contrast(m.group("contrast"))
            line = _canonical_line(m.group("line"))
            ext = m.group("ext").lower()
            if contrast and line:
                key = InVitroKey(line=line, contrast=contrast)
                out.setdefault(key, {})[ext] = p
            continue

        m2 = _FILE_RE_LEGACY.match(name)
        if m2:
            contrast_raw = m2.group("contrast")
            line = _canonical_line(m2.group("line"))
            ext = m2.group("ext").lower()

            contrast = _canonical_contrast(contrast_raw)
            if contrast is None:
                # allow legacy contrast strings that already match our tokens with minor differences
                contrast = _canonical_contrast(contrast_raw.replace("-", "").replace("_", ""))
            if contrast and line:
                key = InVitroKey(line=line, contrast=contrast)
                out.setdefault(key, {})[ext] = p
            continue

    return out


# -----------------------------------------------------------------------------
# Gene mapping
# -----------------------------------------------------------------------------

def _normalise_ensembl_id(x: object) -> str:
    """
    Normalise Ensembl IDs:
    - uppercase
    - strip version suffix e.g. ENSG... .12 -> ENSG...
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = s.split(".")[0]
    return s.upper()


def load_gene_mapping() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Load gene mapping from stem_cell_model/gene_mapping.(csv|parquet) if present.

    Returns:
      ens_to_symbol: {ENSG... : SYMBOL}
      symbol_to_ens: {SYMBOL : [ENSG..., ...]}  (list to handle 1-to-many safely)
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return {}, {}

    # locate mapping file
    mapping_path = None
    for cand in ["gene_mapping.csv", "gene_mapping.parquet", "Gene_mapping.csv", "GENE_MAPPING.csv"]:
        fp = root / cand
        if fp.exists() and fp.is_file():
            mapping_path = fp
            break

    if mapping_path is None:
        # also try any file that contains "gene_mapping"
        for fp in root.iterdir():
            if fp.is_file() and "gene_mapping" in fp.name.lower():
                mapping_path = fp
                break

    if mapping_path is None:
        return {}, {}

    try:
        if mapping_path.suffix.lower() == ".parquet":
            mdf = pd.read_parquet(mapping_path)
        else:
            mdf = pd.read_csv(mapping_path)
    except Exception:
        return {}, {}

    if mdf is None or mdf.empty:
        return {}, {}

    cols = {c.lower(): c for c in mdf.columns}
    ens_col = None
    sym_col = None

    # common patterns from Ensembl export
    for c in ["gene stable id", "gene_stable_id", "ensembl", "ensembl_id", "gene", "gene_id"]:
        if c in cols:
            ens_col = cols[c]
            break

    for c in ["gene name", "gene_symbol", "symbol", "hgnc symbol", "hgnc_symbol", "name"]:
        if c in cols:
            sym_col = cols[c]
            break

    if ens_col is None or sym_col is None:
        return {}, {}

    ens = mdf[ens_col].map(_normalise_ensembl_id)
    sym = mdf[sym_col].astype(str).str.strip().str.upper()

    ens_to_symbol: Dict[str, str] = {}
    symbol_to_ens: Dict[str, List[str]] = {}

    for e, s in zip(ens, sym):
        if not e:
            continue
        if not s or s.lower() == "nan":
            continue
        if e not in ens_to_symbol:
            ens_to_symbol[e] = s
        symbol_to_ens.setdefault(s, []).append(e)

    return ens_to_symbol, symbol_to_ens


# -----------------------------------------------------------------------------
# Robust DEG table loading + normalisation
# -----------------------------------------------------------------------------

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


def _read_parquet_or_csv(preferred: Dict[str, Path]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Try parquet first (if present), otherwise csv.
    If parquet read fails, fall back to csv (if present) before emitting an error.
    """
    # try parquet
    if "parquet" in preferred:
        try:
            return pd.read_parquet(preferred["parquet"]), None
        except Exception as e:
            # fall back to csv if available
            if "csv" in preferred:
                try:
                    return pd.read_csv(preferred["csv"]), None
                except Exception as e2:
                    return None, f"Could not read parquet or csv for {preferred['parquet'].name}: {type(e).__name__}: {e}; csv error: {type(e2).__name__}: {e2}"
            msg = (
                f"Could not read parquet file: {preferred['parquet'].name}\n\n"
                f"Underlying error: {type(e).__name__}: {e}\n\n"
                "Fix: add a parquet engine to your environment (requirements.txt), e.g.:\n"
                "  pyarrow\n"
                "Then redeploy/restart the app."
            )
            return None, msg

    # csv only
    if "csv" in preferred:
        try:
            return pd.read_csv(preferred["csv"]), None
        except Exception as e:
            return None, f"Could not read csv file: {preferred['csv'].name}: {type(e).__name__}: {e}"

    return None, "No readable file found (expected parquet or csv)."


def normalise_deg_table(df: pd.DataFrame, ens_to_symbol: Dict[str, str]) -> pd.DataFrame:
    """
    Standardises key columns to:
      Ensembl, Gene (display), logFC, pval, padj, stat, baseMean
    Leaves other columns intact.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # If "Gene" is the Ensembl ID column (most common), capture it.
    gene_col = None
    cols = list(out.columns)
    lower_map = {c.lower(): c for c in cols}
    if "gene" in lower_map:
        gene_col = lower_map["gene"]

    # if Ensembl IDs are in the index
    if gene_col is None and out.index.name and out.index.name.lower() == "gene":
        out = out.reset_index()
        cols = list(out.columns)
        lower_map = {c.lower(): c for c in cols}
        gene_col = lower_map.get("gene")

    if gene_col is not None:
        out["Ensembl"] = out[gene_col].map(_normalise_ensembl_id)
    else:
        # if we can't find a gene column, bail early
        return pd.DataFrame()

    # map to symbol for display
    def _disp(e: str) -> str:
        sym = ens_to_symbol.get(e, "")
        return sym if sym else e

    out["Gene"] = out["Ensembl"].map(_disp).astype(str).str.strip().str.upper()

    # rename metrics
    cols = list(out.columns)
    logfc_col = _pick_col(cols, _LOGFC_COL_CANDIDATES)
    pval_col = _pick_col(cols, _PVAL_COL_CANDIDATES)
    padj_col = _pick_col(cols, _PADJ_COL_CANDIDATES)
    stat_col = _pick_col(cols, _STAT_COL_CANDIDATES)
    base_col = _pick_col(cols, _BASEMEAN_COL_CANDIDATES)

    ren = {}
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

    for c in ["logFC", "pval", "padj", "stat", "baseMean"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def load_all_invitro_deg_tables() -> Tuple[Dict[InVitroKey, pd.DataFrame], List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Load all discovered DEG tables.

    Returns:
      tables: {InVitroKey: normalised_df}
      errors: list[str]
      ens_to_symbol, symbol_to_ens: mapping dicts
    """
    ens_to_symbol, symbol_to_ens = load_gene_mapping()

    files = discover_invitro_deg_files()
    tables: Dict[InVitroKey, pd.DataFrame] = {}
    errors: List[str] = []

    for key, paths in files.items():
        df, err = _read_parquet_or_csv(paths)
        if err is not None:
            errors.append(err)
            continue
        norm = normalise_deg_table(df, ens_to_symbol)
        if norm is None or norm.empty:
            errors.append(f"Loaded {list(paths.values())[0].name} but normalisation produced an empty table (missing 'Gene' column?).")
            continue
        tables[key] = norm

    return tables, errors, ens_to_symbol, symbol_to_ens


# -----------------------------------------------------------------------------
# Gene lookup + summaries
# -----------------------------------------------------------------------------

def _resolve_query_to_ensembl(query: str, symbol_to_ens: Dict[str, List[str]]) -> List[str]:
    q = str(query).strip().upper()
    if not q:
        return []
    if q.startswith("ENSG"):
        return [_normalise_ensembl_id(q)]
    if symbol_to_ens and q in symbol_to_ens:
        return [_normalise_ensembl_id(x) for x in symbol_to_ens[q]]
    return []


def _get_gene_row(df: pd.DataFrame, query: str, symbol_to_ens: Dict[str, List[str]]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    ens_hits = _resolve_query_to_ensembl(query, symbol_to_ens)

    if ens_hits and "Ensembl" in df.columns:
        hit = df.loc[df["Ensembl"].isin(ens_hits)]
        if not hit.empty:
            return hit.iloc[0]

    # fall back to display name matching (could be Ensembl if unmapped)
    if "Gene" in df.columns:
        q = str(query).strip().upper()
        hit2 = df.loc[df["Gene"] == q]
        if not hit2.empty:
            return hit2.iloc[0]

    return None


def gene_summary_table(tables: Dict[InVitroKey, pd.DataFrame], query: str,
                       symbol_to_ens: Dict[str, List[str]], ens_to_symbol: Dict[str, str]) -> pd.DataFrame:
    """
    6-row summary for the selected query (symbol or Ensembl):
      line x contrast, with logFC/padj/pval + direction
    """
    rows = []

    # decide "display gene" for header
    q = str(query).strip().upper()
    ens_resolved = _resolve_query_to_ensembl(q, symbol_to_ens)
    display_gene = q
    if ens_resolved:
        # if user typed ENSG, show symbol if available
        sym = ens_to_symbol.get(ens_resolved[0], "")
        display_gene = sym if sym else ens_resolved[0]

    for contrast in _CONTRAST_ORDER:
        for line in _LINE_ORDER:
            k = InVitroKey(line=line, contrast=contrast)
            df = tables.get(k)
            r = _get_gene_row(df, q, symbol_to_ens) if df is not None else None

            if r is None:
                rows.append({
                    "Contrast": _CONTRAST_LABELS.get(contrast, contrast),
                    "iHeps line": line,
                    "Gene": display_gene,
                    "Ensembl": (ens_resolved[0] if ens_resolved else np.nan),
                    "logFC": np.nan,
                    "padj": np.nan,
                    "pval": np.nan,
                    "Direction": "missing",
                })
                continue

            logfc = float(r["logFC"]) if "logFC" in r and pd.notna(r["logFC"]) else np.nan
            padj = float(r["padj"]) if "padj" in r and pd.notna(r["padj"]) else np.nan
            pval = float(r["pval"]) if "pval" in r and pd.notna(r["pval"]) else np.nan
            ensg = str(r["Ensembl"]) if "Ensembl" in r and pd.notna(r["Ensembl"]) else (ens_resolved[0] if ens_resolved else "")

            direction = "Up in model" if pd.notna(logfc) and logfc > 0 else "Down in model" if pd.notna(logfc) and logfc < 0 else "missing"

            rows.append({
                "Contrast": _CONTRAST_LABELS.get(contrast, contrast),
                "iHeps line": line,
                "Gene": str(r.get("Gene", display_gene)),
                "Ensembl": ensg,
                "logFC": logfc,
                "padj": padj,
                "pval": pval,
                "Direction": direction,
            })

    out = pd.DataFrame(rows)

    # order nicely
    out["__c_rank__"] = out["Contrast"].map({v: i for i, v in enumerate([_CONTRAST_LABELS[c] for c in _CONTRAST_ORDER])})
    out["__l_rank__"] = out["iHeps line"].map({v: i for i, v in enumerate(_LINE_ORDER)})
    out = out.sort_values(["__c_rank__", "__l_rank__"]).drop(columns=["__c_rank__", "__l_rank__"])

    return out


def gene_direction_consensus(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For the gene: per contrast, compare direction (sign of logFC) between 1b and 5a.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    tmp = summary_df.copy()
    # pivot by line
    piv = tmp.pivot(index="Contrast", columns="iHeps line", values="logFC").reset_index()
    if piv.empty:
        return pd.DataFrame()

    def _dir(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "missing"
        if float(x) > 0:
            return "Up"
        if float(x) < 0:
            return "Down"
        return "0"

    out = pd.DataFrame({
        "Contrast": piv["Contrast"],
        "logFC_1b": piv.get("1b"),
        "logFC_5a": piv.get("5a"),
    })
    out["Direction_1b"] = out["logFC_1b"].apply(_dir)
    out["Direction_5a"] = out["logFC_5a"].apply(_dir)
    out["Agreement"] = np.where(
        (out["Direction_1b"].isin(["Up", "Down"])) & (out["Direction_1b"] == out["Direction_5a"]),
        "Yes",
        np.where(out["Direction_1b"].eq("missing") | out["Direction_5a"].eq("missing"), "missing", "No")
    )

    return out[["Contrast", "Direction_1b", "Direction_5a", "Agreement", "logFC_1b", "logFC_5a"]]


def contrast_global_direction_concordance(tables: Dict[InVitroKey, pd.DataFrame]) -> pd.DataFrame:
    """
    Genome-wide direction concordance between 1b and 5a for each contrast.
    Uses all genes with non-missing logFC in both lines; significance is ignored.
    """
    rows = []
    for contrast in _CONTRAST_ORDER:
        k1 = InVitroKey(line="1b", contrast=contrast)
        k2 = InVitroKey(line="5a", contrast=contrast)
        d1 = tables.get(k1)
        d2 = tables.get(k2)
        if d1 is None or d2 is None or d1.empty or d2.empty:
            rows.append({
                "Contrast": _CONTRAST_LABELS.get(contrast, contrast),
                "n_overlap": 0,
                "n_compared": 0,
                "agreement_pct": np.nan,
                "note": "missing dataset(s)",
            })
            continue

        m = d1[["Ensembl", "logFC"]].merge(d2[["Ensembl", "logFC"]], on="Ensembl", suffixes=("_1b", "_5a"))
        m = m.dropna(subset=["logFC_1b", "logFC_5a"])
        if m.empty:
            rows.append({
                "Contrast": _CONTRAST_LABELS.get(contrast, contrast),
                "n_overlap": int(d1["Ensembl"].nunique()),
                "n_compared": 0,
                "agreement_pct": np.nan,
                "note": "no comparable genes (missing logFC?)",
            })
            continue

        # compare sign only, treat 0 as neither
        s1 = np.sign(m["logFC_1b"].astype(float))
        s2 = np.sign(m["logFC_5a"].astype(float))
        comp = (s1 != 0) & (s2 != 0)
        m2 = m.loc[comp].copy()

        if m2.empty:
            rows.append({
                "Contrast": _CONTRAST_LABELS.get(contrast, contrast),
                "n_overlap": int(m.shape[0]),
                "n_compared": 0,
                "agreement_pct": np.nan,
                "note": "all zeros or missing",
            })
            continue

        agree = (np.sign(m2["logFC_1b"]) == np.sign(m2["logFC_5a"])).mean() * 100.0
        rows.append({
            "Contrast": _CONTRAST_LABELS.get(contrast, contrast),
            "n_overlap": int(m.shape[0]),
            "n_compared": int(m2.shape[0]),
            "agreement_pct": float(agree),
            "note": "",
        })

    out = pd.DataFrame(rows)
    return out


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_gene_logfc_heatmap(summary_df: pd.DataFrame, gene_label: str) -> Optional[go.Figure]:
    if summary_df is None or summary_df.empty:
        return None
    if "logFC" not in summary_df.columns:
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
        title=dict(text=f"{gene_label} logFC across iHeps lines and contrasts", font=dict(size=14)),
        height=280,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_gene_dotplot(summary_df: pd.DataFrame, gene_label: str) -> Optional[go.Figure]:
    if summary_df is None or summary_df.empty:
        return None
    if "logFC" not in summary_df.columns:
        return None

    df = summary_df.copy()

    def neglog10(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)) or x <= 0:
                return np.nan
            return -math.log10(float(x))
        except Exception:
            return np.nan

    if "padj" in df.columns:
        df["neglog10_padj"] = df["padj"].apply(neglog10)
    else:
        df["neglog10_padj"] = np.nan

    fig = go.Figure()
    for _, r in df.iterrows():
        sz = r.get("neglog10_padj", np.nan)
        size = 8 if pd.isna(sz) else float(min(20, 6 + 3.0 * sz))
        fig.add_trace(go.Scatter(
            x=[r["Contrast"]],
            y=[r["logFC"]],
            mode="markers",
            marker=dict(size=size, line=dict(width=1, color="white")),
            hovertemplate=(
                "Line: " + str(r["iHeps line"]) + "<br>"
                "Contrast: " + str(r["Contrast"]) + "<br>"
                "logFC: " + (f"{r['logFC']:.3f}" if pd.notna(r["logFC"]) else "missing") + "<br>"
                "padj: " + (f"{r['padj']:.3g}" if ("padj" in r and pd.notna(r["padj"])) else "missing") +
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.update_layout(
        title=dict(text=f"{gene_label} effect size (logFC) with significance (dot size)", font=dict(size=14)),
        xaxis_title="Contrast",
        yaxis_title="logFC (model vs HCM)",
        height=360,
        margin=dict(l=40, r=20, t=60, b=90),
    )
    return fig


def make_volcano(df: pd.DataFrame, title: str, highlight_gene: Optional[str] = None,
                fdr_thresh: float = 0.05, abs_logfc_thresh: float = 1.0) -> Optional[go.Figure]:
    if df is None or df.empty or "logFC" not in df.columns:
        return None

    tmp = df.copy()

    # Compute -log10(padj) if available, else pval
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

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=tmp["logFC"],
        y=tmp["neglog10p"],
        mode="markers",
        marker=dict(size=4),
        # IMPORTANT: do NOT use f-strings here; Plotly placeholders look like %{y:.3f}
        hovertemplate=(
            "Gene: %{text}<br>"
            "logFC: %{x:.3f}<br>"
            + y_label + ": %{y:.3f}<extra></extra>"
        ),
        text=tmp["Gene"] if "Gene" in tmp.columns else (tmp["Ensembl"] if "Ensembl" in tmp.columns else None),
        showlegend=False
    ))

    # thresholds (visual)
    fig.add_vline(x=abs_logfc_thresh, line_dash="dash", line_width=1)
    fig.add_vline(x=-abs_logfc_thresh, line_dash="dash", line_width=1)
    if fdr_thresh and fdr_thresh > 0:
        fig.add_hline(y=-math.log10(float(fdr_thresh)), line_dash="dash", line_width=1)

    if highlight_gene and "Gene" in tmp.columns:
        g = str(highlight_gene).strip().upper()
        # highlight by symbol OR Ensembl
        hit = tmp.loc[(tmp["Gene"] == g) | (tmp.get("Ensembl", pd.Series("", index=tmp.index)) == _normalise_ensembl_id(g))]
        if not hit.empty:
            fig.add_trace(go.Scatter(
                x=hit["logFC"],
                y=hit["neglog10p"],
                mode="markers+text",
                text=[hit.iloc[0]["Gene"]],
                textposition="top center",
                marker=dict(size=10),
                hovertemplate=(
                    "Gene: %{text}<br>"
                    "logFC: %{x:.3f}<br>"
                    + y_label + ": %{y:.3f}<extra></extra>"
                ),
                showlegend=False
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="logFC (model vs HCM)",
        yaxis_title=y_label,
        height=420,
        margin=dict(l=50, r=20, t=60, b=50)
    )
    return fig


def top_deg_tables(df: pd.DataFrame, n: int = 25, padj_thresh: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or "logFC" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    tmp = df.copy()
    if "padj" in tmp.columns:
        tmp = tmp.loc[tmp["padj"].isna() | (tmp["padj"] <= float(padj_thresh))].copy()

    up = tmp.sort_values("logFC", ascending=False).head(int(n)).copy()
    down = tmp.sort_values("logFC", ascending=True).head(int(n)).copy()

    keep = [c for c in ["Gene", "Ensembl", "logFC", "padj", "pval", "stat", "baseMean"] if c in tmp.columns]
    return up[keep] if keep else up, down[keep] if keep else down


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_invitro_tab(query: str) -> None:
    """
    Streamlit rendering for the in vitro model tab.
    """
    import streamlit as st  # local import keeps module usable outside Streamlit

    files = discover_invitro_deg_files()
    if not files:
        st.warning("No in vitro DEG files were found. Expected: data_dir/stem_cell_model/ with OAPAvsHCM_1b.(parquet|csv) etc.")
        return

    tables, errors, ens_to_symbol, symbol_to_ens = load_all_invitro_deg_tables()
    if errors:
        st.error(errors[0])
        if len(errors) > 1:
            st.caption(f"(+{len(errors)-1} more load/parse errors)")
        return

    if not tables:
        st.warning("In vitro DEG files were found, but none could be loaded.")
        return

    st.markdown("### In vitro (stem-cell-derived) MASLD model")
    st.caption("Two iHeps lines (1b, 5a) under three perturbations vs HCM. PBMCs were not sequenced.")

    view = st.radio("Choose view", ["Gene summary", "Volcano explorer"], index=0, horizontal=True)

    q = str(query).strip().upper()
    if view == "Gene summary":
        # Summary table
        summary = gene_summary_table(tables, q, symbol_to_ens, ens_to_symbol)

        # pick label for titles
        gene_label = summary["Gene"].dropna().iloc[0] if not summary.empty else q

        st.markdown(f"#### {gene_label}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("#### Direction agreement between 1b and 5a (per contrast)")
        cons = gene_direction_consensus(summary)
        st.dataframe(cons, use_container_width=True, hide_index=True)

        st.markdown("#### Genome-wide direction concordance between 1b and 5a (per contrast)")
        global_cons = contrast_global_direction_concordance(tables)
        st.dataframe(global_cons, use_container_width=True, hide_index=True)

        fig_hm = make_gene_logfc_heatmap(summary, gene_label)
        if fig_hm is not None:
            st.plotly_chart(fig_hm, use_container_width=True)

        fig_dot = make_gene_dotplot(summary, gene_label)
        if fig_dot is not None:
            st.plotly_chart(fig_dot, use_container_width=True)

        st.markdown("#### Contrast notes")
        for lbl, expl in _CONTRAST_HELP.items():
            st.caption(f"{lbl}: {expl}")

        return

    # Volcano explorer
    st.markdown("#### Volcano explorer")

    # build options in the exact desired order
    options = []
    for contrast in _CONTRAST_ORDER:
        for line in _LINE_ORDER:
            options.append((contrast, line))

    labels = [f"{c}_{l}" for c, l in options]
    sel = st.selectbox("Dataset", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)
    contrast_token, line_sel = options[int(sel)]

    key = InVitroKey(line=line_sel, contrast=contrast_token)
    df = tables.get(key)
    if df is None or df.empty:
        st.warning("Selected dataset is empty or missing.")
        return

    t1, t2, t3 = st.columns(3)
    with t1:
        fdr = st.number_input("FDR threshold", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")
    with t2:
        lfc_thr = st.number_input("|logFC| threshold", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    with t3:
        topn = st.number_input("Top N genes (tables)", min_value=5, max_value=200, value=25, step=5)

    title = f"{line_sel} — {_CONTRAST_LABELS.get(contrast_token, contrast_token)}"
    fig = make_volcano(df, title=title, highlight_gene=q, fdr_thresh=float(fdr), abs_logfc_thresh=float(lfc_thr))
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    up, down = top_deg_tables(df, n=int(topn), padj_thresh=float(fdr))
    st.markdown("#### Top genes (FDR-filtered where available)")
    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("Upregulated")
        st.dataframe(up, use_container_width=True, hide_index=True)
    with c_down:
        st.markdown("Downregulated")
        st.dataframe(down, use_container_width=True, hide_index=True)
