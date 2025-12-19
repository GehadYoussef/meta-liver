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
DEG tables may use Ensembl IDs or symbols. If the user searches by symbol and the table
stores Ensembl IDs, we map symbol -> Ensembl using gene_mapping.csv (found recursively).
If mapping fails, we fall back to using the entered identifier as-is.

Parquet note:
Reading Parquet requires pyarrow or fastparquet. If missing, the UI will show a clear hint.
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

# Optional mapping file name
GENE_MAPPING_FILENAME = "gene_mapping.csv"

# Regexes for DEG files
_RE_PROCESSED = re.compile(r"^processed_degs_(?P<line>[^_]+)_(?P<contrast>.+)$", flags=re.IGNORECASE)
_RE_CONTRAST_LINE = re.compile(r"^(?P<contrast>.+)_(?P<line>[^_]+)$", flags=re.IGNORECASE)

# Ensembl-like patterns (human + mouse, just in case)
_RE_ENSEMBL = re.compile(r"^(ENSG|ENSMUSG)\d+", flags=re.IGNORECASE)


@dataclass(frozen=True)
class InVitroKey:
    line: str        # "1b" or "5a"
    contrast: str    # canonical token e.g. "OAPAvsHCM"
    ext: str         # "csv" or "parquet" (for debugging)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _clean_gene_id(x: str) -> str:
    """
    Normalise gene identifiers for matching:
    - upper-case
    - strip whitespace
    - drop Ensembl version suffix "ENSG... .12" -> "ENSG..."
    """
    s = str(x).strip().upper()
    if not s:
        return s
    if s.startswith("ENSG") or s.startswith("ENSMUSG"):
        s = s.split(".", 1)[0]
    return s


def _looks_like_ensembl(x: str) -> bool:
    return bool(_RE_ENSEMBL.match(_clean_gene_id(x)))


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

def _parse_deg_stem(stem: str, ext: str) -> Optional[InVitroKey]:
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
            return InVitroKey(line=line, contrast=contrast, ext=ext)
        return None

    m = _RE_CONTRAST_LINE.match(s)
    if m:
        contrast = _canon_contrast(m.group("contrast"))
        line = _canon_line(m.group("line"))
        if line and contrast:
            return InVitroKey(line=line, contrast=contrast, ext=ext)
        return None

    return None


def discover_invitro_deg_files() -> Dict[InVitroKey, Path]:
    """
    Discovers DEG files in stem_cell_model/. Returns {key -> path}.
    Searches recursively so subfolders like stem_cell_model/parquet/ work.
    Ignores gene_mapping.csv.
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

        ext = p.suffix.lower().lstrip(".")
        if ext not in ("csv", "parquet"):
            continue

        key = _parse_deg_stem(p.stem, ext)
        if key is None:
            continue

        # If both CSV and Parquet exist, prefer Parquet
        if key in out:
            if out[key].suffix.lower() == ".csv" and p.suffix.lower() == ".parquet":
                out[key] = p
        else:
            out[key] = p

    return out


# -----------------------------------------------------------------------------
# Gene mapping
# -----------------------------------------------------------------------------

def _find_gene_mapping_file(root: Path) -> Optional[Path]:
    """
    Find gene_mapping.csv anywhere under stem_cell_model/ (recursive),
    because users often keep it inside a parquet/ or misc/ subfolder.
    """
    direct = root / GENE_MAPPING_FILENAME
    if direct.exists() and direct.is_file():
        return direct

    hits = sorted([p for p in root.rglob("*") if p.is_file() and p.name.lower() == GENE_MAPPING_FILENAME.lower()])
    return hits[0] if hits else None


def _load_gene_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Loads gene_mapping.csv if present (searched recursively).
    Returns (symbol_to_ensg, ensg_to_symbol).

    Expected columns:
      'Gene stable ID' (Ensembl) and 'Gene name' (symbol)
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return {}, {}

    fp = _find_gene_mapping_file(root)
    if fp is None:
        return {}, {}

    try:
        gm = pd.read_csv(fp)
    except Exception:
        return {}, {}

    cols = {c.lower(): c for c in gm.columns}
    ensg_col = cols.get("gene stable id") or cols.get("ensembl id") or cols.get("ensembl_id") or cols.get("ensembl")
    sym_col = cols.get("gene name") or cols.get("symbol") or cols.get("gene_symbol") or cols.get("gene")

    if ensg_col is None or sym_col is None:
        return {}, {}

    tmp = gm[[ensg_col, sym_col]].copy()
    tmp[ensg_col] = tmp[ensg_col].astype(str).map(_clean_gene_id)
    tmp[sym_col] = tmp[sym_col].astype(str).str.strip().str.upper()

    symbol_to_ensg: Dict[str, str] = {}
    ensg_to_symbol: Dict[str, str] = {}

    for _, r in tmp.iterrows():
        ensg = str(r[ensg_col]).strip().upper()
        sym = str(r[sym_col]).strip().upper()
        if sym and sym != "NAN" and ensg and ensg != "NAN":
            symbol_to_ensg.setdefault(sym, ensg)
            ensg_to_symbol.setdefault(ensg, sym)

    return symbol_to_ensg, ensg_to_symbol


def _resolve_query_to_gene_id(query: str, symbol_to_ensg: Dict[str, str]) -> Tuple[str, Optional[str]]:
    """
    Returns (gene_id_to_search_in_deg_table, resolved_symbol_if_any)
    """
    q = str(query).strip().upper()
    if not q:
        return "", None

    q_clean = _clean_gene_id(q)
    if _looks_like_ensembl(q_clean):
        return q_clean, None

    ensg = symbol_to_ensg.get(q_clean)
    if ensg:
        return _clean_gene_id(ensg), q_clean

    return q_clean, q_clean


# -----------------------------------------------------------------------------
# Robust readers + normalisation
# -----------------------------------------------------------------------------

# Common DEG formats vary a lot; include many realistic candidates
_GENE_COL_CANDIDATES = [
    "Gene", "gene",
    "GeneID", "gene_id", "geneid",
    "Ensembl ID", "Ensembl_ID", "ensembl id", "ensembl_id", "ensembl_gene_id",
    "Gene stable ID", "Gene stable id",
    "ensg", "ensembl",
    "feature", "id",
    "Name", "NAME"
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


def _autodetect_gene_col_by_values(df: pd.DataFrame) -> Optional[str]:
    """
    If we can't find a gene column by name, inspect values and pick the first column
    where a decent fraction of entries look like ENSG... (after cleaning).
    """
    if df is None or df.empty:
        return None

    sample_n = min(200, len(df))
    for c in df.columns:
        s = df[c].head(sample_n).astype(str).map(_clean_gene_id)
        if s.empty:
            continue
        frac = float((s.map(_looks_like_ensembl)).mean())
        if frac >= 0.3:
            return c
    return None


def _read_deg_file_safe(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read CSV/Parquet robustly.

    Key fix:
    - For CSVs where the Ensembl IDs are stored in an unnamed first column, keep it
      (do NOT treat it as an index at read time), then normalise_deg_table() will
      detect/rename it to 'Gene'.
    """
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)

            # Some CSV exports store Ensembl IDs in an unnamed first column.
            # pandas will call it "Unnamed: 0" (or similar). We keep it and let
            # normalise_deg_table() turn it into 'Gene'.
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



def normalise_deg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises key columns to:
      Gene, logFC, pval, padj, stat, baseMean

    Key fixes:
    - If Ensembl IDs are in the index, reset_index() and name it 'Gene'
    - If Ensembl IDs are in an unnamed first column (e.g. 'Unnamed: 0' or ''), rename it to 'Gene'
    - If we still can't find a gene column by name, auto-detect by ENSG-like values
    - Strip Ensembl version suffixes (ENSG... .12 -> ENSG...)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # 1) If Ensembl IDs are in the index (common with parquet), materialise them
    try:
        idx_vals = pd.Series(out.index).astype(str).map(_clean_gene_id)
        if len(idx_vals) > 0 and float(idx_vals.map(_looks_like_ensembl).mean()) >= 0.3:
            out = out.reset_index()
            # pandas names this column "index" unless index.name exists
            first_col = out.columns[0]
            out = out.rename(columns={first_col: "Gene"})
    except Exception:
        pass

    cols = list(out.columns)

    # 2) If the first column is unnamed and looks like ENSG..., treat it as Gene
    if cols:
        c0 = cols[0]
        c0_str = str(c0)
        if c0_str.strip() == "" or c0_str.lower().startswith("unnamed"):
            s = out[c0].head(min(200, len(out))).astype(str).map(_clean_gene_id)
            if len(s) > 0 and float(s.map(_looks_like_ensembl).mean()) >= 0.3:
                out = out.rename(columns={c0: "Gene"})

    cols = list(out.columns)

    # 3) Pick/auto-detect a gene column if not yet called 'Gene'
    gene_col = _pick_col(cols, _GENE_COL_CANDIDATES)
    if gene_col is None and "Gene" not in cols:
        gene_col = _autodetect_gene_col_by_values(out)

    logfc_col = _pick_col(cols, _LOGFC_COL_CANDIDATES)
    pval_col = _pick_col(cols, _PVAL_COL_CANDIDATES)
    padj_col = _pick_col(cols, _PADJ_COL_CANDIDATES)
    stat_col = _pick_col(cols, _STAT_COL_CANDIDATES)
    base_col = _pick_col(cols, _BASEMEAN_COL_CANDIDATES)

    ren: Dict[str, str] = {}
    if gene_col is not None:
        ren[gene_col] = "Gene"
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

    # 4) Final clean-up: standardise Gene IDs for matching
    if "Gene" in out.columns:
        out["Gene"] = out["Gene"].astype(str).map(_clean_gene_id)

    for c in ["logFC", "pval", "padj", "stat", "baseMean"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def load_all_invitro_deg_tables() -> Tuple[Dict[InVitroKey, pd.DataFrame], List[str], Dict[str, str], Dict[str, str]]:
    """
    Loads all discovered DEG tables.
    Returns (tables, errors, symbol_to_ensg, ensg_to_symbol).
    """
    files = discover_invitro_deg_files()
    tables: Dict[InVitroKey, pd.DataFrame] = {}
    errors: List[str] = []

    symbol_to_ensg, ensg_to_symbol = _load_gene_mapping()

    for key, fp in files.items():
        df, err = _read_deg_file_safe(fp)
        if err is not None:
            errors.append(err)
            continue
        tables[key] = normalise_deg_table(df)

    return tables, errors, symbol_to_ensg, ensg_to_symbol


# -----------------------------------------------------------------------------
# Gene-centric summaries + direction consensus between lines
# -----------------------------------------------------------------------------

def _get_gene_row(df: pd.DataFrame, gene_id: str) -> Optional[pd.Series]:
    if df is None or df.empty or "Gene" not in df.columns:
        return None
    g = _clean_gene_id(gene_id)
    hit = df.loc[df["Gene"] == g]
    if hit.empty:
        return None
    return hit.iloc[0]


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
        r = _get_gene_row(df, gene_id)

        # If we mapped SYMBOL->ENSG but the table stores symbols, try symbol too.
        if r is None and resolved_symbol and resolved_symbol != gene_id:
            r = _get_gene_row(df, resolved_symbol)

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

        logfc = float(r["logFC"]) if "logFC" in r and pd.notna(r["logFC"]) else np.nan
        padj = float(r["padj"]) if "padj" in r and pd.notna(r["padj"]) else np.nan
        pval = float(r["pval"]) if "pval" in r and pd.notna(r["pval"]) else np.nan

        direction = (
            "Up in model" if pd.notna(logfc) and logfc > 0
            else "Down in model" if pd.notna(logfc) and logfc < 0
            else "missing"
        )

        rows.append({
            "iHeps line": k.line,
            "Contrast": label,
            "Gene ID": gene_id,
            "Gene symbol": (resolved_symbol or ensg_to_symbol.get(gene_id)),
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

    # Hover gene label: prefer Symbol if we can map Ensembl IDs
    if "Gene" in tmp.columns:
        tmp["Gene"] = tmp["Gene"].astype(str).map(_clean_gene_id)
        if ensg_to_symbol:
            tmp["Symbol"] = tmp["Gene"].map(lambda g: ensg_to_symbol.get(str(g).upper(), ""))
        else:
            tmp["Symbol"] = ""
        hover_text = tmp["Symbol"].where(tmp["Symbol"].astype(str) != "", tmp["Gene"])
    else:
        hover_text = pd.Series([""] * len(tmp), index=tmp.index)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=tmp["logFC"],
        y=tmp["neglog10p"],
        mode="markers",
        marker=dict(size=4),
        text=hover_text,
        hovertemplate=(
            "Gene: %{text}<br>"
            "logFC: %{x:.3f}<br>"
            f"{y_label}: %{{y:.3f}}<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.add_vline(x=abs_logfc_thresh, line_dash="dash", line_width=1)
    fig.add_vline(x=-abs_logfc_thresh, line_dash="dash", line_width=1)
    if fdr_thresh and fdr_thresh > 0:
        fig.add_hline(y=-math.log10(fdr_thresh), line_dash="dash", line_width=1)

    if highlight_gene_id and "Gene" in tmp.columns:
        g = _clean_gene_id(highlight_gene_id)
        hit = tmp.loc[tmp["Gene"] == g]

        # If user typed a symbol and table stores symbols, try symbol too
        if hit.empty and highlight_label:
            hit = tmp.loc[tmp["Gene"] == _clean_gene_id(highlight_label)]

        if not hit.empty:
            label = highlight_label or (ensg_to_symbol.get(g) if ensg_to_symbol else g) or g
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
                    f"{y_label}: %{{y:.3f}}<extra></extra>"
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
    if "Gene" in tmp.columns:
        tmp["Gene"] = tmp["Gene"].astype(str).map(_clean_gene_id)

    if "padj" in tmp.columns:
        tmp = tmp.loc[tmp["padj"].isna() | (tmp["padj"] <= float(padj_thresh))].copy()

    if ensg_to_symbol and "Gene" in tmp.columns:
        tmp.insert(1, "Gene symbol", tmp["Gene"].map(lambda g: ensg_to_symbol.get(str(g).upper(), "")))

    up = tmp.sort_values("logFC", ascending=False).head(int(n)).copy()
    down = tmp.sort_values("logFC", ascending=True).head(int(n)).copy()

    keep = [c for c in ["Gene", "Gene symbol", "logFC", "padj", "pval", "stat", "baseMean"] if c in tmp.columns]
    return up[keep] if keep else up, down[keep] if keep else down


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_invitro_tab(query: str) -> None:
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

    # Quick availability + debug info
    st.markdown("### Dataset availability")
    st.caption(f"Gene mapping loaded: {len(symbol_to_ensg):,} symbols")
    avail_rows = []
    for c in CONTRAST_TOKENS:
        for l in LINE_TOKENS:
            hit = [k for k in files.keys() if k.contrast == c and k.line == l]
            avail_rows.append({
                "iHeps line": l,
                "Contrast": CONTRAST_LABELS[c],
                "File": files[hit[0]].name if hit else "missing",
            })
    st.dataframe(pd.DataFrame(avail_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    view = st.radio("Choose view", ["Gene summary", "Volcano explorer"], index=0, horizontal=True)

    if view == "Gene summary":
        st.markdown("### Gene summary")
        st.caption("Direction consensus is computed from logFC sign between 1b and 5a for the same contrast (significance not required).")

        summ = gene_summary_table(tables, query, symbol_to_ensg, ensg_to_symbol)
        if summ is None or summ.empty:
            st.warning("No rows could be generated for this query.")
            return

        gene_id, resolved_symbol = _resolve_query_to_gene_id(query, symbol_to_ensg)
        label = resolved_symbol or ensg_to_symbol.get(gene_id) or str(query).strip().upper()

        st.dataframe(summ, use_container_width=True, hide_index=True)

        cons = direction_consensus_by_contrast(summ)
        if cons is not None and not cons.empty:
            st.markdown("### Direction consensus between lines (1b vs 5a)")
            st.dataframe(cons, use_container_width=True, hide_index=True)

        fig_hm = make_gene_logfc_heatmap(summ, label)
        if fig_hm is not None:
            st.plotly_chart(fig_hm, use_container_width=True)

        fig_dot = make_gene_dotplot(summ, label)
        if fig_dot is not None:
            st.plotly_chart(fig_dot, use_container_width=True)

        st.markdown("### Contrast notes")
        for lbl, expl in CONTRAST_HELP.items():
            st.caption(f"{lbl}: {expl}")

        return

    st.markdown("### Volcano explorer")

    lines = sorted({k.line for k in tables.keys()})
    contrasts = [c for c in CONTRAST_TOKENS if any(k.contrast == c for k in tables.keys())]

    c1, c2 = st.columns(2)
    with c1:
        line_sel = st.selectbox("iHeps line", options=lines, index=0)
    with c2:
        contrast_sel = st.selectbox("Contrast", options=[CONTRAST_LABELS[c] for c in contrasts], index=0)

    contrast_token = None
    for c in contrasts:
        if CONTRAST_LABELS[c] == contrast_sel:
            contrast_token = c
            break
    if contrast_token is None:
        contrast_token = contrasts[0]

    key = None
    for k in tables.keys():
        if k.line == line_sel and k.contrast == contrast_token:
            key = k
            break
    if key is None:
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
        lfc_thr = st.number_input("|logFC| threshold", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.1f")
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
        st.plotly_chart(fig, use_container_width=True)

    up, down = top_deg_tables(df, ensg_to_symbol=ensg_to_symbol, n=int(topn), padj_thresh=float(fdr))
    st.markdown("### Top genes (FDR-filtered where available)")
    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("Upregulated")
        st.dataframe(up, use_container_width=True, hide_index=True)
    with c_down:
        st.markdown("Downregulated")
        st.dataframe(down, use_container_width=True, hide_index=True)
