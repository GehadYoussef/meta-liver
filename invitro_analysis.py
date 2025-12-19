"""
In vitro (human stem cell-derived) MASLD model analysis for Meta Liver.

This module backs a dedicated Streamlit tab that summarises DEGs from two iHeps
lines (1b, 5a) under three conditions compared with healthy controls (HCM):
- OA+PA vs HCM
- OA+PA + Resistin/Myostatin vs HCM
- OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM
(PBMCs were not included in RNA-seq; the DEGs reflect iHeps only.)

Expected data location (inside the app data directory):
  stem_cell_model/processed_degs_<LINE>_<CONTRAST>.parquet
Optionally, CSVs can exist alongside and are used as a fallback if parquet cannot be read.

Your DEG exports typically contain:
  external_gene_name (gene symbol) and/or Spalte1
  log2FoldChange (logFC)
  pvalue (pval)
  padj (FDR)
This module normalises these to: Gene, logFC, pval, padj, stat, baseMean.
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
# File discovery
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

# Accept either parquet or csv
_FILE_RE = re.compile(
    r"processed_degs_(?P<line>[^_]+)_(?P<contrast>.+)\.(?P<ext>parquet|csv)$",
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

    candidates = [
        data_dir / "stem_cell_model",
        data_dir / "stem-cell-model",
        data_dir / "stemcell_model",
        data_dir / "stemcell",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    for p in data_dir.iterdir():
        if p.is_dir():
            nm = p.name.lower().replace("-", "_")
            if "stem" in nm and "cell" in nm:
                return p

    return None


def discover_invitro_deg_files() -> List[Path]:
    """
    Returns DEG files present under stem_cell_model/ (csv or parquet).
    Does not attempt to read them.
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return []

    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if _FILE_RE.search(p.name):
            out.append(p)

    # Prefer parquet over csv if both exist (same stem)
    def _rank(path: Path) -> Tuple[str, int]:
        stem = path.stem.lower()
        ext = path.suffix.lower()
        return (stem, 0 if ext == ".parquet" else 1)

    out = sorted(out, key=_rank)

    # Deduplicate by stem, keep best-ranked
    seen = set()
    uniq = []
    for p in out:
        st = p.stem.lower()
        if st in seen:
            continue
        seen.add(st)
        uniq.append(p)

    return uniq


def _parse_file(path: Path) -> Optional[InVitroKey]:
    m = _FILE_RE.search(path.name)
    if not m:
        return None
    line = str(m.group("line")).strip()
    contrast = str(m.group("contrast")).strip()

    # Normalise contrast token casing to our canonical keys
    for k in list(_CONTRAST_LABELS.keys()):
        if k.lower() == contrast.lower():
            contrast = k
            break

    return InVitroKey(line=line, contrast=contrast)


# -----------------------------------------------------------------------------
# Robust DEG table normalisation
# -----------------------------------------------------------------------------

# Your files: external_gene_name and Spalte1
_GENE_COL_CANDIDATES = [
    "external_gene_name", "Spalte1",
    "gene", "Gene", "symbol", "Symbol", "gene_symbol", "gene_name", "GeneSymbol", "hgnc_symbol"
]
_LOGFC_COL_CANDIDATES = ["log2FoldChange", "logFC", "log2FC", "log2_fc", "log2foldchange", "lfc"]
_PVAL_COL_CANDIDATES = ["pvalue", "pval", "PValue", "p_value", "p.val", "p"]
_PADJ_COL_CANDIDATES = ["padj", "FDR", "adj_pval", "adj_pvalue", "qvalue", "q_value", "fdr"]
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


def _read_table_with_fallback(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Try to read parquet; if that fails, try CSV with same stem.
    Also supports being given a CSV directly.
    """
    if path is None or not path.exists():
        return None, f"File not found: {path}"

    suf = path.suffix.lower()

    if suf == ".csv":
        try:
            return pd.read_csv(path), None
        except Exception as e:
            return None, f"Could not read CSV file: {path.name}\n{type(e).__name__}: {e}"

    if suf == ".parquet":
        try:
            return pd.read_parquet(path), None
        except Exception as e:
            # CSV fallback with same stem
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                try:
                    return pd.read_csv(csv_path), None
                except Exception as e2:
                    return None, (
                        f"Parquet read failed and CSV fallback also failed for {path.stem}.\n"
                        f"Parquet error: {type(e).__name__}: {e}\n"
                        f"CSV error: {type(e2).__name__}: {e2}"
                    )

            return None, (
                f"Could not read parquet file: {path.name}\n\n"
                f"Underlying error: {type(e).__name__}: {e}\n\n"
                "If you want parquet support on Streamlit Cloud, add a parquet engine to requirements.txt, e.g.:\n"
                "  pyarrow\n"
                "Alternatively, provide the CSV alongside the parquet."
            )

    return None, f"Unsupported file type: {path.name}"


def _rescale_prob_like_series(s: pd.Series) -> pd.Series:
    """
    Some exports accidentally store probabilities as large integers that represent
    the decimal digits of a 0.x number, e.g. 841526188339379 -> 0.841526188339379.

    Heuristic:
      if value > 1 and looks integer-like, divide by 10^(number of digits).
    """
    if s is None:
        return s

    out = pd.to_numeric(s, errors="coerce")

    def _fix(v: float) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        try:
            fv = float(v)
        except Exception:
            return np.nan

        if fv <= 1.0:
            return fv

        # If it is very large but not astronomical, treat as "digits of 0.xxx"
        if fv < 1e18:
            iv = int(fv)
            # Only apply if it is truly integer-like (within tiny tolerance)
            if abs(fv - float(iv)) < 1e-6 and iv > 0:
                digits = len(str(iv))
                return fv / (10.0 ** digits)

        return fv

    return out.map(_fix)


def normalise_deg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises key columns to:
      Gene, logFC, pval, padj, stat, baseMean
    Keeps any other columns intact.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    cols = list(out.columns)

    gene_col = _pick_col(cols, _GENE_COL_CANDIDATES)
    logfc_col = _pick_col(cols, _LOGFC_COL_CANDIDATES)
    pval_col = _pick_col(cols, _PVAL_COL_CANDIDATES)
    padj_col = _pick_col(cols, _PADJ_COL_CANDIDATES)
    stat_col = _pick_col(cols, _STAT_COL_CANDIDATES)
    base_col = _pick_col(cols, _BASEMEAN_COL_CANDIDATES)

    # If gene is in the index, try to bring it back
    if gene_col is None and out.index.name is not None:
        idxn = str(out.index.name).lower()
        if idxn in {c.lower() for c in _GENE_COL_CANDIDATES}:
            out = out.reset_index()
            cols = list(out.columns)
            gene_col = _pick_col(cols, _GENE_COL_CANDIDATES)

    ren = {}
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

    if "Gene" in out.columns:
        out["Gene"] = out["Gene"].astype(str).str.strip().str.upper()

    # Coerce numerics
    for c in ["logFC", "pval", "padj", "stat", "baseMean"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Heuristic fixes for p-values that were exported as digit-strings
    if "pval" in out.columns:
        out["pval"] = _rescale_prob_like_series(out["pval"])
    if "padj" in out.columns:
        out["padj"] = _rescale_prob_like_series(out["padj"])

    return out


def load_all_invitro_deg_tables() -> Tuple[Dict[InVitroKey, pd.DataFrame], List[str]]:
    """
    Loads all discovered DEG tables.
    Returns (tables, errors).
    """
    tables: Dict[InVitroKey, pd.DataFrame] = {}
    errors: List[str] = []

    for f in discover_invitro_deg_files():
        key = _parse_file(f)
        if key is None:
            continue

        df, err = _read_table_with_fallback(f)
        if err is not None:
            errors.append(err)
            continue
        if df is None or df.empty:
            continue

        norm = normalise_deg_table(df)
        tables[key] = norm

    return tables, errors


# -----------------------------------------------------------------------------
# Gene-centric summarisation
# -----------------------------------------------------------------------------

def _get_gene_row(df: pd.DataFrame, gene: str) -> Optional[pd.Series]:
    if df is None or df.empty or "Gene" not in df.columns:
        return None
    g = str(gene).strip().upper()
    hit = df.loc[df["Gene"] == g]
    if hit.empty:
        return None
    return hit.iloc[0]


def gene_summary_table(tables: Dict[InVitroKey, pd.DataFrame], gene: str) -> pd.DataFrame:
    rows = []
    for k, df in tables.items():
        r = _get_gene_row(df, gene)
        contrast_lbl = _CONTRAST_LABELS.get(k.contrast, k.contrast)

        if r is None:
            rows.append({
                "iHeps line": k.line,
                "Contrast": contrast_lbl,
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
            "Up in model" if pd.notna(logfc) and logfc > 0 else
            "Down in model" if pd.notna(logfc) and logfc < 0 else
            "missing"
        )

        rows.append({
            "iHeps line": k.line,
            "Contrast": contrast_lbl,
            "logFC": logfc,
            "padj": padj,
            "pval": pval,
            "Direction": direction,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    contrast_order = list(_CONTRAST_LABELS.values())
    out["__c_rank__"] = out["Contrast"].astype(str).apply(lambda x: contrast_order.index(x) if x in contrast_order else 999)
    out = out.sort_values(["iHeps line", "__c_rank__"], ascending=[True, True]).drop(columns=["__c_rank__"])
    return out


def make_gene_logfc_heatmap(summary_df: pd.DataFrame, gene: str) -> Optional[go.Figure]:
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
        hovertemplate="Line: %{y}<br>Contrast: %{x}<br>logFC: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=f"{gene} logFC across iHeps lines and contrasts", font=dict(size=14)),
        height=280,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig


def make_gene_dotplot(summary_df: pd.DataFrame, gene: str) -> Optional[go.Figure]:
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
        title=dict(text=f"{gene} effect size (logFC) with significance (dot size)", font=dict(size=14)),
        xaxis_title="Contrast",
        yaxis_title="logFC (model vs HCM)",
        height=360,
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


# -----------------------------------------------------------------------------
# Dataset-centric volcano explorer
# -----------------------------------------------------------------------------

def make_volcano(
    df: pd.DataFrame,
    title: str,
    highlight_gene: Optional[str] = None,
    fdr_thresh: float = 0.05,
    abs_logfc_thresh: float = 1.0
) -> Optional[go.Figure]:
    if df is None or df.empty or "logFC" not in df.columns:
        return None

    tmp = df.copy()

    if "padj" in tmp.columns:
        p = pd.to_numeric(tmp["padj"], errors="coerce")
    elif "pval" in tmp.columns:
        p = pd.to_numeric(tmp["pval"], errors="coerce")
    else:
        p = pd.Series(np.nan, index=tmp.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp["neglog10p"] = -np.log10(p)

    sig = pd.Series(False, index=tmp.index)
    if "padj" in tmp.columns:
        sig = (tmp["padj"] <= float(fdr_thresh)) & (tmp["logFC"].abs() >= float(abs_logfc_thresh))
    tmp["sig"] = sig

    text = tmp["Gene"] if "Gene" in tmp.columns else None

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=tmp["logFC"],
        y=tmp["neglog10p"],
        mode="markers",
        marker=dict(size=4),
        hovertemplate=(
            "Gene: %{text}<br>"
            "logFC: %{x:.3f}<br>"
            "-log10(p): %{y:.3f}<extra></extra>"
        ),
        text=text,
        showlegend=False
    ))

    fig.add_vline(x=abs_logfc_thresh, line_dash="dash", line_width=1)
    fig.add_vline(x=-abs_logfc_thresh, line_dash="dash", line_width=1)
    fig.add_hline(y=-math.log10(fdr_thresh) if fdr_thresh > 0 else 0, line_dash="dash", line_width=1)

    if highlight_gene and "Gene" in tmp.columns:
        g = str(highlight_gene).strip().upper()
        hit = tmp.loc[tmp["Gene"] == g]
        if not hit.empty:
            fig.add_trace(go.Scatter(
                x=hit["logFC"],
                y=hit["neglog10p"],
                mode="markers+text",
                text=[g],
                textposition="top center",
                marker=dict(size=10),
                hovertemplate=(
                    "Gene: %{text}<br>"
                    "logFC: %{x:.3f}<br>"
                    "-log10(p): %{y:.3f}<extra></extra>"
                ),
                showlegend=False
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="logFC (model vs HCM)",
        yaxis_title="-log10(FDR)",
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

    keep = [c for c in ["Gene", "logFC", "padj", "pval", "stat", "baseMean"] if c in tmp.columns]
    return (up[keep] if keep else up), (down[keep] if keep else down)


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_invitro_tab(gene: str) -> None:
    """
    Streamlit rendering for the in vitro model tab.
    """
    import streamlit as st

    files = discover_invitro_deg_files()
    if not files:
        st.warning("No in vitro DEG files were found. Expected: data_dir/stem_cell_model/processed_degs_*.(parquet|csv)")
        return

    tables, errors = load_all_invitro_deg_tables()
    if errors:
        st.error(errors[0])
        if len(errors) > 1:
            st.info("More DEG read errors were encountered for other files as well.")
        return

    if not tables:
        st.warning("In vitro DEG files were found, but none could be loaded.")
        return

    st.markdown("### Data view")
    view = st.radio("Choose view", ["Gene summary", "Volcano explorer"], index=0, horizontal=True)

    lines = sorted({k.line for k in tables.keys()})
    # Keep the contrast order as defined in _CONTRAST_LABELS
    contrast_tokens = [k for k in _CONTRAST_LABELS.keys() if any(t.contrast == k for t in tables.keys())]
    if not contrast_tokens:
        contrast_tokens = sorted({k.contrast for k in tables.keys()})

    if view == "Gene summary":
        g = str(gene).strip().upper()
        st.markdown(f"### {g} in the in vitro model")
        st.caption("logFC is reported for model vs HCM within each iHeps line and contrast.")

        summary = gene_summary_table(tables, g)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        fig_hm = make_gene_logfc_heatmap(summary, g)
        if fig_hm is not None:
            st.plotly_chart(fig_hm, use_container_width=True)

        fig_dot = make_gene_dotplot(summary, g)
        if fig_dot is not None:
            st.plotly_chart(fig_dot, use_container_width=True)

        st.markdown("### Contrast notes")
        for lbl, expl in _CONTRAST_HELP.items():
            st.caption(f"{lbl}: {expl}")

        return

    st.markdown("### Volcano explorer")
    c1, c2 = st.columns(2)
    with c1:
        line_sel = st.selectbox("iHeps line", options=lines, index=0)
    with c2:
        contrast_sel = st.selectbox(
            "Contrast",
            options=[_CONTRAST_LABELS.get(t, t) for t in contrast_tokens],
            index=0
        )

    contrast_token = None
    for t in contrast_tokens:
        if _CONTRAST_LABELS.get(t, t) == contrast_sel:
            contrast_token = t
            break
    if contrast_token is None:
        contrast_token = contrast_tokens[0]

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
    fig = make_volcano(df, title=title, highlight_gene=gene, fdr_thresh=float(fdr), abs_logfc_thresh=float(lfc_thr))
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    up, down = top_deg_tables(df, n=int(topn), padj_thresh=float(fdr))
    st.markdown("### Top genes (FDR-filtered where available)")
    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("Upregulated")
        st.dataframe(up, use_container_width=True, hide_index=True)
    with c_down:
        st.markdown("Downregulated")
        st.dataframe(down, use_container_width=True, hide_index=True)
