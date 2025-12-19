"""
In vitro (human stem cell-derived) MASLD model analysis for Meta Liver.

This module is designed to back a dedicated Streamlit tab that summarises DEGs from
two iHeps lines (1b, 5a) under three conditions compared with healthy controls (HCM):
- OA+PA vs HCM
- OA+PA + Resistin/Myostatin vs HCM
- OA+PA + Resistin/Myostatin + PBMC co-culture vs HCM
(PBMCs were not included in RNA-seq; the DEGs reflect iHeps only.)

Expected data location (inside the app data directory):
  stem_cell_model/processed_degs_<LINE>_<CONTRAST>.parquet
Example:
  stem_cell_model/processed_degs_1b_OAPAvsHCM.parquet
  stem_cell_model/processed_degs_5a_OAPAResMyovsHCM.parquet

Important:
Parquet reading requires an optional engine (pyarrow or fastparquet). If missing,
the UI will show a clear install hint rather than crashing the whole app.
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

_FILE_RE = re.compile(r"processed_degs_(?P<line>[^_]+)_(?P<contrast>.+)\.parquet$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class InVitroKey:
    line: str        # "1b" or "5a"
    contrast: str    # canonical contrast token e.g. "OAPAvsHCM"


def _find_stem_cell_model_dir() -> Optional[Path]:
    data_dir = find_data_dir()
    if data_dir is None:
        return None

    # Case-insensitive folder match
    candidates = [
        data_dir / "stem_cell_model",
        data_dir / "stem-cell-model",
        data_dir / "stemcell_model",
        data_dir / "stemcell",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    # Fallback: walk one level deep for anything containing 'stem' and 'cell'
    for p in data_dir.iterdir():
        if p.is_dir():
            nm = p.name.lower().replace("-", "_")
            if "stem" in nm and "cell" in nm:
                return p

    return None


def discover_invitro_deg_files() -> List[Path]:
    """
    Returns a list of DEG parquet files, if present.
    This does not read parquet (safe to call even if parquet engines are missing).
    """
    root = _find_stem_cell_model_dir()
    if root is None:
        return []

    out: List[Path] = []
    for p in sorted(root.glob("*.parquet")):
        if _FILE_RE.search(p.name):
            out.append(p)
    return out


def _parse_file(path: Path) -> Optional[InVitroKey]:
    m = _FILE_RE.search(path.name)
    if not m:
        return None
    line = str(m.group("line")).strip()
    contrast = str(m.group("contrast")).strip()
    # Normalise contrast token casing to the keys we use in _CONTRAST_LABELS
    for k in list(_CONTRAST_LABELS.keys()):
        if k.lower() == contrast.lower():
            contrast = k
            break
    return InVitroKey(line=line, contrast=contrast)


# -----------------------------------------------------------------------------
# Robust DEG table normalisation
# -----------------------------------------------------------------------------

_GENE_COL_CANDIDATES = [
    "gene", "Gene", "symbol", "Symbol", "gene_symbol", "gene_name", "GeneSymbol",
    "external_gene_name", "External_Gene_Name", "externalGeneName",
    "Spalte1", "spalte1",  # often gene symbol
    "Column1", "column1"   # often Ensembl ID
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



def _read_parquet_safe(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read a DEG table from parquet, with a CSV fallback if parquet can't be read
    (e.g., missing pyarrow on Streamlit Cloud).
    """
    try:
        df = pd.read_parquet(path)
        return df, None
    except Exception as e:
        # Fallback: try a CSV with the same stem if present
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return df, None
            except Exception as e2:
                msg = (
                    f"Could not read parquet file: {path.name} and CSV fallback failed: {csv_path.name}

"
                    f"Parquet error: {type(e).__name__}: {e}
"
                    f"CSV error: {type(e2).__name__}: {e2}

"
                    "Fix: add a parquet engine to your environment (requirements.txt), e.g.
"
                    "  pyarrow
"
                    "Then redeploy/restart the app."
                )
                return None, msg

        msg = (
            f"Could not read parquet file: {path.name}

"
            f"Underlying error: {type(e).__name__}: {e}

"
            "Fix: add a parquet engine to your environment, for example include this in requirements.txt:
"
            "  pyarrow
"
            "Then redeploy/restart the app.
"
            "Optional: you can also place a CSV with the same name next to the parquet file for fallback loading."
        )
        return None, msg



def _auto_scale_p_like(s: pd.Series) -> pd.Series:
    """
    Heuristic: some exports store p-values/padj as large integers that represent
    the true value scaled by 1e15 (e.g. 841526188339379 -> 0.8415...).
    We rescale values > 1 by 1e15 when that fixes the range.
    """
    if s is None:
        return s
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.dropna().empty:
        return s2

    mask = s2 > 1
    if not mask.any():
        return s2

    scaled = s2.copy()
    cand = s2[mask] / 1e15
    ok_frac = float(((cand >= 0) & (cand <= 1)).mean()) if len(cand) else 0.0
    if ok_frac >= 0.8:
        scaled.loc[mask] = cand
    return scaled


def _auto_scale_logfc(s: pd.Series) -> pd.Series:
    """
    Heuristic: some exports store log2FC as large integers that represent the true
    log2FC scaled by 1e15 (e.g. -999772864250975 -> -0.9997...).
    """
    if s is None:
        return s
    s2 = pd.to_numeric(s, errors="coerce")
    v = s2.dropna()
    if v.empty:
        return s2

    med_abs = float(np.median(np.abs(v)))
    if med_abs > 1000:
        return s2 / 1e15
    return s2


def _auto_scale_basemean(s: pd.Series) -> pd.Series:
    """
    Heuristic: baseMean sometimes appears scaled by 1e12 in exported tables.
    """
    if s is None:
        return s
    s2 = pd.to_numeric(s, errors="coerce")
    v = s2.dropna()
    if v.empty:
        return s2

    med = float(np.median(v))
    if med > 1e8:
        return s2 / 1e12
    return s2

def normalise_deg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises key columns to:
      Gene, logFC, pval, padj, stat, baseMean
    Leaves any other columns intact.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols = list(df.columns)

    gene_col = _pick_col(cols, _GENE_COL_CANDIDATES)
    logfc_col = _pick_col(cols, _LOGFC_COL_CANDIDATES)
    pval_col = _pick_col(cols, _PVAL_COL_CANDIDATES)
    padj_col = _pick_col(cols, _PADJ_COL_CANDIDATES)
    stat_col = _pick_col(cols, _STAT_COL_CANDIDATES)
    base_col = _pick_col(cols, _BASEMEAN_COL_CANDIDATES)

    out = df.copy()

    # If gene is in the index, try to bring it back
    if gene_col is None and out.index.name is not None and str(out.index.name).lower() in [c.lower() for c in _GENE_COL_CANDIDATES]:
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

    # Clean up types
    if "Gene" in out.columns:
        out["Gene"] = out["Gene"].astype(str).str.strip().str.upper()

    for c in ["logFC", "pval", "padj", "stat", "baseMean"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Fix scaled exports (common in some CSV/Parquet conversions)
    if "pval" in out.columns:
        out["pval"] = _auto_scale_p_like(out["pval"])
    if "padj" in out.columns:
        out["padj"] = _auto_scale_p_like(out["padj"])
    if "logFC" in out.columns:
        out["logFC"] = _auto_scale_logfc(out["logFC"])
    if "baseMean" in out.columns:
        out["baseMean"] = _auto_scale_basemean(out["baseMean"])

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
        df, err = _read_parquet_safe(f)
        if err is not None:
            errors.append(err)
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
    # If duplicates, take the first
    return hit.iloc[0]


def gene_summary_table(tables: Dict[InVitroKey, pd.DataFrame], gene: str) -> pd.DataFrame:
    rows = []
    for k, df in tables.items():
        r = _get_gene_row(df, gene)
        if r is None:
            rows.append({
                "iHeps line": k.line,
                "Contrast": _CONTRAST_LABELS.get(k.contrast, k.contrast),
                "logFC": np.nan,
                "padj": np.nan,
                "pval": np.nan,
                "Direction": "missing",
            })
            continue

        logfc = float(r["logFC"]) if "logFC" in r and pd.notna(r["logFC"]) else np.nan
        padj = float(r["padj"]) if "padj" in r and pd.notna(r["padj"]) else np.nan
        pval = float(r["pval"]) if "pval" in r and pd.notna(r["pval"]) else np.nan

        direction = "Up in model" if pd.notna(logfc) and logfc > 0 else "Down in model" if pd.notna(logfc) and logfc < 0 else "missing"

        rows.append({
            "iHeps line": k.line,
            "Contrast": _CONTRAST_LABELS.get(k.contrast, k.contrast),
            "logFC": logfc,
            "padj": padj,
            "pval": pval,
            "Direction": direction,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Nice ordering
    contrast_order = list(_CONTRAST_LABELS.values())
    out["__c__"] = out["Contrast"].astype(str)
    out["__c_rank__"] = out["__c__"].apply(lambda x: contrast_order.index(x) if x in contrast_order else 999)
    out = out.sort_values(["iHeps line", "__c_rank__"], ascending=[True, True]).drop(columns=["__c__", "__c_rank__"])
    return out


def make_gene_logfc_heatmap(summary_df: pd.DataFrame, gene: str) -> Optional[go.Figure]:
    if summary_df is None or summary_df.empty or "logFC" not in summary_df.columns:
        return None

    # Pivot into matrix: line x contrast
    mat = summary_df.pivot(index="iHeps line", columns="Contrast", values="logFC")
    if mat.empty:
        return None

    z = mat.values.astype(float)
    y = list(mat.index)
    x = list(mat.columns)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
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

    df = summary_df.copy()
    if "padj" not in df.columns or "logFC" not in df.columns:
        return None

    # Size encodes -log10(padj)
    def neglog10(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)) or x <= 0:
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
            name=str(r["iHeps line"]),
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

def make_volcano(df: pd.DataFrame, title: str, highlight_gene: Optional[str] = None,
                fdr_thresh: float = 0.05, abs_logfc_thresh: float = 1.0) -> Optional[go.Figure]:
    if df is None or df.empty or "logFC" not in df.columns:
        return None

    tmp = df.copy()

    # Compute -log10(padj) if available, else pval
    if "padj" in tmp.columns:
        p = tmp["padj"].astype(float)
    elif "pval" in tmp.columns:
        p = tmp["pval"].astype(float)
    else:
        p = pd.Series(np.nan, index=tmp.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp["neglog10p"] = -np.log10(p)

    sig = pd.Series(False, index=tmp.index)
    if "padj" in tmp.columns:
        sig = (tmp["padj"] <= float(fdr_thresh)) & (tmp["logFC"].abs() >= float(abs_logfc_thresh))

    tmp["sig"] = sig

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
        text=tmp["Gene"] if "Gene" in tmp.columns else None,
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
    return up[keep] if keep else up, down[keep] if keep else down


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_invitro_tab(gene: str) -> None:
    """
    Streamlit rendering for the in vitro model tab.
    """
    import streamlit as st  # local import to keep module usable outside Streamlit

    files = discover_invitro_deg_files()
    if not files:
        st.warning("No in vitro DEG parquet files were found. Expected: data_dir/stem_cell_model/*.parquet")
        return

    tables, errors = load_all_invitro_deg_tables()
    if errors:
        # Show only the first error in full; others as short lines
        st.error(errors[0])
        if len(errors) > 1:
            st.info("More parquet read errors were encountered for other files as well.")
        return

    if not tables:
        st.warning("In vitro DEG files were found, but none could be loaded.")
        return

    # Controls
    st.markdown("### Data view")
    view = st.radio("Choose view", ["Gene summary", "Volcano explorer"], index=0, horizontal=True)

    # Make options
    lines = sorted({k.line for k in tables.keys()})
    contrast_tokens = sorted({k.contrast for k in tables.keys()}, key=lambda x: list(_CONTRAST_LABELS.keys()).index(x) if x in _CONTRAST_LABELS else 999)

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

    # Volcano explorer
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

    # Map back to token
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

    # Thresholds
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
