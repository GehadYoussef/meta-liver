"""
Bulk Omics analysis module for Meta Liver.

Folder layout expected (inside app data directory):
  meta-liver-data/Bulk_Omics/
      MASLD vs Control/
          GSE126848_MASLD_Control.tsv
          ...
      early MASLD vs control/
          ...
      MASH vs control/
          ...
      Early MASLD vs MASH/
          ...

Each DEG table should include (case-insensitive):
  - Symbol
  - log2FoldChange
  - pvalue
  - padj

This module:
  - loads all bulk DEG tables grouped by contrast folder
  - normalises schemas and resolves duplicate gene symbols
  - supports gene-centric summary and meta-analysis across studies per contrast
  - provides "top genes" ranking per contrast (fast vectorised implementation)
  - provides gene-summary plots (forest-style log2FC CI; log2FC vs -log10(p))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import math

import numpy as np
import pandas as pd

from robust_data_loader import find_data_dir, find_subfolder


# -----------------------------------------------------------------------------
# Constants + normalisation helpers
# -----------------------------------------------------------------------------

_BULK_SYMBOL_CANDS = ["Symbol", "symbol", "gene", "Gene", "gene_symbol", "GeneSymbol"]
_BULK_LOGFC_CANDS = ["log2FoldChange", "logFC", "log2FC", "log2_fc", "log2foldchange"]
_BULK_PVAL_CANDS = ["pvalue", "pval", "p_value", "PValue", "P.Value"]
_BULK_PADJ_CANDS = ["padj", "FDR", "adj_pval", "adj_pvalue", "qvalue", "q_value"]


def _pick_col_ci(cols: List[str], cands: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in cols:
            return cand
        if cand.lower() in m:
            return m[cand.lower()]
    return None


def _normalise_symbol(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("", "NAN", "NONE"):
        return ""
    return s


def _safe_read_table(fp: Path) -> pd.DataFrame:
    if fp is None or not fp.exists() or not fp.is_file():
        return pd.DataFrame()

    suf = fp.suffix.lower()
    try:
        if suf == ".parquet":
            return pd.read_parquet(fp)
        if suf == ".csv":
            return pd.read_csv(fp)
        if suf in (".tsv", ".txt"):
            return pd.read_csv(fp, sep="\t")
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def normalise_bulk_deg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a bulk DEG table to ensure:
    - Symbol column present
    - log2FoldChange / pvalue / padj standardised (if present)
    - duplicate Symbols resolved deterministically
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    sym_col = _pick_col_ci(list(out.columns), _BULK_SYMBOL_CANDS)
    lfc_col = _pick_col_ci(list(out.columns), _BULK_LOGFC_CANDS)
    p_col = _pick_col_ci(list(out.columns), _BULK_PVAL_CANDS)
    q_col = _pick_col_ci(list(out.columns), _BULK_PADJ_CANDS)

    if sym_col is None:
        return pd.DataFrame()

    ren: Dict[str, str] = {sym_col: "Symbol"}
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
        out["log2FoldChange"] = pd.to_numeric(out["log2FoldChange"], errors="coerce")
    if "pvalue" in out.columns:
        out["pvalue"] = pd.to_numeric(out["pvalue"], errors="coerce")
    if "padj" in out.columns:
        out["padj"] = pd.to_numeric(out["padj"], errors="coerce")

    # Resolve duplicates by best significance then largest absolute effect
    if out["Symbol"].duplicated().any():

        def _rank_row(r) -> Tuple[float, float]:
            q = r.get("padj", float("nan"))
            p = r.get("pvalue", float("nan"))
            lfc = r.get("log2FoldChange", float("nan"))

            best_sig = q if pd.notna(q) else (p if pd.notna(p) else float("inf"))
            abs_lfc = abs(lfc) if pd.notna(lfc) else 0.0
            return (best_sig, -abs_lfc)

        out["__rk__"] = out.apply(_rank_row, axis=1)
        out = out.sort_values("__rk__", ascending=True)
        out = out.drop_duplicates(subset=["Symbol"], keep="first").drop(columns=["__rk__"])

    return out


# -----------------------------------------------------------------------------
# Discovery + loading
# -----------------------------------------------------------------------------

def _find_bulk_omics_dir() -> Optional[Path]:
    data_dir = find_data_dir()
    if data_dir is None:
        return None
    return find_subfolder(data_dir, "bulk_omics")


def discover_bulk_contrasts() -> List[str]:
    root = _find_bulk_omics_dir()
    if root is None or not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_bulk_omics() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Returns:
      {contrast_name: {study_id: df}}
    """
    root = _find_bulk_omics_dir()
    if root is None or not root.exists():
        return {}

    out: Dict[str, Dict[str, pd.DataFrame]] = {}

    for contrast_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        contrast = contrast_dir.name
        studies: Dict[str, pd.DataFrame] = {}

        for fp in sorted(contrast_dir.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in (".tsv", ".csv", ".txt", ".parquet"):
                continue

            df = _safe_read_table(fp)
            if df.empty:
                continue

            df = normalise_bulk_deg_table(df)
            if df.empty:
                continue

            studies[fp.stem] = df

        if studies:
            out[contrast] = studies

    return out


# -----------------------------------------------------------------------------
# Meta-analysis helpers (stable normal inverse CDF approximation; vectorised)
# -----------------------------------------------------------------------------

def _norm_ppf_approx(u: np.ndarray) -> np.ndarray:
    """
    Vectorised Acklam-ish rational approximation to Φ^{-1}(u).
    u must be in (0,1). Caller is responsible for clipping.
    """
    u = np.asarray(u, dtype=float)

    a = np.array(
        [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00],
        dtype=float,
    )
    b = np.array(
        [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01],
        dtype=float,
    )
    c = np.array(
        [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00],
        dtype=float,
    )
    d = np.array(
        [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00],
        dtype=float,
    )

    plow = 0.02425
    phigh = 1.0 - plow

    z = np.empty_like(u, dtype=float)

    low = u < plow
    high = u > phigh
    mid = ~(low | high)

    if np.any(low):
        ul = u[low]
        q = np.sqrt(-2.0 * np.log(ul))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        z[low] = num / den

    if np.any(high):
        uh = u[high]
        q = np.sqrt(-2.0 * np.log(1.0 - uh))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        z[high] = -(num / den)

    if np.any(mid):
        um = u[mid]
        q = um - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        z[mid] = num / den

    return z


def _signed_z_from_two_sided_p(p: np.ndarray, lfc: np.ndarray) -> np.ndarray:
    """
    Signed z from a two-sided p-value, with sign from log2FoldChange.
    z = sign(lfc) * Φ^{-1}(1 - p/2)
    """
    p = np.asarray(p, dtype=float)
    lfc = np.asarray(lfc, dtype=float)

    ok = np.isfinite(p) & np.isfinite(lfc) & (p > 0.0) & (p <= 1.0)
    z = np.full_like(p, np.nan, dtype=float)
    if not np.any(ok):
        return z

    # Avoid u rounding to exactly 1.0 when p is extremely small
    eps_u = 1e-16
    u = 1.0 - (p[ok] / 2.0)
    u = np.clip(u, eps_u, 1.0 - eps_u)

    base = _norm_ppf_approx(u)
    sgn = np.where(lfc[ok] >= 0.0, 1.0, -1.0)
    z[ok] = sgn * base
    return z


def _z_to_p_two_sided(z: np.ndarray) -> np.ndarray:
    """
    Two-sided p from z.
    Uses scipy.special.erfc if available; otherwise falls back to math.erfc element-wise.
    """
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.nan, dtype=float)

    ok = np.isfinite(z)
    if not np.any(ok):
        return out

    za = np.abs(z[ok]) / math.sqrt(2.0)

    try:
        from scipy.special import erfc as sp_erfc  # type: ignore
        out[ok] = np.clip(sp_erfc(za), 0.0, 1.0)
    except Exception:
        out[ok] = np.clip(np.array([math.erfc(float(v)) for v in za], dtype=float), 0.0, 1.0)

    return out



# -----------------------------------------------------------------------------
# Gene-centric summaries (single gene; simple and readable)
# -----------------------------------------------------------------------------

def per_study_gene_rows(studies: Dict[str, pd.DataFrame], gene_symbol: str) -> pd.DataFrame:
    g = _normalise_symbol(gene_symbol)
    if not g:
        return pd.DataFrame()

    rows = []
    for study_id, df in (studies or {}).items():
        if df is None or df.empty or "Symbol" not in df.columns:
            continue
        hit = df.loc[df["Symbol"] == g]
        if hit.empty:
            continue
        r = hit.iloc[0].to_dict()
        r["Study"] = study_id
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    front = [c for c in ["Study", "Symbol", "log2FoldChange", "padj", "pvalue"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    sig = "padj" if ("padj" in out.columns and out["padj"].notna().any()) else ("pvalue" if "pvalue" in out.columns else None)
    if sig:
        out = out.sort_values(sig, ascending=True, na_position="last")
    return out


def direction_agreement(per_study: pd.DataFrame) -> float:
    if per_study is None or per_study.empty or "log2FoldChange" not in per_study.columns:
        return float("nan")
    s = pd.to_numeric(per_study["log2FoldChange"], errors="coerce").dropna()
    if s.empty:
        return float("nan")
    signs = np.sign(s.values)
    signs = signs[signs != 0]
    if signs.size == 0:
        return float("nan")
    pos = float((signs > 0).mean())
    neg = float((signs < 0).mean())
    return float(max(pos, neg))


def bulk_gene_summary(studies: Dict[str, pd.DataFrame], gene_symbol: str) -> Dict[str, object]:
    per = per_study_gene_rows(studies, gene_symbol)
    if per is None or per.empty:
        return {
            "per_study": pd.DataFrame(),
            "meta_log2FoldChange": float("nan"),
            "meta_p": float("nan"),
            "meta_Z": float("nan"),
            "agreement": float("nan"),
            "n_studies": 0,
        }

    # row-wise p-eff: prefer padj, else pvalue
    lfc = pd.to_numeric(per.get("log2FoldChange", np.nan), errors="coerce").to_numpy(dtype=float)
    padj = pd.to_numeric(per.get("padj", np.nan), errors="coerce").to_numpy(dtype=float) if "padj" in per.columns else np.full_like(lfc, np.nan)
    pval = pd.to_numeric(per.get("pvalue", np.nan), errors="coerce").to_numpy(dtype=float) if "pvalue" in per.columns else np.full_like(lfc, np.nan)
    p_eff = np.where(np.isfinite(padj), padj, pval)

    z = _signed_z_from_two_sided_p(p_eff, lfc)
    ok = np.isfinite(z)
    if not np.any(ok):
        meta_Z = float("nan")
        meta_p = float("nan")
        meta_lfc = float(np.nanmean(lfc))
    else:
        w = np.maximum(1.0, np.abs(z[ok]))
        meta_Z = float(np.sum(w * z[ok]) / math.sqrt(float(np.sum(w * w))))
        meta_p = float(_z_to_p_two_sided(np.array([meta_Z], dtype=float))[0])

        # logFC weighted by abs(z)
        meta_lfc = float(np.sum(w * lfc[ok]) / np.sum(w))

    agree = direction_agreement(per)

    return {
        "per_study": per,
        "meta_log2FoldChange": meta_lfc,
        "meta_p": meta_p,
        "meta_Z": meta_Z,
        "agreement": agree,
        "n_studies": int(per["Study"].nunique()) if "Study" in per.columns else int(len(per)),
    }


# -----------------------------------------------------------------------------
# Top genes per contrast (fast vectorised implementation)
# -----------------------------------------------------------------------------

def _long_table_from_studies(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for study_id, df in (studies or {}).items():
        if df is None or df.empty or "Symbol" not in df.columns:
            continue
        cols = ["Symbol"]
        if "log2FoldChange" in df.columns:
            cols.append("log2FoldChange")
        if "padj" in df.columns:
            cols.append("padj")
        if "pvalue" in df.columns:
            cols.append("pvalue")
        sub = df.loc[:, [c for c in cols if c in df.columns]].copy()
        sub["Study"] = str(study_id)
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    long = pd.concat(frames, ignore_index=True)

    if "log2FoldChange" in long.columns:
        long["log2FoldChange"] = pd.to_numeric(long["log2FoldChange"], errors="coerce")
    if "padj" in long.columns:
        long["padj"] = pd.to_numeric(long["padj"], errors="coerce")
    if "pvalue" in long.columns:
        long["pvalue"] = pd.to_numeric(long["pvalue"], errors="coerce")

    # p_eff: prefer padj, else pvalue
    padj = long["padj"].to_numpy(dtype=float) if "padj" in long.columns else np.full(len(long), np.nan, dtype=float)
    pval = long["pvalue"].to_numpy(dtype=float) if "pvalue" in long.columns else np.full(len(long), np.nan, dtype=float)
    long["p_eff"] = np.where(np.isfinite(padj), padj, pval)

    return long


def top_genes_for_contrast(
    studies: Dict[str, pd.DataFrame],
    *,
    min_studies: int = 2,
    max_genes: int = 200,
) -> pd.DataFrame:
    """
    Compute a ranked table of genes for one contrast.

    Evidence score = (-log10(meta_p)) * |meta_logFC| * agreement

    meta_p: signed Stouffer (signed z per study; weights ~ |z|)
    meta_logFC: weighted mean logFC (weights ~ |z|)
    agreement: majority sign proportion across studies
    """
    if studies is None or not studies:
        return pd.DataFrame()

    long = _long_table_from_studies(studies)
    if long.empty or "Symbol" not in long.columns or "log2FoldChange" not in long.columns:
        return pd.DataFrame()

    lfc = long["log2FoldChange"].to_numpy(dtype=float)
    p_eff = pd.to_numeric(long["p_eff"], errors="coerce").to_numpy(dtype=float)
    z = _signed_z_from_two_sided_p(p_eff, lfc)

    long["z"] = z
    long["w"] = np.where(np.isfinite(z), np.maximum(1.0, np.abs(z)), np.nan)

    # Keep only rows with needed values for meta
    m = long[np.isfinite(long["z"]) & np.isfinite(long["w"]) & np.isfinite(long["log2FoldChange"])].copy()
    if m.empty:
        return pd.DataFrame()

    # Studies per gene
    n_st = long.groupby("Symbol", dropna=False)["Study"].nunique()
    n_st = n_st.rename("n_studies")

    # Agreement per gene
    sgn = np.sign(pd.to_numeric(long["log2FoldChange"], errors="coerce"))
    long["_sgn_"] = sgn
    g2 = long.dropna(subset=["_sgn_"]).copy()
    g2 = g2[g2["_sgn_"] != 0]
    if g2.empty:
        agree = pd.Series(dtype=float, name="agreement")
    else:
        pos = g2.assign(_pos_=(g2["_sgn_"] > 0)).groupby("Symbol")["_pos_"].mean()
        neg = g2.assign(_neg_=(g2["_sgn_"] < 0)).groupby("Symbol")["_neg_"].mean()
        agree = pd.concat([pos, neg], axis=1).max(axis=1)
        agree.name = "agreement"

    # Meta Z per gene: sum(w*z)/sqrt(sum(w^2))
    m["wz"] = m["w"] * m["z"]
    m["w2"] = m["w"] * m["w"]

    sum_wz = m.groupby("Symbol")["wz"].sum()
    sum_w2 = m.groupby("Symbol")["w2"].sum()
    meta_Z = sum_wz / np.sqrt(sum_w2)
    meta_Z.name = "meta_Z"

    meta_p = _z_to_p_two_sided(meta_Z.to_numpy(dtype=float))
    meta_p = pd.Series(meta_p, index=meta_Z.index, name="meta_p")

    # Meta logFC per gene: sum(w*lfc)/sum(w)
    m["wlfc"] = m["w"] * m["log2FoldChange"]
    sum_wlfc = m.groupby("Symbol")["wlfc"].sum()
    sum_w = m.groupby("Symbol")["w"].sum()
    meta_lfc = sum_wlfc / sum_w
    meta_lfc.name = "meta_log2FoldChange"

    out = pd.concat([n_st, meta_lfc, meta_Z, meta_p, agree], axis=1).reset_index().rename(columns={"index": "Symbol"})

    # Filter by min studies
    out = out[out["n_studies"].fillna(0).astype(int) >= int(min_studies)].copy()
    if out.empty:
        return out

    # Evidence score
    mp = pd.to_numeric(out["meta_p"], errors="coerce").to_numpy(dtype=float)
    ml = pd.to_numeric(out["meta_log2FoldChange"], errors="coerce").to_numpy(dtype=float)
    ag = pd.to_numeric(out["agreement"], errors="coerce").to_numpy(dtype=float)

    score = np.full(len(out), np.nan, dtype=float)
    ok = np.isfinite(mp) & (mp > 0) & np.isfinite(ml) & np.isfinite(ag)
    score[ok] = (-np.log10(mp[ok])) * np.abs(ml[ok]) * ag[ok]
    out["evidence_score"] = score

    out = out.sort_values("evidence_score", ascending=False, na_position="last").head(int(max_genes))
    return out


# -----------------------------------------------------------------------------
# Plot helpers (gene summary)
# -----------------------------------------------------------------------------

def _forest_plot_from_per_study(per: pd.DataFrame, *, contrast_label: str):
    import plotly.graph_objects as go

    if per is None or per.empty or "log2FoldChange" not in per.columns:
        return None

    df = per.copy()
    lfc = pd.to_numeric(df["log2FoldChange"], errors="coerce").to_numpy(dtype=float)

    padj = pd.to_numeric(df["padj"], errors="coerce").to_numpy(dtype=float) if "padj" in df.columns else np.full_like(lfc, np.nan)
    pval = pd.to_numeric(df["pvalue"], errors="coerce").to_numpy(dtype=float) if "pvalue" in df.columns else np.full_like(lfc, np.nan)
    p_eff = np.where(np.isfinite(padj), padj, pval)

    ok = np.isfinite(lfc) & np.isfinite(p_eff) & (p_eff > 0) & (p_eff <= 1)
    if not np.any(ok):
        return None

    z = _signed_z_from_two_sided_p(p_eff[ok], lfc[ok])
    zabs = np.abs(z)

    # Approx SE from Wald relationship: z ≈ lfc/SE => SE ≈ |lfc|/|z|
    se = np.full_like(zabs, np.nan, dtype=float)
    good = zabs > 1e-12
    se[good] = np.abs(lfc[ok][good]) / zabs[good]

    ci_lo = lfc[ok] - 1.96 * se
    ci_hi = lfc[ok] + 1.96 * se

    study = df.loc[ok, "Study"].astype(str).tolist() if "Study" in df.columns else [f"Study {i+1}" for i in range(int(ok.sum()))]

    # Sort by effect for readability
    order = np.argsort(lfc[ok])
    study = [study[i] for i in order]
    x = lfc[ok][order]
    lo = ci_lo[order]
    hi = ci_hi[order]

    err_plus = hi - x
    err_minus = x - lo

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=study,
            mode="markers",
            error_x=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, thickness=1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "log2FC=%{x:.3f}<br>"
                "95% CI [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
            ),
            customdata=np.column_stack([lo, hi]),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_width=1)

    fig.update_layout(
        title=f"Forest-style view: per-study log2FC (approx 95% CI) — {contrast_label}",
        xaxis_title="log2FoldChange",
        yaxis_title="Study",
        height=max(320, 26 * len(study) + 140),
        margin=dict(l=80, r=20, t=50, b=40),
    )
    return fig


def _scatter_logfc_vs_sig(per: pd.DataFrame, *, contrast_label: str):
    import plotly.graph_objects as go

    if per is None or per.empty or "log2FoldChange" not in per.columns:
        return None

    df = per.copy()
    lfc = pd.to_numeric(df["log2FoldChange"], errors="coerce").to_numpy(dtype=float)

    padj = pd.to_numeric(df["padj"], errors="coerce").to_numpy(dtype=float) if "padj" in df.columns else np.full_like(lfc, np.nan)
    pval = pd.to_numeric(df["pvalue"], errors="coerce").to_numpy(dtype=float) if "pvalue" in df.columns else np.full_like(lfc, np.nan)
    p_eff = np.where(np.isfinite(padj), padj, pval)

    ok = np.isfinite(lfc) & np.isfinite(p_eff) & (p_eff > 0)
    if not np.any(ok):
        return None

    y = -np.log10(p_eff[ok])
    study = df.loc[ok, "Study"].astype(str).tolist() if "Study" in df.columns else [f"Study {i+1}" for i in range(int(ok.sum()))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lfc[ok],
            y=y,
            mode="markers+text",
            text=study,
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>log2FC=%{x:.3f}<br>-log10(p)=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_width=1)
    fig.update_layout(
        title=f"Per-study signal: log2FC vs −log10(p) — {contrast_label}",
        xaxis_title="log2FoldChange",
        yaxis_title="−log10(p) (padj if available, else pvalue)",
        height=340,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_bulk_omics_tab(
    query: str,
    *,
    bulk_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
) -> None:
    import streamlit as st

    def _st_df(df: pd.DataFrame, *, hide_index: bool = True):
        try:
            st.dataframe(df, width="stretch", hide_index=hide_index)
        except TypeError:
            st.dataframe(df, use_container_width=True, hide_index=hide_index)

    def _st_plotly(fig):
        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)

    root = _find_bulk_omics_dir()

    # Prefer preloaded bulk_data (faster; avoids disk I/O inside tab)
    data = bulk_data if isinstance(bulk_data, dict) and len(bulk_data) > 0 else load_bulk_omics()

    if (bulk_data is None or (isinstance(bulk_data, dict) and len(bulk_data) == 0)) and (root is None or not root.exists()):
        st.warning("No Bulk Omics folder found. Expected: meta-liver-data/Bulk_Omics/")
        return

    if not data:
        st.warning("Bulk Omics found, but no valid TSV/CSV/Parquet DEG tables could be loaded.")
        if root is not None:
            st.caption(f"Looking in: {root}")
        return

    st.markdown("### Dataset availability")
    if root is not None:
        st.caption(f"Bulk Omics root: {root}")

    avail_rows = []
    for contrast, studies in data.items():
        avail_rows.append({"Contrast": str(contrast), "Study tables": len(studies or {})})
    _st_df(pd.DataFrame(avail_rows), hide_index=True)

    st.markdown("---")

    contrasts = sorted(list(data.keys()))
    contrast_sel = st.selectbox("Contrast", options=contrasts, index=0)

    studies = data.get(contrast_sel, {}) or {}
    if not studies:
        st.warning("No study tables found for this contrast.")
        return

    view = st.radio("Choose view", ["Gene summary", "Top genes"], index=0, horizontal=True)

    if view == "Top genes":
        st.markdown(f"### Top genes (meta + consistency) — {contrast_sel}")
        min_st = st.number_input("Minimum studies per gene", min_value=1, max_value=50, value=2, step=1)
        max_g = st.number_input("Max genes to show", min_value=50, max_value=5000, value=200, step=50)

        with st.spinner("Scoring genes across studies..."):
            top = top_genes_for_contrast(studies, min_studies=int(min_st), max_genes=int(max_g))

        if top.empty:
            st.info("No genes met the minimum study requirement (or no usable p-values/logFC).")
            return

        _st_df(top, hide_index=True)

        # Optional quick overview plot (helps sanity-check the ranking)
        try:
            import plotly.graph_objects as go

            dfp = top.copy()
            mp = pd.to_numeric(dfp["meta_p"], errors="coerce")
            ml = pd.to_numeric(dfp["meta_log2FoldChange"], errors="coerce")
            ok = mp.notna() & (mp > 0) & ml.notna()
            if ok.sum() >= 5:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=ml[ok],
                        y=-np.log10(mp[ok]),
                        mode="markers",
                        text=dfp.loc[ok, "Symbol"],
                        hovertemplate="<b>%{text}</b><br>meta_log2FC=%{x:.3f}<br>-log10(meta_p)=%{y:.2f}<extra></extra>",
                    )
                )
                fig.add_vline(x=0, line_dash="dash", line_width=1)
                fig.update_layout(
                    title=f"Top genes overview: meta log2FC vs −log10(meta p) — {contrast_sel}",
                    xaxis_title="meta log2FoldChange",
                    yaxis_title="−log10(meta p)",
                    height=340,
                    margin=dict(l=60, r=20, t=50, b=40),
                )
                _st_plotly(fig)
        except Exception:
            pass

        return

    st.markdown(f"### Gene summary — {contrast_sel}")
    gene = str(query).strip()
    if not gene:
        st.info("Enter a gene symbol in the search box above.")
        return

    summ = bulk_gene_summary(studies, gene)
    per = summ["per_study"]

    if per.empty:
        st.warning("Gene not found in any study tables for this contrast.")
        return

    per_show = per.copy()
    if "Contrast" not in per_show.columns:
        per_show.insert(0, "Contrast", str(contrast_sel))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Studies found", int(summ["n_studies"]))
    with c2:
        st.metric("Meta log2FC", f"{summ['meta_log2FoldChange']:.3f}" if pd.notna(summ["meta_log2FoldChange"]) else "missing")
    with c3:
        st.metric("Meta p-value", f"{summ['meta_p']:.2e}" if pd.notna(summ["meta_p"]) else "missing")
    with c4:
        st.metric("Direction agreement", f"{100*summ['agreement']:.1f}%" if pd.notna(summ["agreement"]) else "missing")

    st.caption(f"Comparison / contrast shown here: **{contrast_sel}** (from the Bulk_Omics folder label).")

    fig_forest = _forest_plot_from_per_study(per, contrast_label=str(contrast_sel))
    if fig_forest is not None:
        _st_plotly(fig_forest)

    fig_sig = _scatter_logfc_vs_sig(per, contrast_label=str(contrast_sel))
    if fig_sig is not None:
        _st_plotly(fig_sig)

    st.markdown("#### Per-study table")
    _st_df(per_show, hide_index=True)

    st.caption(
        "Meta p-value uses a signed Stouffer combination of per-study p-values (padj if available, else pvalue), "
        "with direction from log2FoldChange. Forest plot CI is an approximation derived from p-values (Wald-style)."
    )
