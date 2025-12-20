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
  - loads all bulk DEG tables grouped by contrast folder (if bulk_data not provided)
  - normalises schemas and resolves duplicate gene symbols
  - supports gene-centric summary and meta-analysis across studies per contrast
  - provides "top genes" ranking per contrast (fast vectorised)
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
# Discovery + loading (only used if bulk_data not supplied from streamlit_app)
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
# Numerics: fast normal inverse CDF (vectorised Acklam-ish)
# -----------------------------------------------------------------------------

def _norm_ppf_vec(u: np.ndarray) -> np.ndarray:
    """
    Vectorised inverse normal CDF approximation.
    Input u in (0,1). Output z such that Phi(z)=u.
    """
    u = np.asarray(u, dtype=float)

    # Keep away from 0/1 to avoid log(0) + float cancellation at u ~ 1
    eps = 1e-15
    u = np.clip(u, eps, 1.0 - eps)

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
        q = np.sqrt(-2.0 * np.log(u[low]))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        z[low] = num / den

    if np.any(high):
        q = np.sqrt(-2.0 * np.log1p(-u[high]))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        z[high] = -(num / den)

    if np.any(mid):
        q = u[mid] - 0.5
        r = q*q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        z[mid] = num / den

    return z


def _signed_z_from_two_sided_p(p: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """
    p: two-sided p-values
    sign: sign source (e.g. logFC); >=0 => positive z, <0 => negative z
    """
    p = np.asarray(p, dtype=float)
    sign = np.asarray(sign, dtype=float)

    p = np.where(np.isfinite(p), p, np.nan)
    p = np.clip(p, 0.0, 1.0)

    u = 1.0 - (p / 2.0)
    z_abs = _norm_ppf_vec(u)

    s = np.where(sign >= 0.0, 1.0, -1.0)
    return s * z_abs


def _p_from_z_two_sided(z: float) -> float:
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return float("nan")
    return float(min(max(math.erfc(abs(float(z)) / math.sqrt(2.0)), 0.0), 1.0))


# -----------------------------------------------------------------------------
# Gene-centric summaries (single gene: OK to be simple)
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

    sig = "padj" if "padj" in out.columns else ("pvalue" if "pvalue" in out.columns else None)
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
    pos = float((signs > 0).mean())
    neg = float((signs < 0).mean())
    return float(max(pos, neg))


def meta_combine_stouffer(per_study: pd.DataFrame, *, weight_mode: str = "abs_z") -> Dict[str, float]:
    """
    Signed Stouffer meta for a single gene (few rows, scalar path is fine).
    Uses padj if present (and has any non-NA) else pvalue.
    """
    if per_study is None or per_study.empty:
        return {"meta_Z": float("nan"), "meta_p": float("nan")}

    df = per_study.copy()
    pcol = "padj" if ("padj" in df.columns and df["padj"].notna().any()) else ("pvalue" if "pvalue" in df.columns else None)
    if pcol is None or "log2FoldChange" not in df.columns:
        return {"meta_Z": float("nan"), "meta_p": float("nan")}

    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
    lfc = pd.to_numeric(df["log2FoldChange"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(p) & np.isfinite(lfc)
    if not np.any(ok):
        return {"meta_Z": float("nan"), "meta_p": float("nan")}

    z = _signed_z_from_two_sided_p(p[ok], lfc[ok])
    if weight_mode == "abs_z":
        w = np.maximum(1.0, np.abs(z))
    else:
        w = np.ones_like(z, dtype=float)

    Z = float(np.sum(w * z) / math.sqrt(float(np.sum(w**2))))
    return {"meta_Z": Z, "meta_p": _p_from_z_two_sided(Z)}


def meta_logfc_weighted(per_study: pd.DataFrame, *, weight_mode: str = "abs_z") -> float:
    if per_study is None or per_study.empty or "log2FoldChange" not in per_study.columns:
        return float("nan")

    df = per_study.copy()
    lfc = pd.to_numeric(df["log2FoldChange"], errors="coerce").to_numpy(dtype=float)

    pcol = "padj" if ("padj" in df.columns and df["padj"].notna().any()) else ("pvalue" if "pvalue" in df.columns else None)
    if pcol is None:
        return float(np.nanmean(lfc))

    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(lfc) & np.isfinite(p)
    if not np.any(ok):
        return float(np.nanmean(lfc))

    z = _signed_z_from_two_sided_p(p[ok], lfc[ok])
    w = np.ones_like(z, dtype=float)
    if weight_mode == "abs_z":
        w = np.maximum(1.0, np.abs(z))

    return float(np.sum(w * lfc[ok]) / np.sum(w))


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

    meta = meta_combine_stouffer(per, weight_mode="abs_z")
    meta_lfc = meta_logfc_weighted(per, weight_mode="abs_z")
    agree = direction_agreement(per)

    return {
        "per_study": per,
        "meta_log2FoldChange": meta_lfc,
        "meta_p": meta["meta_p"],
        "meta_Z": meta["meta_Z"],
        "agreement": agree,
        "n_studies": int(per["Study"].nunique()) if "Study" in per.columns else int(len(per)),
    }


# -----------------------------------------------------------------------------
# Fast top-genes: build one long table then groupby (vectorised)
# -----------------------------------------------------------------------------

def _build_long_for_contrast(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for study_id, df in (studies or {}).items():
        if df is None or df.empty:
            continue
        if "Symbol" not in df.columns or "log2FoldChange" not in df.columns:
            continue
        pcol = "padj" if ("padj" in df.columns and df["padj"].notna().any()) else ("pvalue" if "pvalue" in df.columns else None)
        if pcol is None:
            continue

        tmp = df[["Symbol", "log2FoldChange", pcol]].copy()
        tmp = tmp.rename(columns={pcol: "p"})
        tmp["Study"] = study_id
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    long = pd.concat(frames, ignore_index=True)
    long["Symbol"] = long["Symbol"].astype(str).map(_normalise_symbol)

    long["log2FoldChange"] = pd.to_numeric(long["log2FoldChange"], errors="coerce")
    long["p"] = pd.to_numeric(long["p"], errors="coerce")

    long = long.dropna(subset=["Symbol", "log2FoldChange", "p"])
    long = long.loc[long["Symbol"] != ""].copy()
    return long


def compute_contrast_summary(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Returns a gene-level summary dataframe with:
    Symbol, n_studies, meta_Z, meta_p, meta_log2FoldChange, agreement, evidence_score
    """
    long = _build_long_for_contrast(studies)
    if long.empty:
        return pd.DataFrame()

    # z + weights
    p = long["p"].to_numpy(dtype=float)
    lfc = long["log2FoldChange"].to_numpy(dtype=float)

    z = _signed_z_from_two_sided_p(p, lfc)
    w = np.maximum(1.0, np.abs(z))

    long["__z__"] = z
    long["__w__"] = w
    long["__wz__"] = w * z
    long["__w2__"] = w * w
    long["__wlfc__"] = w * lfc
    long["__pos__"] = (lfc > 0).astype(np.int8)
    long["__neg__"] = (lfc < 0).astype(np.int8)

    g = long.groupby("Symbol", sort=False)

    n_studies = g["Study"].nunique()
    num = g["__wz__"].sum()
    den = np.sqrt(g["__w2__"].sum())
    Z = num / den

    # meta p (scalar erfc per gene; fast enough for ~10^4 genes)
    Z_vals = Z.to_numpy(dtype=float)
    meta_p = np.array([_p_from_z_two_sided(float(x)) for x in Z_vals], dtype=float)
    meta_p = np.clip(meta_p, 1e-300, 1.0)

    meta_lfc = g["__wlfc__"].sum() / g["__w__"].sum()

    pos_frac = g["__pos__"].mean()
    neg_frac = g["__neg__"].mean()
    agreement = np.maximum(pos_frac, neg_frac)

    # Evidence score = (-log10(meta_p)) * |meta_logFC| * agreement
    score = (-np.log10(meta_p)) * np.abs(meta_lfc.to_numpy(dtype=float)) * agreement.to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Symbol": n_studies.index,
            "n_studies": n_studies.to_numpy(dtype=int),
            "meta_Z": Z_vals,
            "meta_p": meta_p,
            "meta_log2FoldChange": meta_lfc.to_numpy(dtype=float),
            "agreement": agreement.to_numpy(dtype=float),
            "evidence_score": score,
        }
    )

    out = out.sort_values("evidence_score", ascending=False, na_position="last")
    return out


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_bulk_omics_tab(query: str, *, bulk_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None) -> None:
    import streamlit as st
    import plotly.graph_objects as go

    # Use preloaded data if supplied (fastest, avoids re-reading disk)
    if bulk_data is None:
        root = _find_bulk_omics_dir()
        if root is None or not root.exists():
            st.warning("No Bulk Omics folder found. Expected: meta-liver-data/Bulk_Omics/")
            return
        data = load_bulk_omics()
        if not data:
            st.warning("Bulk Omics folder found, but no valid TSV/CSV/Parquet DEG tables could be loaded.")
            st.caption(f"Looking in: {root}")
            return
        st.caption(f"Bulk Omics root: {root}")
    else:
        data = bulk_data
        if not data:
            st.warning("Bulk-omics not available.")
            return
        st.caption("Bulk Omics source: preloaded tables (from robust_data_loader)")

    st.markdown("### Dataset availability")
    avail_rows = [{"Contrast": c, "Study tables": len(v)} for c, v in (data or {}).items()]
    st.dataframe(pd.DataFrame(avail_rows), width="stretch", hide_index=True)

    st.markdown("---")

    contrasts = sorted(list(data.keys()))
    if not contrasts:
        st.warning("No bulk contrasts found.")
        return

    contrast_sel = st.selectbox("Contrast", options=contrasts, index=0)
    studies = data.get(contrast_sel, {})
    if not studies:
        st.warning("No study tables found for this contrast.")
        return

    view = st.radio("Choose view", ["Gene summary", "Top genes"], index=0, horizontal=True, key="bulk_view")

    # Cache per-contrast summary inside session_state (fast and avoids hashing huge dicts)
    def _fingerprint(sts: Dict[str, pd.DataFrame]) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted((k, int(v.shape[0])) for k, v in (sts or {}).items() if isinstance(v, pd.DataFrame)))

    cache = st.session_state.setdefault("_bulk_omics_cache", {})
    fp = _fingerprint(studies)
    sum_key = f"summary::{contrast_sel}"

    if sum_key in cache and cache[sum_key].get("fp") == fp:
        summary_df = cache[sum_key]["df"]
    else:
        with st.spinner("Preparing bulk meta-summary for this contrast…"):
            summary_df = compute_contrast_summary(studies)
        cache[sum_key] = {"fp": fp, "df": summary_df}

    if view == "Top genes":
        st.markdown("### Top genes (meta + consistency)")

        # Use a form so we do not recompute on every tiny widget change
        with st.form(key=f"bulk_top_form::{contrast_sel}"):
            min_st = st.number_input("Minimum studies per gene", min_value=1, max_value=20, value=2, step=1)
            max_g = st.number_input("Max genes to show", min_value=50, max_value=2000, value=200, step=50)
            submitted = st.form_submit_button("Compute")

        # Persist last result so the page does not look empty
        last_key = f"top_last::{contrast_sel}"
        if submitted or (last_key in cache):
            if submitted:
                if summary_df is None or summary_df.empty:
                    st.info("No valid genes were found for meta-analysis in this contrast (check columns).")
                    cache[last_key] = pd.DataFrame()
                else:
                    top = summary_df.loc[summary_df["n_studies"] >= int(min_st)].copy()
                    top = top.head(int(max_g)).copy()
                    cache[last_key] = top
            else:
                top = cache[last_key]

            top = cache.get(last_key, pd.DataFrame())
            if top is None or top.empty:
                st.info("No genes met the minimum study requirement.")
                return

            # Simple volcano-style plot (meta logFC vs -log10(meta_p)), sized by agreement
            y = -np.log10(np.clip(top["meta_p"].to_numpy(dtype=float), 1e-300, 1.0))
            x = top["meta_log2FoldChange"].to_numpy(dtype=float)
            size = 6.0 + 10.0 * np.clip(top["agreement"].to_numpy(dtype=float), 0.0, 1.0)

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=size, opacity=0.75),
                    text=top["Symbol"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "meta_log2FC=%{x:.3f}<br>"
                        "-log10(meta_p)=%{y:.2f}<br>"
                        "<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                height=360,
                xaxis_title="Meta log2FoldChange",
                yaxis_title="-log10(meta p-value)",
                title=f"Top genes: {contrast_sel}",
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(top, width="stretch", hide_index=True)
            return

        st.info("Adjust settings and click **Compute** to generate the ranked table and plot.")
        return

    # Gene summary
    st.markdown("### Gene summary")
    gene = str(query).strip()
    if not gene:
        st.info("Enter a gene symbol in the search box above.")
        return

    summ = bulk_gene_summary(studies, gene)
    per = summ["per_study"]

    if per.empty:
        st.warning("Gene not found in any study tables for this contrast.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Studies found", int(summ["n_studies"]))
    with c2:
        st.metric("Meta log2FC", f"{summ['meta_log2FoldChange']:.3f}" if pd.notna(summ["meta_log2FoldChange"]) else "missing")
    with c3:
        st.metric("Meta p-value", f"{summ['meta_p']:.2e}" if pd.notna(summ["meta_p"]) else "missing")
    with c4:
        st.metric("Direction agreement", f"{100*summ['agreement']:.1f}%" if pd.notna(summ["agreement"]) else "missing")

    st.dataframe(per, width="stretch", hide_index=True)

    st.caption(
        "Meta p-value uses a signed Stouffer combination of per-study p-values (padj if available, else pvalue), "
        "with direction from log2FoldChange. Meta log2FC is a weighted average (weights ~ |z|)."
    )
