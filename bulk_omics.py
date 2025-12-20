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
  - provides "top genes" ranking per contrast
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
    # tolerant to naming because robust loader normalises
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
# Meta-analysis
# -----------------------------------------------------------------------------

def _p_to_z_two_sided(p: float, sign: float) -> float:
    """
    Convert a two-sided p-value to a signed z score.

    Fixes:
    - Handles p==0 (common in exported tables due to underflow) by treating it as extremely small
    - Avoids u==1.0 leading to log(0) in the tail approximation by clipping u away from {0,1}
    - Uses log1p(-u) in the upper tail for stability
    """
    if p is None:
        return float("nan")

    try:
        p = float(p)
    except Exception:
        return float("nan")

    if math.isnan(p) or p > 1.0:
        return float("nan")

    # Treat p==0 (or negative due to file quirks) as extremely small rather than dropping.
    if p <= 0.0:
        p = 0.0

    # Convert two-sided p to upper-tail probability u = 1 - p/2.
    # For extremely small p, 1 - p/2 can round to exactly 1.0 in float arithmetic.
    u = 1.0 - (p / 2.0)

    # Clip u away from {0,1} to avoid log(0). Use an eps that matters at float precision near 1.0.
    eps = 1e-15
    if not (0.0 < u < 1.0):
        u = 1.0 - eps
    else:
        u = min(max(u, eps), 1.0 - eps)

    def _norm_ppf(u_: float) -> float:
        # Defensive clipping
        u_ = min(max(float(u_), eps), 1.0 - eps)

        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]

        plow = 0.02425
        phigh = 1.0 - plow

        if u_ < plow:
            q = math.sqrt(-2.0 * math.log(u_))
            return (
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            )

        if u_ > phigh:
            # Use log1p for stability near 1.0
            q = math.sqrt(-2.0 * math.log1p(-u_))
            return -(
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            )

        q = u_ - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    z = _norm_ppf(u)
    sgn = 1.0 if float(sign) >= 0 else -1.0
    return sgn * float(z)


def _z_to_p_two_sided(z: float) -> float:
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return float("nan")
    z = abs(float(z))
    p = math.erfc(z / math.sqrt(2.0))
    return float(min(max(p, 0.0), 1.0))


def meta_combine_stouffer(
    per_study: pd.DataFrame,
    *,
    weight_mode: str = "equal",
) -> Dict[str, float]:
    """
    Signed Stouffer meta across studies.

    per_study columns expected: log2FoldChange, pvalue (or padj fallback).
    """
    if per_study is None or per_study.empty:
        return {"meta_Z": float("nan"), "meta_p": float("nan")}

    df = per_study.copy()

    pcol = "padj" if "padj" in df.columns and df["padj"].notna().any() else "pvalue"
    if pcol not in df.columns:
        return {"meta_Z": float("nan"), "meta_p": float("nan")}

    z_list: List[float] = []
    w_list: List[float] = []

    for _, r in df.iterrows():
        lfc = r.get("log2FoldChange", float("nan"))
        p = r.get(pcol, float("nan"))
        if pd.isna(lfc) or pd.isna(p):
            continue

        z = _p_to_z_two_sided(float(p), sign=float(lfc))
        if math.isnan(z):
            continue

        if weight_mode == "equal":
            w = 1.0
        elif weight_mode == "abs_z":
            w = max(1.0, abs(z))
        else:
            w = 1.0

        z_list.append(float(z))
        w_list.append(float(w))

    if not z_list:
        return {"meta_Z": float("nan"), "meta_p": float("nan")}

    z_arr = np.asarray(z_list, dtype=float)
    w_arr = np.asarray(w_list, dtype=float)

    Z = float(np.sum(w_arr * z_arr) / math.sqrt(float(np.sum(w_arr ** 2))))
    p_meta = _z_to_p_two_sided(Z)
    return {"meta_Z": float(Z), "meta_p": float(p_meta)}


def meta_logfc_weighted(per_study: pd.DataFrame, *, weight_mode: str = "abs_z") -> float:
    """
    Compute a meta logFC as weighted average of per-study logFC.
    Default weights follow abs(z) so stronger evidence contributes more.
    """
    if per_study is None or per_study.empty or "log2FoldChange" not in per_study.columns:
        return float("nan")

    df = per_study.copy()

    pcol = (
        "padj"
        if "padj" in df.columns and df["padj"].notna().any()
        else ("pvalue" if "pvalue" in df.columns else None)
    )
    if pcol is None:
        return float(pd.to_numeric(df["log2FoldChange"], errors="coerce").mean())

    vals: List[float] = []
    ws: List[float] = []

    for _, r in df.iterrows():
        lfc = r.get("log2FoldChange", float("nan"))
        p = r.get(pcol, float("nan"))
        if pd.isna(lfc):
            continue

        w = 1.0
        if weight_mode == "abs_z" and pd.notna(p):
            z = _p_to_z_two_sided(float(p), sign=float(lfc))
            if not math.isnan(z):
                w = max(1.0, abs(z))

        vals.append(float(lfc))
        ws.append(float(w))

    if not vals:
        return float("nan")

    v = np.asarray(vals, dtype=float)
    w = np.asarray(ws, dtype=float)
    denom = float(np.sum(w))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * v) / denom)


# -----------------------------------------------------------------------------
# Gene-centric summaries
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
# Top genes per contrast
# -----------------------------------------------------------------------------

def list_all_symbols(studies: Dict[str, pd.DataFrame]) -> List[str]:
    syms = set()
    for _, df in (studies or {}).items():
        if df is None or df.empty or "Symbol" not in df.columns:
            continue
        vals = df["Symbol"].astype(str).map(_normalise_symbol)
        syms.update([v for v in vals.tolist() if v])
    return sorted(syms)


def top_genes_for_contrast(
    studies: Dict[str, pd.DataFrame],
    *,
    min_studies: int = 2,
    max_genes: int = 200,
) -> pd.DataFrame:
    """
    Compute a ranked table of genes for one contrast.
    Evidence score = (-log10(meta_p)) * |meta_logFC| * agreement
    """
    if studies is None or not studies:
        return pd.DataFrame()

    all_syms = list_all_symbols(studies)
    if not all_syms:
        return pd.DataFrame()

    rows = []
    for g in all_syms:
        summ = bulk_gene_summary(studies, g)
        if summ["n_studies"] < int(min_studies):
            continue

        meta_p = summ["meta_p"]
        meta_lfc = summ["meta_log2FoldChange"]
        agree = summ["agreement"]

        if meta_p is None or (isinstance(meta_p, float) and (math.isnan(meta_p) or meta_p <= 0)):
            score = float("nan")
        else:
            a = float(agree) if pd.notna(agree) else 0.0
            score = float((-math.log10(float(meta_p))) * abs(float(meta_lfc)) * a)

        rows.append(
            {
                "Symbol": g,
                "n_studies": summ["n_studies"],
                "meta_log2FoldChange": meta_lfc,
                "meta_p": meta_p,
                "agreement": agree,
                "evidence_score": score,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("evidence_score", ascending=False, na_position="last").head(int(max_genes))
    return out


# -----------------------------------------------------------------------------
# Streamlit UI entry point
# -----------------------------------------------------------------------------

def render_bulk_omics_tab(
    query: str,
    *,
    bulk_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
) -> None:
    import streamlit as st

    root = _find_bulk_omics_dir()
    if root is None or not root.exists():
        st.warning("No Bulk Omics folder found. Expected: meta-liver-data/Bulk_Omics/")
        return

    data = bulk_data if isinstance(bulk_data, dict) and len(bulk_data) > 0 else load_bulk_omics()
    if not data:
        st.warning("Bulk Omics folder found, but no valid TSV/CSV/Parquet DEG tables could be loaded.")
        st.caption(f"Looking in: {root}")
        return

    st.markdown("### Dataset availability")
    st.caption(f"Bulk Omics root: {root}")

    avail_rows = []
    for contrast, studies in data.items():
        avail_rows.append({"Contrast": contrast, "Study tables": len(studies)})
    st.dataframe(pd.DataFrame(avail_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    contrasts = sorted(list(data.keys()))
    contrast_sel = st.selectbox("Contrast", options=contrasts, index=0)

    studies = data.get(contrast_sel, {})
    if not studies:
        st.warning("No study tables found for this contrast.")
        return

    view = st.radio("Choose view", ["Gene summary", "Top genes"], index=0, horizontal=True)

    if view == "Top genes":
        st.markdown("### Top genes (meta + consistency)")
        min_st = st.number_input("Minimum studies per gene", min_value=1, max_value=20, value=2, step=1)
        max_g = st.number_input("Max genes to show", min_value=50, max_value=2000, value=200, step=50)

        top = top_genes_for_contrast(studies, min_studies=int(min_st), max_genes=int(max_g))
        if top.empty:
            st.info("No genes met the minimum study requirement.")
            return

        st.dataframe(top, use_container_width=True, hide_index=True)
        return

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

    st.dataframe(per, use_container_width=True, hide_index=True)

    st.caption(
        "Meta p-value uses a signed Stouffer combination of per-study p-values (padj if available, else pvalue) "
        "with direction from log2FoldChange. Meta log2FC is a weighted average (weights ~ |z|)."
    )
