# gene_screener.py
"""
Meta Liver - Gene Screener (backend)

This module is intentionally Streamlit-free. It provides a flexible gene screening
engine that can apply optional filters across:
- Single-omics (evidence score, agreement, AUROC)
- Knowledge graph (cluster membership, composite centrality percentile)
- WGCNA (module membership, module–trait sign, druggability)
- In vitro iHeps model DEGs (per-contrast direction/significance consistency)
- Bulk omics DEGs (per-contrast direction/significance consistency)

Designed to be called from streamlit_app.py, which owns caching + UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def normalise_symbol(x: Any) -> str:
    s = str(x).strip().upper()
    if s in ("", "NAN", "NONE"):
        return ""
    return s


def _is_nanlike(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except Exception:
        return x is None


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _sign_of_lfc(lfc: Any) -> int:
    v = _to_float(lfc)
    if np.isnan(v) or v == 0:
        return 0
    return 1 if v > 0 else -1


def _pick_col_ci(cols: List[str], candidates: List[str]) -> Optional[str]:
    m = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        c2 = cand.lower()
        if c2 in m:
            return m[c2]
    return None


def _infer_gene_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    cands = [
        "Symbol",
        "symbol",
        "Gene",
        "gene",
        "Gene name",
        "Gene_name",
        "gene_name",
        "hgnc_symbol",
        "HGNC",
        "Name",  # KG nodes sometimes
    ]
    gc = _pick_col_ci(cols, cands)
    if gc is not None:
        return gc

    # unnamed first column (DESeq2 rownames saved)
    for c in cols:
        c0 = str(c).strip().lower()
        if c0 in ("unnamed: 0", "unnamed:0", ""):
            return c

    # index as genes
    if not isinstance(df.index, pd.RangeIndex):
        return "__INDEX__"

    return None


def _infer_lfc_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    return _pick_col_ci(
        list(df.columns),
        ["log2FoldChange", "logFC", "log2FC", "log2_fc", "lfc", "LFC"],
    )


def _infer_auc_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    return _pick_col_ci(
        list(df.columns),
        [
            "AUROC",
            "AUC",
            "auc",
            "auroc",
            "roc_auc",
            "ROC_AUC",
            "AUC_mean",
            "AUROC_mean",
        ],
    )


def _infer_padj_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    return _pick_col_ci(list(df.columns), ["padj", "FDR", "qvalue", "q_value", "adj_pval", "adj_pvalue"])


def _read_table(path: Path) -> pd.DataFrame:
    if path is None or (not path.exists()):
        return pd.DataFrame()
    suf = path.suffix.lower()
    try:
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf == ".csv":
            return pd.read_csv(path)
        if suf in (".tsv", ".txt"):
            try:
                return pd.read_csv(path, sep="\t")
            except Exception:
                return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _row_for_gene(df: pd.DataFrame, gene: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    g = normalise_symbol(gene)
    if not g:
        return None

    gene_col = _infer_gene_col(df)
    if gene_col is None:
        return None

    if gene_col == "__INDEX__":
        try:
            idx = df.index.astype(str).map(normalise_symbol)
            m = np.where(idx.values == g)[0]
            if len(m) == 0:
                return None
            return df.iloc[int(m[0])]
        except Exception:
            return None

    try:
        s = df[gene_col].astype(str).map(normalise_symbol)
        m = np.where(s.values == g)[0]
        if len(m) == 0:
            return None
        return df.iloc[int(m[0])]
    except Exception:
        return None


# -----------------------------------------------------------------------------
# In vitro indexing
# -----------------------------------------------------------------------------
def parse_invitro_deg_files(file_paths: Iterable[Any]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Returns a nested index: contrast -> line -> [paths]
    Contrast and line are inferred from filename tokens, but tolerant:
      processed_degs_<LINE>_<CONTRAST>.parquet
      <CONTRAST>_<LINE>.parquet
    """
    out: Dict[str, Dict[str, List[Path]]] = {}
    for fp in (file_paths or []):
        try:
            p = Path(fp)
        except Exception:
            continue
        if not p.exists() or not p.is_file():
            continue
        if p.suffix.lower() not in (".csv", ".parquet", ".tsv", ".txt"):
            continue

        stem = p.stem.strip()
        if not stem:
            continue

        toks = stem.split("_")
        line = None
        contrast = None

        if stem.startswith("processed_degs_") and len(toks) >= 4:
            # processed_degs_1b_OAPAvsHCM
            line = toks[2]
            contrast = "_".join(toks[3:])
        elif len(toks) >= 2:
            # OAPAvsHCM_1b
            # Contrast could itself contain underscores; assume last token is line if short
            maybe_line = toks[-1]
            if len(maybe_line) <= 4:
                line = maybe_line
                contrast = "_".join(toks[:-1])
            else:
                # fallback: treat first token as contrast, second as line
                contrast = toks[0]
                line = toks[1]
        else:
            continue

        line = str(line).strip()
        contrast = str(contrast).strip()
        if not line or not contrast:
            continue

        out.setdefault(contrast, {}).setdefault(line, []).append(p)

    return out


def invitro_gene_summary(
    gene: str,
    invitro_index: Dict[str, Dict[str, List[Path]]],
    selected_contrasts: List[str],
    selected_lines: List[str],
    padj_thr: Optional[float] = None,
    require_significant: bool = False,
    require_consistent_direction: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates a gene across selected in vitro contrasts/lines.
    If no contrasts selected, returns (True, {}) meaning "no constraint".
    """
    g = normalise_symbol(gene)
    if not selected_contrasts:
        return True, {}

    details: Dict[str, Any] = {"gene": g, "hits": []}
    ok_any_hit = False
    dirs: List[int] = []

    lines_req = [str(x).strip() for x in (selected_lines or []) if str(x).strip()]
    contrasts_req = [str(x).strip() for x in (selected_contrasts or []) if str(x).strip()]

    for c in contrasts_req:
        lines_map = invitro_index.get(c, {}) if invitro_index else {}
        if not lines_map:
            details["hits"].append({"contrast": c, "status": "missing_files"})
            continue

        usable_lines = lines_req if lines_req else sorted(lines_map.keys())
        c_any = False

        for ln in usable_lines:
            paths = lines_map.get(ln, [])
            if not paths:
                continue

            # If multiple files exist, accept first successful load with a hit
            hit_row = None
            hit_path = None
            for p in paths:
                df = _read_table(p)
                if df.empty:
                    continue
                row = _row_for_gene(df, g)
                if row is not None:
                    hit_row = row
                    hit_path = p
                    break

            if hit_row is None:
                continue

            lfc_col = _infer_lfc_col(pd.DataFrame([hit_row]))
            padj_col = _infer_padj_col(pd.DataFrame([hit_row]))

            lfc = hit_row.get(lfc_col) if lfc_col else hit_row.get("log2FoldChange", np.nan)
            padj = hit_row.get(padj_col) if padj_col else hit_row.get("padj", np.nan)

            sgn = _sign_of_lfc(lfc)
            sig_ok = True
            if padj_thr is not None and (not _is_nanlike(padj)):
                sig_ok = _to_float(padj) <= float(padj_thr)

            passed = True
            if require_significant and (padj_thr is not None):
                passed = bool(sig_ok)

            details["hits"].append(
                {
                    "contrast": c,
                    "line": ln,
                    "log2FC": _to_float(lfc),
                    "padj": _to_float(padj),
                    "direction": "UP" if sgn > 0 else "DOWN" if sgn < 0 else "FLAT/NA",
                    "file": str(hit_path) if hit_path else "",
                    "pass": bool(passed),
                }
            )

            c_any = True
            ok_any_hit = True
            if passed and sgn != 0:
                dirs.append(sgn)

        if not c_any:
            details["hits"].append({"contrast": c, "status": "no_gene_hit"})

    if not ok_any_hit:
        return False, details

    if require_consistent_direction and len(dirs) >= 2:
        if not all(d == dirs[0] for d in dirs):
            return False, details

    # If require_significant is on, ensure at least one passing row exists for each contrast
    if require_significant and (padj_thr is not None):
        passed_by_contrast: Dict[str, bool] = {}
        for rec in details["hits"]:
            if isinstance(rec, dict) and ("contrast" in rec) and ("pass" in rec):
                passed_by_contrast[str(rec["contrast"])] = passed_by_contrast.get(str(rec["contrast"]), False) or bool(rec["pass"])
        for c in contrasts_req:
            if not passed_by_contrast.get(c, False):
                return False, details

    return True, details


# -----------------------------------------------------------------------------
# Bulk indexing / checks
# -----------------------------------------------------------------------------
def bulk_gene_summary(
    gene: str,
    bulk_omics_data: Dict[str, Any],
    selected_contrasts: List[str],
    padj_thr: Optional[float] = None,
    require_significant: bool = False,
    require_consistent_direction: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    bulk_omics_data expected as: {contrast_group: {study_name: df}} or similar.
    If no contrasts selected, returns (True, {}).
    """
    g = normalise_symbol(gene)
    if not selected_contrasts:
        return True, {}

    details: Dict[str, Any] = {"gene": g, "hits": []}
    dirs: List[int] = []
    ok_any = False

    for contrast in selected_contrasts:
        block = (bulk_omics_data or {}).get(contrast, None)
        if block is None:
            details["hits"].append({"contrast": contrast, "status": "missing_contrast"})
            continue

        # accept either dict-of-dfs or list-of-dfs
        if isinstance(block, dict):
            items = list(block.items())
        elif isinstance(block, list):
            items = [(f"table_{i+1}", x) for i, x in enumerate(block)]
        else:
            items = [("table", block)]

        c_any = False
        for study, df in items:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue

            row = _row_for_gene(df, g)
            if row is None:
                continue

            # prefer already-normalised columns
            lfc = row.get("log2FoldChange", np.nan)
            padj = row.get("padj", np.nan)
            if _is_nanlike(lfc):
                lfc_col = _infer_lfc_col(df)
                if lfc_col:
                    lfc = row.get(lfc_col, np.nan)
            if _is_nanlike(padj):
                padj_col = _infer_padj_col(df)
                if padj_col:
                    padj = row.get(padj_col, np.nan)

            sgn = _sign_of_lfc(lfc)
            sig_ok = True
            if padj_thr is not None and (not _is_nanlike(padj)):
                sig_ok = _to_float(padj) <= float(padj_thr)

            passed = True
            if require_significant and (padj_thr is not None):
                passed = bool(sig_ok)

            details["hits"].append(
                {
                    "contrast": contrast,
                    "study": str(study),
                    "log2FC": _to_float(lfc),
                    "padj": _to_float(padj),
                    "direction": "UP" if sgn > 0 else "DOWN" if sgn < 0 else "FLAT/NA",
                    "pass": bool(passed),
                }
            )
            c_any = True
            ok_any = True
            if passed and sgn != 0:
                dirs.append(sgn)

        if not c_any:
            details["hits"].append({"contrast": contrast, "status": "no_gene_hit"})

    if not ok_any:
        return False, details

    if require_consistent_direction and len(dirs) >= 2:
        if not all(d == dirs[0] for d in dirs):
            return False, details

    if require_significant and (padj_thr is not None):
        passed_by_contrast: Dict[str, bool] = {}
        for rec in details["hits"]:
            if isinstance(rec, dict) and ("contrast" in rec) and ("pass" in rec):
                passed_by_contrast[str(rec["contrast"])] = passed_by_contrast.get(str(rec["contrast"]), False) or bool(rec["pass"])
        for c in selected_contrasts:
            if not passed_by_contrast.get(c, False):
                return False, details

    return True, details


# -----------------------------------------------------------------------------
# WGCNA helpers
# -----------------------------------------------------------------------------
def build_wgcna_gene_to_module(wgcna_module_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Best-effort extraction of gene->module mapping from wgcna_module_data
    without assuming a single canonical structure.
    """
    out: Dict[str, str] = {}
    if not wgcna_module_data:
        return out

    # common: dict of module_name -> df (with "Gene" column)
    for k, v in (wgcna_module_data or {}).items():
        if isinstance(v, pd.DataFrame) and (("Gene" in v.columns) or ("gene" in v.columns)):
            gene_col = "Gene" if "Gene" in v.columns else "gene"
            for g in v[gene_col].astype(str).map(normalise_symbol).tolist():
                if g:
                    out[g] = str(k)
        elif isinstance(v, dict):
            # nested: might contain module_genes df
            for kk, vv in v.items():
                if isinstance(vv, pd.DataFrame) and (("Gene" in vv.columns) or ("gene" in vv.columns)):
                    gene_col = "Gene" if "Gene" in vv.columns else "gene"
                    mod = str(k)
                    for g in vv[gene_col].astype(str).map(normalise_symbol).tolist():
                        if g:
                            out[g] = mod

    # alternative: a single DF with columns ["Gene","module"]
    for v in (wgcna_module_data or {}).values():
        if isinstance(v, pd.DataFrame):
            cols = set(map(str, v.columns))
            if ("Gene" in cols or "gene" in cols) and ("module" in cols or "Module" in cols):
                gc = "Gene" if "Gene" in v.columns else "gene"
                mc = "module" if "module" in v.columns else "Module"
                for _, r in v[[gc, mc]].iterrows():
                    g = normalise_symbol(r[gc])
                    m = str(r[mc]).strip()
                    if g and m:
                        out[g] = m

    return out


def _match_module_row(df: pd.DataFrame, module: str) -> Optional[str]:
    if df is None or df.empty:
        return None
    m = str(module).strip()
    candidates = [m, m.lower(), m.upper(), f"ME{m}", f"ME{m.lower()}", f"ME{m.upper()}"]
    for c in candidates:
        if c in df.index:
            return c
    idx_lower = {str(i).lower(): i for i in df.index}
    for c in candidates:
        key = str(c).lower()
        if key in idx_lower:
            return idx_lower[key]
    return None


def wgcna_gene_summary(
    gene: str,
    gene_to_module: Dict[str, str],
    wgcna_cor: pd.DataFrame,
    trait: Optional[str],
    require_module_in: List[str],
    require_trait_sign: str,  # "any"|"positive"|"negative"
    require_drug_target: bool,
    gene_to_drugs: Dict[str, Any],
    min_abs_corr: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    If no WGCNA constraints are active (no module list, sign=any, require_drug_target=False),
    returns (True, {}).
    """
    g = normalise_symbol(gene)
    constraints_active = bool(require_module_in) or (require_trait_sign in ("positive", "negative")) or bool(require_drug_target) or (min_abs_corr is not None)
    if not constraints_active:
        return True, {}

    mod = gene_to_module.get(g, "")
    has_mod = bool(mod)
    has_drug = False
    try:
        has_drug = bool(gene_to_drugs.get(g, []))
    except Exception:
        has_drug = False

    trait_val = np.nan
    trait_ok = True
    if trait and (wgcna_cor is not None) and (not wgcna_cor.empty) and has_mod:
        row_key = _match_module_row(wgcna_cor, mod)
        if row_key is not None and trait in wgcna_cor.columns:
            trait_val = _to_float(wgcna_cor.loc[row_key, trait])

        if require_trait_sign == "positive":
            trait_ok = (not np.isnan(trait_val)) and trait_val > 0
        elif require_trait_sign == "negative":
            trait_ok = (not np.isnan(trait_val)) and trait_val < 0

        if (min_abs_corr is not None) and (not np.isnan(trait_val)):
            trait_ok = trait_ok and (abs(trait_val) >= float(min_abs_corr))
        elif (min_abs_corr is not None) and np.isnan(trait_val):
            trait_ok = False

    module_ok = True
    if require_module_in:
        module_ok = has_mod and (str(mod) in set(map(str, require_module_in)))

    drug_ok = True
    if require_drug_target:
        drug_ok = has_drug

    passed = bool(module_ok and drug_ok and trait_ok)

    details = {
        "gene": g,
        "module": mod if mod else None,
        "has_module": bool(has_mod),
        "trait": trait,
        "trait_corr": (None if np.isnan(trait_val) else float(trait_val)),
        "has_drug_target": bool(has_drug),
    }
    return passed, details


# -----------------------------------------------------------------------------
# KG checks
# -----------------------------------------------------------------------------
def kg_gene_summary(
    gene: str,
    kg_mod: Any,
    kg_data: Dict[str, Any],
    require_in_cluster: bool,
    min_composite_percentile: Optional[float],
) -> Tuple[bool, Dict[str, Any]]:
    """
    If KG constraints are inactive, returns (True, {}).
    """
    g = normalise_symbol(gene)
    constraints_active = bool(require_in_cluster) or (min_composite_percentile is not None)
    if not constraints_active:
        return True, {}

    info = {}
    try:
        if kg_mod is not None and hasattr(kg_mod, "get_gene_kg_info"):
            info = kg_mod.get_gene_kg_info(g, kg_data) or {}
    except Exception:
        info = {}

    cluster = info.get("cluster", None)
    comp_pct = info.get("composite_percentile", None)
    comp_pct_f = np.nan
    try:
        comp_pct_f = float(comp_pct) if comp_pct is not None else np.nan
    except Exception:
        comp_pct_f = np.nan

    in_cluster_ok = True
    if require_in_cluster:
        in_cluster_ok = (cluster is not None) and (str(cluster).strip() != "") and (str(cluster).lower() != "nan")

    comp_ok = True
    if min_composite_percentile is not None:
        comp_ok = (not np.isnan(comp_pct_f)) and (comp_pct_f >= float(min_composite_percentile))

    passed = bool(in_cluster_ok and comp_ok)

    details = {
        "gene": g,
        "found": bool(info),
        "cluster": cluster if cluster is not None else None,
        "composite_percentile": (None if np.isnan(comp_pct_f) else float(comp_pct_f)),
        "pagerank_percentile": info.get("pagerank_percentile", None),
        "bet_percentile": info.get("bet_percentile", None),
        "eigen_percentile": info.get("eigen_percentile", None),
    }
    return passed, details


# -----------------------------------------------------------------------------
# Single-omics checks (delegate to soa when possible)
# -----------------------------------------------------------------------------
def single_omics_gene_summary(
    gene: str,
    soa: Any,
    single_omics_data: Dict[str, Any],
    min_evidence_score: Optional[float],
    min_direction_agreement: Optional[float],
    min_auc_disc: Optional[float],
    min_n_auc: Optional[int],
) -> Tuple[bool, Dict[str, Any]]:
    """
    If single-omics constraints are inactive, returns (True, {}).
    """
    constraints_active = (min_evidence_score is not None) or (min_direction_agreement is not None) or (min_auc_disc is not None) or (min_n_auc is not None)
    if not constraints_active:
        return True, {}

    g = normalise_symbol(gene)

    score = None
    try:
        if soa is not None and hasattr(soa, "compute_consistency_score"):
            score = soa.compute_consistency_score(g, single_omics_data)
    except Exception:
        score = None

    if not isinstance(score, dict) or not score:
        return False, {"gene": g, "found": False}

    ev = _to_float(score.get("evidence_score", np.nan))
    agr = _to_float(score.get("direction_agreement", np.nan))
    aucd = _to_float(score.get("auc_median_discriminative", np.nan))
    n_auc = score.get("n_auc", None)
    try:
        n_auc_i = int(n_auc) if n_auc is not None and str(n_auc) != "nan" else 0
    except Exception:
        n_auc_i = 0

    ok = True
    if min_evidence_score is not None:
        ok = ok and (not np.isnan(ev)) and (ev >= float(min_evidence_score))
    if min_direction_agreement is not None:
        ok = ok and (not np.isnan(agr)) and (agr >= float(min_direction_agreement))
    if min_auc_disc is not None:
        ok = ok and (not np.isnan(aucd)) and (aucd >= float(min_auc_disc))
    if min_n_auc is not None:
        ok = ok and (n_auc_i >= int(min_n_auc))

    details = {
        "gene": g,
        "found": True,
        "evidence_score": (None if np.isnan(ev) else float(ev)),
        "direction_agreement": (None if np.isnan(agr) else float(agr)),
        "auc_median_discriminative": (None if np.isnan(aucd) else float(aucd)),
        "n_auc": int(n_auc_i),
        "found_count": score.get("found_count", None),
        "auc_median": score.get("auc_median", None),
        "auc_median_oriented": score.get("auc_median_oriented", None),
        "interpretation": score.get("interpretation", None),
    }
    return bool(ok), details


# -----------------------------------------------------------------------------
# Candidate universe
# -----------------------------------------------------------------------------
def collect_candidates(
    *,
    sources: List[str],
    custom_genes_text: str,
    single_omics_data: Dict[str, Any],
    kg_data: Dict[str, Any],
    gene_to_module: Dict[str, str],
    invitro_index: Dict[str, Dict[str, List[Path]]],
    bulk_omics_data: Dict[str, Any],
) -> List[str]:
    """
    sources supports: ["Custom", "WGCNA", "Knowledge Graph", "In vitro", "Bulk Omics", "Single-Omics"]
    """
    genes: set[str] = set()

    src = set([str(s).strip().lower() for s in (sources or [])])

    if "custom" in src:
        txt = (custom_genes_text or "").replace(",", " ").replace(";", " ")
        for tok in txt.split():
            g = normalise_symbol(tok)
            if g:
                genes.add(g)

    if "wgcna" in src:
        for g, _m in (gene_to_module or {}).items():
            gg = normalise_symbol(g)
            if gg:
                genes.add(gg)

    if "knowledge graph" in src or "knowledge_graph" in src or "kg" in src:
        # Best effort: treat any KG table rows with a "Name" column as potential symbols, then filter to uppercase tokens.
        for df in (kg_data or {}).values():
            if isinstance(df, pd.DataFrame) and (not df.empty) and ("Name" in df.columns):
                vals = df["Name"].astype(str).map(normalise_symbol)
                for v in vals.tolist():
                    if v and (v.isalnum() or (v.replace("-", "").isalnum())):
                        genes.add(v)

    if "in vitro" in src or "invitro" in src:
        for contrast, lines in (invitro_index or {}).items():
            for ln, paths in (lines or {}).items():
                for p in paths:
                    df = _read_table(p)
                    if df.empty:
                        continue
                    gc = _infer_gene_col(df)
                    if gc == "__INDEX__":
                        vals = df.index.astype(str).map(normalise_symbol)
                    elif gc is None:
                        continue
                    else:
                        vals = df[gc].astype(str).map(normalise_symbol)
                    for v in vals.tolist():
                        if v:
                            genes.add(v)

    if "bulk omics" in src or "bulk" in src:
        for contrast, block in (bulk_omics_data or {}).items():
            if isinstance(block, dict):
                dfs = list(block.values())
            elif isinstance(block, list):
                dfs = [x for x in block]
            else:
                dfs = [block]
            for df in dfs:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                gc = _infer_gene_col(df)
                if gc == "__INDEX__":
                    vals = df.index.astype(str).map(normalise_symbol)
                elif gc is None:
                    continue
                else:
                    vals = df[gc].astype(str).map(normalise_symbol)
                for v in vals.tolist():
                    if v:
                        genes.add(v)

    if "single-omics" in src or "single omics" in src or "single_omics" in src:
        for df in (single_omics_data or {}).values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            gc = _infer_gene_col(df)
            if gc is None:
                continue
            if gc == "__INDEX__":
                vals = df.index.astype(str).map(normalise_symbol)
            else:
                vals = df[gc].astype(str).map(normalise_symbol)
            for v in vals.tolist():
                if v:
                    genes.add(v)

    return sorted([g for g in genes if g])


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
@dataclass
class ScreenerConfig:
    # Candidate universe
    sources: List[str]
    custom_genes_text: str
    max_candidates: int

    # Single-omics filters (None means inactive)
    min_evidence_score: Optional[float]
    min_direction_agreement: Optional[float]
    min_auc_disc: Optional[float]
    min_n_auc: Optional[int]

    # KG filters
    kg_require_in_cluster: bool
    kg_min_composite_percentile: Optional[float]

    # WGCNA filters
    wgcna_modules: List[str]
    wgcna_trait: Optional[str]
    wgcna_trait_sign: str  # "any"|"positive"|"negative"
    wgcna_min_abs_corr: Optional[float]
    wgcna_require_drug_target: bool

    # In vitro filters
    invitro_contrasts: List[str]
    invitro_lines: List[str]
    invitro_padj_thr: Optional[float]
    invitro_require_significant: bool
    invitro_require_consistent_direction: bool

    # Bulk filters
    bulk_contrasts: List[str]
    bulk_padj_thr: Optional[float]
    bulk_require_significant: bool
    bulk_require_consistent_direction: bool


def run_screener(
    config: ScreenerConfig,
    *,
    soa: Any,
    kg_mod: Any,
    single_omics_data: Dict[str, Any],
    kg_data: Dict[str, Any],
    gene_to_module: Dict[str, str],
    wgcna_cor: pd.DataFrame,
    gene_to_drugs: Dict[str, Any],
    invitro_index: Dict[str, Dict[str, List[Path]]],
    bulk_omics_data: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      results_df: one row per passing gene
      details_by_gene: dict keyed by gene with nested evidence dicts
    """
    candidates = collect_candidates(
        sources=config.sources,
        custom_genes_text=config.custom_genes_text,
        single_omics_data=single_omics_data,
        kg_data=kg_data,
        gene_to_module=gene_to_module,
        invitro_index=invitro_index,
        bulk_omics_data=bulk_omics_data,
    )

    if config.max_candidates and len(candidates) > int(config.max_candidates):
        candidates = candidates[: int(config.max_candidates)]

    details_by_gene: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for g in candidates:
        # Evaluate filters; if a filter category is "inactive", it returns (True,{})
        ok_omics, omics_det = single_omics_gene_summary(
            g,
            soa=soa,
            single_omics_data=single_omics_data,
            min_evidence_score=config.min_evidence_score,
            min_direction_agreement=config.min_direction_agreement,
            min_auc_disc=config.min_auc_disc,
            min_n_auc=config.min_n_auc,
        )

        ok_kg, kg_det = kg_gene_summary(
            g,
            kg_mod=kg_mod,
            kg_data=kg_data,
            require_in_cluster=config.kg_require_in_cluster,
            min_composite_percentile=config.kg_min_composite_percentile,
        )

        ok_wgcna, wgcna_det = wgcna_gene_summary(
            g,
            gene_to_module=gene_to_module,
            wgcna_cor=wgcna_cor,
            trait=config.wgcna_trait,
            require_module_in=config.wgcna_modules,
            require_trait_sign=config.wgcna_trait_sign,
            require_drug_target=config.wgcna_require_drug_target,
            gene_to_drugs=gene_to_drugs,
            min_abs_corr=config.wgcna_min_abs_corr,
        )

        ok_invitro, invitro_det = invitro_gene_summary(
            g,
            invitro_index=invitro_index,
            selected_contrasts=config.invitro_contrasts,
            selected_lines=config.invitro_lines,
            padj_thr=config.invitro_padj_thr,
            require_significant=config.invitro_require_significant,
            require_consistent_direction=config.invitro_require_consistent_direction,
        )

        ok_bulk, bulk_det = bulk_gene_summary(
            g,
            bulk_omics_data=bulk_omics_data,
            selected_contrasts=config.bulk_contrasts,
            padj_thr=config.bulk_padj_thr,
            require_significant=config.bulk_require_significant,
            require_consistent_direction=config.bulk_require_consistent_direction,
        )

        passed = bool(ok_omics and ok_kg and ok_wgcna and ok_invitro and ok_bulk)

        # Always store details for evidence reporting, even if a category was inactive
        details_by_gene[g] = {
            "single_omics": omics_det,
            "kg": kg_det,
            "wgcna": wgcna_det,
            "invitro": invitro_det,
            "bulk": bulk_det,
            "passed": passed,
        }

        if not passed:
            continue

        # Build a compact row (UI can drill into details_by_gene)
        row = {"Gene": g}

        # Single-omics
        row["Evidence score"] = omics_det.get("evidence_score") if isinstance(omics_det, dict) else None
        row["Direction agreement"] = omics_det.get("direction_agreement") if isinstance(omics_det, dict) else None
        row["Median AUC (disc)"] = omics_det.get("auc_median_discriminative") if isinstance(omics_det, dict) else None
        row["n_auc"] = omics_det.get("n_auc") if isinstance(omics_det, dict) else None

        # KG
        row["KG cluster"] = kg_det.get("cluster") if isinstance(kg_det, dict) else None
        row["KG composite %ile"] = kg_det.get("composite_percentile") if isinstance(kg_det, dict) else None

        # WGCNA
        row["WGCNA module"] = wgcna_det.get("module") if isinstance(wgcna_det, dict) else None
        row["WGCNA trait corr"] = wgcna_det.get("trait_corr") if isinstance(wgcna_det, dict) else None
        row["Drug target (WGCNA)"] = wgcna_det.get("has_drug_target") if isinstance(wgcna_det, dict) else None

        # In vitro summary (compact)
        inv_hits = invitro_det.get("hits", []) if isinstance(invitro_det, dict) else []
        row["In vitro hits"] = sum(1 for x in inv_hits if isinstance(x, dict) and "log2FC" in x)

        # Bulk summary (compact)
        b_hits = bulk_det.get("hits", []) if isinstance(bulk_det, dict) else []
        row["Bulk hits"] = sum(1 for x in b_hits if isinstance(x, dict) and "log2FC" in x)

        rows.append(row)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        # Nice ordering
        pref = ["Gene", "Evidence score", "Direction agreement", "Median AUC (disc)", "n_auc", "KG cluster", "KG composite %ile", "WGCNA module", "WGCNA trait corr", "Drug target (WGCNA)", "In vitro hits", "Bulk hits"]
        cols = [c for c in pref if c in results_df.columns] + [c for c in results_df.columns if c not in pref]
        results_df = results_df[cols].copy()

    return results_df, details_by_gene
