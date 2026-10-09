# Does processing drive the Xiao vs Wang disagreement? Recomputes NASH vs control effects for
# processing variants (one step changed at a time) and compares Xiao with Wang and each with bulk.
# Needs raw 10x data and results/<dataset>/*_cell_annotation.csv. Writes data/single_cell/consistency/.
source(here::here("R", "load_all.R"))
cfg <- load_config()
app <- readRDS(project_path("app", "data", "app_data.rds"))

# Bulk reference: NASH F2-F4 vs control, genes significant in at least one contrast
b <- app$bulk[app$bulk$contrast %in% c("NASH F2 vs control", "NASH F3 vs control", "NASH F4 vs control") &
                !is.na(app$bulk$gene_human) & !is.na(app$bulk$padj), ]
bulk_lfc <- tapply(b$log2FC, b$gene_human, mean)
bulk_padj <- tapply(b$padj, b$gene_human, min)
bulk_ref <- names(bulk_lfc)[bulk_padj < 0.05 & abs(bulk_lfc) >= 0.5]

variants <- c(
  "1 all nuclei, nFeature>200 only",
  "2 all nuclei, fixed mt<5% (original QC)",
  "3 marker-gated hepatocytes, fixed mt<5% (original scripts)",
  "4 marker-gated hepatocytes, adaptive mt QC",
  "5 cluster-annotated hepatocytes, adaptive mt QC, no purity filter",
  "6 full pipeline (cluster + per-cell purity, adaptive mt QC)",
  "7 full pipeline but fixed mt<5%"
)

select_cells <- function(v, qc, ann, gate) {
  base <- qc$nFeature > 200
  in_ann <- qc$cell %in% ann$cell
  a <- ann[match(qc$cell, ann$cell), ]
  hep_cluster <- in_ann & a$lineage %in% "hepatocyte"
  pure <- hep_cluster & a$max_nonhep_score < cfg$defaults$lineage_min_score &
    a$max_nonhep_score <= a$hepatocyte_score
  switch(substr(v, 1, 1),
    "1" = base,
    "2" = base & qc$mt < 5,
    "3" = base & qc$mt < 5 & gate,
    "4" = in_ann & gate,
    "5" = hep_cluster,
    "6" = pure,
    "7" = pure & qc$mt < 5)
}

effects_for <- function(name) {
  ds <- dataset_config(cfg, name)
  log_step("==== ", name, " ====")
  obj <- read_10x_samples(ds$raw_dir, ds$samples)
  counts <- SeuratObject::LayerData(obj, layer = "counts")
  meta <- obj@meta.data
  rm(obj)
  invisible(gc())
  mt_genes <- grepl("^(MT-|mt-)", rownames(counts))
  qc <- data.frame(cell = colnames(counts),
                   nFeature = Matrix::colSums(counts > 0),
                   mt = 100 * Matrix::colSums(counts[mt_genes, ]) / pmax(Matrix::colSums(counts), 1))
  gate <- counts["ALB", ] > 0 & (counts["TTR", ] > 0 | counts["CYP2E1", ] > 0)
  ann <- as.data.frame(data.table::fread(project_path("results", name, paste0(name, "_cell_annotation.csv"))))

  lapply(setNames(variants, variants), function(v) {
    keep <- select_cells(v, qc, ann, gate)
    cells <- qc$cell[keep]
    smp <- meta[cells, "sample"]
    cond <- meta[cells, "condition"]
    log_step(v, ": ", length(cells), " cells")
    pb <- pseudobulk_de(counts[, cells], smp, cond, ds$min_cells_per_sample_pseudobulk)
    capped <- cap_cells_per_sample(cells, smp, ds$cells_per_sample_cap, ds$seed)
    is_case <- meta[capped, "condition"] == "disease"
    cm <- counts[, capped]
    lib <- Matrix::colSums(cm)
    naive <- cm
    naive@x <- log1p(naive@x / rep.int(lib, diff(naive@p)) * 1e4)
    auc_naive <- gene_auc(naive, is_case, ds$min_detect_pct)
    auc_dm <- gene_auc(depth_matched_lognorm(cm, is_case, ds$seed), is_case, ds$min_detect_pct)
    list(n_cells = length(cells), pb = pb[, c("gene", "pb_logFC")],
         auc_naive = auc_naive[, c("gene", "auc")], auc_dm = auc_dm[, c("gene", "auc")],
         pct_up_dm = mean(auc_dm$auc > 0.5, na.rm = TRUE))
  })
}

sign_agree <- function(x, y, thr) {
  k <- !is.na(x) & !is.na(y) & abs(x) >= thr & abs(y) >= thr
  c(agree = if (sum(k)) mean(sign(x[k]) == sign(y[k])) else NA, n = sum(k))
}

X <- effects_for("xiao_GSE189600")
invisible(gc())
W <- effects_for("wang_human_GSE212837")
invisible(gc())

res <- do.call(rbind, lapply(variants, function(v) {
  x <- X[[v]]
  w <- W[[v]]
  pb <- merge(x$pb, w$pb, by = "gene", suffixes = c("_x", "_w"))
  pbs <- sign_agree(pb$pb_logFC_x, pb$pb_logFC_w, 0.5)
  an <- merge(x$auc_naive, w$auc_naive, by = "gene", suffixes = c("_x", "_w"))
  ad <- merge(x$auc_dm, w$auc_dm, by = "gene", suffixes = c("_x", "_w"))
  ans <- sign_agree(an$auc_x - 0.5, an$auc_w - 0.5, 0.05)
  ads <- sign_agree(ad$auc_x - 0.5, ad$auc_w - 0.5, 0.05)
  vs_bulk <- function(e) {
    g <- intersect(e$pb$gene[!is.na(e$pb$pb_logFC)], bulk_ref)
    round(stats::cor(e$pb$pb_logFC[match(g, e$pb$gene)], bulk_lfc[g]), 3)
  }
  data.frame(
    variant = v, cells_xiao = x$n_cells, cells_wang = w$n_cells,
    xiao_vs_wang_pb_cor = round(stats::cor(pb$pb_logFC_x, pb$pb_logFC_w, use = "complete.obs"), 3),
    xiao_vs_wang_pb_sign = round(pbs[["agree"]], 3), n_pb = pbs[["n"]],
    xiao_vs_wang_auc_naive = round(ans[["agree"]], 3),
    xiao_vs_wang_auc_depth_matched = round(ads[["agree"]], 3), n_auc = ads[["n"]],
    xiao_vs_bulk_pb_cor = vs_bulk(x), wang_vs_bulk_pb_cor = vs_bulk(w),
    pct_up_xiao = round(100 * x$pct_up_dm, 1), pct_up_wang = round(100 * w$pct_up_dm, 1)
  )
}))
out <- project_path("data", "single_cell", "consistency", "processing_sensitivity.csv")
utils::write.csv(res, out, row.names = FALSE)
print(res, row.names = FALSE)
log_step("Wrote ", out)
