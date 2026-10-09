# Single-cell pipeline steps shared by all datasets (Seurat v5).

# Marker panels for cluster-level lineage annotation.
LINEAGE_MARKERS <- list(
  human = list(
    hepatocyte   = c("ALB", "APOA1", "APOC3", "TTR", "CYP2E1", "HP", "SERPINA1", "TF", "APOH", "FABP1"),
    endothelial  = c("PECAM1", "VWF", "KDR", "STAB2", "CLEC4G", "FLT1", "PTPRB"),
    macrophage   = c("CD68", "CD163", "MARCO", "VSIG4", "C1QA", "C1QB", "CSF1R"),
    stellate     = c("DCN", "COL1A1", "COL3A1", "LRAT", "RELN", "PDGFRB"),
    cholangiocyte = c("KRT19", "KRT7", "SOX9", "EPCAM", "ANXA4"),
    t_nk         = c("CD3D", "CD3E", "NKG7", "GZMA", "CD2"),
    b_plasma     = c("CD79A", "MS4A1", "MZB1", "JCHAIN")
  ),
  mouse = list(
    hepatocyte   = c("Alb", "Apoa1", "Apoc3", "Ttr", "Cyp2e1", "Serpina1a", "Tf", "Ahsg", "Apoh", "Fabp1"),
    endothelial  = c("Pecam1", "Kdr", "Stab2", "Clec4g", "Flt1", "Ptprb", "Cdh5"),
    macrophage   = c("Cd68", "Clec4f", "Vsig4", "Marco", "C1qa", "C1qb", "Csf1r", "Adgre1"),
    stellate     = c("Dcn", "Col1a1", "Col3a1", "Lrat", "Reln", "Pdgfrb"),
    cholangiocyte = c("Krt19", "Krt7", "Sox9", "Epcam", "Spp1"),
    t_nk         = c("Cd3d", "Cd3e", "Nkg7", "Gzma", "Cd2"),
    b_plasma     = c("Cd79a", "Ms4a1", "Mzb1", "Jchain")
  )
)

# `id` becomes $capture (technical batch) and `donor` becomes $sample (the
# statistical unit), so captures from one liver are not counted as replicates.
read_10x_samples <- function(raw_dir, samples) {
  objs <- lapply(samples, function(s) {
    path <- file.path(resolve_path(raw_dir), s$folder)
    if (!dir.exists(path)) stop("Missing 10x folder: ", path)
    mat <- Seurat::Read10X(data.dir = path)
    if (is.list(mat)) mat <- if ("Gene Expression" %in% names(mat)) mat[["Gene Expression"]] else mat[[1]]
    o <- Seurat::CreateSeuratObject(counts = mat, project = s$id)
    o$capture <- s$id
    o$sample <- if (is.null(s$donor)) s$id else s$donor
    o$condition <- s$condition
    log_step("  ", s$id, " [", o$sample[1], ", ", s$condition, "]: ", ncol(o), " cells")
    o
  })
  ids <- vapply(samples, `[[`, "", "id")
  obj <- if (length(objs) == 1) {
    Seurat::RenameCells(objs[[1]], add.cell.id = ids[1])
  } else {
    merge(objs[[1]], y = objs[-1], add.cell.ids = ids)
  }
  SeuratObject::JoinLayers(obj)
}

validate_conditions <- function(obj) {
  bad <- setdiff(unique(obj$condition), c("disease", "control"))
  if (length(bad)) stop("condition must be 'disease' or 'control', found: ", paste(bad, collapse = ", "))
  per_sample <- unique(obj@meta.data[, c("sample", "condition")])
  if (anyDuplicated(per_sample$sample)) stop("A sample (donor) is assigned to more than one condition.")
  log_step("Cells per sample/condition:")
  print(table(obj$sample, obj$condition))
  invisible(obj)
}

# Per-capture mt cut-off (median + n_mads * MAD, within [mt_floor, mt_cap]), since
# one fixed cut-off filters differently prepared captures unevenly.
mt_thresholds <- function(percent_mt, capture, mt_floor, mt_cap, n_mads = 3) {
  vapply(split(percent_mt, capture), function(x) {
    min(mt_cap, max(mt_floor, stats::median(x) + n_mads * stats::mad(x)))
  }, numeric(1))
}

qc_filter <- function(obj, min_features, mt_floor, mt_cap, n_mads = 3) {
  obj[["percent.mt"]] <- Seurat::PercentageFeatureSet(obj, pattern = "^(MT-|mt-|Mt-)")
  capture <- if ("capture" %in% colnames(obj@meta.data)) obj$capture else obj$sample
  thr <- mt_thresholds(obj$percent.mt, capture, mt_floor, mt_cap, n_mads)
  before <- table(capture)
  keep <- obj$nFeature_RNA > min_features & obj$percent.mt < thr[as.character(capture)]
  qc <- data.frame(
    capture = names(before),
    mt_median = vapply(split(obj$percent.mt, capture), stats::median, 0)[names(before)],
    mt_cutoff = thr[names(before)],
    before = as.vector(before),
    after = as.vector(table(factor(capture[keep], levels = names(before)))),
    row.names = NULL
  )
  qc$kept_pct <- round(100 * qc$after / qc$before, 1)
  log_step("QC per capture (mt cut-off = median + ", n_mads, " MAD, within [", mt_floor, ", ", mt_cap, "]%):")
  print(transform(qc, mt_median = round(mt_median, 1), mt_cutoff = round(mt_cutoff, 1)), row.names = FALSE)
  obj <- subset(obj, cells = colnames(obj)[keep])
  attr(obj, "qc_table") <- qc
  obj
}

preprocess <- function(obj, n_pcs = 20, resolution = 0.5, batch_var = NULL, seed = 42) {
  obj <- Seurat::NormalizeData(obj, verbose = FALSE)
  obj <- Seurat::FindVariableFeatures(obj, verbose = FALSE)
  obj <- Seurat::ScaleData(obj, verbose = FALSE)
  obj <- Seurat::RunPCA(obj, npcs = max(n_pcs, 30), verbose = FALSE, seed.use = seed)
  reduction <- "pca"
  if (!is.null(batch_var) && length(unique(obj[[batch_var, drop = TRUE]])) > 1) {
    if (!requireNamespace("harmony", quietly = TRUE)) stop("batch_var set but harmony is not installed")
    obj <- harmony::RunHarmony(obj, group.by.vars = batch_var, verbose = FALSE)
    reduction <- "harmony"
  }
  dims <- seq_len(n_pcs)
  obj <- Seurat::FindNeighbors(obj, reduction = reduction, dims = dims, verbose = FALSE)
  obj <- Seurat::FindClusters(obj, resolution = resolution, random.seed = seed, verbose = FALSE)
  obj <- Seurat::RunUMAP(obj, reduction = reduction, dims = dims, seed.use = seed, verbose = FALSE)
  obj
}

# Ambient hepatocyte RNA inflates the hepatocyte score everywhere, so a cluster is a
# non-hepatocyte lineage when that lineage's score is an outlier (z >= min_z) and large.
annotate_lineages <- function(obj, species, cluster_col = "seurat_clusters", override = NULL,
                              min_z = 3, min_lineage_score = 0.25, min_hep_score = 0.1) {
  panels <- lapply(LINEAGE_MARKERS[[species]], intersect, rownames(obj))
  panels <- panels[lengths(panels) >= 2]
  if (!"hepatocyte" %in% names(panels)) stop("Too few hepatocyte markers found in the data.")

  obj <- Seurat::AddModuleScore(obj, features = panels, name = "lineage_score_", seed = 1)
  score_cols <- paste0("lineage_score_", seq_along(panels))
  scores <- obj@meta.data[, score_cols, drop = FALSE]
  colnames(scores) <- names(panels)
  obj@meta.data <- obj@meta.data[, setdiff(colnames(obj@meta.data), score_cols)]

  cl <- as.character(obj[[cluster_col, drop = TRUE]])
  means <- stats::aggregate(scores, by = list(cluster = cl), FUN = mean)
  m <- as.matrix(means[, -1, drop = FALSE])
  robust_z <- function(x) (x - stats::median(x)) / max(stats::mad(x), 0.02)
  z <- apply(m, 2, robust_z)
  if (is.null(dim(z))) z <- matrix(z, nrow = nrow(m), dimnames = dimnames(m))
  nonhep <- setdiff(colnames(m), "hepatocyte")

  lineage <- vapply(seq_len(nrow(m)), function(k) {
    ok <- nonhep[z[k, nonhep] >= min_z & m[k, nonhep] >= min_lineage_score]
    if (length(ok)) return(ok[which.max(z[k, ok])])
    if (m[k, "hepatocyte"] >= min_hep_score) "hepatocyte" else "ambiguous"
  }, "")

  tab <- data.frame(cluster = means$cluster, lineage = lineage,
                    n_cells = as.vector(table(cl)[means$cluster]),
                    round(m, 3), stringsAsFactors = FALSE)
  # Manual corrections from config.yml
  for (k in names(override)) tab$lineage[tab$cluster == k] <- override[[k]]
  tab <- tab[order(as.numeric(tab$cluster)), ]

  obj$lineage <- tab$lineage[match(cl, tab$cluster)]
  obj$hepatocyte_score <- scores$hepatocyte
  # Used by the purity filter in run_condition_analysis()
  nh <- as.matrix(scores[, nonhep, drop = FALSE])
  obj$max_nonhep_score <- apply(nh, 1, max)
  obj$max_nonhep_lineage <- nonhep[max.col(nh, ties.method = "first")]
  log_step("Cluster annotation:")
  print(tab[, c("cluster", "lineage", "n_cells", colnames(m))], row.names = FALSE)
  list(obj = obj, cluster_table = tab)
}

save_overview_plots <- function(obj, out_dir, prefix) {
  grDevices::pdf(file.path(out_dir, paste0(prefix, "_overview_umaps.pdf")), width = 9, height = 7)
  for (g in c("condition", "sample", "seurat_clusters", "lineage")) {
    if (g %in% colnames(obj@meta.data)) {
      print(Seurat::DimPlot(obj, group.by = g, label = g %in% c("seurat_clusters", "lineage"),
                            pt.size = 0.2, raster = TRUE) + ggplot2::ggtitle(paste(prefix, "-", g)))
    }
  }
  print(Seurat::FeaturePlot(obj, "hepatocyte_score", raster = TRUE) + ggplot2::ggtitle("Hepatocyte module score"))
  grDevices::dev.off()
}

# Disease-vs-control analysis within hepatocytes.
run_condition_analysis <- function(obj, ds, name, target_genes = character()) {
  out_dir <- ensure_dir(project_path("results", name))
  tab_dir <- ensure_dir(if (is.null(ds$output_dir)) project_path("data", "single_cell", "results")
                        else resolve_path(ds$output_dir))

  in_hep_cluster <- obj$lineage == "hepatocyte"
  keep <- in_hep_cluster
  if (isTRUE(ds$cell_purity_filter)) {
    # Drop contaminating cells inside hepatocyte clusters
    m <- obj@meta.data
    impure <- m$max_nonhep_score >= ds$lineage_min_score | m$max_nonhep_score > m$hepatocyte_score
    keep <- in_hep_cluster & !impure
    status <- ifelse(!impure, "kept", paste0("removed: ", m$max_nonhep_lineage))
    purity <- as.data.frame.matrix(table(m$condition[in_hep_cluster], status[in_hep_cluster]))
    log_step("Cells in hepatocyte clusters, kept vs removed by per-cell purity filter:")
    print(purity)
    utils::write.csv(purity, file.path(out_dir, paste0(name, "_hepatocyte_cluster_purity.csv")))
  }
  hep <- subset(obj, cells = colnames(obj)[keep])
  hep <- SeuratObject::JoinLayers(hep)
  log_step("Hepatocytes per sample:")
  print(table(hep$sample, hep$condition))

  counts <- SeuratObject::LayerData(hep, assay = "RNA", layer = "counts")
  data   <- SeuratObject::LayerData(hep, assay = "RNA", layer = "data")

  meta <- hep@meta.data
  cells <- cap_cells_per_sample(colnames(hep), meta$sample, ds$cells_per_sample_cap, ds$seed)
  is_case <- meta[cells, "condition"] == "disease"
  auc_data <- data[, cells]
  if (isTRUE(ds$depth_match_auc)) {
    auc_data <- depth_matched_lognorm(counts[, cells], is_case, ds$seed)
    log_step("Depth matching for AUC: shallower/deeper median depth = ",
             round(attr(auc_data, "depth_ratio"), 3))
  }
  auc_tab <- timed("Per-gene AUC", gene_auc(auc_data, is_case, ds$min_detect_pct))

  auc_tab$mean_logexpr_disease <- Matrix::rowMeans(auc_data[auc_tab$gene, is_case, drop = FALSE])
  auc_tab$mean_logexpr_control <- Matrix::rowMeans(auc_data[auc_tab$gene, !is_case, drop = FALSE])
  auc_tab$n_cells_disease <- sum(is_case)
  auc_tab$n_cells_control <- sum(!is_case)

  pb <- timed("Pseudobulk edgeR", pseudobulk_de(counts, meta$sample, meta$condition,
                                                 ds$min_cells_per_sample_pseudobulk))
  res <- merge(auc_tab, pb, by = "gene", all.x = TRUE, sort = FALSE)
  res <- cbind(dataset = name, species = ds$species, res)
  res <- res[order(-res$auc_power, res$gene, na.last = TRUE), ]
  out_csv <- file.path(tab_dir, paste0(name, "_hepatocyte_gene_stats.csv"))
  utils::write.csv(res, out_csv, row.names = FALSE)
  log_step("Wrote ", out_csv, " (", nrow(res), " genes, ", sum(res$tested), " tested)")

  sig <- NULL
  if (length(target_genes)) {
    sig <- heldout_signature_roc(auc_data, is_case, meta[cells, "sample"], target_genes,
                                 ds$min_detect_pct, ds$signature_min_auc, ds$seed)
    if (!is.null(sig)) {
      grDevices::pdf(file.path(out_dir, paste0(name, "_target_signature_ROC_heldout.pdf")), 6, 6)
      plot(sig$roc, main = "Target-gene signature (held-out cells)")
      graphics::legend("bottomright", bty = "n", legend = c(
        sprintf("Validation AUC = %.3f (95%% CI %.3f-%.3f)", sig$auc_validation,
                sig$auc_validation_ci95[1], sig$auc_validation_ci95[2]),
        sprintf("In-sample AUC (optimistic) = %.3f", sig$auc_discovery_insample),
        paste("Split:", sub(" \\(.*", "", sig$split))))
      grDevices::dev.off()

      score <- rep(NA_real_, ncol(hep))
      score[match(cells, colnames(hep))] <- sig$score
      hep$target_signature_score <- score
      grDevices::pdf(file.path(out_dir, paste0(name, "_target_signature_UMAP.pdf")), 11, 5)
      print(Seurat::FeaturePlot(hep, "target_signature_score", split.by = "condition", raster = TRUE))
      grDevices::dev.off()
      utils::write.csv(data.frame(gene = sig$genes),
                       file.path(out_dir, paste0(name, "_target_signature_genes.csv")), row.names = FALSE)
    }
  }

  summary <- list(
    n_hepatocytes = ncol(hep),
    median_depth_disease = stats::median(Matrix::colSums(counts[, cells[is_case]])),
    median_depth_control = stats::median(Matrix::colSums(counts[, cells[!is_case]])),
    pct_tested_genes_up = round(100 * mean(res$auc[res$tested] > 0.5), 1),
    n_removed_by_cell_purity = sum(in_hep_cluster) - sum(keep),
    n_cells_auc = length(cells),
    n_genes_tested = sum(res$tested),
    pseudobulk = unique(res$pb_method),
    n_fdr_below_0.05 = sum(res$pb_fdr < 0.05, na.rm = TRUE),
    target_signature = if (is.null(sig)) NULL else list(
      n_genes = length(sig$genes), split = sig$split,
      auc_validation = round(sig$auc_validation, 4),
      auc_validation_ci95 = round(sig$auc_validation_ci95, 4),
      auc_discovery_insample = round(sig$auc_discovery_insample, 4))
  )
  write_run_info(out_dir, ds, list(summary = summary))
  invisible(list(table = res, signature = sig, hepatocytes = hep, summary = summary))
}

read_target_genes <- function(path) {
  path <- resolve_path(path)
  if (!file.exists(path)) return(character())
  g <- trimws(readLines(path, warn = FALSE))
  unique(g[nzchar(g)])
}

# The curated list is mouse. Human symbols come from the ortholog table, or
# upper-casing when there is none.
target_genes_for <- function(ds) {
  mouse <- read_target_genes(ds$target_genes_mouse)
  if (ds$species == "mouse") return(mouse)
  unique(target_genes_human(mouse))
}

target_genes_human <- function(mouse, orth = load_orthologs()) {
  if (is.null(orth)) return(unique(toupper(mouse)))
  unique(orth$human_symbol[orth$mouse_symbol %in% mouse])
}

load_orthologs <- function(path = project_path("data", "reference", "orthologs_mouse_human.csv")) {
  if (!file.exists(path)) return(NULL)
  utils::read.csv(path, stringsAsFactors = FALSE)
}

# decontX rather than SoupX because GEO has only filtered matrices (no empty droplets).
# Counts are rounded so tiny residuals do not count as detection.
ambient_correct <- function(obj, seed = 42) {
  if (!requireNamespace("decontX", quietly = TRUE)) {
    stop("ambient_correction: decontx needs the decontX package (BiocManager::install(\"decontX\"))")
  }
  counts <- SeuratObject::LayerData(obj, assay = "RNA", layer = "counts")
  batch <- if ("capture" %in% colnames(obj@meta.data)) obj$capture else obj$sample
  res <- decontX::decontX(counts, z = as.integer(obj$seurat_clusters), batch = as.character(batch),
                          seed = seed, verbose = FALSE)
  corrected <- round(res$decontXcounts)
  corrected <- methods::as(Matrix::drop0(corrected), "CsparseMatrix")
  dimnames(corrected) <- dimnames(counts)
  SeuratObject::LayerData(obj, assay = "RNA", layer = "counts") <- corrected
  obj <- Seurat::NormalizeData(obj, verbose = FALSE)
  obj$ambient_contamination <- res$contamination
  summ <- stats::aggregate(ambient_contamination ~ capture + condition,
                           data.frame(capture = as.character(batch), condition = obj$condition,
                                      ambient_contamination = obj$ambient_contamination), stats::median)
  log_step("Median estimated ambient contamination per capture:")
  print(transform(summ, ambient_contamination = round(ambient_contamination, 3)), row.names = FALSE)
  attr(obj, "ambient_summary") <- summ
  obj
}

# QC, clustering, optional ambient correction and lineage annotation.
prepare_dataset <- function(obj, ds, name) {
  validate_conditions(obj)
  obj <- timed("QC", qc_filter(obj, ds$min_features, ds$mt_floor, ds$mt_cap, ds$mt_n_mads))
  utils::write.csv(attr(obj, "qc_table"), file.path(ensure_dir(project_path("results", name)),
                                                    paste0(name, "_qc_per_capture.csv")), row.names = FALSE)
  obj <- timed("Normalise, PCA, clustering, UMAP",
               preprocess(obj, ds$n_pcs, ds$cluster_resolution, ds$batch_var, ds$seed))
  if (identical(ds$ambient_correction, "decontx")) {
    obj <- timed("Ambient RNA correction (decontX)", ambient_correct(obj, ds$seed))
    utils::write.csv(attr(obj, "ambient_summary"),
                     file.path(ensure_dir(project_path("results", name)), paste0(name, "_ambient_contamination.csv")),
                     row.names = FALSE)
  }
  ann <- annotate_lineages(obj, ds$species, override = ds$lineage_override,
                           min_z = ds$lineage_min_z, min_lineage_score = ds$lineage_min_score,
                           min_hep_score = ds$hepatocyte_min_score)
  out_dir <- ensure_dir(project_path("results", name))
  utils::write.csv(ann$cluster_table, file.path(out_dir, paste0(name, "_cluster_annotation.csv")),
                   row.names = FALSE)
  save_overview_plots(ann$obj, out_dir, name)
  # Per-cell decisions, reused by scripts/13_processing_sensitivity.R
  m <- ann$obj@meta.data
  cell_tab <- data.frame(cell = rownames(m), m[, intersect(c("sample", "capture", "condition", "nCount_RNA",
    "nFeature_RNA", "percent.mt", "seurat_clusters", "lineage", "hepatocyte_score", "max_nonhep_score",
    "max_nonhep_lineage", "ambient_contamination"), colnames(m))])
  data.table::fwrite(cell_tab, file.path(out_dir, paste0(name, "_cell_annotation.csv")))
  ann$obj
}

# Full run for a dataset stored as one 10x folder per sample.
run_10x_dataset <- function(name, cfg = load_config()) {
  ds <- dataset_config(cfg, name)
  set.seed(ds$seed)
  log_step("==== ", name, " ====")
  obj <- timed("Reading 10x", read_10x_samples(ds$raw_dir, ds$samples))
  obj <- prepare_dataset(obj, ds, name)
  res <- run_condition_analysis(obj, ds, name, target_genes_for(ds))
  saveRDS(res$hepatocytes, project_path("results", name, paste0(name, "_hepatocytes.rds")))
  str(res$summary)
  invisible(res)
}
