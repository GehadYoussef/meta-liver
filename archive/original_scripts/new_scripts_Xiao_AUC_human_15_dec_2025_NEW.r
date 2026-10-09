# ============================================================
# Xiao et al / GSE189600 (human snRNA-seq) — Hepatocyte-only + balanced Healthy vs NASH
#
# Assumes you already created and saved:
#   "XIAO_GSE189600_merged_postQC.rds"
# from your loader+merge script.
#
# Steps:
# - Load merged object
# - Seurat v5: JoinLayers robustly (SeuratObject::JoinLayers)
# - (Optional) redo HVG/PCA/UMAP on full object for sanity plots
# - Hepatocyte gating (ALB > 0 AND (TTR > 0 OR CYP2E1 > 0), with fallbacks)
# - Recompute HVG/PCA/UMAP on hepatocytes-only
# - Balance Healthy vs NASH
# - JoinLayers on balanced hepatocytes
# - Genome-wide: detection %, AUC(roc), direction (avg expr), Wilcoxon p-values
#   Rule: if max(pct) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# - Target gene list: DE + AUC table + combined upregulated score + ROC + UMAP overlays
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(pROC)
  library(ggplot2)
})

setwd("/scratch_data/gy260/new_MASH_April_2025/Xiao_GSE189600")  # change if needed

# ----------------------------
# SETTINGS
# ----------------------------
RANDOM_SEED <- 42
MIN_DETECT_PCT_FOR_AUC <- 0.10
CAP_PER_GROUP <- NA_integer_        # e.g. 20000L; NA keeps all of smaller group

# For sanity plots (full object)
PCA_DIMS_FULL <- 1:20
CLUSTER_RES_FULL <- 1.0

# Hep-only preprocessing
PCA_DIMS_HEP <- 1:20
CLUSTER_RES_HEP <- 1.0

# Hepatocyte markers (human)
HEP_MARKERS_MAIN <- c("ALB", "TTR", "CYP2E1")
HEP_MARKERS_BACKUP <- c("APOA1", "APOA2", "TF", "HP", "FGA", "FGB", "FGG")

# Target genes file (human symbols recommended)
TARGET_GENE_FILE <- "my_genes_human.txt"
COMBINED_SCORE_THRESHOLD <- 0.15

cat("\n============================================================\n")
cat("START:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n")

t_all <- Sys.time()

# ----------------------------
# [1] Load merged object
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Loading merged object...\n", sep = "")
obj <- readRDS("XIAO_GSE189600_merged_postQC.rds")

if (!("Condition" %in% colnames(obj[[]]))) stop("Missing metadata column: Condition")
obj$Condition <- as.character(obj$Condition)
obj$Condition[obj$Condition %in% c("Healthy","Control","CTRL")] <- "Healthy"
obj$Condition[obj$Condition %in% c("NASH","MASH")] <- "NASH"
obj$Condition <- factor(obj$Condition, levels = c("Healthy","NASH"))

cat("Loaded -> Cells:", ncol(obj), " Genes:", nrow(obj), "\n")
cat("Condition counts:\n")
print(table(obj$Condition))

DefaultAssay(obj) <- "RNA"

# ----------------------------
# [1b] Seurat v5: JoinLayers robustly (counts + data if present)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers (v5-safe) on merged object...\n", sep = "")
obj[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj[["RNA"]],
  layers = c("counts", "data"),
  new    = c("counts", "data")
)
cat("RNA layers now:\n")
print(Layers(obj[["RNA"]]))

# If no data layer exists (e.g. you saved pre-normalisation), normalise now and re-join
lyr0 <- Layers(obj[["RNA"]])
if (!("data" %in% lyr0)) {
  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] No unified 'data' layer detected -> NormalizeData...\n", sep = "")
  obj <- NormalizeData(obj, verbose = FALSE)
  obj[["RNA"]] <- SeuratObject::JoinLayers(
    object = obj[["RNA"]],
    layers = c("counts", "data"),
    new    = c("counts", "data")
  )
  cat("RNA layers after NormalizeData:\n")
  print(Layers(obj[["RNA"]]))
}

# ----------------------------
# [2] Full-object sanity UMAP (optional but useful)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Full-object HVG/PCA/UMAP for sanity plots...\n", sep = "")
set.seed(RANDOM_SEED)

obj <- FindVariableFeatures(obj, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, verbose = FALSE)
obj <- RunUMAP(obj, dims = PCA_DIMS_FULL, verbose = FALSE)
obj <- FindNeighbors(obj, dims = PCA_DIMS_FULL, verbose = FALSE)
obj <- FindClusters(obj, resolution = CLUSTER_RES_FULL, verbose = FALSE)

pdf("XIAO_full_UMAP_byCondition.pdf", width = 8, height = 6)
print(DimPlot(obj, group.by = "Condition", pt.size = 0.25) + ggtitle("Xiao GSE189600: UMAP (full object)"))
dev.off()

pdf("XIAO_full_UMAP_bySample.pdf", width = 8, height = 6)
gb <- if ("sample" %in% colnames(obj[[]])) "sample" else NULL
if (!is.null(gb)) {
  print(DimPlot(obj, group.by = gb, pt.size = 0.25) + ggtitle("Xiao GSE189600: UMAP by sample (full object)"))
}
dev.off()

# ----------------------------
# [3] Hepatocyte gating (marker-based)
# Rule: ALB > 0 AND (TTR > 0 OR CYP2E1 > 0)
# Fallback: if these aren’t all present, use AND of the first two present markers.
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Hepatocyte gating...\n", sep = "")

genes_rn <- rownames(obj)
hep_markers <- intersect(HEP_MARKERS_MAIN, genes_rn)

if (length(hep_markers) < 2) {
  # Try case variants if needed
  hep_markers2 <- intersect(toupper(HEP_MARKERS_MAIN), toupper(genes_rn))
  if (length(hep_markers2) >= 2) {
    # map back to actual rownames
    rn_upper <- toupper(genes_rn)
    hep_markers <- genes_rn[match(hep_markers2, rn_upper)]
  }
}

if (length(hep_markers) < 2) {
  cat("Main hepatocyte markers not found well; trying backup markers...\n")
  hep_markers <- intersect(HEP_MARKERS_BACKUP, genes_rn)
  if (length(hep_markers) < 2) {
    hep_markers2 <- intersect(toupper(HEP_MARKERS_BACKUP), toupper(genes_rn))
    if (length(hep_markers2) >= 2) {
      rn_upper <- toupper(genes_rn)
      hep_markers <- genes_rn[match(hep_markers2, rn_upper)]
    }
  }
}

if (length(hep_markers) < 2) {
  cat("\nCould not find enough hepatocyte markers in rownames(obj). First 30 genes:\n")
  print(head(genes_rn, 30))
  stop("Hepatocyte marker symbols not found (or you have Ensembl IDs). Provide a symbol-mapped matrix or an ID mapping step.")
}

cat("Markers used for gating:", paste(hep_markers, collapse = ", "), "\n")

m <- FetchData(obj, vars = hep_markers)

# Apply preferred rule if ALB/TTR/CYP2E1 are all available (case-insensitive)
cn_upper <- toupper(colnames(m))
has_ALB <- "ALB" %in% cn_upper
has_TTR <- "TTR" %in% cn_upper
has_CYP2E1 <- "CYP2E1" %in% cn_upper

if (has_ALB && has_TTR && has_CYP2E1) {
  alb_col <- colnames(m)[match("ALB", cn_upper)]
  ttr_col <- colnames(m)[match("TTR", cn_upper)]
  cyp_col <- colnames(m)[match("CYP2E1", cn_upper)]
  hep_cells <- rownames(m)[ m[[alb_col]] > 0 & (m[[ttr_col]] > 0 | m[[cyp_col]] > 0) ]
} else {
  # fallback: AND of first two markers present
  hep_cells <- rownames(m)[ m[[colnames(m)[1]]] > 0 & m[[colnames(m)[2]]] > 0 ]
}

cat("Hepatocyte-gated cells:", length(hep_cells), "\n")
if (length(hep_cells) < 200) cat("WARNING: very few hepatocytes gated. Consider loosening thresholds or using cluster-based selection.\n")

obj_hep <- subset(obj, cells = hep_cells)

cat("Hepatocytes-only condition counts:\n")
print(table(obj_hep$Condition))
cat("Hepatocyte object -> Cells:", ncol(obj_hep), " Genes:", nrow(obj_hep), "\n")

# JoinLayers on hepatocyte subset (v5-safe)
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers on hepatocyte subset...\n", sep = "")
DefaultAssay(obj_hep) <- "RNA"
obj_hep[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj_hep[["RNA"]],
  layers = c("counts", "data"),
  new    = c("counts", "data")
)
cat("Hep RNA layers now:\n")
print(Layers(obj_hep[["RNA"]]))

# Recompute HVG/PCA/UMAP on hepatocytes-only (recommended)
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Hepatocytes-only HVG/PCA/UMAP...\n", sep = "")
set.seed(RANDOM_SEED)

# Ensure normalised data exists on hep subset
lyr_hep <- Layers(obj_hep[["RNA"]])
if (!("data" %in% lyr_hep)) {
  obj_hep <- NormalizeData(obj_hep, verbose = FALSE)
  obj_hep[["RNA"]] <- SeuratObject::JoinLayers(
    object = obj_hep[["RNA"]],
    layers = c("counts", "data"),
    new    = c("counts", "data")
  )
}

obj_hep <- FindVariableFeatures(obj_hep, verbose = FALSE)
obj_hep <- ScaleData(obj_hep, verbose = FALSE)
obj_hep <- RunPCA(obj_hep, verbose = FALSE)
obj_hep <- RunUMAP(obj_hep, dims = PCA_DIMS_HEP, verbose = FALSE)
obj_hep <- FindNeighbors(obj_hep, dims = PCA_DIMS_HEP, verbose = FALSE)
obj_hep <- FindClusters(obj_hep, resolution = CLUSTER_RES_HEP, verbose = FALSE)

pdf("XIAO_hepatocytes_UMAP_byCondition.pdf", width = 8, height = 6)
print(DimPlot(obj_hep, group.by = "Condition", pt.size = 0.3) + ggtitle("Xiao: UMAP (hepatocytes-only)"))
dev.off()

# ----------------------------
# [4] Balance Healthy vs NASH within hepatocytes
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Balancing Healthy vs NASH within hepatocytes...\n", sep = "")
set.seed(RANDOM_SEED)

Idents(obj_hep) <- obj_hep$Condition
cells_h <- WhichCells(obj_hep, idents = "Healthy")
cells_n <- WhichCells(obj_hep, idents = "NASH")

cat("Hep cells Healthy:", length(cells_h), "  NASH:", length(cells_n), "\n")

n_bal <- min(length(cells_h), length(cells_n))
if (!is.na(CAP_PER_GROUP)) n_bal <- min(n_bal, CAP_PER_GROUP)

cat("Balanced n per group:", n_bal, "\n")
h_sub <- sample(cells_h, n_bal)
n_sub <- sample(cells_n, n_bal)

obj_hep_bal <- subset(obj_hep, cells = c(h_sub, n_sub))

cat("After balancing:\n")
print(table(obj_hep_bal$Condition))
cat("Balanced hepatocyte object -> Cells:", ncol(obj_hep_bal), " Genes:", nrow(obj_hep_bal), "\n")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers on balanced hepatocytes...\n", sep = "")
DefaultAssay(obj_hep_bal) <- "RNA"
obj_hep_bal[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj_hep_bal[["RNA"]],
  layers = c("counts", "data"),
  new    = c("counts", "data")
)
cat("Balanced hep RNA layers now:\n")
print(Layers(obj_hep_bal[["RNA"]]))

# Recompute UMAP on balanced object for clean overlays
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Balanced hepatocytes HVG/PCA/UMAP...\n", sep = "")
set.seed(RANDOM_SEED)
obj_hep_bal <- FindVariableFeatures(obj_hep_bal, verbose = FALSE)
obj_hep_bal <- ScaleData(obj_hep_bal, verbose = FALSE)
obj_hep_bal <- RunPCA(obj_hep_bal, verbose = FALSE)
obj_hep_bal <- RunUMAP(obj_hep_bal, dims = PCA_DIMS_HEP, verbose = FALSE)

pdf("XIAO_hepatocytes_balanced_UMAP_byCondition.pdf", width = 8, height = 6)
print(DimPlot(obj_hep_bal, group.by = "Condition", pt.size = 0.35) + ggtitle("Xiao: UMAP (hepatocytes-only, balanced)"))
dev.off()

# ----------------------------
# [5] Genome-wide AUROC + direction + Wilcoxon p-values (balanced hepatocytes)
# Rule: if max(pct) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Genome-wide AUROC + direction + p-values (balanced hepatocytes)...\n", sep = "")

DefaultAssay(obj_hep_bal) <- "RNA"
Idents(obj_hep_bal) <- obj_hep_bal$Condition

cells_h <- WhichCells(obj_hep_bal, idents = "Healthy")
cells_n <- WhichCells(obj_hep_bal, idents = "NASH")

cat("Balanced hep cells Healthy:", length(cells_h), "  NASH:", length(cells_n), "\n")
cat("Processing genes:", nrow(obj_hep_bal), "\n")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Computing detection rates...\n", sep = "")
t0 <- Sys.time()

counts_mat_all <- GetAssayData(obj_hep_bal, assay = "RNA", layer = "counts")
pct_h <- Matrix::rowSums(counts_mat_all[, cells_h, drop = FALSE] > 0) / length(cells_h)
pct_n <- Matrix::rowSums(counts_mat_all[, cells_n, drop = FALSE] > 0) / length(cells_n)
pct_max <- pmax(pct_h, pct_n)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Detection rates done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

keep_genes_all <- names(pct_max)[pct_max >= MIN_DETECT_PCT_FOR_AUC]
cat("Genes total:", length(pct_max), "\n")
cat("Genes with max(pct) >= ", MIN_DETECT_PCT_FOR_AUC, ": ", length(keep_genes_all),
    " | forced AUC=0: ", length(pct_max) - length(keep_genes_all), "\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression for direction...\n", sep = "")
t0 <- Sys.time()

avg_expr <- AverageExpression(
  obj_hep_bal,
  assays = "RNA",
  group.by = "Condition",
  layer = "data",
  verbose = FALSE
)$RNA

cat("[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

if (!all(c("Healthy","NASH") %in% colnames(avg_expr))) {
  stop("AverageExpression did not return both Healthy and NASH columns. Check obj_hep_bal$Condition.")
}

avg_h <- avg_expr[, "Healthy"]
avg_n2 <- avg_expr[, "NASH"]
avg_logFC <- avg_n2 - avg_h

direction <- ifelse(avg_logFC > 0, "enriched_in_NASH",
                    ifelse(avg_logFC < 0, "enriched_in_Healthy", "no_change"))

genes_all <- rownames(avg_expr)
auc_vec <- setNames(rep(0, length(genes_all)), genes_all)

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] ROC AUC on ", length(keep_genes_all), " genes...\n", sep = "")
t0 <- Sys.time()

if (length(keep_genes_all) > 0) {
  auc_keep <- FindMarkers(
    obj_hep_bal,
    ident.1 = "NASH",
    ident.2 = "Healthy",
    test.use = "roc",
    features = keep_genes_all,
    min.pct = 0,
    logfc.threshold = 0,
    verbose = TRUE
  )
  auc_col <- intersect(c("myAUC","AUC"), colnames(auc_keep))[1]
  if (is.na(auc_col)) stop("Could not find myAUC/AUC in Seurat ROC output.")
  auc_vec[rownames(auc_keep)] <- auc_keep[[auc_col]]
}

cat("[", format(Sys.time(), "%H:%M:%S"), "] ROC AUC done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Wilcoxon p-values on ", length(keep_genes_all), " genes...\n", sep = "")
t0 <- Sys.time()

de_keep <- FindMarkers(
  obj_hep_bal,
  ident.1 = "NASH",
  ident.2 = "Healthy",
  test.use = "wilcox",
  features = keep_genes_all,
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Wilcoxon done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Building genome-wide results table...\n", sep = "")

res_auc <- data.frame(
  gene = genes_all,
  AUC = as.numeric(auc_vec[genes_all]),
  avg_logFC = as.numeric(avg_logFC[genes_all]),
  direction = direction[genes_all],
  mean_logexpr_Healthy = as.numeric(avg_h[genes_all]),
  mean_logexpr_NASH = as.numeric(avg_n2[genes_all]),
  pct_Healthy = as.numeric(pct_h[genes_all]),
  pct_NASH = as.numeric(pct_n[genes_all]),
  max_pct = as.numeric(pct_max[genes_all]),
  p_val = NA_real_,
  p_val_adj = NA_real_,
  tested = FALSE,
  stringsAsFactors = FALSE
)

res_auc$tested[res_auc$gene %in% rownames(de_keep)] <- TRUE
res_auc$p_val[match(rownames(de_keep), res_auc$gene)] <- de_keep$p_val
if ("p_val_adj" %in% colnames(de_keep)) {
  res_auc$p_val_adj[match(rownames(de_keep), res_auc$gene)] <- de_keep$p_val_adj
}

res_auc$AUC[res_auc$max_pct < MIN_DETECT_PCT_FOR_AUC] <- 0
res_auc <- res_auc[order(-res_auc$AUC, -abs(res_auc$avg_logFC)), ]

write.csv(res_auc, "XIAO_HEPATOCYTES_gene_AUC_direction_stats_balanced_genomewide.csv", row.names = FALSE)

cat("\nTop 20 genes:\n")
print(head(res_auc[, c("gene","AUC","direction","avg_logFC","pct_Healthy","pct_NASH","p_val_adj")], 20))

# ----------------------------
# [6] Target gene list: DE + AUC table + combined upregulated score + ROC + UMAP overlays
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Target gene list analysis...\n", sep = "")

if (!file.exists(TARGET_GENE_FILE)) {
  cat("Target gene file not found: ", TARGET_GENE_FILE, "\n", sep = "")
  cat("Create it (one gene symbol per line) or change TARGET_GENE_FILE.\n")
} else {

  target_genes <- readLines(TARGET_GENE_FILE) %>% trimws()
  target_genes <- target_genes[target_genes != ""]

  valid_genes <- intersect(target_genes, rownames(obj_hep_bal))

  if (length(valid_genes) == 0) {
    # try case-insensitive mapping
    rn_upper <- toupper(rownames(obj_hep_bal))
    tg_upper <- toupper(target_genes)
    hit <- intersect(tg_upper, rn_upper)
    valid_genes <- rownames(obj_hep_bal)[match(hit, rn_upper)]
  }

  missing_genes <- setdiff(target_genes, valid_genes)
  write.csv(data.frame(MissingGene = missing_genes),
            "XIAO_HEPATOCYTES_target_genes_missing.csv",
            row.names = FALSE)

  cat("Target genes:", length(target_genes), " | present:", length(valid_genes), " | missing:", length(missing_genes), "\n")

  if (length(valid_genes) > 0) {

    deg_target <- FindMarkers(
      obj_hep_bal,
      ident.1 = "NASH",
      ident.2 = "Healthy",
      features = valid_genes,
      test.use = "wilcox",
      min.pct = 0,
      logfc.threshold = 0,
      verbose = TRUE
    )
    deg_target$Gene <- rownames(deg_target)
    deg_target <- deg_target %>% relocate(Gene)
    write.csv(deg_target, "XIAO_HEPATOCYTES_DE_NASH_vs_Healthy_target_genes_balanced.csv", row.names = FALSE)

    counts_targets <- GetAssayData(obj_hep_bal, assay = "RNA", layer = "counts")[valid_genes, , drop = FALSE]
    pct_h_t <- Matrix::rowMeans(counts_targets[, cells_h, drop = FALSE] > 0)
    pct_n_t <- Matrix::rowMeans(counts_targets[, cells_n, drop = FALSE] > 0)
    pct_max_t <- pmax(pct_h_t, pct_n_t)

    keep_targets <- names(pct_max_t)[pct_max_t >= MIN_DETECT_PCT_FOR_AUC]
    auc_t <- setNames(rep(0, length(valid_genes)), valid_genes)

    if (length(keep_targets) > 0) {
      auc_keep_t <- FindMarkers(
        obj_hep_bal,
        ident.1 = "NASH",
        ident.2 = "Healthy",
        test.use = "roc",
        features = keep_targets,
        min.pct = 0,
        logfc.threshold = 0,
        verbose = TRUE
      )
      auc_col_t <- intersect(c("myAUC","AUC"), colnames(auc_keep_t))[1]
      if (is.na(auc_col_t)) stop("Could not find myAUC/AUC for target-gene ROC output.")
      auc_t[rownames(auc_keep_t)] <- auc_keep_t[[auc_col_t]]
    }

    target_auc_table <- data.frame(
      Gene = valid_genes,
      AUC = as.numeric(auc_t[valid_genes]),
      pct_Healthy = as.numeric(pct_h_t[valid_genes]),
      pct_NASH = as.numeric(pct_n_t[valid_genes]),
      max_pct = as.numeric(pct_max_t[valid_genes]),
      stringsAsFactors = FALSE
    ) %>%
      left_join(deg_target, by = "Gene")

    if ("p_val_adj" %in% colnames(target_auc_table)) {
      target_auc_table <- target_auc_table %>% arrange(desc(AUC), p_val_adj)
    } else {
      target_auc_table <- target_auc_table %>% arrange(desc(AUC))
    }

    write.csv(target_auc_table, "XIAO_HEPATOCYTES_target_genes_AUC_stats_balanced.csv", row.names = FALSE)

    # Combined upregulated score from target DE (avg_log2FC > 0)
    upregulated_genes <- deg_target %>%
      filter(avg_log2FC > 0) %>%
      pull(Gene) %>%
      intersect(rownames(obj_hep_bal))

    cat("Upregulated genes from target list (avg_log2FC>0) present:", length(upregulated_genes), "\n")

    if (length(upregulated_genes) >= 1) {
      expr_up <- GetAssayData(obj_hep_bal, assay = "RNA", layer = "data")[upregulated_genes, , drop = FALSE]
      obj_hep_bal$Upregulated_Score <- as.numeric(Matrix::colMeans(expr_up))

      labels <- ifelse(obj_hep_bal$Condition == "NASH", 1, 0)
      combined_roc <- roc(response = labels, predictor = obj_hep_bal$Upregulated_Score, quiet = TRUE)

      pdf("XIAO_HEPATOCYTES_ROC_Combined_Upregulated_Genes_balanced.pdf", width = 6, height = 6)
      plot(combined_roc, col = "darkblue", main = "ROC: Combined Upregulated Gene Score (hepatocytes, balanced)")
      legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)),
             col = "darkblue", lwd = 2)
      dev.off()

      obj_hep_bal$ExpressionStatus <- ifelse(
        obj_hep_bal$Upregulated_Score > COMBINED_SCORE_THRESHOLD, "Expressing",
        as.character(obj_hep_bal$Condition)
      )

      umap_df <- as.data.frame(Embeddings(obj_hep_bal, "umap"))
      colnames(umap_df) <- c("UMAP_1","UMAP_2")
      umap_df$Status <- obj_hep_bal$ExpressionStatus
      umap_df$Condition <- obj_hep_bal$Condition

      pdf("XIAO_HEPATOCYTES_UMAP_Three_Color_Overlay_balanced.pdf", width = 8, height = 6)
      print(
        ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
          geom_point(size = 0.4, alpha = 0.8) +
          ggtitle(paste0("UMAP (hepatocytes): Healthy vs NASH vs Expressing (thr=", COMBINED_SCORE_THRESHOLD, ")")) +
          theme_minimal()
      )
      dev.off()

      pdf("XIAO_HEPATOCYTES_UMAP_Healthy_NASH_only_balanced.pdf", width = 8, height = 6)
      print(
        ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Condition)) +
          geom_point(size = 0.4, alpha = 0.8) +
          ggtitle("UMAP (hepatocytes): Healthy vs NASH (balanced)") +
          theme_minimal()
      )
      dev.off()
    }
  }
}

# ----------------------------
# [7] Save objects
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Saving RDS objects...\n", sep = "")
saveRDS(obj, file = "XIAO_full_processed_postQC_clustered.rds")
saveRDS(obj_hep, file = "XIAO_hepatocytes_raw.rds")
saveRDS(obj_hep_bal, file = "XIAO_hepatocytes_balanced.rds")

cat("\n============================================================\n")
cat("DONE:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total elapsed mins:", round(as.numeric(difftime(Sys.time(), t_all, units = "mins")), 2), "\n")
cat("============================================================\n")
