# ============================================================
# COASSOLO mouse liver scRNA-seq (Chow vs NASH) — SU-style + HEPATOCYTE ONLY
#
# Pipeline:
# - Load 10X + merge
# - JoinLayers early (Seurat v5-safe)
# - QC -> Normalise -> PCA -> UMAP -> cluster (for plots + to sanity check hepatocyte clusters)
# - Hepatocyte-only subset (marker gating: Alb>0 AND (Ttr>0 OR Cyp2e1>0))
# - JoinLayers on hepatocyte subset (Seurat v5-safe)
# - BALANCE Chow vs NASH within hepatocytes (downsample)
# - JoinLayers again on BALANCED hepatocytes
# - Genome-wide (hepatocytes only): AUROC + direction + Wilcoxon p-values with rule:
#     if max(pct.NASH, pct.Chow) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# - Target gene list (hepatocytes only): DE + target AUC table (same rule) + combined upregulated score + ROC + UMAP overlays
# - Live console progress (timestamps + elapsed time)
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(patchwork)
  library(pROC)
  library(ggplot2)
})

setwd("/scratch_data/gy260/new_MASH_April_2025/coassollo_paper/")

# ----------------------------
# GLOBAL SETTINGS
# ----------------------------
RANDOM_SEED <- 42
MIN_DETECT_PCT_FOR_AUC <- 0.10
CAP_PER_GROUP <- NA_integer_        # e.g. 20000L to cap per condition; NA keeps all of smaller group

MIN_FEATURES <- 200
MAX_MT_PCT   <- 5

PCA_DIMS <- 1:10
CLUSTER_RES <- 1.2

# Hepatocyte gating (marker-based)
HEP_MARKERS_TRY1 <- c("Alb","Ttr","Cyp2e1")
HEP_MARKERS_TRY2 <- c("ALB","TTR","CYP2E1")

TARGET_GENE_FILE <- "my_genes_mus.txt"
COMBINED_SCORE_THRESHOLD <- 0.15

CHOW_DIR <- "GSM6431458_Chow/"
NASH_DIR <- "GSM6431459_NASH/"

cat("\n============================================================\n")
cat("START:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n")

t_all <- Sys.time()

# ----------------------------
# [1] Load 10X + create objects + merge
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Reading 10X...\n", sep = "")
t0 <- Sys.time()

chow_data <- Read10X(data.dir = CHOW_DIR)
nash_data <- Read10X(data.dir = NASH_DIR)

chow <- CreateSeuratObject(counts = chow_data, project = "Chow")
nash <- CreateSeuratObject(counts = nash_data, project = "NASH")
chow$Condition <- "Chow"
nash$Condition <- "NASH"

obj <- merge(chow, y = nash, add.cell.ids = c("Chow", "NASH"))
obj$Condition <- factor(obj$Condition, levels = c("Chow","NASH"))

cat("[", format(Sys.time(), "%H:%M:%S"), "] Loaded+merged in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")
cat("Merged -> Cells:", ncol(obj), " Genes:", nrow(obj), "\n")
cat("Condition counts:\n")
print(table(obj$Condition))

# ----------------------------
# [1b] JoinLayers early (Seurat v5-safe)
# Merge keeps split layers for integration; join if you are NOT integrating.
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers after merge (early)...\n", sep = "")
t0 <- Sys.time()
DefaultAssay(obj) <- "RNA"
obj[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj[["RNA"]],
  layers = "counts",
  new    = "counts"
)
cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers(early) done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")
cat("RNA layers now:\n")
print(Layers(obj[["RNA"]]))

# ----------------------------
# [2] QC + normalisation + PCA/UMAP + clustering (for plots only)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] QC + normalise + PCA/UMAP + clustering...\n", sep = "")
t0 <- Sys.time()

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^(mt-|Mt-|MT-)")
obj <- subset(obj, subset = nFeature_RNA > MIN_FEATURES & percent.mt < MAX_MT_PCT)

cat("After QC -> Cells:", ncol(obj), " Genes:", nrow(obj), "\n")
cat("Condition counts after QC:\n")
print(table(obj$Condition))

obj <- NormalizeData(obj, verbose = FALSE)

# Ensure a single counts+data layer exists after normalisation/subsetting
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers after QC+NormalizeData (safety)...\n", sep = "")
DefaultAssay(obj) <- "RNA"
obj[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj[["RNA"]],
  layers = c("counts", "data"),
  new    = c("counts", "data")
)

obj <- FindVariableFeatures(obj, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, verbose = FALSE)
obj <- RunUMAP(obj, dims = PCA_DIMS, verbose = FALSE)
obj <- FindNeighbors(obj, dims = PCA_DIMS, verbose = FALSE)
obj <- FindClusters(obj, resolution = CLUSTER_RES, verbose = FALSE)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Preprocess done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")
cat("Clusters:\n")
print(table(obj$seurat_clusters))

pdf("COASSOLO_UMAP_By_Condition_raw.pdf", width = 8, height = 6)
print(DimPlot(obj, reduction = "umap", group.by = "Condition", pt.size = 0.4) +
        ggtitle("UMAP (raw, QC-passed): Chow vs NASH"))
dev.off()

pdf("COASSOLO_UMAP_By_Cluster_IDs_raw.pdf", width = 8, height = 6)
print(DimPlot(obj, reduction = "umap", label = TRUE) + ggtitle("UMAP: Cluster IDs (raw)"))
dev.off()

pdf("COASSOLO_marker_FeaturePlot_Alb_Ttr_Cyp2e1_raw.pdf", width = 10, height = 6)
print(FeaturePlot(obj, features = c("Alb","Ttr","Cyp2e1")))
dev.off()

# ----------------------------
# [3] Hepatocyte-only subset (marker gating)
# Alb > 0 AND (Ttr > 0 OR Cyp2e1 > 0)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Hepatocyte gating (marker-based)...\n", sep = "")

DefaultAssay(obj) <- "RNA"

hep_markers <- intersect(HEP_MARKERS_TRY1, rownames(obj))
if (length(hep_markers) == 0) {
  hep_markers <- intersect(HEP_MARKERS_TRY2, rownames(obj))
}

if (length(hep_markers) < 2) {
  cat("\nCould not find hepatocyte markers in rownames(obj). First 30 rownames are:\n")
  print(head(rownames(obj), 30))
  stop("Hepatocyte marker symbols not found. Matrix may use Ensembl IDs or different symbols.")
}

cat("Hepatocyte markers found:", paste(hep_markers, collapse = ", "), "\n")

t0 <- Sys.time()
m <- FetchData(obj, vars = hep_markers)
cat("[", format(Sys.time(), "%H:%M:%S"), "] FetchData done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), " secs\n", sep = "")

if (all(c("Alb","Ttr","Cyp2e1") %in% colnames(m))) {
  hep_cells <- rownames(m)[ m[["Alb"]] > 0 & (m[["Ttr"]] > 0 | m[["Cyp2e1"]] > 0) ]
} else if (all(c("ALB","TTR","CYP2E1") %in% colnames(m))) {
  hep_cells <- rownames(m)[ m[["ALB"]] > 0 & (m[["TTR"]] > 0 | m[["CYP2E1"]] > 0) ]
} else if (length(colnames(m)) >= 2) {
  hep_cells <- rownames(m)[ m[[colnames(m)[1]]] > 0 & m[[colnames(m)[2]]] > 0 ]
} else {
  stop("Not enough hepatocyte markers found to gate.")
}

cat("Hepatocyte-gated cells:", length(hep_cells), "\n")
if (length(hep_cells) < 100) {
  cat("WARNING: very few hepatocytes gated. Consider using cluster-based hepatocyte selection instead.\n")
}

obj_hep <- subset(obj, cells = hep_cells)
cat("Hepatocyte-only counts (before balancing):\n")
print(table(obj_hep$Condition))
cat("Hepatocyte-only object -> Cells:", ncol(obj_hep), " Genes:", nrow(obj_hep), "\n")

# JoinLayers on hepatocyte subset (v5-safe)
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers on hepatocyte subset...\n", sep = "")
t0 <- Sys.time()
DefaultAssay(obj_hep) <- "RNA"
obj_hep[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj_hep[["RNA"]],
  layers = c("counts", "data"),
  new    = c("counts", "data")
)
cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers(hepatocytes) done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")
cat("Hep RNA layers now:\n")
print(Layers(obj_hep[["RNA"]]))

pdf("COASSOLO_UMAP_By_Condition_hepatocytes_raw.pdf", width = 8, height = 6)
print(DimPlot(obj_hep, reduction = "umap", group.by = "Condition", pt.size = 0.4) +
        ggtitle("UMAP (hepatocytes-only, raw): Chow vs NASH"))
dev.off()

# ----------------------------
# [4] BALANCE Chow vs NASH within hepatocytes
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Balancing Chow vs NASH within hepatocytes...\n", sep = "")
set.seed(RANDOM_SEED)

Idents(obj_hep) <- obj_hep$Condition
cells_chow <- WhichCells(obj_hep, idents = "Chow")
cells_nash <- WhichCells(obj_hep, idents = "NASH")

cat("Hep cells Chow:", length(cells_chow), "  NASH:", length(cells_nash), "\n")

n_bal <- min(length(cells_chow), length(cells_nash))
if (!is.na(CAP_PER_GROUP)) n_bal <- min(n_bal, CAP_PER_GROUP)

cat("Balanced n per group:", n_bal, "\n")

chow_sub <- sample(cells_chow, n_bal)
nash_sub <- sample(cells_nash, n_bal)

obj_hep_bal <- subset(obj_hep, cells = c(chow_sub, nash_sub))

cat("After balancing (hepatocytes-only):\n")
print(table(obj_hep_bal$Condition))
cat("Balanced hepatocyte object -> Cells:", ncol(obj_hep_bal), " Genes:", nrow(obj_hep_bal), "\n")

# JoinLayers on balanced hepatocytes (v5-safe)
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Joining layers on balanced hepatocytes...\n", sep = "")
t0 <- Sys.time()
DefaultAssay(obj_hep_bal) <- "RNA"
obj_hep_bal[["RNA"]] <- SeuratObject::JoinLayers(
  object = obj_hep_bal[["RNA"]],
  layers = c("counts", "data"),
  new    = c("counts", "data")
)
cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers(hep balanced) done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")
cat("Balanced hep RNA layers now:\n")
print(Layers(obj_hep_bal[["RNA"]]))

pdf("COASSOLO_UMAP_By_Condition_hepatocytes_balanced.pdf", width = 8, height = 6)
print(DimPlot(obj_hep_bal, reduction = "umap", group.by = "Condition", pt.size = 0.4) +
        ggtitle("UMAP (hepatocytes-only, balanced): Chow vs NASH"))
dev.off()

# ----------------------------
# [5] Genome-wide AUROC + direction + p-values (HEPATOCYTES-ONLY, balanced)
# Rule: if max(pct) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Genome-wide AUROC + direction + p-values (hepatocytes-only, balanced)...\n", sep = "")

DefaultAssay(obj_hep_bal) <- "RNA"
Idents(obj_hep_bal) <- obj_hep_bal$Condition

cells_chow <- WhichCells(obj_hep_bal, idents = "Chow")
cells_nash <- WhichCells(obj_hep_bal, idents = "NASH")

cat("Balanced hepatocyte cells Chow:", length(cells_chow), "  NASH:", length(cells_nash), "\n")
cat("Processing genes:", nrow(obj_hep_bal), "\n")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Computing detection rates...\n", sep = "")
t0 <- Sys.time()

counts_mat_all <- GetAssayData(obj_hep_bal, assay = "RNA", layer = "counts")
pct_chow <- Matrix::rowSums(counts_mat_all[, cells_chow, drop = FALSE] > 0) / length(cells_chow)
pct_nash <- Matrix::rowSums(counts_mat_all[, cells_nash, drop = FALSE] > 0) / length(cells_nash)
pct_max  <- pmax(pct_chow, pct_nash)

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
  layer = "data",
  group.by = "Condition",
  verbose = FALSE
)$RNA

cat("[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

if (!all(c("Chow","NASH") %in% colnames(avg_expr))) {
  stop("AverageExpression did not return both Chow and NASH columns. Check obj_hep_bal$Condition.")
}

avg_chow <- avg_expr[, "Chow"]
avg_nash <- avg_expr[, "NASH"]
avg_logFC <- avg_nash - avg_chow

direction <- ifelse(avg_logFC > 0, "enriched_in_NASH",
                    ifelse(avg_logFC < 0, "enriched_in_Chow", "no_change"))

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] ROC AUC on ", length(keep_genes_all), " genes (slow step)...\n", sep = "")
t0 <- Sys.time()

auc_vec <- setNames(rep(0, nrow(avg_expr)), rownames(avg_expr))

if (length(keep_genes_all) > 0) {
  auc_keep <- FindMarkers(
    obj_hep_bal,
    ident.1 = "NASH",
    ident.2 = "Chow",
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
  ident.2 = "Chow",
  test.use = "wilcox",
  features = keep_genes_all,
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Wilcoxon done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Building genome-wide results table (hepatocytes-only)...\n", sep = "")
t0 <- Sys.time()

res_auc <- data.frame(
  gene = rownames(avg_expr),
  AUC = as.numeric(auc_vec[rownames(avg_expr)]),
  avg_logFC = as.numeric(avg_logFC[rownames(avg_expr)]),
  direction = direction[rownames(avg_expr)],
  mean_logexpr_Chow = as.numeric(avg_chow[rownames(avg_expr)]),
  mean_logexpr_NASH = as.numeric(avg_nash[rownames(avg_expr)]),
  pct_Chow = as.numeric(pct_chow[rownames(avg_expr)]),
  pct_NASH = as.numeric(pct_nash[rownames(avg_expr)]),
  max_pct = as.numeric(pct_max[rownames(avg_expr)]),
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

cat("[", format(Sys.time(), "%H:%M:%S"), "] Table built in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), " secs\n", sep = "")

write.csv(res_auc, "COASSOLO_HEPATOCYTES_gene_AUC_direction_stats_balanced_genomewide.csv", row.names = FALSE)

cat("\nTop 20 genes (AUC, direction, avg_logFC, padj):\n")
print(head(res_auc[, c("gene","AUC","direction","avg_logFC","pct_Chow","pct_NASH","p_val_adj")], 20))

# ----------------------------
# [6] Target gene list: DE + AUC table (same rule) + combined score + ROC + UMAP overlays
# (hepatocytes-only, balanced)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Target gene list (hepatocytes-only): DE + target-gene AUC + combined score...\n", sep = "")

target_genes <- readLines(TARGET_GENE_FILE) %>% trimws()
target_genes <- target_genes[target_genes != ""]

valid_genes <- intersect(target_genes, rownames(obj_hep_bal))
missing_genes <- setdiff(target_genes, rownames(obj_hep_bal))

write.csv(data.frame(MissingGene = missing_genes),
          "COASSOLO_HEPATOCYTES_target_genes_missing.csv",
          row.names = FALSE)

cat("Target genes:", length(target_genes), " | present:", length(valid_genes), " | missing:", length(missing_genes), "\n")
if (length(valid_genes) == 0) stop("No target genes found in the balanced hepatocyte object.")

deg_target <- FindMarkers(
  obj_hep_bal,
  ident.1 = "NASH",
  ident.2 = "Chow",
  features = valid_genes,
  test.use = "wilcox",
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)

deg_target$Gene <- rownames(deg_target)
deg_target <- deg_target %>% relocate(Gene)
write.csv(deg_target, "COASSOLO_HEPATOCYTES_DE_NASH_vs_Chow_target_genes_balanced.csv", row.names = FALSE)

# Detection for target genes
counts_targets <- GetAssayData(obj_hep_bal, assay = "RNA", layer = "counts")[valid_genes, , drop = FALSE]
pct_chow_t <- Matrix::rowMeans(counts_targets[, cells_chow, drop = FALSE] > 0)
pct_nash_t <- Matrix::rowMeans(counts_targets[, cells_nash, drop = FALSE] > 0)
pct_max_t  <- pmax(pct_chow_t, pct_nash_t)
keep_targets <- names(pct_max_t)[pct_max_t >= MIN_DETECT_PCT_FOR_AUC]

auc_t <- setNames(rep(0, length(valid_genes)), valid_genes)

if (length(keep_targets) > 0) {
  auc_keep_t <- FindMarkers(
    obj_hep_bal,
    ident.1 = "NASH",
    ident.2 = "Chow",
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
  pct_Chow = as.numeric(pct_chow_t[valid_genes]),
  pct_NASH = as.numeric(pct_nash_t[valid_genes]),
  max_pct = as.numeric(pct_max_t[valid_genes]),
  stringsAsFactors = FALSE
) %>%
  left_join(deg_target, by = "Gene")

if ("p_val_adj" %in% colnames(target_auc_table)) {
  target_auc_table <- target_auc_table %>% arrange(desc(AUC), p_val_adj)
} else {
  target_auc_table <- target_auc_table %>% arrange(desc(AUC))
}

write.csv(target_auc_table, "COASSOLO_HEPATOCYTES_target_genes_AUC_stats_balanced.csv", row.names = FALSE)

# Combined upregulated score (from target DE, avg_log2FC > 0)
upregulated_genes <- deg_target %>%
  filter(avg_log2FC > 0) %>%
  pull(Gene) %>%
  intersect(rownames(obj_hep_bal))

cat("Upregulated genes (target list, avg_log2FC>0) present:", length(upregulated_genes), "\n")

if (length(upregulated_genes) >= 1) {
  expr_up <- GetAssayData(obj_hep_bal, assay = "RNA", layer = "data")[upregulated_genes, , drop = FALSE]
  obj_hep_bal$Upregulated_Score <- as.numeric(Matrix::colMeans(expr_up))

  labels <- ifelse(obj_hep_bal$Condition == "NASH", 1, 0)
  combined_roc <- roc(response = labels, predictor = obj_hep_bal$Upregulated_Score, quiet = TRUE)

  pdf("COASSOLO_HEPATOCYTES_ROC_Combined_Upregulated_Genes_balanced.pdf", width = 6, height = 6)
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

  pdf("COASSOLO_HEPATOCYTES_UMAP_Three_Color_Overlay_balanced.pdf", width = 8, height = 6)
  print(
    ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
      geom_point(size = 0.4, alpha = 0.8) +
      ggtitle(paste0("UMAP (hepatocytes): Chow vs NASH vs Expressing (thr=", COMBINED_SCORE_THRESHOLD, ")")) +
      theme_minimal()
  )
  dev.off()

  pdf("COASSOLO_HEPATOCYTES_UMAP_Chow_NASH_only_balanced.pdf", width = 8, height = 6)
  print(
    ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Condition)) +
      geom_point(size = 0.4, alpha = 0.8) +
      ggtitle("UMAP (hepatocytes): Chow vs NASH (balanced)") +
      theme_minimal()
  )
  dev.off()
} else {
  cat("No upregulated genes from target list; skipping combined score/ROC/UMAP overlays.\n")
}

# ----------------------------
# [7] Save objects
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Saving RDS objects...\n", sep = "")
saveRDS(obj, file = "COASSOLO_full_processed_QC_clustered.rds")
saveRDS(obj_hep, file = "COASSOLO_hepatocytes_raw.rds")
saveRDS(obj_hep_bal, file = "COASSOLO_hepatocytes_balanced.rds")

cat("\n============================================================\n")
cat("DONE:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total elapsed mins:", round(as.numeric(difftime(Sys.time(), t_all, units = "mins")), 2), "\n")
cat("============================================================\n")
