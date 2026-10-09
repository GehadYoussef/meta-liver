# ============================================================
# COASSOLO mouse liver scRNA-seq (Chow vs NASH)
# SU-style script:
# - Load 10X + merge
# - QC -> Normalise -> PCA -> UMAP -> cluster (for plots only)
# - BALANCE (downsample) Chow vs NASH for fair per-gene stats
# - JoinLayers (Seurat v5-safe)
# - Per-gene AUROC + direction + p-values with rule:
#     if max(pct.NASH, pct.Chow) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# - Target-gene DE + target-gene AUC table (same rule) + combined score + ROC + UMAP overlays
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
# [2] QC + normalisation + PCA/UMAP + clustering (for plots)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] QC + normalise + PCA/UMAP + clustering...\n", sep = "")
t0 <- Sys.time()

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^(mt-|Mt-)")
obj <- subset(obj, subset = nFeature_RNA > MIN_FEATURES & percent.mt < MAX_MT_PCT)

cat("After QC -> Cells:", ncol(obj), " Genes:", nrow(obj), "\n")
cat("Condition counts after QC:\n")
print(table(obj$Condition))

obj <- NormalizeData(obj, verbose = FALSE)

if ("JoinLayers" %in% getNamespaceExports("Seurat")) {
  cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers after NormalizeData...\n", sep = "")
  obj <- JoinLayers(obj)
}

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

# ----------------------------
# [3] Marker plots (clusters)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Saving marker plots...\n", sep = "")

pdf("Cluster_HepatocyteMarkers_Alb_Cyp2e1.pdf", width = 8, height = 6)
print(VlnPlot(obj, features = c("Alb","Cyp2e1"), group.by = "seurat_clusters", pt.size = 0) +
        ggtitle("Alb and Cyp2e1 Expression by Cluster"))
dev.off()

pdf("Cluster_StellateMarkers_Lrat_Col1a1.pdf", width = 8, height = 6)
print(VlnPlot(obj, features = c("Lrat","Col1a1"), group.by = "seurat_clusters", pt.size = 0) +
        ggtitle("Lrat and Col1a1 Expression by Cluster"))
dev.off()

pdf("UMAP_By_Cluster_IDs.pdf", width = 8, height = 6)
print(DimPlot(obj, reduction = "umap", label = TRUE) + ggtitle("UMAP: Cluster IDs"))
dev.off()

pdf("HSC_vs_Hepatocyte_Marker_UMAP.pdf", width = 10, height = 6)
print(FeaturePlot(obj, features = c("Lrat","Col1a1","Pdgfrb","Alb","Cyp2e1","Ttr")))
dev.off()

marker_genes <- c("Alb","Ttr","Cyp2e1","Col1a1","Lrat","Des","Pdgfrb")
pdf("Heatmap_Hepatocyte_HSC_Markers_By_Cluster.pdf", width = 10, height = 8)
print(DoHeatmap(obj, features = marker_genes) + ggtitle("Marker Gene Expression by Cluster"))
dev.off()

pdf("FeaturePlots_Liver_Cell_Type_Markers.pdf", width = 8, height = 6)
print(FeaturePlot(obj, features = "Alb"))
print(FeaturePlot(obj, features = "Lrat"))
print(FeaturePlot(obj, features = "Csf1r"))
print(FeaturePlot(obj, features = "Pecam1"))
print(FeaturePlot(obj, features = "Lyz2"))
dev.off()

# ----------------------------
# [4] BALANCE Chow vs NASH (SU-style fairness before AUROC/DE)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Balancing Chow vs NASH by downsampling...\n", sep = "")
set.seed(RANDOM_SEED)

Idents(obj) <- obj$Condition

cells_chow <- WhichCells(obj, idents = "Chow")
cells_nash <- WhichCells(obj, idents = "NASH")

cat("Cells Chow:", length(cells_chow), "  NASH:", length(cells_nash), "\n")

n_bal <- min(length(cells_chow), length(cells_nash))
if (!is.na(CAP_PER_GROUP)) n_bal <- min(n_bal, CAP_PER_GROUP)

cat("Balanced n per group:", n_bal, "\n")

chow_sub <- sample(cells_chow, n_bal)
nash_sub <- sample(cells_nash, n_bal)

obj_bal <- subset(obj, cells = c(chow_sub, nash_sub))

cat("After balancing:\n")
print(table(obj_bal$Condition))
cat("Balanced object -> Cells:", ncol(obj_bal), " Genes:", nrow(obj_bal), "\n")

if ("JoinLayers" %in% getNamespaceExports("Seurat")) {
  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers on balanced object...\n", sep = "")
  t0 <- Sys.time()
  obj_bal <- JoinLayers(obj_bal)
  cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers(balanced) done in ",
      round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")
}

pdf("COASSOLO_UMAP_By_Condition_balanced.pdf", width = 8, height = 6)
print(DimPlot(obj_bal, group.by = "Condition", label = TRUE, pt.size = 0.4) +
        ggtitle("UMAP (balanced): Chow vs NASH"))
dev.off()

# ----------------------------
# [5] Per-gene AUROC + direction + p-values (WHOLE TRANSCRIPTOME, SU-style)
# rule: if max(pct) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Per-gene AUROC + direction + p-values (balanced, genome-wide)...\n", sep = "")

DefaultAssay(obj_bal) <- "RNA"
Idents(obj_bal) <- obj_bal$Condition

cells_chow <- WhichCells(obj_bal, idents = "Chow")
cells_nash <- WhichCells(obj_bal, idents = "NASH")

cat("Balanced cells Chow:", length(cells_chow), "  NASH:", length(cells_nash), "\n")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Computing detection rates...\n", sep = "")
t0 <- Sys.time()
counts_mat_all <- GetAssayData(obj_bal, assay = "RNA", slot = "counts")
pct_chow <- Matrix::rowSums(counts_mat_all[, cells_chow, drop = FALSE] > 0) / length(cells_chow)
pct_nash <- Matrix::rowSums(counts_mat_all[, cells_nash, drop = FALSE] > 0) / length(cells_nash)
pct_max  <- pmax(pct_chow, pct_nash)
cat("[", format(Sys.time(), "%H:%M:%S"), "] Detection rates done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

keep_genes_all <- names(pct_max)[pct_max >= MIN_DETECT_PCT_FOR_AUC]
cat("Genes total:", length(pct_max), "\n")
cat("Genes with max(pct) >= ", MIN_DETECT_PCT_FOR_AUC, ": ", length(keep_genes_all), "\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression for direction...\n", sep = "")
t0 <- Sys.time()
avg_expr <- AverageExpression(
  obj_bal,
  assays = "RNA",
  slot = "data",
  group.by = "Condition",
  verbose = FALSE
)$RNA
cat("[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

avg_chow <- avg_expr[, "Chow"]
avg_nash <- avg_expr[, "NASH"]
avg_logFC <- avg_nash - avg_chow
direction <- ifelse(avg_logFC > 0, "enriched_in_NASH",
                    ifelse(avg_logFC < 0, "enriched_in_Chow", "no_change"))

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] ROC AUC on ", length(keep_genes_all), " genes (slow step)...\n", sep = "")
t0 <- Sys.time()
auc_keep <- FindMarkers(
  obj_bal,
  ident.1 = "NASH",
  ident.2 = "Chow",
  test.use = "roc",
  features = keep_genes_all,
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)
cat("[", format(Sys.time(), "%H:%M:%S"), "] ROC AUC done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins\n", sep = "")

auc_col <- intersect(c("myAUC","AUC"), colnames(auc_keep))[1]
if (is.na(auc_col)) stop("Could not find myAUC/AUC in Seurat ROC output.")

auc_vec <- rep(0, nrow(avg_expr))
names(auc_vec) <- rownames(avg_expr)
auc_vec[rownames(auc_keep)] <- auc_keep[[auc_col]]

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Wilcoxon p-values on ", length(keep_genes_all), " genes...\n", sep = "")
t0 <- Sys.time()
de_keep <- FindMarkers(
  obj_bal,
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

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Building genome-wide results table...\n", sep = "")
t0 <- Sys.time()

res_auc <- data.frame(
  gene = rownames(avg_expr),
  AUC = as.numeric(auc_vec),
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

res_auc$tested[match(rownames(de_keep), res_auc$gene)] <- TRUE
res_auc$p_val[match(rownames(de_keep), res_auc$gene)] <- de_keep$p_val
res_auc$p_val_adj[match(rownames(de_keep), res_auc$gene)] <- de_keep$p_val_adj

res_auc$AUC[res_auc$max_pct < MIN_DETECT_PCT_FOR_AUC] <- 0
res_auc <- res_auc[order(-res_auc$AUC, -abs(res_auc$avg_logFC)), ]

cat("[", format(Sys.time(), "%H:%M:%S"), "] Table built in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), " secs\n", sep = "")

write.csv(res_auc, "COASSOLO_gene_AUC_direction_stats_balanced_genomewide.csv", row.names = FALSE)

cat("\nTop 20 genes (AUC, direction, avg_logFC, padj):\n")
print(head(res_auc[, c("gene","AUC","direction","avg_logFC","pct_Chow","pct_NASH","p_val_adj")], 20))

# ----------------------------
# [6] Target gene list: DE + AUC table (same SU logic) + combined score + ROC + UMAP overlays
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Target gene list: DE + target-gene AUC + combined score...\n", sep = "")

target_genes <- readLines(TARGET_GENE_FILE) %>% trimws()
target_genes <- target_genes[target_genes != ""]
valid_genes <- intersect(target_genes, rownames(obj_bal))
missing_genes <- setdiff(target_genes, rownames(obj_bal))

write.csv(data.frame(MissingGene = missing_genes), "COASSOLO_target_genes_missing.csv", row.names = FALSE)

cat("Target genes:", length(target_genes), " | present:", length(valid_genes), " | missing:", length(missing_genes), "\n")
if (length(valid_genes) == 0) stop("No target genes found in the balanced object.")

deg_target <- FindMarkers(
  obj_bal,
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
write.csv(deg_target, "COASSOLO_DE_NASH_vs_Chow_target_genes_balanced.csv", row.names = FALSE)

counts_targets <- GetAssayData(obj_bal, assay = "RNA", slot = "counts")[valid_genes, , drop = FALSE]
pct_chow_t <- Matrix::rowMeans(counts_targets[, cells_chow, drop = FALSE] > 0)
pct_nash_t <- Matrix::rowMeans(counts_targets[, cells_nash, drop = FALSE] > 0)
pct_max_t  <- pmax(pct_chow_t, pct_nash_t)
keep_targets <- names(pct_max_t)[pct_max_t >= MIN_DETECT_PCT_FOR_AUC]

auc_t <- setNames(rep(0, length(valid_genes)), valid_genes)
if (length(keep_targets) > 0) {
  auc_keep_t <- FindMarkers(
    obj_bal,
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
) %>% left_join(deg_target, by = "Gene") %>%
  arrange(desc(AUC), p_val_adj)

write.csv(target_auc_table, "COASSOLO_target_genes_AUC_direction_stats_balanced.csv", row.names = FALSE)

upregulated_genes <- deg_target %>%
  filter(avg_log2FC > 0) %>%
  pull(Gene) %>%
  intersect(rownames(obj_bal))

cat("Upregulated genes (target list, avg_log2FC>0) present:", length(upregulated_genes), "\n")

if (length(upregulated_genes) >= 1) {
  expr_up <- GetAssayData(obj_bal, assay = "RNA", slot = "data")[upregulated_genes, , drop = FALSE]
  obj_bal$Upregulated_Score <- as.numeric(Matrix::colMeans(expr_up))

  labels <- ifelse(obj_bal$Condition == "NASH", 1, 0)
  combined_roc <- roc(response = labels, predictor = obj_bal$Upregulated_Score, quiet = TRUE)

  pdf("COASSOLO_ROC_Combined_Upregulated_Genes_balanced.pdf", width = 6, height = 6)
  plot(combined_roc, col = "darkblue", main = "ROC: Combined Upregulated Gene Score (balanced)")
  legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)), col = "darkblue", lwd = 2)
  dev.off()

  obj_bal$ExpressionStatus <- ifelse(
    obj_bal$Upregulated_Score > COMBINED_SCORE_THRESHOLD, "Expressing",
    as.character(obj_bal$Condition)
  )

  umap_df <- as.data.frame(Embeddings(obj_bal, "umap"))
  colnames(umap_df) <- c("UMAP_1","UMAP_2")
  umap_df$Status <- obj_bal$ExpressionStatus
  umap_df$Condition <- obj_bal$Condition

  pdf("COASSOLO_UMAP_Three_Color_Overlay_balanced.pdf", width = 8, height = 6)
  print(
    ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
      geom_point(size = 0.4, alpha = 0.8) +
      ggtitle(paste0("UMAP: Chow vs NASH vs Expressing (threshold=", COMBINED_SCORE_THRESHOLD, ")")) +
      theme_minimal()
  )
  dev.off()

  pdf("COASSOLO_UMAP_Chow_NASH_only_balanced.pdf", width = 8, height = 6)
  print(
    ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Condition)) +
      geom_point(size = 0.4, alpha = 0.8) +
      ggtitle("UMAP: Chow vs NASH (balanced)") +
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
saveRDS(obj, file = "COASSOLO_full_processed.rds")
saveRDS(obj_bal, file = "COASSOLO_balanced_processed.rds")

cat("\n============================================================\n")
cat("DONE:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total elapsed mins:", round(as.numeric(difftime(Sys.time(), t_all, units = "mins")), 2), "\n")
cat("============================================================\n")
