rm(list = ls())

setwd("/scratch_data/gy260/new_MASH_April_2025/coassollo_paper/")

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(pROC)
  library(patchwork)
  library(ggplot2)
})

set.seed(42)

# =========================
# CONFIG
# =========================
CHOW_DIR <- "GSM6431458_Chow/"
NASH_DIR <- "GSM6431459_NASH/"

MIN_FEATURES <- 200
MAX_MT_PCT   <- 5
PCA_DIMS     <- 1:10

CLUSTER_RES  <- 1.2

TARGET_GENE_FILE <- "my_genes_mus.txt"

MIN_DETECT_PCT_FOR_AUC <- 0.10   # if max(pct_Chow, pct_NASH) < this, force AUC=0
THRESHOLD_EXPRESSION_STATUS <- 0.15  # for "Expressing" in combined score plot

# =========================
# Step 1: Load 10X
# =========================
cat("\n[1/10] Reading 10X...\n")
chow_data <- Read10X(data.dir = CHOW_DIR)
nash_data <- Read10X(data.dir = NASH_DIR)

# If Read10X returns a list (multi-modal), pull Gene Expression explicitly
if (is.list(chow_data)) {
  if ("Gene Expression" %in% names(chow_data)) chow_data <- chow_data[["Gene Expression"]]
}
if (is.list(nash_data)) {
  if ("Gene Expression" %in% names(nash_data)) nash_data <- nash_data[["Gene Expression"]]
}

# =========================
# Step 2: Create objects + merge
# =========================
cat("\n[2/10] Creating Seurat objects + merging...\n")
chow <- CreateSeuratObject(counts = chow_data, project = "Chow")
nash <- CreateSeuratObject(counts = nash_data, project = "NASH")

chow$Condition <- "Chow"
nash$Condition <- "NASH"

obj <- merge(chow, y = nash, add.cell.ids = c("Chow", "NASH"))
obj$Condition <- factor(obj$Condition, levels = c("Chow", "NASH"))

cat("Merged cells:", ncol(obj), " | genes:", nrow(obj), "\n")

# Seurat v5: always join layers after merge
obj <- JoinLayers(obj)

# =========================
# Step 3: Preprocessing
# =========================
cat("\n[3/10] QC + normalisation + PCA/UMAP...\n")
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^(mt-|Mt-)")

obj <- subset(obj, subset = nFeature_RNA > MIN_FEATURES & percent.mt < MAX_MT_PCT)

cat("After QC cells:", ncol(obj), " | genes:", nrow(obj), "\n")
cat("Condition counts:\n")
print(table(obj$Condition))

obj <- NormalizeData(obj)

# Join again after normalisation (common source of "please run JoinLayers")
obj <- JoinLayers(obj)

obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj)
obj <- RunPCA(obj)
obj <- RunUMAP(obj, dims = PCA_DIMS)

# =========================
# Step 4: Clustering (higher resolution)
# =========================
cat("\n[4/10] Neighbours + clustering...\n")
obj <- FindNeighbors(obj, dims = PCA_DIMS)
obj <- FindClusters(obj, resolution = CLUSTER_RES)

cat("Clusters:\n")
print(table(Idents(obj)))

# =========================
# Step 5: Marker plots (clusters)
# =========================
cat("\n[5/10] Saving marker plots...\n")

pdf("Cluster_HepatocyteMarkers_Alb_Cyp2e1.pdf", width = 8, height = 6)
print(
  VlnPlot(obj, features = c("Alb", "Cyp2e1"), group.by = "seurat_clusters", pt.size = 0) +
    ggtitle("Alb and Cyp2e1 Expression by Cluster")
)
dev.off()

pdf("Cluster_StellateMarkers_Lrat_Col1a1.pdf", width = 8, height = 6)
print(
  VlnPlot(obj, features = c("Lrat", "Col1a1"), group.by = "seurat_clusters", pt.size = 0) +
    ggtitle("Lrat and Col1a1 Expression by Cluster")
)
dev.off()

pdf("UMAP_By_Cluster_IDs.pdf", width = 8, height = 6)
print(DimPlot(obj, reduction = "umap", label = TRUE) + ggtitle("UMAP: Cluster IDs"))
dev.off()

pdf("HSC_vs_Hepatocyte_Marker_UMAP.pdf", width = 10, height = 6)
print(FeaturePlot(obj, features = c("Lrat", "Col1a1", "Pdgfrb", "Alb", "Cyp2e1", "Ttr")))
dev.off()

marker_genes <- c("Alb", "Ttr", "Cyp2e1", "Col1a1", "Lrat", "Des", "Pdgfrb")
pdf("Heatmap_Hepatocyte_HSC_Markers_By_Cluster.pdf", width = 10, height = 8)
print(DoHeatmap(obj, features = marker_genes) + ggtitle("Marker Gene Expression by Cluster"))
dev.off()

pdf("FeaturePlots_Liver_Cell_Type_Markers.pdf", width = 8, height = 6)
print(FeaturePlot(obj, features = "Alb"))    # Hepatocyte
print(FeaturePlot(obj, features = "Lrat"))   # HSC
print(FeaturePlot(obj, features = "Csf1r"))  # Kupffer
print(FeaturePlot(obj, features = "Pecam1")) # Endothelial
print(FeaturePlot(obj, features = "Lyz2"))   # Myeloid
dev.off()

# =========================
# IMPORTANT: set identities back to Condition for DE/AUC
# =========================
Idents(obj) <- obj$Condition
cat("\nIdentities set to Condition for DE/AUC. Levels:\n")
print(levels(Idents(obj)))

# =========================
# Step 6: Load target genes + run DE (Wilcoxon) for stats/direction
# =========================
cat("\n[6/10] Loading target genes + DE (Wilcoxon)...\n")
target_genes <- readLines(TARGET_GENE_FILE) %>% trimws()
target_genes <- target_genes[target_genes != ""]

valid_genes <- intersect(target_genes, rownames(obj))
cat("Target genes:", length(target_genes), " | present in object:", length(valid_genes), "\n")
if (length(valid_genes) == 0) stop("No target genes found in the Seurat object.")

deg_target <- FindMarkers(
  obj,
  ident.1 = "NASH",
  ident.2 = "Chow",
  features = valid_genes,
  test.use = "wilcox",
  min.pct = 0,
  logfc.threshold = 0
)

deg_target$Gene <- rownames(deg_target)
deg_target <- deg_target %>% relocate(Gene)

write.csv(deg_target, "Coassolo_DE_NASH_vs_Chow_target_genes.csv", row.names = FALSE)

# =========================
# Step 7: AUC per gene (with min detection rule + full stats table)
# =========================
cat("\n[7/10] Per-gene AUC with detection thresholding...\n")

# Use raw counts for detection %
counts_mat <- GetAssayData(obj, assay = "RNA", slot = "counts")[valid_genes, , drop = FALSE]

cells_chow <- WhichCells(obj, idents = "Chow")
cells_nash <- WhichCells(obj, idents = "NASH")

cat("Cells Chow:", length(cells_chow), " | Cells NASH:", length(cells_nash), "\n")

pct_chow <- Matrix::rowMeans(counts_mat[, cells_chow, drop = FALSE] > 0)
pct_nash <- Matrix::rowMeans(counts_mat[, cells_nash, drop = FALSE] > 0)
pct_max  <- pmax(pct_chow, pct_nash)

keep_genes <- names(pct_max)[pct_max >= MIN_DETECT_PCT_FOR_AUC]
cat("Genes with max detection >= ", MIN_DETECT_PCT_FOR_AUC, ": ", length(keep_genes),
    " | forced-AUC=0 genes: ", length(valid_genes) - length(keep_genes), "\n", sep = "")

# Initialise all AUCs to 0, then fill only for keep_genes
auc_vec <- setNames(rep(0, length(valid_genes)), valid_genes)

if (length(keep_genes) > 0) {
  cat("Computing AUC via Seurat test.use='roc' for keep genes...\n")
  auc_keep <- FindMarkers(
    obj,
    ident.1 = "NASH",
    ident.2 = "Chow",
    features = keep_genes,
    test.use = "roc",
    min.pct = 0,
    logfc.threshold = 0
  )

  auc_keep$Gene <- rownames(auc_keep)
  auc_col <- intersect(c("myAUC", "AUC"), colnames(auc_keep))[1]
  if (is.na(auc_col)) stop("Could not find an AUC column from Seurat ROC test (expected myAUC).")

  auc_vec[auc_keep$Gene] <- auc_keep[[auc_col]]
}

# Build final AUC/statistics table
auc_table <- data.frame(
  Gene = valid_genes,
  AUC = as.numeric(auc_vec[valid_genes]),
  pct_Chow = as.numeric(pct_chow[valid_genes]),
  pct_NASH = as.numeric(pct_nash[valid_genes]),
  pct_max = as.numeric(pct_max[valid_genes]),
  stringsAsFactors = FALSE
)

# Merge in DE stats (p_val, p_val_adj, avg_log2FC, pct.1, pct.2 if present)
auc_table <- auc_table %>%
  left_join(deg_target, by = "Gene")

# Sort by AUC (desc), then by adjusted p (asc) as tie-breaker if present
if ("p_val_adj" %in% colnames(auc_table)) {
  auc_table <- auc_table %>% arrange(desc(AUC), p_val_adj)
} else {
  auc_table <- auc_table %>% arrange(desc(AUC))
}

write.csv(auc_table, "AUC_scores_target_genes_full_stats.csv", row.names = FALSE)
cat("Saved: AUC_scores_target_genes_full_stats.csv\n")

# =========================
# Step 8: Combined upregulated score & ROC
# =========================
cat("\n[8/10] Combined upregulated score + ROC...\n")
upregulated_genes <- deg_target %>%
  filter(avg_log2FC > 0) %>%
  pull(Gene) %>%
  intersect(rownames(obj))

cat("Upregulated genes (avg_log2FC>0) present:", length(upregulated_genes), "\n")
if (length(upregulated_genes) == 0) stop("No upregulated genes found after filtering avg_log2FC > 0.")

expr_up <- GetAssayData(obj, assay = "RNA", slot = "data")[upregulated_genes, , drop = FALSE]
combined_expression <- Matrix::colMeans(expr_up)
obj$Upregulated_Score <- as.numeric(combined_expression)

labels <- ifelse(obj$Condition == "NASH", 1, 0)
combined_roc <- roc(response = labels, predictor = obj$Upregulated_Score, quiet = TRUE)

pdf("Coassollo_ROCCombined_ROC_Upregulated_Genes.pdf", width = 6, height = 6)
plot(combined_roc, col = "darkblue", main = "ROC: Combined Upregulated Gene Score")
legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)),
       col = "darkblue", lwd = 2)
dev.off()

# =========================
# Step 9: Classify "Expressing" cells + UMAP overlays
# =========================
cat("\n[9/10] UMAP overlays...\n")
threshold <- THRESHOLD_EXPRESSION_STATUS

obj$ExpressionStatus <- ifelse(
  obj$Upregulated_Score > threshold, "Expressing",
  as.character(obj$Condition)
)

umap_df <- as.data.frame(Embeddings(obj, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")
umap_df$Status <- obj$ExpressionStatus

pdf("UMAP_Three_Color_Overlay0.15_new.pdf", width = 8, height = 6)
print(
  ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
    geom_point(size = 0.4, alpha = 0.8) +
    scale_color_manual(values = c("Chow" = "blue", "NASH" = "red", "Expressing" = "lightgrey")) +
    ggtitle(paste0("UMAP: Chow vs NASH vs Expressing Cells (threshold=", threshold, ")")) +
    theme_minimal()
)
dev.off()

umap_df2 <- as.data.frame(Embeddings(obj, "umap"))
colnames(umap_df2) <- c("UMAP_1", "UMAP_2")
umap_df2$Condition <- obj$Condition

pdf("UMAP_Chow_NASH_Coloring_Only.pdf", width = 8, height = 6)
print(
  ggplot(umap_df2, aes(x = UMAP_1, y = UMAP_2, color = Condition)) +
    geom_point(size = 0.4, alpha = 0.8) +
    scale_color_manual(values = c("Chow" = "blue", "NASH" = "red")) +
    ggtitle("UMAP: Chow vs NASH (all cells shown)") +
    theme_minimal()
)
dev.off()

# =========================
# Step 10: Save object
# =========================
cat("\n[10/10] Saving Seurat object...\n")
saveRDS(obj, file = "Coassolo_Chow_NASH_processed.rds")
cat("Done.\n")
