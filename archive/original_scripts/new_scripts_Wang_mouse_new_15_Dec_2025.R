# ============================================================
# Wang mouse liver scRNA-seq (mouseNASH_1 + mouseNASH_2)
# Standard Seurat workflow + canonical marker plots
# Cluster markers + heatmap (robust to Seurat column names)
# Manual cluster->celltype annotation
# Target gene enrichment in hepatocytes vs all other cells
# ROC/AUC for target genes: hepatocytes (1) vs non-hepatocytes (0)
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(pROC)
  library(patchwork)
  library(ggplot2)
})

setwd("/scratch_data/gy260/new_MASH_April_2025/Wang_paper/Wang_mus")

RANDOM_SEED <- 42
MIN_DETECT_PCT_FOR_AUC <- 0.10
MARKER_MIN_PCT <- 0.25
MARKER_LOGFC <- 0.25
HEATMAP_TOPN <- 10

cat("\n============================================================\n")
cat("START:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n")

t_all <- Sys.time()

# ----------------------------
# Load 10X
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Reading 10X...\n", sep = "")
t0 <- Sys.time()

data1 <- Read10X(data.dir = "mouseNASH_1/")
data2 <- Read10X(data.dir = "mouseNASH_2/")

cat("[", format(Sys.time(), "%H:%M:%S"), "] Read10X done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")

# ----------------------------
# Create + merge
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Creating Seurat objects + merge...\n", sep = "")
t0 <- Sys.time()

seu1 <- CreateSeuratObject(counts = data1, project = "mouseNASH1")
seu2 <- CreateSeuratObject(counts = data2, project = "mouseNASH2")

seu1$sample <- "mouseNASH1"
seu2$sample <- "mouseNASH2"

mouse_obj <- merge(seu1, y = seu2, add.cell.ids = c("NASH1", "NASH2"))

cat("[", format(Sys.time(), "%H:%M:%S"), "] Merge done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")
cat("Cells:", ncol(mouse_obj), " Genes:", nrow(mouse_obj), "\n")

# ----------------------------
# QC
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] QC + filtering...\n", sep = "")
t0 <- Sys.time()

mouse_obj[["percent.mt"]] <- PercentageFeatureSet(mouse_obj, pattern = "^mt-")
mouse_obj <- subset(mouse_obj, subset = nFeature_RNA > 200 & percent.mt < 5)

cat("[", format(Sys.time(), "%H:%M:%S"), "] QC filter done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")
cat("After QC -> Cells:", ncol(mouse_obj), " Genes:", nrow(mouse_obj), "\n")

# ----------------------------
# Normalise + HVGs + scale + PCA + neighbours + clusters + UMAP
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Normalise -> PCA -> UMAP...\n", sep = "")
t0 <- Sys.time()

mouse_obj <- NormalizeData(mouse_obj, verbose = FALSE)
mouse_obj <- FindVariableFeatures(mouse_obj, verbose = FALSE)
mouse_obj <- ScaleData(mouse_obj, verbose = FALSE)
mouse_obj <- RunPCA(mouse_obj, verbose = FALSE)
mouse_obj <- FindNeighbors(mouse_obj, dims = 1:10, verbose = FALSE)
mouse_obj <- FindClusters(mouse_obj, resolution = 0.5, verbose = FALSE)
mouse_obj <- RunUMAP(mouse_obj, dims = 1:10, verbose = FALSE)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Dim-reduction done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

# ----------------------------
# Canonical marker FeaturePlots
# ----------------------------
marker_genes <- list(
  Hepatocytes = c("Alb", "Ttr", "Cyp2e1"),
  Stellate = c("Lrat", "Col1a1", "Acta2"),
  Endothelial = c("Pecam1", "Cdh5"),
  Macrophages = c("Adgre1", "Cd68", "Csf1r"),
  Monocytes = c("Ly6c1", "Ccr2"),
  T_Cells = c("Cd3d", "Cd3e"),
  B_Cells = c("Cd79a", "Ms4a1"),
  NK_Cells = c("Nkg7", "Klrb1c"),
  Dendritic_Cells = c("Flt3", "Xcr1")
)

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Saving marker FeaturePlots...\n", sep = "")
pdf("mouse_NASH_marker_genes.pdf", width = 12, height = 10)
print(FeaturePlot(mouse_obj, features = unique(unlist(marker_genes)), ncol = 4))
dev.off()

# ----------------------------
# JoinLayers (Seurat v5-safe)
# ----------------------------
if ("JoinLayers" %in% getNamespaceExports("Seurat")) {
  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers detected -> joining...\n", sep = "")
  t0 <- Sys.time()
  mouse_obj <- JoinLayers(mouse_obj)
  cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers done in ",
      round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
      " secs\n", sep = "")
} else {
  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers not available -> skipping.\n", sep = "")
}

# ----------------------------
# Cluster markers + robust heatmap
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] FindAllMarkers...\n", sep = "")
cat("Identities (Idents):", paste(head(levels(Idents(mouse_obj))), collapse = ", "), "\n")
cat("Cells:", ncol(mouse_obj), " Genes:", nrow(mouse_obj), "\n")

t0 <- Sys.time()
markers <- FindAllMarkers(
  mouse_obj,
  only.pos = TRUE,
  min.pct = MARKER_MIN_PCT,
  logfc.threshold = MARKER_LOGFC
)
cat("[", format(Sys.time(), "%H:%M:%S"), "] FindAllMarkers done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

if (nrow(markers) == 0) {
  cat("\nNo markers returned (nrow=0). Lower min.pct/logfc.threshold or check clustering.\n")
} else {
  cat("\nFindAllMarkers columns:\n")
  print(colnames(markers))

  cluster_col <- if ("cluster" %in% colnames(markers)) {
    "cluster"
  } else if ("ident" %in% colnames(markers)) {
    "ident"
  } else if ("group" %in% colnames(markers)) {
    "group"
  } else {
    stop("No cluster/ident/group column found in FindAllMarkers output.")
  }

  fc_col <- if ("avg_log2FC" %in% colnames(markers)) {
    "avg_log2FC"
  } else if ("avg_logFC" %in% colnames(markers)) {
    "avg_logFC"
  } else {
    stop("No avg_log2FC/avg_logFC column found in FindAllMarkers output.")
  }

  topN <- markers %>%
    dplyr::group_by(.data[[cluster_col]]) %>%
    dplyr::slice_max(order_by = .data[[fc_col]], n = HEATMAP_TOPN, with_ties = FALSE) %>%
    dplyr::ungroup()

  genes_to_scale <- unique(topN$gene)
  cat("Heatmap genes:", length(genes_to_scale), "\n")

  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] ScaleData for heatmap genes...\n", sep = "")
  t0 <- Sys.time()
  mouse_obj <- ScaleData(mouse_obj, features = genes_to_scale, verbose = FALSE)
  cat("[", format(Sys.time(), "%H:%M:%S"), "] ScaleData done in ",
      round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
      " secs\n", sep = "")

  pdf("mouse_NASH_cluster_heatmap_updated.pdf", width = 10, height = 12)
  print(DoHeatmap(mouse_obj, features = genes_to_scale) + NoLegend())
  dev.off()

  write.csv(markers, "mouseNASH_markers_FindAllMarkers.csv", row.names = FALSE)
}

# ----------------------------
# Average expression of canonical markers
# ----------------------------
DefaultAssay(mouse_obj) <- "RNA"
avg_expr <- AverageExpression(mouse_obj, features = unique(unlist(marker_genes)), return.seurat = FALSE)$RNA
write.csv(avg_expr, "avg_expression_canonical_markers_mouse.csv")

# ----------------------------
# Manual cluster annotation -> celltype (no plyr)
# ----------------------------
Idents(mouse_obj) <- "seurat_clusters"

cluster_annot <- c(
  "0" = "Hepatocytes", "1" = "Hepatocytes", "2" = "Mixed Hepato/Macrophage",
  "3" = "Stellate Cells", "4" = "Hepatocytes", "5" = "Hepatocytes",
  "6" = "Hepatocytes", "7" = "NK Cells", "8" = "Macrophages",
  "9" = "Endothelial/B Cells", "10" = "Hepatocytes", "11" = "Macrophages",
  "12" = "Macrophages", "13" = "Hepatocytes", "14" = "Stellate Cells"
)

mouse_obj$celltype <- unname(cluster_annot[as.character(mouse_obj$seurat_clusters)])
mouse_obj$celltype[is.na(mouse_obj$celltype)] <- "Unannotated"
mouse_obj$celltype <- factor(mouse_obj$celltype)

Idents(mouse_obj) <- "celltype"

cat("\nCelltype counts:\n")
print(table(mouse_obj$celltype))

# ----------------------------
# Target genes: enrichment in hepatocytes vs all other cells
# ----------------------------
target_genes <- readLines("my_genes_mus.txt") %>% trimws()
target_genes <- target_genes[target_genes != ""]
valid_genes <- intersect(target_genes, rownames(mouse_obj))

cat("\nTarget genes total:", length(target_genes), " | present:", length(valid_genes), "\n")
if (length(valid_genes) == 0) stop("None of the target genes were found in the Seurat object.")

markers_hepato <- FindMarkers(
  object = mouse_obj,
  ident.1 = "Hepatocytes",
  ident.2 = NULL,
  features = valid_genes,
  logfc.threshold = 0,
  min.pct = 0,
  verbose = TRUE
)
write.csv(markers_hepato, "hepatocyte_enrichment_target_genes.csv")

avg_expr_target <- AverageExpression(mouse_obj, features = valid_genes, return.seurat = FALSE)$RNA
write.csv(avg_expr_target, "target_gene_avg_expression_by_celltype.csv")

# ----------------------------
# ROC/AUC per target gene: hepatocytes (1) vs non-hepatocytes (0)
# with detection threshold rule (based on raw counts)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] ROC/AUC per target gene (hepatocytes vs rest)...\n", sep = "")

labels <- ifelse(mouse_obj$celltype == "Hepatocytes", 1, 0)

cells_hep <- WhichCells(mouse_obj, idents = "Hepatocytes")
cells_rest <- setdiff(colnames(mouse_obj), cells_hep)

counts_mat <- GetAssayData(mouse_obj, assay = "RNA", slot = "counts")[valid_genes, , drop = FALSE]
pct_hep  <- Matrix::rowMeans(counts_mat[, cells_hep, drop = FALSE] > 0)
pct_rest <- Matrix::rowMeans(counts_mat[, cells_rest, drop = FALSE] > 0)
pct_max  <- pmax(pct_hep, pct_rest)

expr_data <- GetAssayData(mouse_obj, assay = "RNA", slot = "data")[valid_genes, , drop = FALSE]

gene_auc <- sapply(valid_genes, function(g) {
  if (is.na(pct_max[g]) || pct_max[g] < MIN_DETECT_PCT_FOR_AUC) return(0)
  roc_obj <- try(roc(response = labels, predictor = as.numeric(expr_data[g, ]), quiet = TRUE), silent = TRUE)
  if (inherits(roc_obj, "try-error")) return(NA_real_)
  as.numeric(auc(roc_obj))
})

auc_df <- data.frame(
  Gene = valid_genes,
  AUC = as.numeric(gene_auc[valid_genes]),
  pct_Hepatocytes = as.numeric(pct_hep[valid_genes]),
  pct_NonHepatocytes = as.numeric(pct_rest[valid_genes]),
  max_pct = as.numeric(pct_max[valid_genes]),
  stringsAsFactors = FALSE
) %>% arrange(desc(AUC))

write.csv(auc_df, "ROC_AUC_TargetGenes_Hepatocytes.csv", row.names = FALSE)

# ----------------------------
# Combined score (mean of target genes) + ROC
# ----------------------------
combined_expression <- Matrix::colMeans(expr_data)
mouse_obj$TargetGene_Score <- as.numeric(combined_expression)

combined_roc <- roc(response = labels, predictor = mouse_obj$TargetGene_Score, quiet = TRUE)

pdf("ROC_combined_target_genes_mouse.pdf", width = 6, height = 6)
plot(combined_roc, col = "darkred", main = "ROC: Combined Target Gene Score (Hepatocytes vs Others)")
legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)), col = "darkred", lwd = 2)
dev.off()

# ----------------------------
# UMAP overlay: "Expressing" based on threshold
# ----------------------------
threshold <- 0.15
mouse_obj$ExpressionStatus <- ifelse(
  mouse_obj$TargetGene_Score > threshold, "Expressing",
  as.character(mouse_obj$celltype)
)

umap_df <- as.data.frame(Embeddings(mouse_obj, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")
umap_df$Status <- mouse_obj$ExpressionStatus

pdf("UMAP_Three_Color_Overlay_Mouse.pdf", width = 8, height = 6)
print(
  ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, colour = Status)) +
    geom_point(size = 0.4, alpha = 0.8) +
    ggtitle("UMAP: TargetGene_Score Expressing vs Celltypes") +
    theme_minimal()
)
dev.off()

# ----------------------------
# Save object
# ----------------------------
saveRDS(mouse_obj, "mouseNASH_merged_annotated_object.rds")

cat("\n============================================================\n")
cat("DONE:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total elapsed mins:", round(as.numeric(difftime(Sys.time(), t_all, units = "mins")), 2), "\n")
cat("============================================================\n")
