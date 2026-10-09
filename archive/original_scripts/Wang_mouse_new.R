
setwd("/scratch_data/gy260/new_MASH_April_2025/Wang_paper/Wang_mus")
library(Seurat)
library(Matrix)
library(dplyr)
library(pROC)
library(patchwork)
library(plyr)

data1 <- Read10X(data.dir = "mouseNASH_1/")
data2 <- Read10X(data.dir = "mouseNASH_2/")

seu1 <- CreateSeuratObject(counts = data1, project = "mouseNASH1")
seu2 <- CreateSeuratObject(counts = data2, project = "mouseNASH2")

seu1$sample <- "mouseNASH1"
seu2$sample <- "mouseNASH2"

mouse_obj <- merge(seu1, y = seu2, add.cell.ids = c("NASH1", "NASH2"))

mouse_obj[["percent.mt"]] <- PercentageFeatureSet(mouse_obj, pattern = "^mt-")
mouse_obj <- subset(mouse_obj, subset = nFeature_RNA > 200 & percent.mt < 5)
mouse_obj <- NormalizeData(mouse_obj)
mouse_obj <- FindVariableFeatures(mouse_obj)
mouse_obj <- ScaleData(mouse_obj)
mouse_obj <- RunPCA(mouse_obj)
mouse_obj <- FindNeighbors(mouse_obj, dims = 1:10)
mouse_obj <- FindClusters(mouse_obj, resolution = 0.5)
mouse_obj <- RunUMAP(mouse_obj, dims = 1:10)

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

pdf("mouse_NASH_marker_genes.pdf", width = 12, height = 10)
FeaturePlot(mouse_obj, features = unlist(marker_genes), ncol = 4)
dev.off()

mouse_obj <- JoinLayers(mouse_obj)

markers <- FindAllMarkers(mouse_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
top10 <- markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)
genes_to_scale <- unique(top10$gene)
mouse_obj <- ScaleData(mouse_obj, features = genes_to_scale)

pdf("mouse_NASH_cluster_heatmap_updated.pdf", width = 10, height = 12)
DoHeatmap(mouse_obj, features = genes_to_scale) + NoLegend()
dev.off()

DefaultAssay(mouse_obj) <- "RNA"
avg_expr <- AverageExpression(mouse_obj, features = unlist(marker_genes), return.seurat = FALSE)$RNA
write.csv(avg_expr, "avg_expression_canonical_markers_mouse.csv")

Idents(mouse_obj) <- "seurat_clusters"
cluster_annot <- c(
  "0" = "Hepatocytes", "1" = "Hepatocytes", "2" = "Mixed Hepato/Macrophage",
  "3" = "Stellate Cells", "4" = "Hepatocytes", "5" = "Hepatocytes",
  "6" = "Hepatocytes", "7" = "NK Cells", "8" = "Macrophages",
  "9" = "Endothelial/B Cells", "10" = "Hepatocytes", "11" = "Macrophages",
  "12" = "Macrophages", "13" = "Hepatocytes", "14" = "Stellate Cells"
)
mouse_obj$celltype <- plyr::mapvalues(as.character(mouse_obj$seurat_clusters), from = names(cluster_annot), to = cluster_annot)
Idents(mouse_obj) <- "celltype"

target_genes <- readLines("my_genes_mus.txt") %>% trimws()
valid_genes <- intersect(target_genes, rownames(mouse_obj))

markers_hepato <- FindMarkers(
  object = mouse_obj,
  ident.1 = "Hepatocytes",
  ident.2 = NULL,
  features = valid_genes,
  logfc.threshold = 0,
  min.pct = 0
)
write.csv(markers_hepato, "hepatocyte_enrichment_target_genes.csv")

avg_expr_target <- AverageExpression(mouse_obj, features = valid_genes, return.seurat = FALSE)$RNA
write.csv(avg_expr_target, "target_gene_avg_expression_by_celltype.csv")

# ROC + combined score analysis
Idents(mouse_obj) <- "celltype"
labels <- ifelse(mouse_obj$celltype == "Hepatocytes", 1, 0)
expr_matrix <- GetAssayData(mouse_obj, slot = "data")[valid_genes, , drop = FALSE]

gene_auc <- sapply(valid_genes, function(gene) {
  roc_obj <- try(roc(response = labels, predictor = as.numeric(expr_matrix[gene, ])), silent = TRUE)
  if (inherits(roc_obj, "try-error")) return(NA)
  return(auc(roc_obj))
})
gene_auc <- sort(gene_auc, decreasing = TRUE)
write.csv(data.frame(Gene = names(gene_auc), AUC = gene_auc), "ROC_AUC_TargetGenes_Hepatocytes.csv")

expr_matrix <- GetAssayData(mouse_obj, slot = "data")[valid_genes, , drop = FALSE]
combined_expression <- Matrix::colMeans(expr_matrix)
mouse_obj$Upregulated_Score <- combined_expression

combined_roc <- roc(response = labels, predictor = combined_expression)

pdf("ROC_combined_target_genes_mouse.pdf", width = 6, height = 6)
plot(combined_roc, col = "darkred", main = "ROC: Combined Target Gene Score")
legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)), col = "darkred", lwd = 2)
dev.off()

# Add classification
threshold <- 0.15
mouse_obj$ExpressionStatus <- ifelse(
  mouse_obj$Upregulated_Score > threshold, "Expressing",
  as.character(mouse_obj$celltype)
)

umap_df <- as.data.frame(Embeddings(mouse_obj, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")
umap_df$Status <- mouse_obj$ExpressionStatus

pdf("UMAP_Three_Color_Overlay_Mouse.pdf", width = 8, height = 6)
ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
  geom_point(size = 0.4, alpha = 0.8) +
  scale_color_manual(values = c("Hepatocytes" = "red", "Expressing" = "grey", "Stellate Cells" = "blue", "Macrophages" = "green", "NK Cells" = "purple", "Mixed Hepato/Macrophage" = "orange", "Endothelial/B Cells" = "cyan")) +
  ggtitle("UMAP: Hepatocytes vs Expressing vs Others") +
  theme_minimal()
dev.off()
