setwd("/scratch_data/gy260/new_MASH_April_2025/coassollo_paper/")

library(Seurat)
library(Matrix)
library(dplyr)
library(pROC)
library(patchwork)

# Step 1: Load Chow and NASH data (update path as needed)
chow_data <- Read10X(data.dir = "GSM6431458_Chow/")
nash_data <- Read10X(data.dir = "GSM6431459_NASH/")

# Step 2: Create Seurat objects
chow <- CreateSeuratObject(counts = chow_data, project = "Chow")
nash <- CreateSeuratObject(counts = nash_data, project = "NASH")
chow$Condition <- "Chow"
nash$Condition <- "NASH"

# Step 3: Merge both datasets
obj <- merge(chow, y = nash, add.cell.ids = c("Chow", "NASH"))

# Step 4: Preprocessing
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^mt-")
obj <- subset(obj, subset = nFeature_RNA > 200 & percent.mt < 5)
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj)
obj <- RunPCA(obj)
obj <- RunUMAP(obj, dims = 1:10)
Idents(obj) <- obj$Condition

# 1. Re-run clustering at higher resolution
obj <- FindNeighbors(obj, dims = 1:10)
obj <- FindClusters(obj, resolution = 1.2)  # adjust resolution here

# 2. Check available clusters
print(levels(Idents(obj)))  # See cluster IDs

# 3. Plot hepatocyte marker expression and save
pdf("Cluster_HepatocyteMarkers_Alb_Cyp2e1.pdf", width = 8, height = 6)
VlnPlot(obj, features = c("Alb", "Cyp2e1"), group.by = "seurat_clusters", pt.size = 0) +
  ggtitle("Alb and Cyp2e1 Expression by Cluster")
dev.off()

# 4. Plot stellate markers and save
pdf("Cluster_StellateMarkers_Lrat_Col1a1.pdf", width = 8, height = 6)
VlnPlot(obj, features = c("Lrat", "Col1a1"), group.by = "seurat_clusters", pt.size = 0) +
  ggtitle("Lrat and Col1a1 Expression by Cluster")
dev.off()

# 5. Optional: View cluster IDs on UMAP
pdf("UMAP_By_Cluster_IDs.pdf", width = 8, height = 6)
DimPlot(obj, reduction = "umap", label = TRUE) + ggtitle("UMAP: Cluster IDs")
dev.off()


pdf("VlnPlot_Alb_Cyp2e1_By_Cluster.pdf", width = 8, height = 6)
VlnPlot(obj, features = c("Alb", "Cyp2e1"), group.by = "seurat_clusters", pt.size = 0) +
  ggtitle("Expression of Hepatocyte Markers by Cluster")
dev.off()


FeaturePlot(obj, features = c("Lrat", "Pdgfrb", "Col1a1", "Alb", "Cyp2e1", "Ttr"))

pdf("HSC_vs_Hepatocyte_Marker_UMAP.pdf", width = 10, height = 6)
FeaturePlot(obj, features = c("Lrat", "Col1a1", "Pdgfrb", "Alb", "Cyp2e1", "Ttr"))
dev.off()


# List of genes to include
marker_genes <- c("Alb", "Ttr", "Cyp2e1", "Col1a1", "Lrat", "Des", "Pdgfrb")

# Save heatmap
pdf("Heatmap_Hepatocyte_HSC_Markers_By_Cluster.pdf", width = 10, height = 8)
DoHeatmap(obj, features = marker_genes) + 
  ggtitle("Marker Gene Expression by Cluster")
dev.off()


# Save multiple FeaturePlots to one PDF
pdf("FeaturePlots_Liver_Cell_Type_Markers.pdf", width = 8, height = 6)

FeaturePlot(obj, features = "Alb")       # Hepatocyte
FeaturePlot(obj, features = "Lrat")      # HSC
FeaturePlot(obj, features = "Csf1r")     # Kupffer
FeaturePlot(obj, features = "Pecam1")    # Endothelial
FeaturePlot(obj, features = "Lyz2")      # Myeloid

dev.off()




# Step 5: Load your mouse gene list
target_genes <- readLines("my_genes_mus.txt") %>% trimws()
valid_genes <- intersect(target_genes, rownames(obj))
obj <- JoinLayers(obj)

# Step 6: Differential expression only for valid genes
deg_target <- FindMarkers(obj, ident.1 = "NASH", ident.2 = "Chow", features = valid_genes)
write.csv(deg_target, "Coassolo_DE_NASH_vs_Chow_target_genes.csv")


# Step 7: AUC for individual genes
labels <- ifelse(obj$Condition == "NASH", 1, 0)
expr_matrix <- GetAssayData(obj, slot = "data")[valid_genes, , drop = FALSE]

gene_auc <- sapply(valid_genes, function(gene) {
  roc_obj <- try(roc(response = labels, predictor = as.numeric(expr_matrix[gene, ])), silent = TRUE)
  if (inherits(roc_obj, "try-error")) return(NA)
  return(auc(roc_obj))
})

gene_auc <- sort(gene_auc, decreasing = TRUE)
write.csv(data.frame(Gene = names(gene_auc), AUC = gene_auc), "AUC_scores_target_genes.csv")

# Step 8: Combined upregulated score & ROC
upregulated_genes <- rownames(deg_target[deg_target$avg_log2FC > 0, ])
upregulated_genes <- intersect(upregulated_genes, rownames(obj))

expr_matrix <- GetAssayData(obj, slot = "data")[upregulated_genes, , drop = FALSE]
combined_expression <- Matrix::colMeans(expr_matrix)
obj$Upregulated_Score <- combined_expression

combined_roc <- roc(response = labels, predictor = combined_expression)

pdf("Coassollo_ROCCombined_ROC_Upregulated_Genes.pdf", width = 6, height = 6)
plot(combined_roc, col = "darkblue", main = "ROC: Combined Upregulated Gene Score")
legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)), col = "darkblue", lwd = 2)
dev.off()

# Step 9: Classify "Expressing" cells
threshold <- 0.15  # Or use: threshold <- coords(combined_roc, "best", ret = "threshold")
obj$ExpressionStatus <- ifelse(
  obj$Upregulated_Score > threshold, "Expressing",
  as.character(obj$Condition)
)

# Step 10: UMAP plot with 3 groups
umap_df <- as.data.frame(Embeddings(obj, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")
umap_df$Status <- obj$ExpressionStatus

pdf("UMAP_Three_Color_Overlay0.15_new.pdf", width = 8, height = 6)
ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
  geom_point(size = 0.4, alpha = 0.8) +
  scale_color_manual(values = c("Chow" = "blue", "NASH" = "red", "Expressing" = "lightgrey")) +
  ggtitle("UMAP: Chow vs NASH vs Expressing Cells") +
  theme_minimal()
dev.off()

# Separate d# Prepare UMAP data frame colored only by Condition (Chow or NASH)
umap_df <- as.data.frame(Embeddings(obj, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")
umap_df$Condition <- obj$Condition  # This will be either "Chow" or "NASH"

# Plot
pdf("UMAP_Chow_NASH_Coloring_Only.pdf", width = 8, height = 6)
ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Condition)) +
  geom_point(size = 0.4, alpha = 0.8) +
  scale_color_manual(values = c("Chow" = "blue", "NASH" = "red")) +
  ggtitle("UMAP: Chow vs NASH (all cells shown)") +
  theme_minimal()
dev.off()
