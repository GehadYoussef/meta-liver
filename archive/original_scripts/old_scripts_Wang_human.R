# Load the required libraries
library(Seurat)
library(Matrix)
library(dplyr)
library(pROC)
library(patchwork)
library(plyr)
library(harmony)  # Load Harmony

setwd("/scratch_data/gy260/new_MASH_April_2025/Wang_paper")

# Load Seurat objects for each sample (replace with your actual sample paths)
samples <- c("humanCTRL_1", "humanCTRL_2", "humanCTRL_3", "humanNASH_1", "humanNASH_2", "humanNASH_3", "humanNASH_4", 
             "humanNASH_5", "humanNASH_6", "humanNASH_7", "humanNASH_8", "humanNASH_9")

seu_list <- lapply(samples, function(sample) {
  sample_dir <- file.path("/scratch_data/gy260/new_MASH_April_2025/Wang_paper/raw_data", sample)
  sample_data <- Read10X(data.dir = sample_dir)
  seu_obj <- CreateSeuratObject(counts = sample_data, project = sample)
  seu_obj$sample <- sample  # Add sample information
  return(seu_obj)
})

# Merge Seurat objects into a single Seurat object
obj <- merge(seu_list[[1]], y = seu_list[-1], add.cell.ids = samples)

# Check the structure of the merged Seurat object
head(obj@meta.data)

# Step 1: Preprocessing
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^mt-")
obj <- subset(obj, subset = nFeature_RNA > 200 & percent.mt < 5)

# Normalize the data
obj <- NormalizeData(obj)

# Identify variable features
obj <- FindVariableFeatures(obj)

# Scale the data
obj <- ScaleData(obj)

# PCA
obj <- RunPCA(obj)

# Step 2: Apply Harmony for batch correction
obj <- RunHarmony(obj, group.by.vars = "sample")  # 'sample' is the metadata column indicating different batches

# Step 3: Run UMAP after Harmony integration
obj <- RunUMAP(obj, reduction = "harmony", dims = 1:10)

# Step 4: Plot UMAP with Harmony correction
pdf("UMAP_after_harmony_correction.pdf", width = 8, height = 6)
DimPlot(obj, reduction = "umap", label = TRUE, group.by = "sample") + ggtitle("UMAP After Harmony Correction")
dev.off()

# Continue with further steps for clustering, visualization, etc.
# For example, re-run clustering
obj <- FindNeighbors(obj, dims = 1:10)
obj <- FindClusters(obj, resolution = 0.5)

# Step 5: Visualizing marker genes or performing downstream analyses
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

# FeaturePlot of some canonical markers
pdf("mouse_NASH_marker_genes_after_harmony.pdf", width = 12, height = 10)
FeaturePlot(obj, features = unlist(marker_genes), ncol = 4)
dev.off()

# Optional: Create a heatmap of top markers per cluster
markers <- FindAllMarkers(obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
top10 <- markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)

# Plot heatmap of top markers
pdf("mouse_NASH_cluster_heatmap_after_harmony.pdf", width = 10, height = 12)
DoHeatmap(obj, features = top10$gene) + NoLegend()
dev.off()

# Save results for further analysis
write.csv(markers, "markers_after_harmony.csv")
