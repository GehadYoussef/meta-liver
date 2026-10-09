setwd("/scratch_data/gy260/new_MASH_April_2025/Su_paper")


library(Seurat)
library(tidyverse)
library(patchwork)
library(biomaRt)
# Step 1: Load your data
counts <- read.delim("GSE166504_cell_raw_counts.20220204.txt", row.names = 1)
metadata <- read.delim("GSE166504_cell_metadata.20220204.tsv", sep = "\t", header = TRUE)
head(metadata)
str(metadata)
metadata$FullName <- paste(metadata$FileName, metadata$CellID, sep = "_")

# Step 2: Filter only hepatocytes from 15w and Chow
hep_meta <- metadata %>%
  filter(grepl("^Hepatocyte", FileName) & grepl("15weeks|Chow", FileName)) %>%
  mutate(
    Condition = ifelse(grepl("15weeks", FileName), "NAFLD", "Control"),
    Animal = sub(".*Animal(\\d+)_.*", "Animal\\1", FileName)
  )

hep_counts <- counts[, hep_meta$FullName]



# Step 4: Create and preprocess Seurat object
hep_obj <- CreateSeuratObject(counts = hep_counts, meta.data = hep_meta)
hep_obj[["percent.mt"]] <- PercentageFeatureSet(hep_obj, pattern = "^MT-")
hep_obj <- subset(hep_obj, subset = nFeature_RNA > 200 & percent.mt < 5)
hep_obj <- NormalizeData(hep_obj)
hep_obj <- FindVariableFeatures(hep_obj)
hep_obj <- ScaleData(hep_obj)
hep_obj <- RunPCA(hep_obj)
hep_obj <- RunUMAP(hep_obj, dims = 1:10)

# Step 5: Set group labels
Idents(hep_obj) <- hep_obj$Condition

# Step 6: Load gene list from file (1 gene per line, no quotes)
target_genes <- readLines("my_genes_mus.txt")
target_genes <- trimws(target_genes)
valid_genes <- intersect(target_genes, rownames(hep_obj))
missing_genes <- setdiff(target_genes, rownames(hep_obj))
cat("Missing genes:\n")
print(missing_genes)

# Step 7: Differential expression for target genes only
deg_target <- FindMarkers(
  hep_obj,
  ident.1 = "NAFLD",
  ident.2 = "Control",
  features = valid_genes
)
# View results
head(deg_target)


# Step 8: Save DE results
write.csv(deg_target, "DE_results_SU_NAFLD_vs_Control_target_genes_new.csv")


# Step 9: Visualize gene expression
# (A) Violin plot for all valid genes
pdf("VlnPlot_NAFLD_vs_Control_valid_genes.pdf", width = 10, height = 6)
VlnPlot(hep_obj, features = valid_genes, group.by = "Condition", pt.size = 0)
dev.off()

# 1. Get all upregulated genes
upregulated_genes <- rownames(deg_target[deg_target$avg_log2FC > 0, ])
upregulated_genes <- intersect(upregulated_genes, rownames(hep_obj))

# 2. Get expression matrix for upregulated genes
expr_matrix <- GetAssayData(hep_obj, slot = "data")[upregulated_genes, , drop = FALSE]

# 3. Compute average expression per cell (can also use sum)
combined_expression <- Matrix::colMeans(expr_matrix)

# 4. Add this as metadata to Seurat object
hep_obj$Upregulated_Score <- combined_expression

# 5. Plot UMAP, colored by combined expression, labelled by Condition
pdf("SU_UMAP_Combined_Upregulated_Gene_Score.pdf", width = 8, height = 6)
FeaturePlot(hep_obj, features = "Upregulated_Score", label = TRUE) +
  ggtitle("Combined Expression of Upregulated Genes")
dev.off()
pdf("SU_UMAP_By_Condition.pdf", width = 8, height = 6)
DimPlot(hep_obj, group.by = "Condition", label = TRUE, pt.size = 0.5) +
  ggtitle("UMAP Colored by Condition")
dev.off()



###AUC

library(pROC)

# 1. Binary condition labels: NAFLD = 1, Control = 0
labels <- ifelse(hep_obj$Condition == "NAFLD", 1, 0)

# 2. Upregulated genes
upregulated_genes <- rownames(deg_target[deg_target$avg_log2FC > 0, ])
upregulated_genes <- intersect(upregulated_genes, rownames(hep_obj))

# 3. Get expression matrix
expr_matrix <- GetAssayData(hep_obj, slot = "data")[upregulated_genes, , drop = FALSE]

# 4. Compute per-cell average expression
combined_expression <- Matrix::colMeans(expr_matrix)

# 5. ROC and AUC
combined_roc <- roc(response = labels, predictor = combined_expression)

# 6. Plot ROC curve
pdf("SU_ROCCombined_ROC_Upregulated_Genes.pdf", width = 6, height = 6)
plot(combined_roc, col = "darkblue", main = "ROC: Combined Upregulated Gene Score")
legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)), col = "darkblue", lwd = 2)
dev.off()




# Threshold: consider a cell "expressing" if combined expression > 0.5
threshold <- 0.15
hep_obj$ExpressionStatus <- ifelse(
  hep_obj$Upregulated_Score > threshold, "Expressing",
  as.character(hep_obj$Condition)
)


library(ggplot2)
# Create a combined metadata column
threshold <- 0.15  # you can adjust this if needed
hep_obj$ExpressionStatus <- ifelse(
  hep_obj$Upregulated_Score > threshold, "Expressing",
  as.character(hep_obj$Condition)
)

# Prepare UMAP data frame
umap_df <- as.data.frame(Embeddings(hep_obj, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")  # Fix the names for ggplot
umap_df$Status <- hep_obj$ExpressionStatus

# Plot
library(ggplot2)
pdf("UMAP_Three_Color_Overlay0.15.pdf", width = 8, height = 6)
ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
  geom_point(size = 0.4, alpha = 0.8) +
  scale_color_manual(values = c("Control" = "blue", "NAFLD" = "red", "Expressing" = "darkgreen")) +
  ggtitle("UMAP: Control vs NAFLD vs Expressing Cells") +
  theme_minimal()
dev.off()