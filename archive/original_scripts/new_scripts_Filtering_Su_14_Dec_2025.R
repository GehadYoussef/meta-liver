# ============================================================
# Su / GSE166504 (mouse hepatocytes) NAFLD vs Control
#
# - Load counts + metadata
# - Filter hepatocytes from 15weeks + Chow
# - QC -> Normalise -> PCA -> UMAP
# - BALANCE (downsample) NAFLD vs Control hepatocytes
# - Per-gene AUROC + direction + p-values with rule:
#     if max(pct.NAFLD, pct.Control) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# - Target-genes DE + combined upregulated score + ROC
# - Live console progress (timestamps + elapsed time)
#
# Seurat v5 note:
# - We run JoinLayers() after normalisation AND after balancing, to avoid
#   "data layers are not joined" when using slot="data" or FindMarkers().
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

setwd("/scratch_data/gy260/new_MASH_April_2025/Su_paper")

# ----------------------------
# GLOBAL SETTINGS
# ----------------------------
MIN_DETECT_PCT_FOR_AUC <- 0.10
RANDOM_SEED <- 42
CAP_PER_GROUP <- NA_integer_   # e.g. 20000L to cap for speed; NA keeps all of smaller group

cat("\n============================================================\n")
cat("START:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n")

# ----------------------------
# LOAD DATA
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Loading counts + metadata...\n", sep = "")
t0 <- Sys.time()

counts <- read.delim(
  "GSE166504_cell_raw_counts.20220204.txt",
  row.names = 1,
  check.names = FALSE
)

metadata <- read.delim(
  "GSE166504_cell_metadata.20220204.tsv",
  sep = "\t",
  header = TRUE,
  check.names = FALSE
)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Loaded in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")
cat("Counts dim:", nrow(counts), "genes x", ncol(counts), "cells\n")
cat("Metadata rows:", nrow(metadata), "\n")

# Create column key used in counts (must match colnames(counts))
metadata$FullName <- paste(metadata$FileName, metadata$CellID, sep = "_")

# ----------------------------
# FILTER HEPATOCYTES: 15weeks vs Chow
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Filtering hepatocytes (15weeks vs Chow)...\n", sep = "")
t0 <- Sys.time()

hep_meta <- metadata %>%
  filter(grepl("^Hepatocyte", FileName) & grepl("15weeks|Chow", FileName)) %>%
  mutate(
    Condition = ifelse(grepl("15weeks", FileName), "NAFLD", "Control"),
    Animal = sub(".*Animal(\\d+)_.*", "Animal\\1", FileName)
  )

# Keep only cells that exist in counts
hep_meta <- hep_meta %>% filter(FullName %in% colnames(counts))

cat("[", format(Sys.time(), "%H:%M:%S"), "] Hep meta built in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")
cat("Hepatocyte metadata cells:", nrow(hep_meta), "\n")
cat("Condition breakdown:\n")
print(table(hep_meta$Condition))

hep_counts <- counts[, hep_meta$FullName, drop = FALSE]
cat("Hep counts dim:", nrow(hep_counts), "genes x", ncol(hep_counts), "cells\n")

# ----------------------------
# CREATE SEURAT OBJECT + QC
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Creating Seurat object...\n", sep = "")
t0 <- Sys.time()

hep_obj <- CreateSeuratObject(counts = hep_counts, meta.data = hep_meta)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Created in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")
cat("Seurat object -> Cells:", ncol(hep_obj), " Genes:", nrow(hep_obj), "\n")

# Mito pattern: mouse often mt-; if zero, try MT-
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Computing percent.mt...\n", sep = "")
t0 <- Sys.time()

hep_obj[["percent.mt"]] <- PercentageFeatureSet(hep_obj, pattern = "^mt-")
if (all(hep_obj$percent.mt == 0)) {
  hep_obj[["percent.mt"]] <- PercentageFeatureSet(hep_obj, pattern = "^MT-")
  cat("Using pattern ^MT-\n")
} else {
  cat("Using pattern ^mt-\n")
}

cat("[", format(Sys.time(), "%H:%M:%S"), "] percent.mt done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")

cat("\nQC summary (before filtering):\n")
print(summary(hep_obj$percent.mt))
print(summary(hep_obj$nFeature_RNA))
print(summary(hep_obj$nCount_RNA))

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] QC filtering...\n", sep = "")
t0 <- Sys.time()

hep_obj <- subset(hep_obj, subset = nFeature_RNA > 200 & percent.mt < 5)

cat("[", format(Sys.time(), "%H:%M:%S"), "] QC filter done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")
cat("After QC -> Cells:", ncol(hep_obj), " Genes:", nrow(hep_obj), "\n")

# ----------------------------
# NORMALISE + PCA + UMAP
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] NormalizeData...\n", sep = "")
t0 <- Sys.time()
hep_obj <- NormalizeData(hep_obj, verbose = FALSE)

# Join layers after normalisation (Seurat v5)
if ("JoinLayers" %in% getNamespaceExports("Seurat")) {
  cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers after NormalizeData...\n", sep = "")
  hep_obj <- JoinLayers(hep_obj)
}

cat("[", format(Sys.time(), "%H:%M:%S"), "] NormalizeData done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] FindVariableFeatures...\n", sep = "")
t0 <- Sys.time()
hep_obj <- FindVariableFeatures(hep_obj, verbose = FALSE)
cat("[", format(Sys.time(), "%H:%M:%S"), "] FindVariableFeatures done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] ScaleData...\n", sep = "")
t0 <- Sys.time()
hep_obj <- ScaleData(hep_obj, verbose = FALSE)
cat("[", format(Sys.time(), "%H:%M:%S"), "] ScaleData done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] RunPCA...\n", sep = "")
t0 <- Sys.time()
hep_obj <- RunPCA(hep_obj, verbose = FALSE)
cat("[", format(Sys.time(), "%H:%M:%S"), "] RunPCA done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] RunUMAP...\n", sep = "")
t0 <- Sys.time()
hep_obj <- RunUMAP(hep_obj, dims = 1:10, verbose = FALSE)
cat("[", format(Sys.time(), "%H:%M:%S"), "] RunUMAP done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

pdf("SU_UMAP_By_Condition_raw.pdf", width = 8, height = 6)
print(
  DimPlot(hep_obj, group.by = "Condition", label = TRUE, pt.size = 0.4) +
    ggtitle("UMAP (raw hepatocytes): NAFLD vs Control")
)
dev.off()

# ----------------------------
# BALANCE NAFLD vs Control
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Balancing NAFLD vs Control by downsampling...\n", sep = "")
set.seed(RANDOM_SEED)

Idents(hep_obj) <- hep_obj$Condition

cells_ctrl <- WhichCells(hep_obj, idents = "Control")
cells_nafld <- WhichCells(hep_obj, idents = "NAFLD")

cat("Cells Control:", length(cells_ctrl), "  NAFLD:", length(cells_nafld), "\n")

n_bal <- min(length(cells_ctrl), length(cells_nafld))
if (!is.na(CAP_PER_GROUP)) n_bal <- min(n_bal, CAP_PER_GROUP)

cat("Balanced n per group:", n_bal, "\n")

ctrl_sub <- sample(cells_ctrl, n_bal)
nafld_sub <- sample(cells_nafld, n_bal)

hep_obj_bal <- subset(hep_obj, cells = c(ctrl_sub, nafld_sub))

cat("After balancing:\n")
print(table(hep_obj_bal$Condition))
cat("Balanced object -> Cells:", ncol(hep_obj_bal), " Genes:", nrow(hep_obj_bal), "\n")

pdf("SU_UMAP_By_Condition_balanced.pdf", width = 8, height = 6)
print(
  DimPlot(hep_obj_bal, group.by = "Condition", label = TRUE, pt.size = 0.4) +
    ggtitle("UMAP (balanced hepatocytes): NAFLD vs Control")
)
dev.off()

# ----------------------------
# JOIN LAYERS (Seurat v5-safe)
# ----------------------------
if ("JoinLayers" %in% getNamespaceExports("Seurat")) {
  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers detected -> joining layers (balanced)...\n", sep = "")
  t0 <- Sys.time()
  hep_obj_bal <- JoinLayers(hep_obj_bal)
  cat("[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers done in ",
      round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
      " mins\n", sep = "")
} else {
  cat("\n[", format(Sys.time(), "%H:%M:%S"), "] JoinLayers not available -> skipping.\n", sep = "")
}

# ----------------------------
# PER-GENE AUROC + DIRECTION + P-VALUES (balanced)
# Rule: if max(pct) < MIN_DETECT_PCT_FOR_AUC => AUC = 0
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Per-gene AUROC + direction + p-values (balanced)...\n", sep = "")

DefaultAssay(hep_obj_bal) <- "RNA"
Idents(hep_obj_bal) <- hep_obj_bal$Condition

cells_ctrl <- WhichCells(hep_obj_bal, idents = "Control")
cells_nafld <- WhichCells(hep_obj_bal, idents = "NAFLD")

cat("Balanced cells Control:", length(cells_ctrl), "  NAFLD:", length(cells_nafld), "\n")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Computing detection rates...\n", sep = "")
t0 <- Sys.time()

counts_mat <- GetAssayData(hep_obj_bal, assay = "RNA", slot = "counts")
pct_ctrl <- Matrix::rowSums(counts_mat[, cells_ctrl, drop = FALSE] > 0) / length(cells_ctrl)
pct_nafld <- Matrix::rowSums(counts_mat[, cells_nafld, drop = FALSE] > 0) / length(cells_nafld)
pct_max <- pmax(pct_ctrl, pct_nafld)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Detection rates done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

keep_genes <- names(pct_max)[pct_max >= MIN_DETECT_PCT_FOR_AUC]
cat("Genes total:", length(pct_max), "\n")
cat("Genes with max(pct) >= ", MIN_DETECT_PCT_FOR_AUC, ": ", length(keep_genes), "\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression for direction...\n", sep = "")
t0 <- Sys.time()

avg_expr <- AverageExpression(
  hep_obj_bal,
  assays = "RNA",
  slot = "data",
  group.by = "Condition",
  verbose = FALSE
)$RNA

cat("[", format(Sys.time(), "%H:%M:%S"), "] AverageExpression done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

if (!all(c("Control", "NAFLD") %in% colnames(avg_expr))) {
  stop("AverageExpression did not return both Control and NAFLD columns. Check hep_obj_bal$Condition.")
}

avg_ctrl <- avg_expr[, "Control"]
avg_nafld <- avg_expr[, "NAFLD"]
avg_logFC <- avg_nafld - avg_ctrl
direction <- ifelse(
  avg_logFC > 0, "enriched_in_NAFLD",
  ifelse(avg_logFC < 0, "enriched_in_Control", "no_change")
)

cat("\n[", format(Sys.time(), "%H:%M:%S"),
    "] Running ROC AUC on ", length(keep_genes), " genes (slow step)...\n", sep = "")
t0 <- Sys.time()

auc_keep <- FindMarkers(
  hep_obj_bal,
  ident.1 = "NAFLD",
  ident.2 = "Control",
  test.use = "roc",
  features = keep_genes,
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)

cat("[", format(Sys.time(), "%H:%M:%S"), "] ROC AUC done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

auc_vec <- rep(0, nrow(avg_expr))
names(auc_vec) <- rownames(avg_expr)
auc_vec[rownames(auc_keep)] <- auc_keep$myAUC

cat("\n[", format(Sys.time(), "%H:%M:%S"),
    "] Running Wilcoxon p-values on ", length(keep_genes), " genes...\n", sep = "")
t0 <- Sys.time()

de_keep <- FindMarkers(
  hep_obj_bal,
  ident.1 = "NAFLD",
  ident.2 = "Control",
  test.use = "wilcox",
  features = keep_genes,
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Wilcoxon done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Building full results table (all genes)...\n", sep = "")
t0 <- Sys.time()

res_auc <- data.frame(
  gene = rownames(avg_expr),
  AUC = as.numeric(auc_vec),
  avg_logFC = as.numeric(avg_logFC[rownames(avg_expr)]),
  direction = direction[rownames(avg_expr)],
  mean_logexpr_Control = as.numeric(avg_ctrl[rownames(avg_expr)]),
  mean_logexpr_NAFLD = as.numeric(avg_nafld[rownames(avg_expr)]),
  pct_Control = as.numeric(pct_ctrl[rownames(avg_expr)]),
  pct_NAFLD = as.numeric(pct_nafld[rownames(avg_expr)]),
  max_pct = as.numeric(pct_max[rownames(avg_expr)]),
  p_val = NA_real_,
  p_val_adj = NA_real_,
  tested = FALSE,
  stringsAsFactors = FALSE
)

res_auc$tested[res_auc$gene %in% rownames(de_keep)] <- TRUE
res_auc$p_val[match(rownames(de_keep), res_auc$gene)] <- de_keep$p_val
res_auc$p_val_adj[match(rownames(de_keep), res_auc$gene)] <- de_keep$p_val_adj

# Force AUC=0 for genes below detection threshold
res_auc$AUC[res_auc$max_pct < MIN_DETECT_PCT_FOR_AUC] <- 0

res_auc <- res_auc[order(-res_auc$AUC, -abs(res_auc$avg_logFC)), ]

cat("[", format(Sys.time(), "%H:%M:%S"), "] Table built in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")

write.csv(res_auc, "SU_Hepatocyte_gene_AUC_direction_stats_balanced.csv", row.names = FALSE)

cat("\nTop 20 genes (AUC, direction, avg_logFC, padj):\n")
print(head(res_auc[, c("gene", "AUC", "direction", "avg_logFC", "pct_Control", "pct_NAFLD", "p_val_adj")], 20))

# ----------------------------
# TARGET GENE LIST DE (balanced)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Loading target gene list + DE (balanced)...\n", sep = "")

target_genes <- readLines("my_genes_mus.txt")
target_genes <- trimws(target_genes)

valid_genes <- intersect(target_genes, rownames(hep_obj_bal))
missing_genes <- setdiff(target_genes, rownames(hep_obj_bal))

cat("Missing genes (not in object):\n")
print(missing_genes)
cat("Valid genes:", length(valid_genes), "\n")

cat("\n[", format(Sys.time(), "%H:%M:%S"), "] FindMarkers on target genes (wilcox)...\n", sep = "")
t0 <- Sys.time()

deg_target <- FindMarkers(
  hep_obj_bal,
  ident.1 = "NAFLD",
  ident.2 = "Control",
  features = valid_genes,
  test.use = "wilcox",
  min.pct = 0,
  logfc.threshold = 0,
  verbose = TRUE
)

cat("[", format(Sys.time(), "%H:%M:%S"), "] Target-gene DE done in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
    " mins\n", sep = "")

write.csv(deg_target, "DE_results_SU_NAFLD_vs_Control_target_genes_balanced.csv")

pdf("VlnPlot_NAFLD_vs_Control_valid_genes_balanced.pdf", width = 12, height = 6)
print(
  VlnPlot(hep_obj_bal, features = valid_genes, group.by = "Condition", pt.size = 0) +
    ggtitle("Target genes (balanced hepatocytes)")
)
dev.off()

# ----------------------------
# COMBINED UPREGULATED SCORE + ROC (balanced)
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Combined upregulated score + ROC (balanced)...\n", sep = "")

upregulated_genes <- rownames(deg_target[deg_target$avg_log2FC > 0, , drop = FALSE])
upregulated_genes <- intersect(upregulated_genes, rownames(hep_obj_bal))

cat("Upregulated genes used for score:", length(upregulated_genes), "\n")

if (length(upregulated_genes) >= 1) {
  expr_matrix <- GetAssayData(hep_obj_bal, slot = "data")[upregulated_genes, , drop = FALSE]
  combined_expression <- Matrix::colMeans(expr_matrix)
  hep_obj_bal$Upregulated_Score <- combined_expression

  pdf("SU_UMAP_Combined_Upregulated_Gene_Score_balanced.pdf", width = 8, height = 6)
  print(
    FeaturePlot(hep_obj_bal, features = "Upregulated_Score") +
      ggtitle("Combined Expression of Upregulated Genes (balanced)")
  )
  dev.off()

  labels <- ifelse(hep_obj_bal$Condition == "NAFLD", 1, 0)
  combined_roc <- roc(response = labels, predictor = combined_expression)

  pdf("SU_ROC_Combined_Upregulated_Genes_balanced.pdf", width = 6, height = 6)
  plot(combined_roc, main = "ROC: Combined Upregulated Gene Score (balanced)")
  legend("bottomright", legend = paste("AUC =", round(auc(combined_roc), 3)), lwd = 2)
  dev.off()

  # Three-colour overlay with threshold
  threshold <- 0.15
  hep_obj_bal$ExpressionStatus <- ifelse(
    hep_obj_bal$Upregulated_Score > threshold, "Expressing",
    as.character(hep_obj_bal$Condition)
  )

  umap_df <- as.data.frame(Embeddings(hep_obj_bal, "umap"))
  colnames(umap_df) <- c("UMAP_1", "UMAP_2")
  umap_df$Status <- hep_obj_bal$ExpressionStatus

  pdf("UMAP_Three_Color_Overlay_balanced_thresh0.15.pdf", width = 8, height = 6)
  print(
    ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
      geom_point(size = 0.4, alpha = 0.8) +
      scale_color_manual(values = c("Control" = "blue", "NAFLD" = "red", "Expressing" = "darkgreen")) +
      ggtitle("UMAP: Control vs NAFLD vs Expressing (balanced)") +
      theme_minimal()
  )
  dev.off()
} else {
  cat("No upregulated genes found from target list DE; skipping combined score + ROC plots.\n")
}

# ----------------------------
# SAVE OBJECT
# ----------------------------
cat("\n[", format(Sys.time(), "%H:%M:%S"), "] Saving balanced object...\n", sep = "")
t0 <- Sys.time()

saveRDS(hep_obj_bal, "SU_hepatocytes_balanced_object.rds")

cat("[", format(Sys.time(), "%H:%M:%S"), "] Saved in ",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    " secs\n", sep = "")

cat("\n============================================================\n")
cat("DONE:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n")
