# Wang et al., mouse NASH liver (2 NASH libraries, no controls): are the target genes
# hepatocyte-specific? Compares hepatocytes with all other cells.
source(here::here("R", "load_all.R"))

NAME <- "wang_mouse_hep_specificity"
ds <- dataset_config(load_config(), NAME)
set.seed(ds$seed)
log_step("==== ", NAME, " ====")

obj <- timed("Reading 10x", read_10x_samples(ds$raw_dir, ds$samples))
obj <- prepare_dataset(obj, ds, NAME)
out_dir <- project_path("results", NAME)

targets <- intersect(target_genes_for(ds), rownames(obj))
log_step(length(targets), " target genes present")

# Ambiguous clusters (doublets, ambient RNA) belong to neither group
obj <- subset(obj, cells = colnames(obj)[obj$lineage != "ambiguous"])
data <- SeuratObject::LayerData(obj, assay = "RNA", layer = "data")
cells <- cap_cells_per_sample(colnames(obj), obj@meta.data$sample, ds$cells_per_sample_cap, ds$seed)
is_hep <- obj@meta.data[cells, "lineage"] == "hepatocyte"

# Here gene_auc() "disease" means hepatocytes and "control" means other cells
spec <- gene_auc(data[targets, cells], is_hep, ds$min_detect_pct)
names(spec) <- sub("_disease$", "_hepatocyte", sub("_control$", "_other_cells", names(spec)))
spec$direction <- sub("up_in_disease", "higher_in_hepatocytes",
                      sub("down_in_disease", "higher_in_other_cells", spec$direction))
spec <- spec[order(-spec$auc_power, na.last = TRUE), ]
utils::write.csv(spec, project_path("data", "single_cell", "results", paste0(NAME, "_target_genes.csv")),
                 row.names = FALSE)

avg <- Seurat::AverageExpression(obj, features = targets, group.by = "lineage", layer = "data")$RNA
utils::write.csv(as.matrix(avg), file.path(out_dir, paste0(NAME, "_target_gene_mean_by_lineage.csv")))

# Combined target score. No genes were picked using these labels, so the ROC is not circular.
obj$target_score <- Matrix::colMeans(data[targets, , drop = FALSE])
roc <- pROC::roc(response = obj@meta.data$lineage == "hepatocyte", predictor = obj$target_score,
                 levels = c(FALSE, TRUE), direction = "<", quiet = TRUE)
grDevices::pdf(file.path(out_dir, paste0(NAME, "_target_score_ROC.pdf")), 6, 6)
plot(roc, main = "Target-gene score: hepatocytes vs other cells")
graphics::legend("bottomright", bty = "n", legend = sprintf("AUC = %.3f", pROC::auc(roc)))
grDevices::dev.off()

grDevices::pdf(file.path(out_dir, paste0(NAME, "_target_score_UMAP.pdf")), 11, 5)
print(Seurat::DimPlot(obj, group.by = "lineage", label = TRUE, raster = TRUE) +
        Seurat::FeaturePlot(obj, "target_score", raster = TRUE))
grDevices::dev.off()

saveRDS(obj, file.path(out_dir, paste0(NAME, "_annotated.rds")))
log_step("Done. Combined score AUC = ", round(pROC::auc(roc), 3))
