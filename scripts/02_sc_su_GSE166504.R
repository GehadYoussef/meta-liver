# Su et al. 2021 (GSE166504), mouse hepatocyte scRNA-seq: 15-week HFHFD vs chow.
# The sample is the animal (its captures are pooled). Also writes our lineage calls vs the
# authors' CellType labels to results/su_GSE166504/.
source(here::here("R", "load_all.R"))

NAME <- "su_GSE166504"
ds <- dataset_config(load_config(), NAME)
set.seed(ds$seed)
log_step("==== ", NAME, " ====")

meta <- timed("Reading metadata", data.table::fread(resolve_path(ds$metadata_file), data.table = FALSE))
meta$cell <- paste(meta$FileName, meta$CellID, sep = "_")
meta <- meta[grepl(ds$library_filter, meta$FileName), ]
meta$condition <- ifelse(grepl(ds$disease_pattern, meta$FileName), "disease",
                         ifelse(grepl(ds$control_pattern, meta$FileName), "control", NA))
meta <- meta[!is.na(meta$condition), ]
if (!is.null(ds$control_age_pattern)) {
  meta <- meta[meta$condition == "disease" | grepl(ds$control_age_pattern, meta$FileName), ]
}
meta$capture <- meta$FileName
meta$sample <- sub("_Capture[0-9]+$", "", meta$FileName)

log_step("Animals and captures used:")
print(unique(meta[, c("sample", "capture", "condition")]), row.names = FALSE)

counts <- timed("Reading counts", {
  # Over 2 GB as text, so read only the gene column and the selected cells
  src <- resolve_path(ds$counts_file)
  txt <- if (grepl("[.]gz$", src)) {
    R.utils::gunzip(src, destname = tempfile(fileext = ".txt"), remove = FALSE)
  } else src
  # The header has no gene column name, so header field j is data column j + 1
  hdr <- strsplit(readLines(txt, n = 1), "\t", fixed = TRUE)[[1]]
  keep <- intersect(hdr, meta$cell)
  dt <- data.table::fread(txt, select = c(1L, match(keep, hdr) + 1L), header = FALSE, skip = 1)
  if (!identical(txt, src)) unlink(txt)
  m <- methods::as(as.matrix(dt[, -1]), "CsparseMatrix")
  dimnames(m) <- list(dt[[1]], keep)
  m
})
meta <- meta[match(colnames(counts), meta$cell), ]
log_step("Counts: ", nrow(counts), " genes x ", ncol(counts), " cells")

obj <- Seurat::CreateSeuratObject(counts = counts)
obj$sample <- meta$sample
obj$capture <- meta$capture
obj$condition <- meta$condition
obj$author_celltype <- meta$CellType

obj <- prepare_dataset(obj, ds, NAME)
agree <- as.data.frame.matrix(table(our_lineage = obj$lineage, author_celltype = obj$author_celltype))
utils::write.csv(agree, project_path("results", NAME, paste0(NAME, "_lineage_vs_author_celltype.csv")))
log_step("Our lineage vs authors' CellType:")
print(agree)

# Cluster labels alone leave a few hundred monocytes and DCs, so also require the authors' label
if (isTRUE(ds$require_author_hepatocyte)) {
  drop <- obj$lineage == "hepatocyte" & obj$author_celltype != "Hepatocytes"
  log_step("Removing ", sum(drop), " cells in hepatocyte clusters that the authors did not label hepatocytes")
  obj$lineage[drop] <- "non_hepatocyte_by_author_label"
}

res <- run_condition_analysis(obj, ds, NAME, target_genes_for(ds))
saveRDS(res$hepatocytes, project_path("results", NAME, paste0(NAME, "_hepatocytes.rds")))
str(res$summary)
