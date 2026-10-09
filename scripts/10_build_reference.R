# Build data/reference/: the human Ensembl to symbol map, MGI mouse-human orthologs
# (optional, needs internet) and the target gene list if it is missing.
source(here::here("R", "load_all.R"))
ref_dir <- ensure_dir(project_path("data", "reference"))

ids <- unique(c(
  unlist(lapply(file.path(project_path("data", "bulk"), BULK_CONTRASTS$file),
                function(f) utils::read.csv(f)[[1]])),
  utils::read.csv(project_path("data", "external", "metaliver", "wgcna", "Module-gene-mapping.csv"))[[1]]
))
ens <- unique(strip_ensembl_version(ids[!grepl("_PAR_Y$", ids)]))
sym <- AnnotationDbi::mapIds(org.Hs.eg.db::org.Hs.eg.db, keys = ens, keytype = "ENSEMBL",
                             column = "SYMBOL", multiVals = "first")
emap <- data.frame(ensembl = ens, symbol = unname(sym), stringsAsFactors = FALSE)
utils::write.csv(emap, file.path(ref_dir, "ensembl_to_symbol_human.csv"), row.names = FALSE)
log_step("Ensembl map: ", sum(!is.na(emap$symbol)), " / ", nrow(emap), " IDs have a symbol")

mgi_url <- "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
orth_file <- file.path(ref_dir, "orthologs_mouse_human.csv")
tmp <- tempfile(fileext = ".rpt")
ok <- !nzchar(Sys.getenv("MASH_OFFLINE")) && !file.exists(orth_file) &&
  tryCatch({ utils::download.file(mgi_url, tmp, quiet = TRUE); TRUE },
           error = function(e) FALSE, warning = function(w) FALSE)
if (ok) {
  hom <- data.table::fread(tmp, data.table = FALSE)
  key <- names(hom)[grepl("^DB Class Key", names(hom))][1]
  mm <- hom[hom[["Common Organism Name"]] == "mouse, laboratory", c(key, "Symbol")]
  hs <- hom[hom[["Common Organism Name"]] == "human", c(key, "Symbol")]
  orth <- merge(mm, hs, by = key)
  names(orth) <- c("class_key", "mouse_symbol", "human_symbol")
  # One-to-one orthologs only
  orth <- orth[!duplicated(orth$mouse_symbol) & !duplicated(orth$mouse_symbol, fromLast = TRUE) &
               !duplicated(orth$human_symbol) & !duplicated(orth$human_symbol, fromLast = TRUE), ]
  utils::write.csv(orth[, c("mouse_symbol", "human_symbol")], orth_file, row.names = FALSE)
  log_step("Orthologs: ", nrow(orth), " one-to-one mouse-human pairs")
} else if (!file.exists(orth_file)) {
  log_step("WARNING: could not download MGI orthologs, so mouse genes will be matched to human ",
           "by upper-casing the symbol. Re-run this script with internet access to fix.")
}

tg <- file.path(ref_dir, "target_genes_mouse.txt")
if (!file.exists(tg)) {
  legacy <- utils::read.csv(project_path("data", "single_cell", "legacy",
                                         "GSE210501_Coassolo_mouse_hep_targets_AUC_legacy.csv"))
  writeLines(sort(unique(legacy$Gene)), tg)
  log_step("Wrote ", tg, " (", length(unique(legacy$Gene)), " genes recovered from the Coassolo target table)")
}
