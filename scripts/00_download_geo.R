# Download the Su, Coassolo and Wang raw data from GEO into data/raw/ (about 2 GB).
# Existing files are skipped. Xiao (GSE189600) is not downloaded: it was copied from
# the original project folder.
source(here::here("R", "utils.R"))
options(timeout = 3600)

geo_sample_url <- function(gsm, file) {
  sprintf("https://ftp.ncbi.nlm.nih.gov/geo/samples/%snnn/%s/suppl/%s",
          substr(gsm, 1, nchar(gsm) - 3), gsm, file)
}
geo_series_url <- function(gse, file) {
  sprintf("https://ftp.ncbi.nlm.nih.gov/geo/series/%snnn/%s/suppl/%s",
          substr(gse, 1, nchar(gse) - 3), gse, file)
}

fetch <- function(url, dest) {
  if (file.exists(dest) && file.size(dest) > 0) return(invisible(dest))
  ensure_dir(dirname(dest))
  tmp <- paste0(dest, ".part")
  utils::download.file(url, tmp, mode = "wb", quiet = TRUE)
  file.rename(tmp, dest)
  log_step("  ", basename(dirname(dest)), "/", basename(dest), " (", round(file.size(dest) / 1e6, 1), " MB)")
  invisible(dest)
}

fetch_10x <- function(gsm, prefix, out_dir) {
  for (part in c("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")) {
    fetch(geo_sample_url(gsm, paste0(prefix, "_", part)), file.path(out_dir, part))
  }
}

raw <- project_path("data", "raw")

log_step("Su GSE166504")
for (f in c("GSE166504_cell_metadata.20220204.tsv.gz", "GSE166504_cell_raw_counts.20220204.txt.gz")) {
  fetch(geo_series_url("GSE166504", f), file.path(raw, "GSE166504_Su", f))
}

log_step("Coassolo GSE210501")
for (p in c("GSM6431458_Chow", "GSM6431459_NASH")) {
  fetch_10x(sub("_.*", "", p), p, file.path(raw, "GSE210501_Coassolo", p))
}

log_step("Wang GSE212837")
wang <- c(
  "GSM6556449_humanCTRL_1", "GSM6556450_humanCTRL_2", "GSM6556451_humanCTRL_3",
  "GSM6556452_humanNASH_1_40kNuclei_R1C1_DMSO", "GSM6556453_humanNASH_1_40kNuclei_R1C2_Flash",
  "GSM6556454_humanNASH_2_R1C1_30kNuclei", "GSM6556455_humanNASH_2_R2C1_30kNuclei",
  "GSM6556456_humanNASH_3_R4C1_30kNuclei", "GSM6556457_humanNASH_3_R5C2_20kNoFacsNuclei",
  "GSM6556458_humanNASH_3_R6C1_20kNoFacsNuclei", "GSM6556459_humanNASH_4_R4C1_20kNoFacsNuclei",
  "GSM6556460_humanNASH_4_R6C1_20kNoFacsNuclei", "GSM6556461_humanNASH_5_R3C8_20kNoFacsNuclei",
  "GSM6556462_humanNASH_5_R5C5_20kNoFacsNuclei", "GSM6556463_humanNASH_6",
  "GSM6556464_humanNASH_7", "GSM6556465_humanNASH_8", "GSM6556466_humanNASH_9",
  "GSM6556467_mouseNASH_1", "GSM6556468_mouseNASH_2"
)
for (p in wang) fetch_10x(sub("_.*", "", p), p, file.path(raw, "GSE212837_Wang", p))

log_step("Done")
