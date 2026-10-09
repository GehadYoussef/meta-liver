# Run the whole pipeline. Fetch raw data first with scripts/00_download_geo.R.
# Single-cell steps without raw data are skipped and the app falls back to the legacy tables.
source(here::here("R", "load_all.R"))
cfg <- load_config()

raw_available <- function(ds) {
  paths <- c(ds$raw_dir, ds$counts_file)
  length(paths) > 0 && all(file.exists(vapply(paths, resolve_path, "")))
}

steps <- c(xiao_GSE189600 = "01_sc_xiao_GSE189600.R", su_GSE166504 = "02_sc_su_GSE166504.R",
           coassolo_GSE210501 = "03_sc_coassolo_GSE210501.R",
           wang_human_GSE212837 = "04_sc_wang_human_GSE212837.R",
           wang_human_GSE212837_unsorted = "04b_sc_wang_human_unsorted_sensitivity.R",
           wang_mouse_hep_specificity = "05_sc_wang_mouse_hep_specificity.R")

source(project_path("scripts", "10_build_reference.R"))
for (nm in names(steps)) {
  if (raw_available(dataset_config(cfg, nm))) {
    source(project_path("scripts", steps[[nm]]), local = new.env())
  } else {
    log_step("Skipping ", nm, ": raw data not found (set paths in config/config.yml)")
  }
}
if (requireNamespace("decontX", quietly = TRUE)) {
  system2(file.path(R.home("bin"), "Rscript"), shQuote(project_path("scripts", "04c_sc_human_decontx_sensitivity.R")))
} else {
  log_step("Skipping decontX sensitivity analysis: decontX not installed")
}
source(project_path("scripts", "11_build_app_data.R"))
