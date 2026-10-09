# Sensitivity analysis: Xiao and Wang with decontX ambient RNA correction. Writes to
# data/single_cell/sensitivity/ and results/<name>/*_ambient_contamination.csv.
# Usage: Rscript scripts/04c_sc_human_decontx_sensitivity.R [dataset]  (no argument runs both)
args <- commandArgs(trailingOnly = TRUE)
if (length(args)) {
  source(here::here("R", "load_all.R"))
  run_10x_dataset(args[1])
} else {
  # One R process per dataset so the memory is freed after Wang (about 250k nuclei)
  script <- here::here("scripts", "04c_sc_human_decontx_sensitivity.R")
  for (name in c("xiao_GSE189600_decontx", "wang_human_GSE212837_decontx")) {
    status <- system2(file.path(R.home("bin"), "Rscript"), c(shQuote(script), name))
    if (status != 0) stop(name, " failed (exit ", status, ")")
  }
}
