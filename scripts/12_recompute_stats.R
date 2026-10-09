# Recompute per-gene statistics from the saved hepatocyte objects after changing statistics
# settings in config.yml. QC, annotation or sample changes need the full dataset scripts.
# n_removed_by_cell_purity is 0 in the new run_info.yml because the objects are already filtered.
source(here::here("R", "load_all.R"))
cfg <- load_config()
for (name in c("xiao_GSE189600", "su_GSE166504", "coassolo_GSE210501",
               "wang_human_GSE212837", "wang_human_GSE212837_unsorted")) {
  f <- project_path("results", name, paste0(name, "_hepatocytes.rds"))
  if (!file.exists(f)) {
    log_step("Skipping ", name, ": no saved hepatocyte object")
    next
  }
  log_step("==== ", name, " ====")
  ds <- dataset_config(cfg, name)
  res <- run_condition_analysis(readRDS(f), ds, name, target_genes_for(ds))
  str(res$summary[c("median_depth_disease", "median_depth_control", "pct_tested_genes_up",
                    "n_genes_tested", "n_fdr_below_0.05")])
}
