# Wang et al. (GSE212837), human liver snRNA-seq (3 control vs 9 NASH): NASH vs control hepatocytes.
# Harmony is used for clustering only. Statistics use the uncorrected counts.
source(here::here("R", "load_all.R"))
run_10x_dataset("wang_human_GSE212837")
