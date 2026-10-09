# Coassolo et al. (GSE210501), mouse liver scRNA-seq: NASH vs chow hepatocytes.
# One library per condition, so disease is confounded with batch. AUCs are descriptive
# and p-values are left empty.
source(here::here("R", "load_all.R"))
run_10x_dataset("coassolo_GSE210501")
