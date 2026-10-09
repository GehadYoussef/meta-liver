# Xiao et al. (GSE189600), human liver snRNA-seq: NASH vs healthy hepatocytes.
# Raw 10x folders: data/raw/GSE189600_Xiao (see config/config.yml).
source(here::here("R", "load_all.R"))
run_10x_dataset("xiao_GSE189600")
