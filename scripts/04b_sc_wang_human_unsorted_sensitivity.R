# Sensitivity analysis: Wang human with unsorted nuclei only, to check whether FACS sorting
# (sorted NASH captures, unsorted controls) drives the NASH vs control directions.
# Output: data/single_cell/sensitivity/wang_human_GSE212837_unsorted_hepatocyte_gene_stats.csv
source(here::here("R", "load_all.R"))
run_10x_dataset("wang_human_GSE212837_unsorted")
