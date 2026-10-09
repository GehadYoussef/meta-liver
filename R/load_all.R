# Load packages and source the project's R functions.
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
})
for (f in c("utils.R", "stats.R", "sc_pipeline.R", "harmonise.R", "harmonise_extra.R")) {
  source(here::here("R", f))
}
