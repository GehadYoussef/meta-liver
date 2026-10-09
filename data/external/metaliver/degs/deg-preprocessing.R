# DEG Data Preprocessing Script
# This script processes the Excel file containing DEG data from multiple sheets
# and converts ENSG gene IDs to gene symbols using biomaRt

# Load required libraries
library(readxl)
library(biomaRt)
library(dplyr)
library(purrr)

# Set up the file path
setwd("/workspaces/inst/extdata/degs/")
excel_file <- "DEGs from Julian.xlsx"

ensembl <- useEnsembl(
  biomart = "ensembl",
  dataset = "hsapiens_gene_ensembl"
)


# Function to convert ENSG IDs to gene symbols
convert_ensg_to_symbols <- function(df, mart) {
  # Remove version numbers from ENSG IDs if present (e.g., ENSG00000123456.1 -> ENSG00000123456)
  clean_ensg_ids <- gsub("\\.\\d+$", "", df$Column1)

  # Query biomaRt
  gene_info <- getBM(
    attributes = c("ensembl_gene_id", "external_gene_name"),
    filters = "ensembl_gene_id",
    values = unique(clean_ensg_ids),
    mart = mart
  )

  inner_join(
    df,
    gene_info,
    by = c("Column1" = "ensembl_gene_id")
  ) |>
    subset(!external_gene_name == "")
}

sheet_names <- excel_sheets(excel_file)
for (sheet in sheet_names) {
  degs <- read_xlsx(excel_file, sheet = sheet)
  degs_processed <- convert_ensg_to_symbols(degs, ensembl)
  filename = paste0("processed_degs_", sheet, ".csv")
  write.csv(degs_processed, filename, row.names = FALSE)
}
