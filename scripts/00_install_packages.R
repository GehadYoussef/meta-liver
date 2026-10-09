# Install the R packages used by the pipeline, the app and the tests.
cran <- c("Seurat", "SeuratObject", "Matrix", "data.table", "harmony", "pROC", "yaml", "here",
          "shiny", "bslib", "DT", "ggplot2", "dplyr", "testthat")
bioc <- c("edgeR", "AnnotationDbi", "org.Hs.eg.db", "decontX")
missing <- setdiff(cran, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
missing_bioc <- setdiff(bioc, rownames(installed.packages()))
if (length(missing_bioc)) BiocManager::install(missing_bioc, update = FALSE)
