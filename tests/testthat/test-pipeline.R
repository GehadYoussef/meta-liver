write_fake_10x <- function(dir, n_cells = 30, seed = 1) {
  set.seed(seed)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  m <- Matrix::Matrix(matrix(rpois(50 * n_cells, 1), 50), sparse = TRUE)
  Matrix::writeMM(m, file.path(dir, "matrix.mtx"))
  writeLines(paste0("ENSG", 1:50, "\tGENE", 1:50, "\tGene Expression"), file.path(dir, "features.tsv"))
  writeLines(paste0("BC", seq_len(n_cells), "-1"), file.path(dir, "barcodes.tsv"))
  for (f in c("matrix.mtx", "features.tsv", "barcodes.tsv")) R.utils::gzip(file.path(dir, f))
}

test_that("captures from one donor share a sample id and capture is kept as batch", {
  root <- file.path(tempdir(), "fake10x")
  for (i in 1:3) write_fake_10x(file.path(root, paste0("cap", i)), seed = i)
  samples <- list(
    list(id = "cap1", donor = "D1", folder = "cap1", condition = "disease"),
    list(id = "cap2", donor = "D1", folder = "cap2", condition = "disease"),
    list(id = "cap3", folder = "cap3", condition = "control")
  )
  obj <- suppressWarnings(read_10x_samples(root, samples))
  expect_equal(ncol(obj), 90)
  expect_setequal(unique(obj$sample), c("D1", "cap3"))
  expect_setequal(unique(obj$capture), c("cap1", "cap2", "cap3"))
  expect_equal(sum(obj$sample == "D1"), 60)
  expect_equal(sum(obj$sample == "cap3"), 30)
  expect_silent(capture.output(validate_conditions(obj)))

  obj$condition[obj$capture == "cap2"] <- "control"   # same donor in two conditions
  expect_error(capture.output(validate_conditions(obj)), "more than one condition")
})

test_that("mitochondrial cut-off adapts per capture within floor and cap", {
  set.seed(3)
  mt <- c(rnorm(500, 10, 2), abs(rnorm(500, 0, 0.1)), rnorm(500, 40, 20))
  cap <- rep(c("ctrl_unsorted", "nash_facs", "messy"), each = 500)
  thr <- mt_thresholds(mt, cap, mt_floor = 5, mt_cap = 20)
  expect_gt(thr[["ctrl_unsorted"]], 14)     # median 10 + 3 MAD ~ 16: keeps most control nuclei
  expect_lt(thr[["ctrl_unsorted"]], 20)
  expect_equal(thr[["nash_facs"]], 5)        # near-zero mt: floor
  expect_equal(thr[["messy"]], 20)           # very spread: cap
})
