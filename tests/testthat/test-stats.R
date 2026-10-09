sim_counts <- function(n_genes = 60, n_cells = 400, seed = 1) {
  set.seed(seed)
  is_case <- rep(c(TRUE, FALSE), each = n_cells / 2)
  lambda <- matrix(0.3, n_genes, n_cells)
  lambda[1:10, is_case] <- 2      # up in disease
  lambda[11:20, !is_case] <- 2    # down in disease
  lambda[21:25, ] <- 0.01         # barely detected
  m <- matrix(rpois(n_genes * n_cells, lambda), n_genes, n_cells)
  rownames(m) <- paste0("g", seq_len(n_genes))
  list(m = Matrix::Matrix(m, sparse = TRUE), is_case = is_case)
}

test_that("gene_auc matches pROC on sparse data with many ties", {
  s <- sim_counts()
  res <- gene_auc(s$m, s$is_case, min_detect = 0)
  ref <- apply(as.matrix(s$m), 1, function(x) {
    if (all(x == x[1])) return(0.5)
    as.numeric(pROC::auc(pROC::roc(s$is_case, x, levels = c(FALSE, TRUE), direction = "<", quiet = TRUE)))
  })
  expect_equal(res$auc, unname(ref), tolerance = 1e-10)
})

test_that("gene_auc is chunk-size invariant and works on dense input", {
  s <- sim_counts()
  a <- gene_auc(s$m, s$is_case, chunk_size = 7)
  b <- gene_auc(as.matrix(s$m), s$is_case)
  expect_equal(a, b)
})

test_that("undetected genes get NA (not 0) and direction follows AUC", {
  s <- sim_counts()
  res <- gene_auc(s$m, s$is_case, min_detect = 0.10)
  expect_true(all(is.na(res$auc[21:25])))
  expect_true(all(res$direction[21:25] == "not_tested"))
  expect_false(any(res$auc == 0, na.rm = TRUE))
  expect_true(all(res$direction[1:10] == "up_in_disease"))
  expect_true(all(res$direction[11:20] == "down_in_disease"))
  # down-markers rank as strongly as up-markers by auc_power
  top20 <- res$gene[order(-res$auc_power)][1:20]
  expect_setequal(top20, paste0("g", 1:20))
})

test_that("pseudobulk_de finds true DE genes with replicated samples", {
  set.seed(2)
  samples <- paste0("s", 1:6)
  cond <- rep(c("control", "disease"), each = 3)
  cell_sample <- rep(samples, each = 80)
  cell_cond <- cond[match(cell_sample, samples)]
  mu <- matrix(5, 200, length(cell_sample))
  mu[1:20, cell_cond == "disease"] <- 20
  # donor-level noise so the test has real between-sample variance
  donor <- exp(matrix(rnorm(200 * 6, 0, 0.2), 200, 6))[, match(cell_sample, samples)]
  m <- matrix(rpois(length(mu), mu * donor), 200)
  rownames(m) <- paste0("g", 1:200)
  res <- pseudobulk_de(Matrix::Matrix(m, sparse = TRUE), cell_sample, cell_cond)
  expect_true(all(res$pb_fdr[1:20] < 0.01))
  expect_lt(mean(res$pb_fdr[21:200] < 0.05), 0.05)
  expect_true(all(res$pb_logFC[1:20] > 1))
  expect_match(res$pb_method[1], "3 disease vs 3 control")
})

test_that("pseudobulk_de refuses to test with one sample per group", {
  m <- Matrix::Matrix(matrix(rpois(2000, 3), 20), sparse = TRUE)
  rownames(m) <- paste0("g", 1:20)
  res <- pseudobulk_de(m, rep(c("a", "b"), each = 50), rep(c("control", "disease"), each = 50))
  expect_true(all(is.na(res$pb_pvalue)))
  expect_match(res$pb_method[1], "not testable")
})

test_that("held-out signature ROC splits by sample when possible", {
  s <- sim_counts(n_cells = 600)
  smp <- c(rep(paste0("d", 1:3), each = 100), rep(paste0("c", 1:3), each = 100))
  sig <- suppressWarnings(heldout_signature_roc(log1p(s$m), s$is_case, smp, rownames(s$m)))
  expect_match(sig$split, "^by sample")
  expect_setequal(sig$genes, paste0("g", 1:10))
  expect_gt(sig$auc_validation, 0.9)

  sig1 <- suppressWarnings(heldout_signature_roc(log1p(s$m), s$is_case, ifelse(s$is_case, "d", "c"), rownames(s$m)))
  expect_match(sig1$split, "^by cell")
})

test_that("cap_cells_per_sample limits each sample", {
  cells <- paste0("c", 1:300)
  smp <- rep(c("a", "b", "c"), c(200, 50, 50))
  kept <- cap_cells_per_sample(cells, smp, 60)
  expect_equal(as.vector(table(smp[cells %in% kept])), c(60, 50, 50))
})

test_that("depth matching removes a pure depth effect from the AUC", {
  set.seed(5)
  n <- 400; is_case <- rep(c(TRUE, FALSE), each = n / 2)
  base <- matrix(rgamma(200, 1, 2), 200, 1)[, rep(1, n)]           # identical expression profiles
  depth <- ifelse(is_case, 0.4, 1)                                  # disease cells sequenced 2.5x shallower
  m <- Matrix::Matrix(matrix(rpois(200 * n, base * rep(depth, each = 200)), 200), sparse = TRUE)
  rownames(m) <- paste0("g", 1:200)
  naive <- gene_auc(log1p(m), is_case, 0.05)
  matched <- gene_auc(depth_matched_lognorm(m, is_case), is_case, 0.05)
  expect_gt(mean(naive$auc < 0.5, na.rm = TRUE), 0.9)                # naive: almost everything "down"
  expect_lt(abs(mean(matched$auc < 0.5, na.rm = TRUE) - 0.5), 0.15)  # matched: no systematic direction
  expect_lt(abs(median(matched$auc, na.rm = TRUE) - 0.5), 0.02)
})
