# Per-gene statistics for disease-vs-control comparisons.

# Rank-based (Mann-Whitney) AUC per gene: P(case cell > control cell), ties count 1/2.
# Genes detected in fewer than min_detect of cells in both groups get AUC = NA.
gene_auc <- function(mat, is_case, min_detect = 0.10, chunk_size = 2000L) {
  stopifnot(ncol(mat) == length(is_case), is.logical(is_case), !anyNA(is_case))
  n1 <- sum(is_case)
  n0 <- sum(!is_case)
  if (n1 == 0 || n0 == 0) stop("Both groups need at least one cell.")
  n <- n1 + n0
  genes <- rownames(mat)
  if (is.null(genes)) genes <- as.character(seq_len(nrow(mat)))

  out <- vector("list", ceiling(length(genes) / chunk_size))
  for (ci in seq_along(out)) {
    rows <- ((ci - 1L) * chunk_size + 1L):min(ci * chunk_size, length(genes))
    # cells x genes, so each column holds one gene's non-zeros
    tm <- methods::as(Matrix::t(methods::as(mat[rows, , drop = FALSE], "CsparseMatrix")), "CsparseMatrix")
    dt <- data.table::data.table(
      g    = rep.int(seq_along(rows), diff(tm@p)),
      case = is_case[tm@i + 1L],
      v    = tm@x
    )[v > 0]
    dt[, r := data.table::frank(v, ties.method = "average"), by = g]
    s <- dt[, list(k = .N, k1 = sum(case), rs1 = sum(r[case])), by = g]

    k <- integer(length(rows))
    k1 <- integer(length(rows))
    rs1 <- numeric(length(rows))
    k[s$g] <- s$k
    k1[s$g] <- s$k1
    rs1[s$g] <- s$rs1

    nzero <- n - k
    # Non-zero values rank above all zeros, which share rank (nzero + 1) / 2.
    r1 <- rs1 + k1 * nzero + (n1 - k1) * (nzero + 1) / 2
    auc <- (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)

    out[[ci]] <- data.frame(
      gene = genes[rows],
      pct_disease = k1 / n1,
      pct_control = (k - k1) / n0,
      auc = auc,
      stringsAsFactors = FALSE
    )
  }
  res <- do.call(rbind, out)
  res$tested <- pmax(res$pct_disease, res$pct_control) >= min_detect
  res$auc[!res$tested] <- NA_real_
  res$auc_power <- 2 * abs(res$auc - 0.5)
  res$direction <- auc_direction(res$auc)
  rownames(res) <- NULL
  res
}

auc_direction <- function(auc) {
  ifelse(is.na(auc), "not_tested",
         ifelse(auc > 0.5, "up_in_disease",
                ifelse(auc < 0.5, "down_in_disease", "no_change")))
}

# Cap cells per sample so one large sample cannot dominate the pooled AUC.
cap_cells_per_sample <- function(cells, sample, cap, seed = 42) {
  if (is.null(cap) || is.na(cap)) return(cells)
  set.seed(seed)
  keep <- unlist(lapply(split(cells, sample), function(x) {
    if (length(x) > cap) sample(x, cap) else x
  }), use.names = FALSE)
  cells[cells %in% keep]
}

# Pseudobulk edgeR quasi-likelihood test. Returns NA statistics when a group has
# fewer than two samples, since no between-sample test is possible.
pseudobulk_de <- function(counts, sample, condition, min_cells_per_sample = 20) {
  stopifnot(ncol(counts) == length(sample), length(sample) == length(condition))
  cells_per_sample <- table(sample)
  ok_samples <- names(cells_per_sample)[cells_per_sample >= min_cells_per_sample]
  keep_cells <- sample %in% ok_samples
  counts <- counts[, keep_cells, drop = FALSE]
  sample <- as.character(sample[keep_cells])
  condition <- as.character(condition[keep_cells])

  sample_cond <- unique(data.frame(sample = sample, condition = condition))
  if (anyDuplicated(sample_cond$sample)) stop("A sample maps to more than one condition.")
  n_dis <- sum(sample_cond$condition == "disease")
  n_ctl <- sum(sample_cond$condition == "control")

  na_result <- function(note) {
    data.frame(gene = rownames(counts), pb_logFC = NA_real_, pb_logCPM = NA_real_,
               pb_pvalue = NA_real_, pb_fdr = NA_real_, pb_method = note,
               n_samples_disease = n_dis, n_samples_control = n_ctl,
               stringsAsFactors = FALSE)
  }
  if (n_dis < 2 || n_ctl < 2) {
    return(na_result(sprintf(
      "not testable: %d disease vs %d control samples (need >= 2 per group)", n_dis, n_ctl)))
  }

  f <- factor(sample, levels = sample_cond$sample)
  design_cells <- Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f), x = 1,
                                       dims = c(length(f), nlevels(f)),
                                       dimnames = list(NULL, levels(f)))
  pb <- round(as.matrix(counts %*% design_cells))

  group <- factor(sample_cond$condition, levels = c("control", "disease"))
  y <- edgeR::DGEList(pb, group = group)
  keep <- edgeR::filterByExpr(y, group = group)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- edgeR::calcNormFactors(y)
  design <- stats::model.matrix(~ group)
  fit <- edgeR::glmQLFit(y, design)
  tt <- edgeR::topTags(edgeR::glmQLFTest(fit, coef = 2), n = Inf, sort.by = "none")$table

  res <- na_result(sprintf("edgeR QL pseudobulk (%d disease vs %d control samples)", n_dis, n_ctl))
  idx <- match(rownames(tt), res$gene)
  res$pb_logFC[idx]  <- tt$logFC
  res$pb_logCPM[idx] <- tt$logCPM
  res$pb_pvalue[idx] <- tt$PValue
  res$pb_fdr[idx]    <- tt$FDR
  res
}

# Signature ROC with genes chosen on a discovery split and AUC measured on a
# separate validation split (by sample when each condition has >= 2 samples).
heldout_signature_roc <- function(data_mat, is_case, sample, candidate_genes,
                                  min_detect = 0.10, min_auc = 0.6, seed = 42) {
  candidate_genes <- intersect(candidate_genes, rownames(data_mat))
  if (length(candidate_genes) == 0) return(NULL)
  set.seed(seed)

  cond <- ifelse(is_case, "disease", "control")
  sample_cond <- unique(data.frame(sample = sample, cond = cond))
  per_cond <- split(sample_cond$sample, sample_cond$cond)

  if (all(lengths(per_cond) >= 2)) {
    disc_samples <- unlist(lapply(per_cond, function(s) sample(s, floor(length(s) / 2))))
    in_disc <- sample %in% disc_samples
    split_type <- "by sample (discovery and validation donors are disjoint)"
  } else {
    in_disc <- logical(length(is_case))
    for (grp in c(TRUE, FALSE)) {
      idx <- which(is_case == grp)
      in_disc[sample(idx, floor(length(idx) / 2))] <- TRUE
    }
    split_type <- "by cell (only 1 sample in a condition: validation shares donors, AUC is optimistic)"
  }

  disc <- gene_auc(data_mat[candidate_genes, in_disc, drop = FALSE], is_case[in_disc], min_detect)
  up_genes <- disc$gene[!is.na(disc$auc) & disc$auc >= min_auc]
  if (length(up_genes) == 0) return(NULL)

  score <- Matrix::colMeans(data_mat[up_genes, , drop = FALSE])
  roc_val <- pROC::roc(response = is_case[!in_disc], predictor = score[!in_disc],
                       levels = c(FALSE, TRUE), direction = "<", quiet = TRUE)
  roc_in  <- pROC::roc(response = is_case[in_disc], predictor = score[in_disc],
                       levels = c(FALSE, TRUE), direction = "<", quiet = TRUE)
  ci <- as.numeric(pROC::ci.auc(roc_val))

  list(
    genes = up_genes,
    split = split_type,
    auc_validation = as.numeric(pROC::auc(roc_val)),
    auc_validation_ci95 = ci[c(1, 3)],
    auc_discovery_insample = as.numeric(pROC::auc(roc_in)),
    n_cells_discovery = sum(in_disc),
    n_cells_validation = sum(!in_disc),
    roc = roc_val,
    score = score
  )
}

# Thin the deeper group's counts to the other group's median depth, then
# log-normalise. Without this, lower depth alone pushes AUCs below 0.5.
depth_matched_lognorm <- function(counts, is_case, seed = 42) {
  stopifnot(ncol(counts) == length(is_case))
  counts <- methods::as(counts, "CsparseMatrix")
  # Mitochondrial reads track cell quality, not mRNA content, so depth excludes them
  nuclear <- !grepl("^(MT-|mt-|Mt-)", rownames(counts))
  depth <- Matrix::colSums(counts[nuclear, , drop = FALSE])
  med <- c(case = stats::median(depth[is_case]), ctrl = stats::median(depth[!is_case]))
  ratio <- min(med) / max(med)
  thin <- if (med[["case"]] > med[["ctrl"]]) is_case else !is_case
  if (ratio < 0.95) {
    set.seed(seed)
    col_of_x <- rep.int(seq_len(ncol(counts)), diff(counts@p))
    sel <- thin[col_of_x]
    counts@x[sel] <- stats::rbinom(sum(sel), size = as.integer(round(counts@x[sel])), prob = ratio)
    counts <- Matrix::drop0(counts)
  }
  lib <- Matrix::colSums(counts)
  lib[lib == 0] <- 1
  out <- counts
  out@x <- log1p(out@x / rep.int(lib, diff(out@p)) * 1e4)
  attr(out, "depth_ratio") <- ratio
  out
}
