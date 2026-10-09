test_that("legacy tables: AUC 0 becomes NA, direction follows AUC, no p-values", {
  for (nm in names(LEGACY_SC_SPECS)) {
    d <- harmonise_legacy_sc(nm, LEGACY_SC_SPECS[[nm]])
    expect_true(nrow(d) > 0, info = nm)
    expect_false(any(d$auc == 0 & !d$tested, na.rm = TRUE), info = nm)
    expect_true(all(is.na(d$auc[!d$tested])), info = nm)
    expect_true(all(d$direction[d$tested & d$auc > 0.5] == "up_in_disease"), info = nm)
    expect_true(all(d$direction[d$tested & d$auc < 0.5] == "down_in_disease"), info = nm)
    expect_true(all(is.na(d$pb_fdr)), info = nm)
  }
})

test_that("Coassolo legacy has a direction column", {
  d <- harmonise_legacy_sc("coassolo_GSE210501", LEGACY_SC_SPECS$coassolo_GSE210501)
  expect_true(any(d$direction == "up_in_disease"))
  expect_true(any(d$direction == "down_in_disease"))
})

test_that("mouse symbols map to human symbols", {
  expect_equal(to_human_symbol(c("Uroc1", "Cyp2e1"), "mouse", orth = NULL), c("UROC1", "CYP2E1"))
  orth <- data.frame(mouse_symbol = c("Serpina1a", "Zfp213"), human_symbol = c("SERPINA1", "ZNF213"))
  expect_equal(to_human_symbol(c("Serpina1a", "Zfp213"), "mouse", orth), c("SERPINA1", "ZNF213"))
  # no one-to-one ortholog: keep the mouse symbol rather than guess
  expect_equal(to_human_symbol("Saa3", "mouse", orth), "Saa3")
})

test_that("python-style list columns are parsed", {
  expect_equal(parse_py_list("[['TP53', 'EGFR']]"), "TP53, EGFR")
  expect_equal(parse_py_list("[]"), "")
})

test_that("direction consistency classifies genes and pairwise agreement", {
  sc <- data.frame(
    dataset = rep(c("a", "b", "c"), each = 4), species = rep(c("mouse", "mouse", "human"), each = 4),
    gene_human = rep(c("UP3", "DOWN2", "CONFLICT", "UNTESTED"), 3),
    auc = c(0.7, 0.3, 0.7, NA,
            0.6, 0.35, 0.3, NA,
            0.8, 0.52, 0.65, NA))
  cons <- build_consistency(sc)
  g <- cons$genes[match(c("UP3", "DOWN2", "CONFLICT", "UNTESTED"), cons$genes$gene_human), ]
  expect_equal(g$n_up, c(3, 0, 2, 0))
  expect_equal(g$n_down, c(0, 2, 1, 0))
  expect_match(g$consistency[1], "consistent up (3/3)", fixed = TRUE)
  expect_match(g$consistency[2], "consistent down")
  expect_match(g$consistency[3], "conflicting")
  expect_equal(g$mouse[3], "conflict")
  expect_equal(g$human[1], "up")
  s <- consistency_summary(cons)
  expect_equal(s$consistent_in_all, 1)
  expect_equal(s$conflicting, 1)
})

test_that("bulk direction needs significance and no opposite significant contrast", {
  bulk <- data.frame(gene_human = c("A", "A", "B", "B", "C"), family = "vs_control",
                     contrast = c("NASH F2", "NASH F3", "NASH F2", "NASH F3", "NASH F2"),
                     log2FC = c(1, 0.5, 1, -1, -2), padj = c(0.01, 0.5, 0.01, 0.01, 0.2))
  b <- bulk_direction(bulk)
  expect_equal(b$bulk[match(c("A", "B", "C"), b$gene_human)], c("up", "none", "none"))
})

test_that("evidence score penalises conflicting directions", {
  sc <- data.frame(gene_human = rep(c("CONSISTENT", "SPLIT"), each = 2), dataset = rep(c("a", "b"), 2),
                   auc = c(0.75, 0.72, 0.75, 0.25), pb_fdr = NA)
  ev <- evidence_scores(sc, c("a", "b"))
  expect_gt(ev$evidence[ev$gene_human == "CONSISTENT"], 0.5)
  expect_equal(ev$evidence[ev$gene_human == "SPLIT"], 0)
})

test_that("evidence score ranks three consistent datasets above one strong one", {
  sc <- data.frame(gene_human = c("ONE", rep("THREE", 3)), dataset = c("a", "a", "b", "c"),
                   auc = c(0.95, 0.80, 0.78, 0.82), pb_fdr = NA)
  ev <- evidence_scores(sc, c("a", "b", "c"))
  expect_gt(ev$evidence[ev$gene_human == "THREE"], ev$evidence[ev$gene_human == "ONE"])
  expect_equal(ev$coverage[ev$gene_human == "ONE"], 1 / 3)
})

test_that("random-effects meta-analysis widens the CI when cohorts disagree in size", {
  dir <- file.path(tempdir(), "cohorts", "MASHvsControl"); dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  for (dd in c("MASLDvsControl", "EarlyMASLDvsControl", "MASHvsEarlyMASLD")) dir.create(file.path(tempdir(), "cohorts", dd), showWarnings = FALSE)
  w <- function(f, lfc) utils::write.table(data.frame(Symbol = c("HET", "HOM"), log2FoldChange = lfc, lfcSE = 0.1,
                                                      pvalue = 0.01, padj = 0.01), file.path(dir, f), sep = "\t", row.names = FALSE, quote = FALSE)
  w("GSE1_MASH_Control.tsv", c(0.2, 1)); w("GSE2_MASH_Control.tsv", c(2.0, 1))
  m <- build_bulk_cohorts(file.path(tempdir(), "cohorts"))$meta
  het <- m[m$gene_human == "HET", ]; hom <- m[m$gene_human == "HOM", ]
  expect_equal(hom$meta_log2FC, 1); expect_equal(hom$meta_se, 0.1 / sqrt(2), tolerance = 1e-8)   # same as fixed effect
  expect_gt(het$i2, 0.9); expect_gt(het$meta_se, 0.5)                    # heterogeneity widens the CI
})

test_that("in-vitro: missing padj is never significant", {
  iv <- build_invitro()
  expect_false(any(iv$significant & is.na(iv$padj)))
  named <- iv[!is.na(iv$gene_human), ]
  expect_false(anyDuplicated(named[, c("line", "contrast", "gene_human")]) > 0)   # one row per symbol per line
})
