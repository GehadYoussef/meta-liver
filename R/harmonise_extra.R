# Bulk cohort meta-analysis, in-vitro iHeps model, single-cell evidence score,
# knowledge-graph percentiles, drug-target index and PPI network.

# Bulk cohorts

BULK_COHORT_CONTRASTS <- data.frame(
  folder = c("MASLDvsControl", "EarlyMASLDvsControl", "MASHvsControl", "MASHvsEarlyMASLD"),
  contrast = c("MASLD vs control", "Early MASLD vs control", "MASH vs control", "MASH vs early MASLD"),
  stringsAsFactors = FALSE
)

# SE is lfcSE (DESeq2) or |log2FC / t| (limma). One row per symbol, best p-value.
read_bulk_cohort_table <- function(path) {
  d <- utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE, quote = "")
  se <- if ("lfcSE" %in% names(d)) d$lfcSE else if ("t" %in% names(d)) abs(d$log2FoldChange / d$t) else NA_real_
  out <- data.frame(gene_human = toupper(trimws(d$Symbol)), log2FC = as.numeric(d$log2FoldChange),
                    se = as.numeric(se), pvalue = as.numeric(d$pvalue), padj = as.numeric(d$padj),
                    stringsAsFactors = FALSE)
  out <- out[!is.na(out$gene_human) & nzchar(out$gene_human) & is.finite(out$log2FC) & is.finite(out$se) & out$se > 0, ]
  out <- out[order(out$gene_human, out$pvalue), ]
  out[!duplicated(out$gene_human), ]
}

# Random-effects DerSimonian-Laird meta-analysis per contrast and gene, BH FDR within contrast
build_bulk_cohorts <- function(dir = project_path("data", "bulk_cohorts")) {
  studies <- do.call(rbind, lapply(seq_len(nrow(BULK_COHORT_CONTRASTS)), function(i) {
    files <- list.files(file.path(dir, BULK_COHORT_CONTRASTS$folder[i]), pattern = "[.]tsv$", full.names = TRUE)
    do.call(rbind, lapply(files, function(f) cbind(
      contrast = BULK_COHORT_CONTRASTS$contrast[i],
      study = regmatches(basename(f), regexpr("GSE[0-9]+", basename(f))),
      read_bulk_cohort_table(f), stringsAsFactors = FALSE)))
  }))
  studies$contrast <- factor(studies$contrast, levels = BULK_COHORT_CONTRASTS$contrast)

  key <- paste(studies$contrast, studies$gene_human, sep = "\r")
  w <- 1 / studies$se^2
  sw <- tapply(w, key, sum)
  sw2 <- tapply(w^2, key, sum)
  fe <- tapply(w * studies$log2FC, key, sum) / sw
  k <- tapply(w, key, length)
  q <- tapply(seq_along(key), key, function(i) sum(w[i] * (studies$log2FC[i] - fe[key[i[1]]])^2))
  tau2 <- ifelse(k > 1, pmax(0, (q - (k - 1)) / (sw - sw2 / sw)), 0)
  wr <- 1 / (studies$se^2 + tau2[key])
  swr <- tapply(wr, key, sum)
  meta_lfc <- tapply(wr * studies$log2FC, key, sum) / swr
  n_up <- tapply(studies$log2FC > 0, key, sum)
  meta <- data.frame(
    contrast = sub("\r.*", "", names(sw)), gene_human = sub(".*\r", "", names(sw)),
    n_studies = as.integer(k), meta_log2FC = as.numeric(meta_lfc), meta_se = as.numeric(1 / sqrt(swr)),
    i2 = ifelse(k > 1 & q > 0, pmax(0, (q - (k - 1)) / q), NA_real_), tau2 = as.numeric(tau2),
    agreement = pmax(as.numeric(n_up), as.numeric(k - n_up)) / as.numeric(k),
    stringsAsFactors = FALSE)
  meta$z <- meta$meta_log2FC / meta$meta_se
  meta$p <- 2 * stats::pnorm(-abs(meta$z))
  meta$fdr <- stats::ave(meta$p, meta$contrast, FUN = function(p) stats::p.adjust(p, "BH"))
  meta$contrast <- factor(meta$contrast, levels = BULK_COHORT_CONTRASTS$contrast)
  list(studies = studies, meta = meta)
}

# In-vitro iHeps model (stem-cell-derived hepatocytes)

INVITRO_CONTRASTS <- data.frame(
  key = c("OAPAvsHCM", "OAPAResMyovsHCM", "OAPAResMyoPBMCsvsHCM"),
  contrast = c("OA+PA", "OA+PA + resistin/myostatin", "OA+PA + resistin/myostatin + PBMC"),
  stringsAsFactors = FALSE
)

build_invitro <- function(dir = project_path("data", "invitro")) {
  map <- utils::read.csv(file.path(dir, "gene_mapping.csv.gz"), stringsAsFactors = FALSE, check.names = FALSE)
  map <- map[!duplicated(map[["Gene stable ID"]]), ]
  files <- list.files(dir, pattern = "^OAPA.*[.]csv[.]gz$", full.names = TRUE)
  tab <- do.call(rbind, lapply(files, function(f) {
    parts <- strsplit(sub("[.]csv[.]gz$", "", basename(f)), "_")[[1]]
    d <- utils::read.csv(f, stringsAsFactors = FALSE)
    ens <- sub("[.][0-9]+$", "", d$Gene)
    data.frame(line = parts[2], contrast = INVITRO_CONTRASTS$contrast[match(parts[1], INVITRO_CONTRASTS$key)],
               ensembl = ens, gene_human = map[["Gene name"]][match(ens, map[["Gene stable ID"]])],
               log2FC = d$log2FoldChange, lfcSE = d$lfcSE, pvalue = d$pvalue, padj = d$padj,
               baseMean = d$baseMean, stringsAsFactors = FALSE)
  }))
  tab$contrast <- factor(tab$contrast, levels = INVITRO_CONTRASTS$contrast)
  # One row per symbol, line and exposure (smallest padj)
  tab <- tab[order(tab$line, tab$contrast, tab$gene_human, tab$padj, na.last = TRUE), ]
  tab <- tab[is.na(tab$gene_human) | !duplicated(tab[, c("line", "contrast", "gene_human")]), ]
  tab$significant <- !is.na(tab$padj) & tab$padj < 0.05
  tab
}

# Per gene x contrast: do the two iHeps lines agree?
invitro_line_agreement <- function(tab) {
  t <- tab[!is.na(tab$gene_human), ]
  key <- paste(t$gene_human, t$contrast, sep = "\r")
  sig_up <- tapply(t$significant & t$log2FC > 0, key, sum)
  sig_dn <- tapply(t$significant & t$log2FC < 0, key, sum)
  lines <- tapply(t$line, key, function(x) length(unique(x)))
  data.frame(gene_human = sub("\r.*", "", names(sig_up)), contrast = sub(".*\r", "", names(sig_up)),
             n_lines = as.integer(lines),
             call = ifelse(sig_up >= 2, "up (both lines)", ifelse(sig_dn >= 2, "down (both lines)",
                    ifelse(sig_up == 1 & sig_dn == 0, "up (one line)", ifelse(sig_dn == 1 & sig_up == 0, "down (one line)",
                    ifelse(sig_up >= 1 & sig_dn >= 1, "lines disagree", "not significant"))))),
             stringsAsFactors = FALSE)
}

# Single-cell evidence score = geometric mean of strength, stability and net agreement, times coverage
evidence_scores <- function(sc, datasets) {
  x <- sc[sc$dataset %in% datasets & !is.na(sc$auc), c("gene_human", "dataset", "auc", "pb_fdr")]
  x$disc <- pmax(x$auc, 1 - x$auc)
  x$dir <- sign(x$auc - 0.5)
  by <- split(x, x$gene_human)
  rows <- lapply(by, function(g) {
    n <- nrow(g)
    med <- stats::median(g$disc)
    strength <- max(0, (med - 0.5) / 0.5)
    stability <- if (n > 1) max(0, 1 - stats::IQR(g$disc) / 0.5) else 1
    nd <- sum(g$dir != 0)
    agreement <- if (nd > 0) max(sum(g$dir > 0), sum(g$dir < 0)) / nd else NA_real_
    coverage <- n / length(datasets)
    net <- if (is.na(agreement)) 0 else 2 * agreement - 1
    parts <- c(strength, stability, net)
    score <- if (any(parts == 0)) 0 else coverage * exp(mean(log(parts)))
    up <- sum(g$dir > 0)
    dn <- sum(g$dir < 0)
    c(n_datasets = n, median_auc_disc = med, strength = strength, stability = stability,
      agreement = agreement, coverage = coverage, evidence = score,
      direction = if (up > dn) 1 else if (dn > up) -1 else 0,
      n_pseudobulk_sig = sum(!is.na(g$pb_fdr) & g$pb_fdr < 0.05))
  })
  out <- as.data.frame(do.call(rbind, rows))
  out$gene_human <- names(by)
  out$label <- with(out, ifelse(strength > 0.7 & stability > 0.7 & agreement > 0.7, "Strong, stable, same direction",
                       ifelse(strength > 0.7 & stability > 0.7, "Strong and stable, mixed direction",
                       ifelse(strength > 0.7 & agreement > 0.7, "Strong, same direction, variable effect",
                       ifelse(stability > 0.7 & agreement > 0.7, "Stable, same direction, weak effect",
                              "Weak or inconsistent")))))
  out$tier <- cut(out$evidence, c(-Inf, 0.25, 0.5, 0.75, Inf), c("very low", "low", "moderate", "high"), right = FALSE)
  out[order(-out$evidence), c("gene_human", setdiff(names(out), "gene_human"))]
}

# Knowledge graph percentiles within node type

build_kg_nodes <- function(dir = project_path("data", "knowledge_graph")) {
  n <- read_kg_nodes(file.path(dir, "MASH_subgraph_nodes.csv"))
  pct <- function(x) 100 * rank(x, na.last = "keep", ties.method = "average") / sum(!is.na(x))
  n$pr_pct <- stats::ave(n$pagerank_score, n$type, FUN = pct)
  n$bet_pct <- stats::ave(n$betweenness_score, n$type, FUN = pct)
  n$eig_pct <- stats::ave(n$eigen_score, n$type, FUN = pct)
  # Weighted geometric mean of percentiles (PageRank 0.5, betweenness 0.25, eigenvector 0.25)
  g <- exp(0.5 * log(pmax(n$pr_pct, 0.01) / 100) + 0.25 * log(pmax(n$bet_pct, 0.01) / 100) +
             0.25 * log(pmax(n$eig_pct, 0.01) / 100))
  n$composite_pct <- stats::ave(g, n$type, FUN = pct)
  n[, c("name", "drugbank_accession", "type", "cluster", "pagerank_score", "pr_pct", "bet_pct", "eig_pct", "composite_pct")]
}

# Drugs with targets (WGCNA active drugs)

build_active_drugs <- function(path = project_path("data", "drugs", "wgcna_active_drugs.csv")) {
  d <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  drugs <- data.frame(drugbank = d$DrugBank_Accession, drug = d[["Drug Name"]], moa = d[["Mechanism of Action"]],
                      indication = d$Indication, distance = d$distance, z = d[["z-score"]],
                      targets = d[["Drug Targets"]], stringsAsFactors = FALSE)
  tg <- strsplit(drugs$targets, "[;,|]")
  index <- data.frame(gene_human = toupper(trimws(unlist(tg))),
                      drugbank = rep(drugs$drugbank, lengths(tg)), stringsAsFactors = FALSE)
  index <- unique(index[nzchar(index$gene_human), ])
  list(drugs = drugs[order(drugs$z), ], index = index)
}

# PPI network as an integer edge list with degree

build_ppi_network <- function(path = project_path("data", "external", "metaliver", "wgcna",
                                                  "PPI_network_largest_component.csv")) {
  e <- data.table::fread(path, select = c("prot1_hgnc_id", "prot2_hgnc_id"), data.table = FALSE)
  nodes <- sort(unique(c(e$prot1_hgnc_id, e$prot2_hgnc_id)))
  a <- match(e$prot1_hgnc_id, nodes)
  b <- match(e$prot2_hgnc_id, nodes)
  deg <- tabulate(c(a, b), nbins = length(nodes))
  list(nodes = nodes, from = a, to = b,
       degree = data.frame(gene_human = nodes, degree = deg,
                           degree_pct = 100 * rank(deg) / length(deg), stringsAsFactors = FALSE))
}
