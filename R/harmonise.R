# Bring every result source into one tidy format for the app.

SC_COLUMNS <- c(
  "dataset", "species", "gene", "gene_human", "auc", "auc_power", "direction", "tested",
  "pct_disease", "pct_control", "mean_logexpr_disease", "mean_logexpr_control",
  "pb_logFC", "pb_fdr", "pb_method", "legacy_cell_logFC", "legacy_cell_padj",
  "result_version", "cell_population"
)

# Column names in each legacy result file.
LEGACY_SC_SPECS <- list(
  su_GSE166504 = list(
    file = "GSE166504_Su_mouse_hep_AUC_legacy.csv", species = "mouse",
    gene = "Gene", pct_disease = "pct_NAFLD", pct_control = "pct_Control",
    mean_disease = "mean_logexpr_NAFLD", mean_control = "mean_logexpr_Control",
    logfc = "avg_logFC", padj = "p_val_adj",
    population = "hepatocyte libraries (legacy, balanced)"),
  xiao_GSE189600 = list(
    file = "GSE189600_Xiao_human_hep_AUC_legacy.csv", species = "human",
    gene = "gene", pct_disease = "pct_NASH", pct_control = "pct_Healthy",
    mean_disease = "mean_logexpr_NASH", mean_control = "mean_logexpr_Healthy",
    logfc = "avg_logFC", padj = "p_val_adj",
    population = "marker-gated hepatocytes (legacy, balanced)"),
  coassolo_GSE210501 = list(
    file = "GSE210501_Coassolo_mouse_hep_targets_AUC_legacy.csv", species = "mouse",
    gene = "Gene", pct_disease = "pct_NASH", pct_control = "pct_Chow",
    logfc = "avg_log2FC", padj = "p_val_adj",
    population = "marker-gated hepatocytes, target genes only (legacy)"),
  wang_human_GSE212837 = list(
    file = "GSE212837_Wang_human_AUC_legacy.csv", species = "human",
    gene = "Gene", pct_disease = "pct.1", pct_control = "pct.2", logfc = "avg_logFC",
    population = "unknown (legacy, ident.1 assumed = NASH)")
)

# Mouse to human symbol via MGI one-to-one orthologs. Genes without one keep their
# mouse symbol. Upper-cases if the ortholog table is missing.
to_human_symbol <- function(gene, species, orth = load_orthologs()) {
  if (species == "human") return(gene)
  if (is.null(orth)) return(toupper(gene))
  hit <- orth$human_symbol[match(gene, orth$mouse_symbol)]
  ifelse(is.na(hit), gene, hit)
}

col_or_na <- function(df, col) if (!is.null(col) && col %in% names(df)) df[[col]] else NA

harmonise_legacy_sc <- function(name, spec, min_detect = 0.10,
                                dir = project_path("data", "single_cell", "legacy")) {
  df <- utils::read.csv(file.path(dir, spec$file), check.names = FALSE, stringsAsFactors = FALSE)
  out <- data.frame(
    dataset = name, species = spec$species, gene = df[[spec$gene]],
    auc = as.numeric(df$AUC),
    pct_disease = as.numeric(df[[spec$pct_disease]]),
    pct_control = as.numeric(df[[spec$pct_control]]),
    mean_logexpr_disease = as.numeric(col_or_na(df, spec$mean_disease)),
    mean_logexpr_control = as.numeric(col_or_na(df, spec$mean_control)),
    legacy_cell_logFC = as.numeric(col_or_na(df, spec$logfc)),
    legacy_cell_padj = as.numeric(col_or_na(df, spec$padj)),
    stringsAsFactors = FALSE
  )
  # Undetected genes are untested, not AUC = 0
  out$tested <- pmax(out$pct_disease, out$pct_control) >= min_detect
  out$auc[!out$tested] <- NA_real_
  out$auc_power <- 2 * abs(out$auc - 0.5)
  out$direction <- auc_direction(out$auc)
  out$pb_logFC <- NA_real_
  out$pb_fdr <- NA_real_
  out$pb_method <- "none (legacy result, rerun pipeline for pseudobulk p-values)"
  out$result_version <- "legacy"
  out$cell_population <- spec$population
  out
}

read_v2_sc <- function(path) {
  df <- utils::read.csv(path, stringsAsFactors = FALSE)
  df$legacy_cell_logFC <- NA_real_
  df$legacy_cell_padj <- NA_real_
  df$result_version <- "v2"
  df$cell_population <- "hepatocytes (cluster + per-cell marker annotation)"
  df
}

# v2 results (data/single_cell/results) take precedence over legacy files.
build_sc_table <- function(min_detect = 0.10) {
  orth <- load_orthologs()
  v2_files <- list.files(project_path("data", "single_cell", "results"),
                         pattern = "_hepatocyte_gene_stats\\.csv$", full.names = TRUE)
  v2_names <- sub("_hepatocyte_gene_stats\\.csv$", "", basename(v2_files))
  tabs <- lapply(v2_files, read_v2_sc)
  for (nm in setdiff(names(LEGACY_SC_SPECS), v2_names)) {
    tabs[[length(tabs) + 1]] <- harmonise_legacy_sc(nm, LEGACY_SC_SPECS[[nm]], min_detect)
  }
  tabs <- lapply(tabs, function(d) {
    d$gene_human <- to_human_symbol(d$gene, d$species[1], orth)
    d[, SC_COLUMNS]
  })
  do.call(rbind, tabs)
}

# Bulk RNA-seq (DESeq2 result tables)

BULK_CONTRASTS <- data.frame(
  file = c("1vs0_all.csv", "2vs0_all.csv", "3vs0_all.csv", "4vs0_all.csv",
           "NAFLvsControl_nofilter.csv", "NASH_F0_F1vsControl_nofilter.csv",
           "NASH_F2vsControl_nofilter.csv", "NASH_F3vsControl_nofilter.csv",
           "NASH_F4vsControl_nofilter.csv"),
  contrast = c("Stage 1 vs 0", "Stage 2 vs 0", "Stage 3 vs 0", "Stage 4 vs 0",
               "NAFL vs control", "NASH F0-F1 vs control", "NASH F2 vs control",
               "NASH F3 vs control", "NASH F4 vs control"),
  family = rep(c("stage_vs_stage0", "vs_control"), c(4, 5)),
  order = 1:9,
  stringsAsFactors = FALSE
)

strip_ensembl_version <- function(x) sub("\\.\\d+$", "", x)

load_ensembl_map <- function(path = project_path("data", "reference", "ensembl_to_symbol_human.csv")) {
  if (!file.exists(path)) stop("Run scripts/10_build_reference.R first (missing ", path, ")")
  utils::read.csv(path, stringsAsFactors = FALSE)
}

build_bulk_table <- function(dir = project_path("data", "bulk")) {
  emap <- load_ensembl_map()
  tabs <- lapply(seq_len(nrow(BULK_CONTRASTS)), function(i) {
    df <- utils::read.csv(file.path(dir, BULK_CONTRASTS$file[i]), stringsAsFactors = FALSE)
    id_raw <- df[[1]]
    df <- df[!grepl("_PAR_Y$", id_raw), ]
    ens <- strip_ensembl_version(df[[1]])
    data.frame(
      ensembl = ens, gene_human = emap$symbol[match(ens, emap$ensembl)],
      contrast = BULK_CONTRASTS$contrast[i], family = BULK_CONTRASTS$family[i],
      baseMean = df$baseMean, log2FC = df$log2FoldChange, lfcSE = df$lfcSE,
      pvalue = df$pvalue, padj = df$padj, stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, tabs)
  out$contrast <- factor(out$contrast, levels = BULK_CONTRASTS$contrast)
  out
}

# Knowledge graph (Supp Table 6)

read_kg_nodes <- function(path) {
  df <- data.table::fread(path, data.table = FALSE, na.strings = c("", "NA"))
  names(df) <- tolower(gsub("\\s+", "_", names(df)))
  df <- df[!is.na(df$type) & nzchar(df$type), ]   # the export has ~1M blank rows
  df
}

build_kg_tables <- function(dir = project_path("data", "knowledge_graph")) {
  nodes <- read_kg_nodes(file.path(dir, "MASH_subgraph_nodes.csv"))
  sp <- function(f) unique(utils::read.csv(file.path(dir, f), check.names = FALSE)$DrugBank_Accession)
  nash <- sp("NASH_shortest_paths.csv")
  hep  <- sp("Hepatic_steatosis_shortest_paths.csv")
  drugs <- nodes[nodes$type == "drug", ]
  drugs$in_nash_shortest_paths <- drugs$drugbank_accession %in% nash
  drugs$in_steatosis_shortest_paths <- drugs$drugbank_accession %in% hep
  drugs$in_any_shortest_paths <- drugs$in_nash_shortest_paths | drugs$in_steatosis_shortest_paths
  drugs <- drugs[order(-drugs$pagerank_score), ]
  drugs$pagerank_rank <- seq_len(nrow(drugs))
  genes <- nodes[nodes$type == "gene/protein", c("name", "pagerank_score", "betweenness_score", "eigen_score", "cluster")]
  genes$pagerank_rank <- rank(-genes$pagerank_score, ties.method = "min")
  list(drugs = drugs, genes = genes)
}

# PPI network (early MAFLD)

parse_py_list <- function(x) {
  vapply(regmatches(x, gregexpr("'([^']+)'", x)),
         function(m) paste(gsub("'", "", m), collapse = ", "), "")
}

build_ppi_tables <- function(dir = project_path("data", "ppi", "early_mafld_network")) {
  cen <- utils::read.csv(file.path(dir, "Centrality_RWR_result_pvalue.csv"), check.names = FALSE)
  names(cen)[1] <- "gene_human"
  names(cen) <- tolower(gsub("\\s+", "_", names(cen)))
  key <- readLines(file.path(dir, "key_proteins.txt"))[-1]
  cen$key_protein <- cen$gene_human %in% key

  prox <- utils::read.csv(file.path(dir, "drug_network_proximity_results.csv"), check.names = FALSE)
  prox <- data.frame(
    drug_id = prox$Drug,
    id_type = ifelse(grepl("^DB[0-9]+$", prox$Drug), "DrugBank",
                     ifelse(grepl("^CHEMBL", prox$Drug), "ChEMBL", "other")),
    n_targets = prox$n.source, d = prox$d, z = prox$z,
    targets_key_proteins = parse_py_list(prox[["0degree.target.keyprotein"]]),
    n_key_0deg = prox$n.0degree, n_key_1deg = prox$n.1degree, n_key_2deg = prox$n.2degree,
    coverage = prox$coverage.score, specificity = prox$specificity.score,
    stringsAsFactors = FALSE
  )
  prox <- prox[order(prox$z), ]
  list(centrality = cen, proximity = prox, key_proteins = key)
}

# WGCNA on the bulk cohort

build_wgcna_tables <- function(dir = project_path("data", "external", "metaliver", "wgcna")) {
  emap <- load_ensembl_map()
  mm <- utils::read.csv(file.path(dir, "Module-gene-mapping.csv"), stringsAsFactors = FALSE)
  names(mm) <- c("ensembl", "module")
  mm$ensembl <- strip_ensembl_version(mm$ensembl)
  mm$gene_human <- emap$symbol[match(mm$ensembl, emap$ensembl)]

  cor <- utils::read.csv(file.path(dir, "moduleTraitCor.csv"), stringsAsFactors = FALSE)
  pv  <- utils::read.csv(file.path(dir, "moduleTraitPvalue.csv"), stringsAsFactors = FALSE)
  trait <- data.frame(module = sub("^ME", "", cor[[1]]), cor_stage = cor[[2]],
                      p_stage = pv[[2]][match(cor[[1]], pv[[1]])])
  trait$n_genes <- as.vector(table(mm$module)[trait$module])

  enr_files <- list.files(file.path(dir, "pathways"), pattern = "_enrichment\\.csv$", full.names = TRUE)
  enr <- do.call(rbind, lapply(enr_files, function(f) {
    d <- utils::read.csv(f, stringsAsFactors = FALSE)
    if (!nrow(d)) return(NULL)
    data.frame(module = d$query, source = d$source, term_id = d$term_id, term_name = d$term_name,
               p_value = d$p_value, intersection_size = d$intersection_size, term_size = d$term_size)
  }))
  list(membership = mm, trait = trait[order(trait$p_stage), ], enrichment = enr)
}

# Legacy vs v2 single-cell results

compare_legacy_v2 <- function(min_detect = 0.10) {
  v2_dir <- project_path("data", "single_cell", "results")
  rows <- lapply(names(LEGACY_SC_SPECS), function(nm) {
    f <- file.path(v2_dir, paste0(nm, "_hepatocyte_gene_stats.csv"))
    if (!file.exists(f)) return(NULL)
    spec <- LEGACY_SC_SPECS[[nm]]
    old <- harmonise_legacy_sc(nm, spec, min_detect)
    new <- utils::read.csv(f, stringsAsFactors = FALSE)
    m <- merge(old[old$tested, c("gene", "auc", "legacy_cell_padj")],
               new[new$tested, c("gene", "auc")], by = "gene", suffixes = c("_old", "_new"))
    strong <- abs(m$auc_old - 0.5) >= 0.1 | abs(m$auc_new - 0.5) >= 0.1
    top <- function(d, n = 50) head(d$gene[order(-abs(d$auc - 0.5))], n)
    data.frame(
      dataset = nm,
      genes_compared = nrow(m),
      auc_correlation = round(stats::cor(m$auc_old, m$auc_new), 3),
      direction_agreement_strong_genes = round(mean(sign(m$auc_old[strong] - 0.5) == sign(m$auc_new[strong] - 0.5)), 3),
      top50_overlap = length(intersect(top(old[old$tested, ]), top(new[new$tested, ]))),
      old_cell_level_padj_lt_0.05 = sum(old$legacy_cell_padj < 0.05, na.rm = TRUE),
      new_pseudobulk_fdr_lt_0.05 = sum(new$pb_fdr < 0.05, na.rm = TRUE),
      new_pvalue_method = new$pb_method[1],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

# Hepatocyte specificity of target genes in mouse NASH liver (Wang, script 05)
build_hep_specificity <- function(path = project_path("data", "single_cell", "results",
                                                      "wang_mouse_hep_specificity_target_genes.csv")) {
  if (!file.exists(path)) return(NULL)
  d <- utils::read.csv(path, stringsAsFactors = FALSE)
  d$gene_human <- to_human_symbol(d$gene, "mouse")
  d
}

# Direction consistency across datasets

direction_call <- function(auc, delta = 0.05) {
  ifelse(is.na(auc), NA_character_,
         ifelse(auc >= 0.5 + delta, "up", ifelse(auc <= 0.5 - delta, "down", "none")))
}

# Up or down if any NASH-vs-control contrast has padj < 0.05 in only that direction
bulk_direction <- function(bulk, family = "vs_control", contrast_pattern = "^NASH") {
  b <- bulk[bulk$family == family & grepl(contrast_pattern, bulk$contrast) &
              !is.na(bulk$gene_human) & !is.na(bulk$padj), ]
  sig_up <- tapply(b$padj < 0.05 & b$log2FC > 0, b$gene_human, any)
  sig_dn <- tapply(b$padj < 0.05 & b$log2FC < 0, b$gene_human, any)
  out <- ifelse(sig_up & !sig_dn, "up", ifelse(sig_dn & !sig_up, "down", "none"))
  data.frame(gene_human = names(out), bulk = as.vector(out), stringsAsFactors = FALSE)
}

build_consistency <- function(sc, bulk = NULL, delta = 0.05) {
  # Several source genes can map to one human symbol: keep the strongest
  sc <- sc[order(sc$dataset, sc$gene_human, -abs(sc$auc - 0.5), na.last = TRUE), ]
  sc <- sc[!duplicated(sc[, c("dataset", "gene_human")]), ]
  sc$call <- direction_call(sc$auc, delta)
  datasets <- sort(unique(sc$dataset))
  w <- stats::reshape(sc[, c("gene_human", "dataset", "call")], idvar = "gene_human",
                      timevar = "dataset", direction = "wide")
  names(w) <- sub("^call[.]", "", names(w))
  for (ds in setdiff(datasets, names(w))) w[[ds]] <- NA_character_
  calls <- as.matrix(w[, datasets, drop = FALSE])
  w$n_up <- rowSums(calls == "up", na.rm = TRUE)
  w$n_down <- rowSums(calls == "down", na.rm = TRUE)
  w$n_tested <- rowSums(!is.na(calls))
  species <- unique(sc[, c("dataset", "species")])
  sp_dir <- function(sp) {
    cols <- intersect(species$dataset[species$species == sp], datasets)
    u <- rowSums(calls[, cols, drop = FALSE] == "up", na.rm = TRUE)
    d <- rowSums(calls[, cols, drop = FALSE] == "down", na.rm = TRUE)
    ifelse(u > 0 & d == 0, "up", ifelse(d > 0 & u == 0, "down", ifelse(u > 0 & d > 0, "conflict", NA)))
  }
  w$mouse <- sp_dir("mouse")
  w$human <- sp_dir("human")
  n_called <- w$n_up + w$n_down
  w$consistency <- ifelse(n_called < 2, "called in < 2 datasets",
                   ifelse(w$n_down == 0, paste0("consistent up (", w$n_up, "/", length(datasets), ")"),
                   ifelse(w$n_up == 0, paste0("consistent down (", w$n_down, "/", length(datasets), ")"),
                          paste0("conflicting (", w$n_up, " up, ", w$n_down, " down)"))))
  w$score <- ifelse(n_called > 0, (w$n_up - w$n_down) / length(datasets), NA)
  if (!is.null(bulk)) {
    bd <- bulk_direction(bulk)
    w$bulk <- bd$bulk[match(w$gene_human, bd$gene_human)]
  }
  w <- w[order(-abs(w$score), -n_called, w$gene_human, na.last = TRUE), ]
  rownames(w) <- NULL

  pairwise <- do.call(rbind, lapply(datasets, function(a) do.call(rbind, lapply(datasets, function(b) {
    k <- w[[a]] %in% c("up", "down") & w[[b]] %in% c("up", "down")
    data.frame(dataset_a = a, dataset_b = b, n_genes = sum(k),
               agreement = if (sum(k) >= 10) mean(w[[a]][k] == w[[b]][k]) else NA_real_)
  }))))
  list(genes = w, pairwise = pairwise, datasets = datasets, delta = delta)
}

consistency_summary <- function(cons, genes = NULL, label = "all genes") {
  w <- cons$genes
  if (!is.null(genes)) w <- w[w$gene_human %in% genes, ]
  called2 <- (w$n_up + w$n_down) >= 2
  k <- length(cons$datasets)
  data.frame(
    gene_set = label,
    called_in_2plus = sum(called2),
    consistent = sum(called2 & (w$n_up == 0 | w$n_down == 0)),
    conflicting = sum(w$n_up > 0 & w$n_down > 0),
    consistent_in_all = sum(w$n_up == k | w$n_down == k),
    mouse_and_human_agree = sum(w$mouse %in% c("up", "down") & w$human %in% c("up", "down") & w$mouse == w$human),
    mouse_and_human_disagree = sum(w$mouse %in% c("up", "down") & w$human %in% c("up", "down") & w$mouse != w$human)
  )
}

# Direction agreement of each dataset with bulk NASH F2-F4 vs control
# (padj < 0.05, |log2FC| >= 0.5). 0.5 is chance.
dataset_bulk_agreement <- function(sc, bulk, delta = 0.05) {
  b <- bulk[bulk$family == "vs_control" & bulk$contrast %in%
              c("NASH F2 vs control", "NASH F3 vs control", "NASH F4 vs control") &
              !is.na(bulk$gene_human) & !is.na(bulk$padj), ]
  lfc <- tapply(b$log2FC, b$gene_human, mean)
  padj <- tapply(b$padj, b$gene_human, min)
  ref <- names(lfc)[padj < 0.05 & abs(lfc) >= 0.5]
  do.call(rbind, lapply(sort(unique(sc$dataset)), function(ds) {
    x <- sc[sc$dataset == ds & sc$gene_human %in% ref, ]
    k <- !is.na(x$auc) & abs(x$auc - 0.5) >= delta
    kp <- !is.na(x$pb_logFC)
    data.frame(
      dataset = ds,
      genes_auc = sum(k),
      agreement_auc = round(mean(sign(x$auc[k] - 0.5) == sign(lfc[x$gene_human[k]])), 3),
      genes_pseudobulk = sum(kp),
      agreement_pseudobulk = if (any(kp)) round(mean(sign(x$pb_logFC[kp]) == sign(lfc[x$gene_human[kp]])), 3) else NA,
      cor_pseudobulk_logFC = if (sum(kp) > 10) round(stats::cor(x$pb_logFC[kp], lfc[x$gene_human[kp]]), 3) else NA
    )
  }))
}

# Directions are unreliable when more than 70% (or under 30%) of tested genes go up,
# which points to a technical difference between groups.
direction_qc <- function(sc, lower = 0.30, upper = 0.70) {
  do.call(rbind, lapply(sort(unique(sc$dataset)), function(ds) {
    x <- sc[sc$dataset == ds & sc$tested & !is.na(sc$auc), ]
    up <- mean(x$auc > 0.5)
    data.frame(dataset = ds, genes_tested = nrow(x), pct_up = round(100 * up, 1),
               directions_reliable = up >= lower && up <= upper)
  }))
}

# AUC sign agreement, pseudobulk log2FC correlation and sign agreement between two tables
pairwise_effect_agreement <- function(a, b, delta = 0.05, lfc = 0.5) {
  m <- merge(a[, c("gene_human", "auc", "pb_logFC")], b[, c("gene_human", "auc", "pb_logFC")],
             by = "gene_human", suffixes = c("_a", "_b"))
  k <- !is.na(m$auc_a) & !is.na(m$auc_b) & abs(m$auc_a - 0.5) >= delta & abs(m$auc_b - 0.5) >= delta
  kp <- !is.na(m$pb_logFC_a) & !is.na(m$pb_logFC_b)
  kl <- kp & abs(m$pb_logFC_a) >= lfc & abs(m$pb_logFC_b) >= lfc
  data.frame(
    auc_sign_agreement = if (sum(k)) round(mean(sign(m$auc_a[k] - 0.5) == sign(m$auc_b[k] - 0.5)), 3) else NA,
    n_auc = sum(k),
    pb_cor = if (sum(kp) > 10) round(stats::cor(m$pb_logFC_a[kp], m$pb_logFC_b[kp]), 3) else NA,
    pb_sign_agreement = if (sum(kl)) round(mean(sign(m$pb_logFC_a[kl]) == sign(m$pb_logFC_b[kl])), 3) else NA,
    n_pb = sum(kl)
  )
}
