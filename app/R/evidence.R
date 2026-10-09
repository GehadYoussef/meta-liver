# Cross-layer evidence for one gene (gene summary) and for all genes (screener).

SUMMARY_BULK_CONTRAST <- "MASH vs control"
SUMMARY_INVITRO_CONTRAST <- "OA+PA + resistin/myostatin + PBMC"

fold_phrase <- function(lfc) {
  if (is.na(lfc)) return("no clear direction")
  if (abs(lfc) < 0.05) return("about the same")
  fold <- 2^abs(lfc)
  sprintf("%s (%s)", if (lfc > 0) "higher" else "lower",
          if (fold > 50) "more than 50\u00d7" else sprintf("about %s\u00d7", format(signif(fold, 2))))
}

gene_evidence <- function(g, d) {
  ev <- d$evidence[d$evidence$gene_human == g, ]
  sc_rows <- d$sc[d$sc$gene_human == g & d$sc$dataset %in% d$consistency$datasets, ]
  bm <- d$bulk_cohorts$meta
  bulk <- bm[bm$gene_human == g, ]
  iva <- d$invitro_agreement[d$invitro_agreement$gene_human == g, ]
  wm <- d$wgcna$membership[!is.na(d$wgcna$membership$gene_human) & d$wgcna$membership$gene_human == g, ]
  wt <- d$wgcna$trait[d$wgcna$trait$module %in% wm$module, ]
  kg <- d$kg_nodes[d$kg_nodes$type == "gene/protein" & toupper(d$kg_nodes$name) == toupper(g), ]
  ppi <- d$ppi_network$degree[d$ppi_network$degree$gene_human == g, ]
  drugs <- d$active_drugs$drugs[d$active_drugs$drugs$drugbank %in% d$active_drugs$index$drugbank[d$active_drugs$index$gene_human == g], ]
  list(gene = g, evidence = ev, sc = sc_rows, bulk = bulk, invitro = iva, wgcna_module = wm, wgcna_trait = wt,
       kg = kg, ppi = ppi, drugs = drugs, key_protein = g %in% d$ppi$key_proteins)
}

# Direction per layer as -1, 0 or 1 (NA = no evidence)
layer_directions <- function(e) {
  sc <- if (nrow(e$evidence) && e$evidence$evidence > 0) e$evidence$direction else NA
  b <- e$bulk[e$bulk$contrast == SUMMARY_BULK_CONTRAST, ]
  bulk <- if (nrow(b) && b$fdr < 0.05) sign(b$meta_log2FC) else if (nrow(b)) 0 else NA
  iv <- e$invitro[e$invitro$contrast == SUMMARY_INVITRO_CONTRAST, ]
  inv <- if (!nrow(iv)) NA else if (grepl("^up", iv$call)) 1 else if (grepl("^down", iv$call)) -1 else 0
  c(single_cell = sc, bulk = bulk, invitro = inv)
}

# Plain-language summary: headline, one row per layer and copyable text
gene_narrative <- function(e) {
  g <- e$gene
  rows <- list()

  # Single-cell hepatocytes
  ev <- e$evidence
  sc_txt <- if (!nrow(ev)) "Not detected in enough hepatocytes to test in any usable dataset." else {
    up <- sum(e$sc$auc > 0.5, na.rm = TRUE)
    dn <- sum(e$sc$auc < 0.5, na.rm = TRUE)
    n <- ev$n_datasets
    lead <- if (ev$direction > 0) sprintf("Up in NASH hepatocytes in %d of %d datasets", up, n)
            else if (ev$direction < 0) sprintf("Down in NASH hepatocytes in %d of %d datasets", dn, n)
            else sprintf("Datasets disagree (%d up, %d down)", up, dn)
    auc <- if (ev$direction < 0) 1 - ev$median_auc_disc else ev$median_auc_disc
    sprintf("%s (median AUC %.2f). %s", lead, auc,
            if (ev$n_pseudobulk_sig > 0) sprintf("Donor-level FDR < 0.05 in %d dataset%s.", ev$n_pseudobulk_sig,
                                                   if (ev$n_pseudobulk_sig > 1) "s" else "")
            else "No dataset reaches donor-level significance.")
  }
  rows$single_cell <- list(area = "Hepatocytes (single-cell)", icon = "microscope", text = sc_txt,
                           strength = if (nrow(ev)) as.character(ev$tier) else "none",
                           dir = if (nrow(ev) && ev$evidence > 0) ev$direction else NA)

  # Bulk cohorts
  b <- e$bulk[e$bulk$contrast == SUMMARY_BULK_CONTRAST, ]
  bulk_txt <- if (!nrow(b)) "Not measured in the bulk liver cohorts." else
    sprintf("Whole liver, MASH vs control (%d cohort%s): %s, FDR %s%s.", b$n_studies, if (b$n_studies > 1) "s" else "",
            fold_phrase(b$meta_log2FC), formatC(b$fdr, format = "g", digits = 2),
            if (!is.na(b$i2) && b$i2 > 0.75) ". The cohorts differ a lot in effect size" else "")
  rows$bulk <- list(area = "Whole liver (bulk cohorts)", icon = "flask", text = bulk_txt,
                    strength = if (!nrow(b)) "none" else if (b$fdr < 0.001 && b$n_studies > 1) "high"
                               else if (b$fdr < 0.05) "moderate" else "low",
                    dir = if (nrow(b) && b$fdr < 0.05) sign(b$meta_log2FC) else NA)

  # In-vitro iHeps
  iv <- e$invitro
  iv_main <- iv[iv$contrast == SUMMARY_INVITRO_CONTRAST, ]
  iv_txt <- if (!nrow(iv)) "Not measured in the iHeps model." else
    sprintf("iHeps model, %s vs untreated: %s. Earlier steps: %s.", SUMMARY_INVITRO_CONTRAST,
            if (nrow(iv_main)) iv_main$call else "not measured",
            paste(sprintf("%s %s", iv$contrast[iv$contrast != SUMMARY_INVITRO_CONTRAST],
                          iv$call[iv$contrast != SUMMARY_INVITRO_CONTRAST]), collapse = ", "))
  rows$invitro <- list(area = "Stem-cell model (iHeps)", icon = "vial", text = iv_txt,
                       strength = if (!nrow(iv_main)) "none" else if (grepl("both", iv_main$call)) "high"
                                  else if (grepl("one line", iv_main$call)) "moderate" else "low",
                       dir = if (nrow(iv_main) && grepl("^up", iv_main$call)) 1 else if (nrow(iv_main) && grepl("^down", iv_main$call)) -1 else NA)

  # Co-expression module
  wm <- e$wgcna_module
  wt <- e$wgcna_trait
  w_txt <- if (!nrow(wm)) "Not in the co-expression analysis." else if (!nrow(wt)) "Not assigned to a co-expression module." else
    sprintf("In the %s module, whose activity %s with fibrosis stage (r = %+.2f, p = %s). This is a module-level result.",
            wm$module[1], if (wt$cor_stage > 0) "rises" else "falls", wt$cor_stage, formatC(wt$p_stage, format = "g", digits = 2))
  rows$wgcna <- list(area = "Co-expression (fibrosis stage)", icon = "diagram-project", text = w_txt,
                     strength = if (nrow(wt) && wt$p_stage < 0.001) "high" else if (nrow(wt) && wt$p_stage < 0.05) "moderate" else "low",
                     dir = NA)

  # Networks
  k_txt <- c(if (nrow(e$kg)) sprintf("More central than %.0f%% of genes in the MASH knowledge graph (cluster %s).",
                                     e$kg$composite_pct, e$kg$cluster) else "Not a node in the MASH knowledge graph.",
             if (nrow(e$ppi)) sprintf("Interacts with %d proteins (more partners than %.0f%% of proteins)%s.", e$ppi$degree,
                                      e$ppi$degree_pct, if (e$key_protein) ". It is an early-MAFLD key protein" else "") else NULL,
             if (nrow(e$drugs)) sprintf("Targeted by %d network-active drug%s (%s%s).", nrow(e$drugs), if (nrow(e$drugs) > 1) "s" else "",
                                        if (nrow(e$drugs) > 3) "including " else "", paste(utils::head(e$drugs$drug, 3), collapse = ", ")) else NULL)
  rows$networks <- list(area = "Networks and drugs", icon = "share-nodes", text = paste(k_txt, collapse = " "),
                        strength = if (nrow(e$kg) && e$kg$composite_pct >= 90) "high" else if (nrow(e$kg) && e$kg$composite_pct >= 50) "moderate" else "low",
                        dir = NA)

  # Headline and cross-layer check
  dirs <- layer_directions(e)
  called <- dirs[!is.na(dirs) & dirs != 0]
  names(called) <- c(single_cell = "hepatocytes", bulk = "whole liver", invitro = "iHeps model")[names(called)]
  anchor <- if (!is.na(dirs["bulk"]) && dirs["bulk"] != 0) sprintf("%s in MASH liver than in healthy liver", fold_phrase(b$meta_log2FC))
            else if (!is.na(dirs["single_cell"]) && dirs["single_cell"] != 0) sprintf("%s in NASH hepatocytes", if (dirs["single_cell"] > 0) "higher" else "lower")
            else "not clearly changed in MASH"
  agreement <- if (length(called) < 2) "Too few layers with a clear direction to check agreement."
               else if (length(unique(called)) == 1) sprintf("The layers with a clear direction agree (%s).", paste(names(called), collapse = ", "))
               else sprintf("The layers disagree: %s.", paste(sprintf("%s %s", names(called), ifelse(called > 0, "up", "down")), collapse = ", "))
  headline <- sprintf("%s is %s.", g, anchor)
  copy <- paste(c(headline, agreement, vapply(rows, function(r) sprintf("- %s: %s", r$area, r$text), "")), collapse = "\n")
  list(headline = headline, agreement = agreement, rows = rows, copy = copy,
       consistent = length(called) >= 2 && length(unique(called)) == 1)
}

# One row per gene with every screener criterion
gene_master <- function(d) {
  genes <- sort(unique(c(d$evidence$gene_human, d$bulk_cohorts$meta$gene_human, d$invitro_agreement$gene_human)), method = "radix")
  m <- data.frame(gene = genes, stringsAsFactors = FALSE)
  ev <- d$evidence[match(genes, d$evidence$gene_human), ]
  m$evidence <- ev$evidence
  m$sc_dir <- ev$direction
  m$sc_n <- ev$n_datasets
  m$sc_pb <- ev$n_pseudobulk_sig
  m$sc_auc <- ev$median_auc_disc
  kg <- d$kg_nodes[d$kg_nodes$type == "gene/protein", ]
  i <- match(toupper(genes), toupper(kg$name))
  m$kg_pct <- kg$composite_pct[i]
  m$kg_cluster <- kg$cluster[i]
  wm <- d$wgcna$membership[!is.na(d$wgcna$membership$gene_human), ]
  m$module <- wm$module[match(genes, wm$gene_human)]
  m$module_r <- d$wgcna$trait$cor_stage[match(m$module, d$wgcna$trait$module)]
  m$module_p <- d$wgcna$trait$p_stage[match(m$module, d$wgcna$trait$module)]
  nd <- table(d$active_drugs$index$gene_human)
  m$n_drugs <- as.integer(nd[genes])
  m$n_drugs[is.na(m$n_drugs)] <- 0L
  m$ppi_pct <- d$ppi_network$degree$degree_pct[match(genes, d$ppi_network$degree$gene_human)]
  m$key_protein <- genes %in% d$ppi$key_proteins
  m$target <- genes %in% d$target_genes_human
  m
}

# NULL entries in crit switch a layer off
run_screener <- function(master, d, crit, max_n = 250) {
  m <- master
  keep <- rep(TRUE, nrow(m))
  dir_want <- switch(crit$direction %||% "any", up = 1, down = -1, NA)
  if (!is.null(crit$sc)) {
    keep <- keep & !is.na(m$evidence) & m$evidence >= crit$sc$min_evidence & m$sc_n >= crit$sc$min_n
    if (isTRUE(crit$sc$require_pb)) keep <- keep & m$sc_pb >= 1
    if (!is.na(dir_want)) keep <- keep & m$sc_dir == dir_want
  }
  bulk_lfc <- bulk_fdr <- rep(NA_real_, nrow(m))
  if (!is.null(crit$bulk)) {
    b <- d$bulk_cohorts$meta[d$bulk_cohorts$meta$contrast == crit$bulk$contrast, ]
    i <- match(m$gene, b$gene_human)
    bulk_lfc <- b$meta_log2FC[i]
    bulk_fdr <- b$fdr[i]
    ok <- !is.na(i) & b$fdr[i] < crit$bulk$max_fdr & b$n_studies[i] >= crit$bulk$min_studies
    if (isTRUE(crit$bulk$consistent)) ok <- ok & b$agreement[i] == 1
    if (!is.na(dir_want)) ok <- ok & sign(b$meta_log2FC[i]) == dir_want
    keep <- keep & ok
  }
  iv_call <- rep(NA_character_, nrow(m))
  if (!is.null(crit$invitro)) {
    a <- d$invitro_agreement[d$invitro_agreement$contrast == crit$invitro$contrast, ]
    iv_call <- a$call[match(m$gene, a$gene_human)]
    pat <- if (isTRUE(crit$invitro$both_lines)) "both lines" else "(both lines|one line)"
    ok <- !is.na(iv_call) & grepl(pat, iv_call)
    if (!is.na(dir_want)) ok <- ok & grepl(if (dir_want > 0) "^up" else "^down", iv_call)
    keep <- keep & ok
  }
  if (!is.null(crit$kg)) keep <- keep & !is.na(m$kg_pct) & m$kg_pct >= crit$kg$min_pct
  if (!is.null(crit$wgcna)) {
    ok <- !is.na(m$module_r) & abs(m$module_r) >= crit$wgcna$min_r & m$module_p < crit$wgcna$max_p
    if (identical(crit$wgcna$trend, "rises")) ok <- ok & m$module_r > 0
    if (identical(crit$wgcna$trend, "falls")) ok <- ok & m$module_r < 0
    if (isTRUE(crit$wgcna$drug_target)) ok <- ok & m$n_drugs > 0
    keep <- keep & ok
  }
  if (!is.null(crit$ppi)) {
    ok <- !is.na(m$ppi_pct) & m$ppi_pct >= crit$ppi$min_pct
    if (isTRUE(crit$ppi$key_only)) ok <- ok & m$key_protein
    keep <- keep & ok
  }
  if (isTRUE(crit$targets_only)) keep <- keep & m$target
  if (isTRUE(crit$same_direction)) {
    iv_dir <- ifelse(is.na(iv_call), NA, ifelse(grepl("^up", iv_call), 1, ifelse(grepl("^down", iv_call), -1, NA)))
    dirs <- cbind(if (!is.null(crit$sc)) m$sc_dir else NA, if (!is.null(crit$bulk)) sign(bulk_lfc) else NA,
                  if (!is.null(crit$invitro)) iv_dir else NA)
    same <- apply(dirs, 1, function(x) length(unique(x[!is.na(x) & x != 0])) <= 1)
    keep <- keep & same
  }
  res <- m[keep, ]
  res$bulk_log2FC <- bulk_lfc[keep]
  res$bulk_fdr <- bulk_fdr[keep]
  res$iv_call <- iv_call[keep]
  # Rank by strength before cutting to max_n
  score <- ifelse(is.na(res$evidence), 0, res$evidence) +
           ifelse(is.na(res$bulk_fdr), 0, pmin(1, -log10(pmax(res$bulk_fdr, 1e-300)) / 10))
  res <- res[order(-score, res$gene, method = "radix"), ]
  list(n_total = nrow(res), table = utils::head(res, max_n))
}
