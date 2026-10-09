# Assemble everything the Shiny app needs into app/data/app_data.rds.
# Run after 10_build_reference.R and whenever single-cell results change.
source(here::here("R", "load_all.R"))
cfg <- load_config()

sc <- timed("Single-cell tables", build_sc_table(cfg$defaults$min_detect_pct))
bulk <- timed("Bulk tables", build_bulk_table())
kg <- timed("Knowledge graph", build_kg_tables())
ppi <- timed("PPI network", build_ppi_tables())
wgcna <- timed("WGCNA", build_wgcna_tables())

# Link the two drug analyses through DrugBank IDs
prox <- ppi$proximity
kg$drugs$ppi_proximity_z <- prox$z[match(kg$drugs$drugbank_accession, prox$drug_id)]
ppi$proximity$drug_name <- kg$drugs$name[match(prox$drug_id, kg$drugs$drugbank_accession)]
ppi$proximity$kg_pagerank_rank <- kg$drugs$pagerank_rank[match(prox$drug_id, kg$drugs$drugbank_accession)]
log_step("Drugs in both the KG and the PPI proximity analysis: ", sum(!is.na(kg$drugs$ppi_proximity_z)))

sc_info <- unique(sc[, c("dataset", "species", "result_version", "cell_population", "pb_method")])
sc_info$n_genes <- as.vector(table(sc$dataset)[sc_info$dataset])
sc_info$n_tested <- as.vector(tapply(sc$tested, sc$dataset, sum)[sc_info$dataset])

comparison <- compare_legacy_v2(cfg$defaults$min_detect_pct)
if (!is.null(comparison)) {
  utils::write.csv(comparison, project_path("data", "single_cell", "legacy_vs_v2_comparison.csv"), row.names = FALSE)
  log_step("Legacy vs v2:")
  print(comparison, row.names = FALSE)
}

# Direction consistency for v2 and, for comparison, the legacy tables
tg_human <- target_genes_human(read_target_genes(cfg$defaults$target_genes_mouse))
dir_qc <- direction_qc(sc)
log_step("Direction QC (share of tested genes with AUC > 0.5):")
print(dir_qc, row.names = FALSE)
reliable <- dir_qc$dataset[dir_qc$directions_reliable]
consistency <- build_consistency(sc[sc$dataset %in% reliable, ], bulk)
consistency$pairwise <- build_consistency(sc)$pairwise    # pairwise uses all datasets
legacy_sc <- do.call(rbind, lapply(names(LEGACY_SC_SPECS), function(nm) {
  x <- harmonise_legacy_sc(nm, LEGACY_SC_SPECS[[nm]], cfg$defaults$min_detect_pct)
  x$gene_human <- to_human_symbol(x$gene, x$species[1])
  x
}))
# Same datasets on both sides so the comparison is like for like
consistency_legacy <- build_consistency(legacy_sc[legacy_sc$dataset %in% reliable, ])
consistency_legacy$pairwise <- build_consistency(legacy_sc)$pairwise
consistency_compare <- rbind(
  cbind(version = "v2 (fixed)", rbind(consistency_summary(consistency),
                                      consistency_summary(consistency, tg_human, "target genes"))),
  cbind(version = "legacy (original)", rbind(consistency_summary(consistency_legacy),
                                             consistency_summary(consistency_legacy, tg_human, "target genes")))
)
cons_dir <- ensure_dir(project_path("data", "single_cell", "consistency"))
utils::write.csv(consistency$genes, file.path(cons_dir, "direction_consistency_by_gene.csv"), row.names = FALSE)
utils::write.csv(consistency$pairwise, file.path(cons_dir, "pairwise_direction_agreement.csv"), row.names = FALSE)
utils::write.csv(consistency_compare, file.path(cons_dir, "consistency_v2_vs_legacy.csv"), row.names = FALSE)
utils::write.csv(dir_qc, file.path(cons_dir, "direction_qc_by_dataset.csv"), row.names = FALSE)
# Sensitivity variants (04b, 04c) against each main dataset and the bulk cohort
sens_files <- list.files(project_path("data", "single_cell", "sensitivity"),
                         pattern = "_hepatocyte_gene_stats[.]csv$", full.names = TRUE)
sensitivity <- NULL
if (length(sens_files)) {
  main <- split(sc, sc$dataset)
  ref_bulk <- {
    bb <- bulk[bulk$contrast %in% c("NASH F2 vs control", "NASH F3 vs control", "NASH F4 vs control") &
                 !is.na(bulk$gene_human) & !is.na(bulk$padj), ]
    l <- tapply(bb$log2FC, bb$gene_human, mean)
    q <- tapply(bb$padj, bb$gene_human, min)
    g <- names(l)[q < 0.05 & abs(l) >= 0.5]
    data.frame(gene_human = g, auc = 0.5 + sign(l[g]) * 0.25, pb_logFC = as.numeric(l[g]))
  }
  sensitivity <- do.call(rbind, lapply(sens_files, function(f) {
    v <- read_v2_sc(f)
    v$gene_human <- to_human_symbol(v$gene, v$species[1])
    vname <- sub("_hepatocyte_gene_stats[.]csv$", "", basename(f))
    rbind(
      do.call(rbind, lapply(names(main), function(ds) cbind(variant = vname, compared_with = ds,
                                                             pairwise_effect_agreement(v, main[[ds]])))),
      cbind(variant = vname, compared_with = "bulk (NASH F2-F4 vs control)", pairwise_effect_agreement(v, ref_bulk))
    )
  }))
  # decontX variants against each other
  dx <- sens_files[grepl("_decontx_", basename(sens_files))]
  if (length(dx) == 2) {
    a <- read_v2_sc(dx[1])
    b <- read_v2_sc(dx[2])
    a$gene_human <- a$gene
    b$gene_human <- b$gene
    sensitivity <- rbind(sensitivity, cbind(
      variant = sub("_hepatocyte_gene_stats[.]csv$", "", basename(dx[1])),
      compared_with = sub("_hepatocyte_gene_stats[.]csv$", "", basename(dx[2])),
      pairwise_effect_agreement(a, b)))
  }
  utils::write.csv(sensitivity, file.path(cons_dir, "sensitivity_agreement.csv"), row.names = FALSE)
  log_step("Sensitivity analyses: agreement with main datasets and bulk:")
  print(sensitivity, row.names = FALSE)
}

bulk_agreement <- dataset_bulk_agreement(sc, bulk)
utils::write.csv(bulk_agreement, file.path(cons_dir, "agreement_with_bulk_by_dataset.csv"), row.names = FALSE)
log_step("Agreement of each dataset with the independent bulk cohort (0.5 = chance):")
print(bulk_agreement, row.names = FALSE)

log_step("Direction consistency (v2 vs legacy):")
print(consistency_compare, row.names = FALSE)

# Dataset cards for the app
datasets_meta <- do.call(rbind, lapply(sort(unique(sc$dataset)), function(ds) {
  dc <- cfg$datasets[[ds]]
  donors <- if (!is.null(dc$samples)) {
    unique(data.frame(donor = vapply(dc$samples, function(s) if (is.null(s$donor)) s$id else s$donor, ""),
                      condition = vapply(dc$samples, `[[`, "", "condition")))
  } else NULL
  info_file <- project_path("results", ds, "run_info.yml")
  n_hep <- if (file.exists(info_file)) yaml::read_yaml(info_file)$summary$n_hepatocytes else NA
  ba <- bulk_agreement[bulk_agreement$dataset == ds, ]
  data.frame(
    dataset = ds,
    label = tools::toTitleCase(sub("_.*", "", ds)),
    geo = regmatches(ds, regexpr("GSE[0-9]+", ds)),
    species = dc$species, assay = dc$assay,
    n_disease = if (is.null(donors)) NA else sum(donors$condition == "disease"),
    n_control = if (is.null(donors)) NA else sum(donors$condition == "control"),
    n_hepatocytes = as.numeric(n_hep),
    directions_reliable = dir_qc$directions_reliable[dir_qc$dataset == ds],
    pct_up = dir_qc$pct_up[dir_qc$dataset == ds],
    pvalues = !all(is.na(sc$pb_fdr[sc$dataset == ds])),
    bulk_agreement = ba$agreement_auc, bulk_cor = ba$cor_pseudobulk_logFC,
    stringsAsFactors = FALSE
  )
}))
# Su's donors come from the metadata, not the config
su <- datasets_meta$dataset == "su_GSE166504"
datasets_meta$n_disease[su] <- 3
datasets_meta$n_control[su] <- 6
print(datasets_meta, row.names = FALSE)

# Layers ported from the Meta Liver app (see R/harmonise_extra.R)
bulk_cohorts <- timed("Bulk cohorts meta-analysis", build_bulk_cohorts())
invitro <- timed("In-vitro iHeps model", build_invitro())
invitro_agreement <- invitro_line_agreement(invitro)
evidence <- evidence_scores(sc, reliable)
kg_nodes <- build_kg_nodes()
active_drugs <- build_active_drugs()
ppi_network <- timed("PPI network", build_ppi_network())
utils::write.csv(evidence, file.path(cons_dir, "single_cell_evidence_scores.csv"), row.names = FALSE)
log_step("Evidence tiers: ", paste(names(table(evidence$tier)), table(evidence$tier), sep = "=", collapse = ", "))

app_data <- list(
  bulk_cohorts = bulk_cohorts,
  invitro = invitro[, c("line", "contrast", "ensembl", "gene_human", "log2FC", "lfcSE", "padj", "significant")],
  invitro_agreement = invitro_agreement,
  evidence = evidence,
  kg_nodes = kg_nodes,
  active_drugs = active_drugs,
  ppi_network = ppi_network,
  datasets_meta = datasets_meta,
  consistency = consistency,
  direction_qc = dir_qc,
  bulk_agreement = bulk_agreement,
  sensitivity = sensitivity,
  consistency_legacy_pairwise = consistency_legacy$pairwise,
  consistency_compare = consistency_compare,
  comparison = comparison,
  hep_specificity = build_hep_specificity(),
  sc = sc, sc_info = sc_info, bulk = bulk, kg = kg, ppi = ppi, wgcna = wgcna,
  target_genes_mouse = read_target_genes(cfg$defaults$target_genes_mouse),
  target_genes_human = target_genes_human(read_target_genes(cfg$defaults$target_genes_mouse)),
  orthologs = if (is.null(load_orthologs())) "upper-case symbol matching (MGI table not built)" else "MGI one-to-one orthologs",
  built_at = format(Sys.time(), "%Y-%m-%d %H:%M")
)
out <- project_path("app", "data", "app_data.rds")
saveRDS(app_data, out, compress = "xz")
print(sc_info, row.names = FALSE)
log_step("Wrote ", out, " (", round(file.size(out) / 1e6, 1), " MB)")
