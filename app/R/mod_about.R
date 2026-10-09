# About: definitions, methods and provenance tables.

aboutUI <- function(id, d) {
  ns <- shiny::NS(id)
  item <- function(icon, title, ...) bslib::accordion_panel(title, icon = shiny::icon(icon), shiny::p(...))
  shiny::tagList(
    page_header("circle-info", "About", "Definitions, methods and data sources"),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header(card_title("How to read the numbers", "book-open")),
        bslib::accordion(open = "AUC",
          item("scale-balanced", "AUC",
               "Chance that a random NASH hepatocyte expresses the gene more than a random control hepatocyte. ",
               "0.5 means no difference, above 0.5 is up in NASH and below is down. Computed on depth-matched counts."),
          item("eye-slash", "Not detected",
               "Genes found in fewer than 10% of cells in both groups are not tested and have no AUC."),
          item("users", "Pseudobulk FDR",
               "Counts summed per donor or animal (captures from one liver pooled), tested with edgeR. ",
               "Cell-level p-values, which treat each cell as a replicate, are not used."),
          item("arrows-up-down", "Direction calls",
               sprintf("Up = AUC ≥ %.2f, down = AUC ≤ %.2f. Datasets where > 70%% of genes move one way are ",
                       0.5 + d$consistency$delta, 0.5 - d$consistency$delta),
               "flagged as unreliable and left out of the consistency calls."),
          item("bullseye", "Agreement with bulk",
               "Share of genes significant in the independent bulk cohort (NASH F2–F4 vs control) that go the ",
               "same way in hepatocytes. 50% = chance.")
        )
      ),
      bslib::card(
        bslib::card_header(card_title("Methods", "gears")),
        bslib::accordion(open = FALSE,
          item("filter", "QC", "Mitochondrial cut-off set per capture (median + 3 MAD, within a floor and cap), ",
               "because control and NASH captures were prepared differently."),
          item("microscope", "Hepatocytes", "Each cluster is scored on 7 lineage marker panels and called non-hepatocyte when ",
               "one lineage's score is a robust outlier. Single contaminating cells are then removed. Su also requires the ",
               "authors' hepatocyte label."),
          item("scale-unbalanced", "Depth matching", "The deeper group's counts are thinned to equal median depth ",
               "before the AUC, so lower sequencing depth in one group cannot make genes look down."),
          item("flask-vial", "Sensitivity analyses", "Seven processing variants, Wang without FACS-sorted captures, and ",
               "DecontX ambient RNA correction all leave the human disagreement unchanged."),
          item("right-left", "Orthologs", "Mouse genes map to human via MGI one-to-one orthologs (", d$orthologs, ").")
        )
      )
    ),
    bslib::navset_card_underline(
      title = card_title("Provenance", "clipboard-list"),
      bslib::nav_panel("Datasets", reactable::reactableOutput(ns("datasets"))),
      bslib::nav_panel("Original vs fixed", reactable::reactableOutput(ns("comparison"))),
      bslib::nav_panel("Sensitivity", reactable::reactableOutput(ns("sensitivity")))
    ),
    shiny::p(class = "text-muted small mt-2", shiny::icon("clock"), sprintf(" App data built %s", d$built_at))
  )
}

aboutServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    output$datasets <- reactable::renderReactable({
      m <- d$datasets_meta
      rt(data.frame(Dataset = m$label, GEO = m$geo, Species = m$species, Assay = m$assay,
                    Design = sprintf("%s vs %s", m$n_disease, m$n_control), Hepatocytes = m$n_hepatocytes,
                    PValues = m$pvalues, Reliable = m$directions_reliable, Bulk = m$bulk_agreement),
         searchable = FALSE, columns = list(
           Dataset = reactable::colDef(style = list(fontWeight = 600)),
           Hepatocytes = reactable::colDef(format = reactable::colFormat(separators = TRUE)),
           PValues = col_check("p-values"), Reliable = col_check("Directions OK"),
           Bulk = col_bar("Agrees with bulk", digits = 2)))
    })
    output$comparison <- reactable::renderReactable({
      shiny::validate(shiny::need(!is.null(d$comparison), "No dataset has been rerun yet."))
      x <- d$comparison
      x$dataset <- d$datasets_meta$label[match(x$dataset, d$datasets_meta$dataset)]
      rt(x, searchable = FALSE, columns = list(
        dataset = reactable::colDef(name = "Dataset", style = list(fontWeight = 600)),
        genes_compared = reactable::colDef(name = "Genes"),
        auc_correlation = col_bar("AUC correlation", digits = 2),
        direction_agreement_strong_genes = col_bar("Same direction", digits = 2),
        top50_overlap = reactable::colDef(name = "Top-50 overlap"),
        old_cell_level_padj_lt_0.05 = reactable::colDef(name = "Significant, original (cell-level)"),
        new_pseudobulk_fdr_lt_0.05 = reactable::colDef(name = "Significant, fixed (pseudobulk FDR < 0.05)"),
        new_pvalue_method = reactable::colDef(name = "p-value method", minWidth = 220,
                                              style = list(fontSize = "0.78rem", color = PAL$muted))))
    })
    output$sensitivity <- reactable::renderReactable({
      shiny::validate(shiny::need(!is.null(d$sensitivity), "No sensitivity analyses have been run."))
      s <- d$sensitivity
      rt(s, searchable = FALSE, page = 12, columns = list(
        variant = reactable::colDef(name = "Variant", minWidth = 200, style = list(fontWeight = 600)),
        compared_with = reactable::colDef(name = "Compared with", minWidth = 180),
        auc_sign_agreement = col_bar("AUC same direction", digits = 2),
        n_auc = reactable::colDef(name = "n"), pb_cor = col_num("Pseudobulk r"),
        pb_sign_agreement = col_bar("Pseudobulk same direction", digits = 2), n_pb = reactable::colDef(name = "n")))
    })
  })
}
