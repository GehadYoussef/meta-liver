# Gene screener: filter genes by criteria from each evidence layer.

screenerUI <- function(id, d) {
  ns <- shiny::NS(id)
  layer <- function(key, icon, title, on, ...) {
    shiny::div(class = "layer-card",
      shiny::div(class = "layer-head",
        shiny::div(class = "layer-title", shiny::span(class = "layer-icon", shiny::icon(icon)), title),
        bslib::input_switch(ns(paste0("use_", key)), NULL, value = on)),
      shiny::div(class = "mt-2", ...))
  }
  shiny::tagList(
    page_header("filter", "Gene screener",
                "Filter genes across evidence layers"),
    shiny::div(class = "toolbar",
      shiny::span(class = "toolbar-label", "Direction"),
      pills(ns("direction"), c("Any" = "any", "▲ Up in disease" = "up", "▼ Down in disease" = "down")),
      shiny::checkboxInput(ns("same_direction"), "Layers must agree in direction", TRUE),
      shiny::checkboxInput(ns("targets_only"), "Target genes only", FALSE)
    ),
    shiny::div(class = "layer-grid",
      layer("sc", "microscope", "Hepatocytes (single-cell)", TRUE,
        shiny::sliderInput(ns("sc_min_evidence"), "Min evidence score", 0, 1, 0.3, step = 0.05),
        shiny::sliderInput(ns("sc_min_n"), "Min datasets measured", 1, 3, 2, step = 1),
        shiny::checkboxInput(ns("sc_require_pb"), "Donor-level FDR < 0.05 in at least one dataset", FALSE)),
      layer("bulk", "flask", "Whole liver (bulk cohorts)", TRUE,
        shiny::selectInput(ns("bulk_contrast"), NULL, BULK_COHORT_LEVELS, selected = "MASH vs control"),
        shiny::sliderInput(ns("bulk_fdr"), "Max meta-analysis FDR", 0.001, 0.2, 0.05, step = 0.005),
        shiny::sliderInput(ns("bulk_min_studies"), "Min cohorts", 1, 3, 2, step = 1),
        shiny::checkboxInput(ns("bulk_consistent"), "Same direction in every cohort", TRUE)),
      layer("invitro", "vial", "Stem-cell model (iHeps)", FALSE,
        shiny::selectInput(ns("iv_contrast"), NULL, levels(d$invitro$contrast), selected = SUMMARY_INVITRO_CONTRAST),
        shiny::checkboxInput(ns("iv_both"), "Significant in both cell lines", TRUE)),
      layer("wgcna", "diagram-project", "Co-expression module", FALSE,
        pills(ns("wg_trend"), c("Rises with stage" = "rises", "Falls" = "falls", "Either" = "either")),
        shiny::sliderInput(ns("wg_min_r"), "Min absolute module correlation with stage", 0, 0.5, 0.15, step = 0.05),
        shiny::checkboxInput(ns("wg_drug"), "Gene is a drug target", FALSE)),
      layer("kg", "share-nodes", "Knowledge graph", FALSE,
        shiny::sliderInput(ns("kg_min_pct"), "Min centrality percentile", 0, 100, 80, step = 5)),
      layer("ppi", "circle-nodes", "Protein interactions", FALSE,
        shiny::sliderInput(ns("ppi_min_pct"), "Min interaction-count percentile", 0, 100, 50, step = 5),
        shiny::checkboxInput(ns("ppi_key"), "Early-MAFLD key proteins only", FALSE))
    ),
    bslib::card(
      bslib::card_header(shiny::div(class = "d-flex justify-content-between align-items-center w-100",
        shiny::uiOutput(ns("count")),
        shiny::div(class = "d-flex gap-2 align-items-center",
          shiny::numericInput(ns("max_n"), NULL, 250, min = 10, max = 5000, step = 50, width = "110px"),
          shiny::downloadButton(ns("download"), "CSV", class = "btn-gear")))),
      table_output(ns("table"))
    )
  )
}

screenerServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    master <- gene_master(d)

    crit <- shiny::reactive({
      list(
        direction = input$direction %||% "any",
        same_direction = isTRUE(input$same_direction),
        targets_only = isTRUE(input$targets_only),
        sc = if (isTRUE(input$use_sc)) list(min_evidence = input$sc_min_evidence %||% 0.3, min_n = input$sc_min_n %||% 2,
                                            require_pb = isTRUE(input$sc_require_pb)),
        bulk = if (isTRUE(input$use_bulk)) list(contrast = input$bulk_contrast %||% "MASH vs control",
                                                max_fdr = input$bulk_fdr %||% 0.05, min_studies = input$bulk_min_studies %||% 2,
                                                consistent = isTRUE(input$bulk_consistent)),
        invitro = if (isTRUE(input$use_invitro)) list(contrast = input$iv_contrast %||% SUMMARY_INVITRO_CONTRAST,
                                                      both_lines = isTRUE(input$iv_both)),
        wgcna = if (isTRUE(input$use_wgcna)) list(trend = input$wg_trend %||% "rises", min_r = input$wg_min_r %||% 0.15,
                                                  max_p = 0.05, drug_target = isTRUE(input$wg_drug)),
        kg = if (isTRUE(input$use_kg)) list(min_pct = input$kg_min_pct %||% 80),
        ppi = if (isTRUE(input$use_ppi)) list(min_pct = input$ppi_min_pct %||% 50, key_only = isTRUE(input$ppi_key))
      )
    }) |> shiny::debounce(300)

    result <- shiny::reactive(run_screener(master, d, crit(), max(10, input$max_n %||% 250)))

    output$count <- shiny::renderUI({
      r <- result()
      layers <- sum(!vapply(crit()[c("sc", "bulk", "invitro", "wgcna", "kg", "ppi")], is.null, TRUE))
      shiny::div(class = "d-flex align-items-baseline gap-2",
        shiny::span(class = "screener-count", format(r$n_total, big.mark = ",")),
        shiny::span(class = "text-muted", sprintf("genes pass %d layer%s%s", layers, if (layers != 1) "s" else "",
                                                  if (r$n_total > nrow(r$table)) sprintf(", showing the strongest %d", nrow(r$table)) else "")))
    })

    output$table <- render_table({
      x <- result()$table
      shiny::validate(shiny::need(nrow(x) > 0, "No gene passes all the selected criteria. Try relaxing a threshold or switching a layer off."))
      link <- sprintf('<a class="gene-link" onclick="Shiny.setInputValue(\'gene-pick\', \'%s\', {priority: \'event\'}); mashGo(\'gene\');">%s</a>',
                      x$gene, x$gene)
      sc_dir <- ifelse(is.na(x$sc_dir), NA, ifelse(x$sc_dir > 0, "up", ifelse(x$sc_dir < 0, "down", "none")))
      bulk_dir <- ifelse(is.na(x$bulk_log2FC) | is.na(x$bulk_fdr) | x$bulk_fdr >= 0.05, ifelse(is.na(x$bulk_log2FC), NA, "none"),
                         ifelse(x$bulk_log2FC > 0, "up", "down"))
      iv_dir <- ifelse(is.na(x$iv_call), NA, ifelse(grepl("^up", x$iv_call), "up", ifelse(grepl("^down", x$iv_call), "down", "none")))
      out <- data.frame(Gene = link, Evidence = x$evidence, SC = sc_dir, Bulk = bulk_dir, BulkFC = x$bulk_log2FC,
                        BulkFDR = x$bulk_fdr, iHeps = iv_dir, KG = x$kg_pct / 100,
                        Module = ifelse(is.na(x$module) | x$module == "grey", "", sprintf("%s|%s", x$module, ifelse(is.na(x$module_r), "", sprintf("%+.2f", x$module_r)))),
                        Drugs = x$n_drugs, Target = x$target, stringsAsFactors = FALSE)
      rt(out, page = 15, searchable = TRUE, columns = list(
        Gene = col_def(html = TRUE, minWidth = 115),
        Evidence = col_bar("Single-cell evidence", digits = 2, width = 150),
        SC = col_call("Hepatocytes"), Bulk = col_call("Whole liver"),
        BulkFC = col_num("Liver log2FC"), BulkFDR = col_p("Liver FDR"), iHeps = col_call("iHeps"),
        KG = col_bar("KG centrality", digits = 2, color = PAL$down, width = 130),
        Module = col_def(name = "Module (r)", minWidth = 120, html = TRUE, cell = function(v) {
          if (!nzchar(v)) return("")
          p <- strsplit(v, "|", fixed = TRUE)[[1]]
          sprintf('<span style="display:inline-block;width:.7rem;height:.7rem;border-radius:3px;background:%s;margin-right:.35rem"></span>%s <span class="cell-id">%s</span>',
                  module_hex(p[1]), esc(p[1]), if (length(p) > 1) p[2] else "")
        }),
        Drugs = col_def(align = "center", maxWidth = 70), Target = col_check("Target")))
    })

    output$download <- shiny::downloadHandler(
      filename = function() sprintf("mash_screener_%s.csv", format(Sys.Date())),
      content = function(file) utils::write.csv(result()$table, file, row.names = FALSE))

    list(result = result, crit = crit)
  })
}
