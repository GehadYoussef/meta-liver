# Markers: strongest NASH-vs-control hepatocyte genes per dataset.

markersUI <- function(id, d) {
  ns <- shiny::NS(id)
  meta <- d$datasets_meta[order(d$datasets_meta$species != "human", -d$datasets_meta$bulk_agreement), ]
  ds_choices <- stats::setNames(meta$dataset, meta$label)
  shiny::tagList(
    page_header("ranking-star", "Markers",
                "The hepatocyte genes that best separate NASH from control in each dataset"),
    shiny::div(class = "toolbar",
      pills(ns("dataset"), ds_choices, selected = meta$dataset[meta$label == "Wang"][1]),
      shiny::span(class = "vr mx-1"),
      pills(ns("dir"), c("Both" = "both", "▲ Up" = "up_in_disease", "▼ Down" = "down_in_disease")),
      shiny::textInput(ns("search"), NULL, placeholder = "Find gene…", width = "160px"),
      filter_popover(
        shiny::sliderInput(ns("min_pct"), "Minimum detection, % of cells (either group)", 0.1, 0.9, 0.1, step = 0.05),
        shiny::checkboxInput(ns("fdr"), "Pseudobulk FDR < 0.05 only", FALSE),
        shiny::checkboxInput(ns("targets"), "Target genes only", FALSE),
        shiny::numericInput(ns("top_n"), "Show top N", 100, min = 10, step = 10),
        title = "Marker filters")
    ),
    shiny::uiOutput(ns("note")),
    bslib::layout_columns(
      col_widths = c(5, 7),
      bslib::navset_card_underline(
        title = card_title("Strongest genes", "chart-simple",
                           "Bars start at AUC 0.5 (no difference): right = higher in NASH, left = lower."),
        bslib::nav_panel("Top 25", plot_output(ns("bars"), height = "560px")),
        bslib::nav_panel("All genes", plot_output(ns("plot"), height = "560px"))),
      bslib::card(bslib::card_header(card_title("Top genes", "ranking-star",
                    "Ranked by distance of the AUC from 0.5, so strong up- and down-regulated genes rank equally.")),
                  reactable::reactableOutput(ns("table")))
    )
  )
}

markersServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    target_human <- unique(c(d$target_genes_human,
                             d$sc$gene_human[d$sc$species == "mouse" & d$sc$gene %in% d$target_genes_mouse]))
    ds_rows <- shiny::reactive(d$sc[d$sc$dataset == shiny::req(input$dataset), ])
    has_pb <- shiny::reactive(any(!is.na(ds_rows()$pb_fdr)))

    output$note <- shiny::renderUI({
      m <- d$datasets_meta[d$datasets_meta$dataset == input$dataset, ]
      shiny::div(class = "mb-2",
        chip(sprintf("%s NASH vs %s control", m$n_disease, m$n_control), "users"),
        chip(sprintf("%s hepatocytes", format(m$n_hepatocytes, big.mark = ",")), "microscope", "muted"),
        if (!has_pb()) chip("No sample-level p-values", "circle-exclamation", "muted"),
        if (!m$directions_reliable) chip(sprintf("Directions unreliable: %.0f%% of genes move one way", m$pct_up),
                                         "triangle-exclamation", "warn"))
    })

    filtered <- shiny::reactive({
      x <- ds_rows()
      x <- x[x$tested & pmax(x$pct_disease, x$pct_control) >= (input$min_pct %||% 0.1), ]
      if (!identical(input$dir, "both")) x <- x[x$direction == input$dir, ]
      if (isTRUE(input$fdr) && has_pb()) x <- x[!is.na(x$pb_fdr) & x$pb_fdr < 0.05, ]
      if (isTRUE(input$targets)) x <- x[x$gene_human %in% target_human, ]
      s <- tolower(trimws(input$search %||% ""))
      if (nzchar(s)) x <- x[grepl(s, tolower(x$gene), fixed = TRUE), ]
      x[order(-x$auc_power, x$gene), ]
    })
    top <- shiny::reactive(utils::head(filtered(), max(1, input$top_n %||% 100)))

    output$bars <- render_plot({
      x <- utils::head(filtered(), 25)
      shiny::validate(shiny::need(nrow(x) > 0, "No genes match the filters."))
      x <- x[order(x$auc), ]
      x$gene <- factor(x$gene, levels = x$gene)
      col <- ifelse(x$auc >= 0.5, PAL$up, PAL$down)
      x$hover <- sprintf("<b>%s</b><br>AUC %.2f<br>detected %.0f%% NASH / %.0f%% control<br>FDR %s",
                         x$gene, x$auc, 100 * x$pct_disease, 100 * x$pct_control, fmt_p(x$pb_fdr))
      lim <- max(0.25, max(abs(x$auc - 0.5)) + 0.05)
      plot_ly(x, y = ~gene, x = ~(auc - 0.5), base = 0.5, type = "bar", orientation = "h",
                      marker = list(color = col, line = list(width = 0)), text = ~hover, hoverinfo = "text",
                      textposition = "none") |>
        plot_style(xaxis = list(title = "AUC (NASH vs control)", range = c(0.5 - lim, 0.5 + lim), zeroline = FALSE),
                   yaxis = list(title = "", tickfont = list(size = 12, color = PAL$ink)),
                   shapes = list(vline(0.5)), bargap = 0.35, showlegend = FALSE)
    })

    output$plot <- render_plot({
      x <- filtered()
      shiny::validate(shiny::need(nrow(x) > 0, "No genes match the filters."))
      in_top <- x$gene %in% top()$gene
      col <- ifelse(!in_top, "#B8C0CC", ifelse(x$auc > 0.5, PAL$up, PAL$down))
      x$hover <- sprintf("<b>%s</b><br>AUC %.2f<br>detected %.0f%% NASH / %.0f%% control<br>FDR %s",
                         x$gene, x$auc, 100 * x$pct_disease, 100 * x$pct_control, fmt_p(x$pb_fdr))
      plot_ly(x, x = ~auc, y = ~(pct_disease - pct_control), type = "scattergl", mode = "markers",
                      marker = list(size = ifelse(in_top, 8, 5), color = col, opacity = ifelse(in_top, 0.95, 0.45), line = list(width = 0)),
                      text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "AUC (NASH vs control)", range = c(0, 1)),
                   yaxis = list(title = "Detection difference (NASH − control)"),
                   shapes = list(vline(0.5), hline(0)), showlegend = FALSE)
    })

    output$table <- reactable::renderReactable({
      x <- top()
      shiny::req(nrow(x) > 0)
      out <- data.frame(Gene = x$gene, AUC = x$auc, Strength = x$auc_power,
                        Detected = sprintf("%f|%f", x$pct_disease, x$pct_control),
                        log2FC = x$pb_logFC, FDR = x$pb_fdr, Target = x$gene_human %in% target_human)
      cols <- list(
        Gene = reactable::colDef(minWidth = 95, style = list(fontWeight = 600)),
        AUC = col_effect(), Strength = col_bar("Strength", digits = 2, width = 120),
        Detected = col_detect(), log2FC = col_num("log2FC"), FDR = col_p("FDR"),
        Target = col_check("Target"))
      if (!has_pb()) {
        out$log2FC <- NULL
        out$FDR <- NULL
        cols$log2FC <- NULL
        cols$FDR <- NULL
      }
      rt(out, columns = cols, page = 12, searchable = FALSE)
    })

    list(filtered = filtered, top = top)
  })
}
