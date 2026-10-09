# Direction consistency: which genes move the same way across datasets and bulk.

consistencyUI <- function(id, d) {
  ns <- shiny::NS(id)
  k <- length(d$consistency$datasets)
  shiny::tagList(
    page_header("arrows-up-down", "Direction consistency",
                "Which genes move the same way across datasets, species and the bulk cohort"),
    shiny::div(class = "toolbar",
      pills(ns("set"), c("All genes" = "all", "Target genes" = "target")),
      shiny::span(class = "vr mx-1"),
      pills(ns("cat"), c("Consistent" = "consistent", "▲ Up" = "up", "▼ Down" = "down",
                         "Conflicting" = "conflicting", "Any" = "all")),
      filter_popover(
        shiny::sliderInput(ns("min"), "Called up/down in at least N datasets", 2, k, 2, step = 1),
        shiny::checkboxInput(ns("species"), "Mouse and human agree", FALSE),
        shiny::checkboxInput(ns("bulk"), "Bulk agrees (padj < 0.05)", FALSE),
        title = "Consistency filters")
    ),
    bslib::layout_columns(
      col_widths = c(5, 7),
      bslib::card(
        bslib::card_header(card_title("Does each dataset agree with bulk?", "bullseye",
          "Share of genes significant in the independent bulk cohort (NASH F2–F4 vs control) that go the same way in hepatocytes. Dashed line = chance.")),
        plot_output(ns("bulk_bars"), height = "260px")),
      bslib::card(
        bslib::card_header(shiny::div(class = "d-flex justify-content-between align-items-center w-100",
          card_title("Pairwise agreement", "code-compare",
                     "Share of genes called in both datasets that go the same way. 50% = chance."),
          pills(ns("version"), c("Fixed" = "v2", "Original" = "legacy")))),
        plot_output(ns("heatmap"), height = "260px"))
    ),
    bslib::card(
      bslib::card_header(shiny::div(class = "d-flex justify-content-between align-items-center w-100",
        card_title("Direction matrix", "table-cells",
                   sprintf("Up means AUC ≥ %.2f and down AUC ≤ %.2f in hepatocytes. Bulk calls use padj < 0.05. Grey is no call, blank is not tested.",
                           0.5 + d$consistency$delta, 0.5 - d$consistency$delta)),
        shiny::div(class = "dir-legend",
                   chip("Up in NASH", "arrow-up", "up"), chip("Down in NASH", "arrow-down", "down"),
                   chip("No call", "minus", "muted")))),
      shiny::uiOutput(ns("count")),
      plot_output(ns("matrix"), height = "300px")
    ),
    bslib::accordion(open = FALSE,
      bslib::accordion_panel("Gene table", icon = shiny::icon("table"),
                             reactable::reactableOutput(ns("table"))),
      bslib::accordion_panel("Fixed pipeline vs original results", icon = shiny::icon("clock-rotate-left"),
                             reactable::reactableOutput(ns("compare"))))
  )
}

consistencyServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    meta <- d$datasets_meta
    lab <- stats::setNames(meta$label, meta$dataset)
    target_human <- unique(c(d$target_genes_human,
                             d$sc$gene_human[d$sc$species == "mouse" & d$sc$gene %in% d$target_genes_mouse]))
    MAX_ROWS <- 60

    genes <- shiny::reactive({
      w <- d$consistency$genes
      if (identical(input$set, "target")) w <- w[w$gene_human %in% target_human, ]
      w <- w[(w$n_up + w$n_down) >= (input$min %||% 2), ]
      w <- switch(input$cat %||% "consistent",
        consistent  = w[w$n_up == 0 | w$n_down == 0, ],
        up          = w[w$n_down == 0 & w$n_up > 0, ],
        down        = w[w$n_up == 0 & w$n_down > 0, ],
        conflicting = w[w$n_up > 0 & w$n_down > 0, ],
        w)
      if (isTRUE(input$species)) {
        w <- w[w$mouse %in% c("up", "down") & w$human %in% c("up", "down") & w$mouse == w$human, ]
      }
      if (isTRUE(input$bulk) && "bulk" %in% names(w)) {
        dir <- ifelse(w$n_up > w$n_down, "up", ifelse(w$n_down > w$n_up, "down", NA))
        w <- w[!is.na(dir) & !is.na(w$bulk) & w$bulk == dir, ]
      }
      w
    })

    output$bulk_bars <- render_plot({
      m <- meta[order(meta$bulk_agreement), ]
      col <- ifelse(!m$directions_reliable, PAL$na, ifelse(m$bulk_agreement >= 0.5, PAL$primary, PAL$warn))
      txt <- sprintf("%.0f%%%s", 100 * m$bulk_agreement, ifelse(m$directions_reliable, "", "  (unreliable)"))
      plot_ly(m, x = ~bulk_agreement, y = ~factor(label, levels = label), type = "bar", orientation = "h",
                      marker = list(color = col), text = txt, textposition = "outside", hoverinfo = "text",
                      hovertext = sprintf("<b>%s</b> (%s)<br>%.0f%% of bulk genes agree<br>pseudobulk r = %s",
                                          m$label, m$species, 100 * m$bulk_agreement, fmt_num(m$bulk_cor))) |>
        plot_style(xaxis = list(range = c(0, 1), tickformat = ".0%", title = ""), yaxis = list(title = ""),
                   shapes = list(vline(0.5)), showlegend = FALSE)
    })

    output$heatmap <- render_plot({
      p <- if (identical(input$version, "legacy")) d$consistency_legacy_pairwise else d$consistency$pairwise
      m <- meta[order(meta$species != "human", -meta$bulk_agreement), ]
      agreement_heatmap(p, m$dataset, m$label)
    })

    output$count <- shiny::renderUI({
      n <- nrow(genes())
      shiny::div(class = "px-3 pt-2",
        chip(sprintf("%d genes", n), "dna", "muted"),
        if (n > MAX_ROWS) chip(sprintf("showing the strongest %d", MAX_ROWS), "filter", "muted"))
    })

    output$matrix <- render_plot({
      w <- utils::head(genes(), MAX_ROWS)
      shiny::validate(shiny::need(nrow(w) > 0, "No genes match the filters."))
      cols <- c(d$consistency$datasets, if ("bulk" %in% names(w)) "bulk")
      xl <- c(unname(lab[d$consistency$datasets]), if ("bulk" %in% names(w)) "Bulk")
      code <- function(v) ifelse(is.na(v), NA, ifelse(v == "up", 1, ifelse(v == "down", -1, 0)))
      z <- sapply(cols, function(cc) code(w[[cc]]))
      if (is.null(dim(z))) z <- matrix(z, nrow = 1)
      # Datasets as rows and genes as columns
      zt <- t(z)
      hover <- t(sapply(seq_along(cols), function(j) sprintf("<b>%s</b>, %s<br>%s", w$gene_human, xl[j],
                                                              ifelse(is.na(w[[cols[j]]]), "not tested", w[[cols[j]]]))))
      if (nrow(w) == 1) hover <- matrix(hover, ncol = 1)
      plot_ly(x = w$gene_human, y = xl, z = zt, type = "heatmap", zmin = -1, zmax = 1, showscale = FALSE,
                      colorscale = list(c(0, PAL$down), c(0.5, "#E4E7EB"), c(1, PAL$up)),
                      text = hover, hoverinfo = "text", xgap = 2, ygap = 4, height = 300) |>
        plot_style(xaxis = list(showgrid = FALSE, fixedrange = TRUE, tickangle = -90, tickfont = list(size = 10),
                                automargin = TRUE),
                   yaxis = list(autorange = "reversed", showgrid = FALSE, fixedrange = TRUE, automargin = TRUE,
                                tickfont = list(size = 12)))
    })

    output$table <- reactable::renderReactable({
      w <- genes()
      shiny::req(nrow(w) > 0)
      cols <- c("gene_human", d$consistency$datasets, intersect("bulk", names(w)), "consistency")
      out <- w[, cols]
      defs <- c(list(gene_human = reactable::colDef(name = "Gene", style = list(fontWeight = 600)),
                     consistency = reactable::colDef(name = "Pattern", minWidth = 170)),
                stats::setNames(lapply(d$consistency$datasets, function(x) col_call(lab[[x]])), d$consistency$datasets),
                if ("bulk" %in% names(w)) list(bulk = col_call("Bulk")))
      rt(out, columns = defs, page = 15)
    })

    output$compare <- reactable::renderReactable({
      x <- d$consistency_compare
      rt(x, searchable = FALSE, columns = list(
        version = reactable::colDef(name = "Version"), gene_set = reactable::colDef(name = "Genes"),
        called_in_2plus = reactable::colDef(name = "Called in 2+"),
        consistent = reactable::colDef(name = "Consistent"), conflicting = reactable::colDef(name = "Conflicting"),
        consistent_in_all = reactable::colDef(name = "In all"),
        mouse_and_human_agree = reactable::colDef(name = "Mouse = human"),
        mouse_and_human_disagree = reactable::colDef(name = "Mouse and human disagree")))
    })

    list(genes = genes)
  })
}
