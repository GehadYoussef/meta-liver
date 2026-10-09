# Co-expression modules (WGCNA on the bulk cohort): click a bar to open a module.

modulesUI <- function(id, d) {
  ns <- shiny::NS(id)
  shiny::tagList(
    page_header("diagram-project", "Co-expression modules",
                "Gene modules in the bulk cohort and how they track disease stage. Click a bar to see its pathways and genes."),
    bslib::layout_columns(
      col_widths = c(5, 7),
      bslib::card(bslib::card_header(card_title("Modules vs disease stage", "diagram-project",
                    "Pearson correlation of the module eigengene with stage. Bars use the module colour.")),
                  plotly::plotlyOutput(ns("bars"), height = "520px")),
      bslib::card(
        bslib::card_header(shiny::uiOutput(ns("title"))),
        bslib::navset_underline(
          bslib::nav_panel(shiny::span(shiny::icon("route"), " Pathways"), reactable::reactableOutput(ns("enrich"))),
          bslib::nav_panel(shiny::span(shiny::icon("dna"), " Genes"), reactable::reactableOutput(ns("genes")))))
    )
  )
}

modulesServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    tr <- d$wgcna$trait
    module <- shiny::reactiveVal(tr$module[which.max(abs(tr$cor_stage))])
    click <- shiny::reactive(suppressWarnings(plotly::event_data("plotly_click", source = session$ns("bars"))))
    shiny::observeEvent(click(), {
      ev <- click()
      if (!is.null(ev$y) && ev$y %in% tr$module) module(ev$y)
    })
    target_human <- unique(c(d$target_genes_human, toupper(d$target_genes_mouse)))

    output$bars <- plotly::renderPlotly({
      t <- tr[order(tr$cor_stage), ]
      t$hover <- sprintf("<b>%s</b>, %d genes<br>r = %+.2f, p = %s", t$module, t$n_genes, t$cor_stage, fmt_p(t$p_stage))
      sel <- t$module == module()
      plotly::plot_ly(t, x = ~cor_stage, y = ~factor(module, levels = module), type = "bar", orientation = "h",
                      source = session$ns("bars"),
                      marker = list(color = module_hex(t$module),
                                    line = list(color = ifelse(sel, PAL$ink, "rgba(0,0,0,0.25)"), width = ifelse(sel, 2.5, 0.5))),
                      text = ~hover, hoverinfo = "text", textposition = "none") |>
        plot_style(xaxis = list(title = "Correlation with stage", zeroline = TRUE), yaxis = list(title = ""),
                   showlegend = FALSE) |>
        plotly::event_register("plotly_click")
    })

    output$title <- shiny::renderUI({
      t <- tr[tr$module == module(), ]
      shiny::div(class = "d-flex align-items-center gap-2",
        shiny::span(style = sprintf("width:1rem;height:1rem;border-radius:4px;background:%s;display:inline-block", module_hex(module()))),
        shiny::span(class = "fw-bold", module()),
        chip(sprintf("r = %+.2f with stage", t$cor_stage), "chart-line", if (t$cor_stage > 0) "up" else "down"),
        chip(sprintf("%d genes", t$n_genes), "dna", "muted"))
    })

    output$enrich <- reactable::renderReactable({
      e <- d$wgcna$enrichment
      e <- e[e$module == module(), ]
      e <- e[order(e$p_value), ]
      shiny::validate(shiny::need(nrow(e) > 0, "No enriched pathways for this module."))
      rt(data.frame(Source = e$source, Term = e$term_name, p = -log10(e$p_value), Overlap = e$intersection_size),
         page = 12, columns = list(
           Source = reactable::colDef(maxWidth = 90, html = TRUE, cell = function(v) chip_html(v, "muted")),
           Term = reactable::colDef(minWidth = 260),
           p = col_bar("−log10 p", max = max(-log10(e$p_value)), color = PAL$primary, digits = 1),
           Overlap = reactable::colDef(align = "right", maxWidth = 90)))
    })

    output$genes <- reactable::renderReactable({
      m <- d$wgcna$membership
      m <- m[m$module == module(), ]
      rt(data.frame(Gene = ifelse(is.na(m$gene_human), m$ensembl, m$gene_human),
                    Target = !is.na(m$gene_human) & m$gene_human %in% target_human),
         page = 15, columns = list(Gene = reactable::colDef(style = list(fontWeight = 600)), Target = col_check("Target")))
    })
    list(module = module)
  })
}
