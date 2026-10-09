# Whole liver: stage volcanoes from GSE135251 and the cohort meta-analysis.

bulkUI <- function(id, d) {
  ns <- shiny::NS(id)
  ct <- levels(d$bulk$contrast)
  short <- sub(" vs control", "", ct)
  shiny::tagList(
    page_header("flask", "Whole liver (bulk RNA-seq)",
                "Disease stages and a three-cohort meta-analysis"),
    bslib::navset_pill(
      bslib::nav_panel(shiny::span(shiny::icon("stairs"), " Disease stages (GSE135251)"),
        shiny::div(class = "mt-3"),
      shiny::div(class = "toolbar",
        pills(ns("contrast"), stats::setNames(ct, short), selected = "NASH F3 vs control"),
        filter_popover(shiny::sliderInput(ns("lfc"), "Highlight genes with absolute log2FC of at least", 0, 3, 1, step = 0.25),
                       shiny::checkboxInput(ns("annotated"), "Only genes with a symbol in the table", TRUE),
                       title = "Volcano settings")
      ),
      bslib::layout_columns(
        col_widths = c(6, 6),
        bslib::card(bslib::card_header(shiny::uiOutput(ns("title"))),
                    plot_output(ns("volcano"), height = "440px")),
        bslib::card(bslib::card_header(shiny::uiOutput(ns("table_title"))),
                    table_output(ns("table")))
      )
      ),
      bslib::nav_panel(shiny::span(shiny::icon("layer-group"), " Cohort meta-analysis"),
        shiny::div(class = "toolbar mt-3",
          pills(ns("meta_contrast"), stats::setNames(BULK_COHORT_LEVELS, BULK_COHORT_LEVELS), selected = "MASH vs control"),
          filter_popover(
            shiny::sliderInput(ns("meta_min_studies"), "Min cohorts", 1, 3, 2, step = 1),
            shiny::checkboxInput(ns("meta_consistent"), "Same direction in every cohort", TRUE),
            title = "Meta-analysis filters")),
        bslib::layout_columns(
          col_widths = c(6, 6),
          bslib::card(bslib::card_header(card_title("Meta-analysis volcano", "chart-simple", "GSE126848 and GSE135251 (RNA-seq) plus GSE151158 (618-gene panel, MASLD vs control only). Random-effects meta-analysis of log2 fold changes.")),
                      plot_output(ns("meta_volcano"), height = "440px")),
          bslib::card(bslib::card_header(shiny::uiOutput(ns("meta_title"))), table_output(ns("meta_table")))
        )
      )
    )
  )
}

bulkServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    contrast <- shiny::reactive(shiny::req(input$contrast))
    rows <- shiny::reactive(d$bulk[d$bulk$contrast == contrast() & !is.na(d$bulk$padj), ])

    output$title <- shiny::renderUI(card_title(contrast(), "flask",
      "DESeq2 results. Stage contrasts compare each stage with stage 0, the other contrasts compare with controls."))

    output$volcano <- render_plot({
      b <- rows()
      hi <- b$padj < 0.05 & abs(b$log2FC) >= (input$lfc %||% 1)
      col <- ifelse(!hi, "#B8C0CC", ifelse(b$log2FC > 0, PAL$up, PAL$down))
      b$hover <- sprintf("<b>%s</b><br>log2FC %+.2f<br>padj %s", ifelse(is.na(b$gene_human), b$ensembl, b$gene_human),
                         b$log2FC, fmt_p(b$padj))
      plot_ly(b, x = ~log2FC, y = ~-log10(pmax(padj, 1e-300)), type = "scattergl", mode = "markers",
                      marker = list(size = ifelse(hi, 6, 4), color = col, opacity = ifelse(hi, 0.9, 0.45), line = list(width = 0)), text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "log2 fold change"), yaxis = list(title = "−log10 adjusted p"),
                   shapes = list(hline(-log10(0.05))), showlegend = FALSE)
    })

    sig <- shiny::reactive({
      b <- rows()
      b <- b[b$padj < 0.05, ]
      if (!isFALSE(input$annotated)) b <- b[!is.na(b$gene_human), ]
      b[order(b$padj), ]
    })
    MAX_TABLE <- 500   # cell renderers run per row, so long tables are slow

    output$table_title <- shiny::renderUI(shiny::div(class = "d-flex align-items-center gap-2",
      card_title("Significant genes (padj < 0.05)", "list-ol"),
      chip(sprintf("%s genes", format(nrow(sig()), big.mark = ",")), "dna", "muted"),
      if (nrow(sig()) > MAX_TABLE) chip(sprintf("showing the strongest %d", MAX_TABLE), "filter", "muted")))

    output$table <- render_table({
      b <- utils::head(sig(), MAX_TABLE)
      out <- data.frame(Gene = paste(ifelse(is.na(b$gene_human), "(no symbol)", b$gene_human), b$ensembl, sep = "|"),
                        Direction = ifelse(b$log2FC > 0, "up_in_disease", "down_in_disease"),
                        log2FC = b$log2FC, padj = b$padj, baseMean = b$baseMean)
      rt(out, page = 12, columns = list(
        Gene = col_name_id("Gene", 140), Direction = col_dir(),
        log2FC = col_num("log2FC"), padj = col_p("padj"), baseMean = col_num("Mean expression", 0)))
    })

    # Cohort meta-analysis
    meta_rows <- shiny::reactive({
      m <- d$bulk_cohorts$meta
      m <- m[m$contrast == shiny::req(input$meta_contrast) & m$n_studies >= (input$meta_min_studies %||% 2), ]
      if (!isFALSE(input$meta_consistent)) m <- m[m$agreement == 1, ]
      m
    })
    output$meta_volcano <- render_plot({
      m <- meta_rows()
      shiny::validate(shiny::need(nrow(m) > 0, "No genes pass the filters."))
      hi <- m$fdr < 0.05 & abs(m$meta_log2FC) >= 1
      col <- ifelse(!hi, "#B8C0CC", ifelse(m$meta_log2FC > 0, PAL$up, PAL$down))
      m$hover <- sprintf("<b>%s</b><br>meta log2FC %+.2f<br>FDR %s<br>%d cohorts, I\u00b2 %s", m$gene_human,
                         m$meta_log2FC, fmt_p(m$fdr), m$n_studies, ifelse(is.na(m$i2), "n/a", sprintf("%.0f%%", 100 * m$i2)))
      plot_ly(m, x = ~meta_log2FC, y = ~-log10(pmax(fdr, 1e-300)), type = "scattergl", mode = "markers",
                      marker = list(size = ifelse(hi, 6, 4), color = col, opacity = ifelse(hi, 0.9, 0.45), line = list(width = 0)), text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "meta-analysis log2 fold change"), yaxis = list(title = "\u2212log10 FDR"),
                   shapes = list(hline(-log10(0.05))), showlegend = FALSE)
    })
    output$meta_title <- shiny::renderUI({
      m <- meta_rows()
      shiny::div(class = "d-flex align-items-center gap-2",
        card_title("Strongest genes", "list-ol"),
        chip(sprintf("%s at FDR < 0.05", format(sum(m$fdr < 0.05), big.mark = ",")), "dna", "muted"))
    })
    output$meta_table <- render_table({
      m <- meta_rows()
      m <- m[m$fdr < 0.05, ]
      m <- utils::head(m[order(m$p), ], 500)
      shiny::validate(shiny::need(nrow(m) > 0, "No genes at FDR < 0.05."))
      link <- sprintf('<a class="gene-link" onclick="Shiny.setInputValue(\'gene-pick\', \'%s\', {priority: \'event\'}); mashGo(\'gene\');">%s</a>',
                      m$gene_human, m$gene_human)
      rt(data.frame(Gene = link, Direction = ifelse(m$meta_log2FC > 0, "up_in_disease", "down_in_disease"),
                    FC = m$meta_log2FC, FDR = m$fdr, N = m$n_studies, I2 = m$i2),
         page = 12, columns = list(
           Gene = col_def(html = TRUE, minWidth = 115), Direction = col_dir(), FC = col_num("meta log2FC"),
           FDR = col_p("FDR"), N = col_def(name = "Cohorts", align = "center", maxWidth = 80),
           I2 = col_bar("Heterogeneity (I\u00b2)", digits = 2, color = PAL$warn, width = 150)))
    })

    list(contrast = contrast, meta_rows = meta_rows)
  })
}
