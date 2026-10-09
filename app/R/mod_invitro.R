# In-vitro model: iHeps (lines 1b and 5a) after each added exposure, vs untreated.

invitroUI <- function(id, d) {
  ns <- shiny::NS(id)
  shiny::tagList(
    page_header("vial", "In-vitro MASLD model",
                "Stem-cell-derived hepatocytes exposed to fatty acids, adipokines and immune cells"),
    bslib::layout_columns(
      col_widths = c(5, 7),
      bslib::card(bslib::card_header(card_title("How many genes respond at each step?", "stairs",
                    "Genes with padj < 0.05 and at least a 2-fold change, per cell line. Each step adds a stimulus to the previous one.")),
                  plot_output(ns("steps"), height = "300px")),
      bslib::card(bslib::card_header(card_title("Do the two cell lines agree?", "code-compare",
                    "Genes significant in both lines in the same direction (replicated), in one line only, or in opposite directions.")),
                  plot_output(ns("agree"), height = "300px"))
    ),
    shiny::div(class = "toolbar",
      shiny::span(class = "toolbar-label", "Exposure"),
      pills(ns("contrast"), stats::setNames(levels(d$invitro$contrast), c("OA+PA", "+ resistin/myostatin", "+ PBMC")),
            selected = SUMMARY_INVITRO_CONTRAST),
      shiny::span(class = "vr mx-1"),
      shiny::span(class = "toolbar-label", "Line"),
      pills(ns("line"), c("1b" = "1b", "5a" = "5a")),
      filter_popover(shiny::sliderInput(ns("lfc"), "Highlight genes with absolute log2FC of at least", 0, 3, 1, step = 0.25),
                     title = "Volcano settings")
    ),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(bslib::card_header(shiny::uiOutput(ns("title"))), plot_output(ns("volcano"), height = "440px")),
      bslib::card(bslib::card_header(card_title("Replicated in both lines", "list-ol",
                    "Genes significant (padj < 0.05) in both lines in the same direction, ranked by the smaller fold change.")),
                  reactable::reactableOutput(ns("table")))
    )
  )
}

invitroServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    iv <- d$invitro
    short <- c("OA+PA" = "OA+PA", "OA+PA + resistin/myostatin" = "+ resistin/myostatin",
               "OA+PA + resistin/myostatin + PBMC" = "+ PBMC")

    output$steps <- render_plot({
      x <- iv[iv$significant & abs(iv$log2FC) >= 1, ]
      cnt <- as.data.frame(table(contrast = x$contrast, line = x$line, dir = ifelse(x$log2FC > 0, "up", "down")))
      cnt$step <- factor(short[as.character(cnt$contrast)], levels = short)
      cnt$y <- ifelse(cnt$dir == "up", cnt$Freq, -cnt$Freq)
      cnt$grp <- paste(cnt$line, cnt$dir)
      plot_ly(cnt, x = ~step, y = ~y, color = ~grp, type = "bar",
                      colors = c("1b down" = "#2F80C3", "5a down" = "#8DB9E0", "1b up" = "#E05263", "5a up" = "#F0A3AD"),
                      text = ~sprintf("line %s, %d genes %s", line, Freq, dir), hoverinfo = "text", textposition = "none") |>
        plot_style(barmode = "relative", xaxis = list(title = ""), yaxis = list(title = "genes (down below 0, up above)"),
                   legend = list(orientation = "h", y = -0.15))
    })

    output$agree <- render_plot({
      a <- d$invitro_agreement
      a <- a[a$call != "not significant", ]
      cats <- c("up (both lines)", "up (one line)", "lines disagree", "down (one line)", "down (both lines)")
      cnt <- as.data.frame(table(step = factor(short[a$contrast], levels = short), call = factor(a$call, levels = cats)))
      plot_ly(cnt, x = ~step, y = ~Freq, color = ~call, type = "bar",
                      colors = c("up (both lines)" = "#E05263", "up (one line)" = "#F4B6BE", "lines disagree" = "#E5A13A",
                                 "down (one line)" = "#A9CBEA", "down (both lines)" = "#2F80C3"),
                      text = ~sprintf("%s: %d genes", call, Freq), hoverinfo = "text", textposition = "none") |>
        plot_style(barmode = "stack", xaxis = list(title = ""), yaxis = list(title = "significant genes"),
                   legend = list(orientation = "h", y = -0.15))
    })

    sel <- shiny::reactive(iv[iv$contrast == shiny::req(input$contrast) & iv$line == shiny::req(input$line) & !is.na(iv$padj), ])

    output$title <- shiny::renderUI(card_title(sprintf("Line %s, %s vs untreated", input$line, input$contrast), "chart-simple"))

    output$volcano <- render_plot({
      x <- sel()
      hi <- x$significant & abs(x$log2FC) >= (input$lfc %||% 1)
      col <- ifelse(!hi, "#B8C0CC", ifelse(x$log2FC > 0, PAL$up, PAL$down))
      x$hover <- sprintf("<b>%s</b><br>log2FC %+.2f<br>padj %s", ifelse(is.na(x$gene_human), x$ensembl, x$gene_human),
                         x$log2FC, fmt_p(x$padj))
      plot_ly(x, x = ~log2FC, y = ~-log10(pmax(padj, 1e-300)), type = "scattergl", mode = "markers",
                      marker = list(size = ifelse(hi, 6, 4), color = col, opacity = ifelse(hi, 0.9, 0.45), line = list(width = 0)), text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "log2 fold change"), yaxis = list(title = "−log10 adjusted p"),
                   shapes = list(hline(-log10(0.05))), showlegend = FALSE)
    })

    output$table <- reactable::renderReactable({
      cc <- shiny::req(input$contrast)
      x <- iv[iv$contrast == cc & !is.na(iv$gene_human), ]
      w <- stats::reshape(x[, c("gene_human", "line", "log2FC", "padj", "significant")], idvar = "gene_human",
                          timevar = "line", direction = "wide")
      both <- w[isTRUE_vec(w$significant.1b) & isTRUE_vec(w$significant.5a) & sign(w$log2FC.1b) == sign(w$log2FC.5a), ]
      both$min_fc <- pmin(abs(both$log2FC.1b), abs(both$log2FC.5a)) * sign(both$log2FC.1b)
      both <- both[order(-abs(both$min_fc)), ]
      shiny::validate(shiny::need(nrow(both) > 0, "No gene is significant in both lines for this exposure."))
      gene_link <- sprintf('<a class="gene-link" onclick="Shiny.setInputValue(\'gene-pick\', \'%s\', {priority: \'event\'}); mashGo(\'gene\');">%s</a>',
                           both$gene_human, both$gene_human)
      rt(data.frame(Gene = gene_link, Direction = ifelse(both$min_fc > 0, "up_in_disease", "down_in_disease"),
                    FC1b = both$log2FC.1b, FC5a = both$log2FC.5a, P = pmax(both$padj.1b, both$padj.5a)),
         page = 12, columns = list(
           Gene = reactable::colDef(html = TRUE, minWidth = 115),
           Direction = reactable::colDef(html = TRUE, minWidth = 135, cell = function(v) pill_html(v, if (v == "up_in_disease") "Up in model" else "Down in model")),
           FC1b = utils::modifyList(col_num("log2FC 1b"), list(minWidth = 80)),
           FC5a = utils::modifyList(col_num("log2FC 5a"), list(minWidth = 80)),
           P = utils::modifyList(col_p("Larger padj"), list(minWidth = 85))))
    })

    list(sel = sel)
  })
}

isTRUE_vec <- function(x) !is.na(x) & x
