# Gene lookup: summary, evidence tiles, charts and context for one gene.

QUICK_PICKS <- c("SREBF1", "PLIN2", "HMGCS1", "ANXA2", "TGFBR3", "DTNA", "AKR1B10", "SAA1")

geneUI <- function(id) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::div(class = "gene-band",
      shiny::div(class = "band-title", shiny::icon("magnifying-glass"), " Gene lookup across all evidence layers"),
      shiny::selectizeInput(ns("gene"), NULL, choices = NULL, width = "100%",
                            options = list(placeholder = "Search a gene (human symbol)…")),
      shiny::div(class = "quick-picks", lapply(QUICK_PICKS, function(g) shiny::tags$button(
        type = "button", class = "btn btn-sm", g,
        onclick = sprintf("Shiny.setInputValue('%s', '%s', {priority: 'event'})", ns("pick"), g))))),
    shiny::uiOutput(ns("header")),
    shiny::uiOutput(ns("summary")),
    shiny::uiOutput(ns("strip")),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(bslib::card_header(card_title("Hepatocytes: NASH vs control", "microscope",
                    "AUC = chance that a NASH hepatocyte expresses the gene more than a control hepatocyte. 0.5 = no difference.")),
                  plotly::plotlyOutput(ns("sc_plot"), height = "280px")),
      bslib::card(bslib::card_header(card_title("Whole liver across disease stages (GSE135251)", "chart-line",
                    "DESeq2 log2 fold change with 95% interval. Filled points have padj < 0.05.")),
                  plotly::plotlyOutput(ns("bulk_plot"), height = "280px"))
    ),
    bslib::navset_card_underline(
      title = card_title("Context", "layer-group"),
      bslib::nav_panel(shiny::span(shiny::icon("flask"), " Bulk cohorts"),
                       shiny::p(class = "small text-muted", "Each cohort's log2 fold change with 95% interval, and the random-effects meta-analysis (diamond, coloured when FDR < 0.05)."),
                       plotly::plotlyOutput(ns("forest"), height = "360px")),
      bslib::nav_panel(shiny::span(shiny::icon("vial"), " iHeps model"),
                       shiny::p(class = "small text-muted", "Stem-cell-derived hepatocytes (lines 1b and 5a) after each exposure, vs untreated. ● = padj < 0.05."),
                       plotly::plotlyOutput(ns("invitro"), height = "260px")),
      bslib::nav_panel(shiny::span(shiny::icon("circle-nodes"), " Interactors"), reactable::reactableOutput(ns("ppi"))),
      bslib::nav_panel(shiny::span(shiny::icon("share-nodes"), " Knowledge-graph cluster"), reactable::reactableOutput(ns("cluster"))),
      bslib::nav_panel(shiny::span(shiny::icon("capsules"), " Drugs"), reactable::reactableOutput(ns("drugs"))),
      bslib::nav_panel(shiny::span(shiny::icon("table"), " Single-cell table"), reactable::reactableOutput(ns("table")))
    )
  )
}

geneServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    meta <- d$datasets_meta[order(d$datasets_meta$species != "human", -d$datasets_meta$bulk_agreement), ]
    all_genes <- sort(unique(stats::na.omit(c(d$sc$gene_human, d$bulk$gene_human, d$wgcna$membership$gene_human,
                                              d$ppi$centrality$gene_human, d$kg$genes$name,
                                              d$bulk_cohorts$meta$gene_human, d$invitro$gene_human))))
    target_human <- unique(c(d$target_genes_human,
                             d$sc$gene_human[d$sc$species == "mouse" & d$sc$gene %in% d$target_genes_mouse]))
    delta <- d$consistency$delta

    shiny::updateSelectizeInput(session, "gene", choices = all_genes, server = TRUE,
                                selected = if ("SREBF1" %in% all_genes) "SREBF1" else all_genes[1])
    shiny::observeEvent(input$pick, {
      shiny::updateSelectizeInput(session, "gene", choices = all_genes, server = TRUE, selected = input$pick)
    })

    gene <- shiny::reactive(shiny::req(input$gene))
    gene_sc <- shiny::reactive(d$sc[d$sc$gene_human == gene(), ])
    gene_bulk <- shiny::reactive(d$bulk[!is.na(d$bulk$gene_human) & d$bulk$gene_human == gene(), ])
    ev <- shiny::reactive(gene_evidence(gene(), d))
    story <- shiny::reactive(gene_narrative(ev()))

    output$header <- shiny::renderUI({
      g <- gene()
      shiny::div(class = "gene-head",
        shiny::h3(class = "gene-name", g),
        if (nrow(ev()$evidence)) chip(sprintf("single-cell evidence: %s", ev()$evidence$tier), "microscope",
                                      if (ev()$evidence$evidence > 0) (if (ev()$evidence$direction > 0) "up" else "down") else "muted"),
        if (g %in% target_human) chip("Target gene", "bullseye"),
        if (isTRUE(ev()$key_protein)) chip("Early-MAFLD key protein", "circle-nodes"),
        if (nrow(ev()$drugs)) chip(sprintf("%d drug%s", nrow(ev()$drugs), if (nrow(ev()$drugs) > 1) "s" else ""), "capsules"))
    })

    # Summary card
    output$summary <- shiny::renderUI({
      st <- story()
      strength_chip <- function(s) chip(s, NULL, switch(s, high = "", moderate = "", low = "muted", "very low" = "muted", none = "muted", "muted"))
      dir_icon <- function(x) if (is.na(x)) shiny::span(class = "sum-dir none", shiny::icon("minus"))
                              else shiny::span(class = paste("sum-dir", if (x > 0) "up" else "down"), shiny::icon(if (x > 0) "arrow-up" else "arrow-down"))
      copy_id <- session$ns("copytext")
      bslib::card(class = "summary-card",
        bslib::card_body(
          shiny::div(class = "sum-headline", shiny::icon("quote-left"), shiny::span(st$headline)),
          shiny::div(class = paste("sum-agree", if (st$consistent) "ok" else ""),
                     shiny::icon(if (st$consistent) "circle-check" else "scale-unbalanced"), st$agreement),
          shiny::div(class = "sum-rows", lapply(st$rows, function(r) shiny::div(class = "sum-row",
            shiny::div(class = "sum-area", shiny::icon(r$icon), r$area),
            dir_icon(r$dir),
            shiny::div(class = "sum-text", r$text),
            strength_chip(r$strength)))),
          shiny::div(class = "d-flex justify-content-end mt-2",
            shiny::tags$textarea(id = copy_id, class = "visually-hidden", st$copy),
            shiny::tags$button(type = "button", class = "btn btn-gear",
              onclick = sprintf("navigator.clipboard.writeText(document.getElementById('%s').value); this.innerHTML='\\u2713 Copied';", copy_id),
              shiny::icon("copy"), " Copy summary"))
        ))
    })

    # Evidence tiles
    tile <- function(source, icon, call, value, sub, dim = FALSE, flag = NULL) {
      cls <- if (is.na(call)) "" else call
      arrow <- switch(cls, up = "arrow-up", down = "arrow-down", none = "minus", NULL)
      shiny::div(class = paste("ev-tile", cls, if (dim) "dim"),
        if (!is.null(flag)) bslib::tooltip(shiny::span(class = "ev-flag", shiny::icon("triangle-exclamation")), flag),
        shiny::div(class = "ev-source", shiny::icon(icon), source),
        shiny::div(class = "ev-main",
          if (!is.null(arrow)) shiny::span(class = "ev-badge", shiny::icon(arrow)),
          shiny::span(class = "ev-value", value)),
        shiny::div(class = "ev-sub", sub))
    }
    info_tile <- function(source, icon, value, sub, dim = FALSE, swatch = NULL) {
      shiny::div(class = paste("ev-tile info", if (dim) "dim"),
        shiny::div(class = "ev-source", shiny::icon(icon), source),
        shiny::div(class = "ev-main", swatch, shiny::span(class = "ev-value", value)),
        shiny::div(class = "ev-sub", sub))
    }

    output$strip <- shiny::renderUI({
      g <- gene()
      s <- gene_sc()
      e <- ev()
      sc_tiles <- lapply(seq_len(nrow(meta)), function(i) {
        m <- meta[i, ]
        r <- s[s$dataset == m$dataset, ]
        flag <- if (!m$directions_reliable) "Directions unreliable in this dataset (global skew)" else NULL
        icon <- if (m$species == "human") "user" else "paw"
        if (!nrow(r)) return(tile(m$label, icon, NA, "n/a", "not measured", dim = TRUE, flag = flag))
        r <- r[1, ]
        if (is.na(r$auc)) return(tile(m$label, icon, NA, "n/a",
                                      sprintf("detected in %.0f%% / %.0f%%", 100 * r$pct_disease, 100 * r$pct_control),
                                      dim = TRUE, flag = flag))
        call <- if (r$auc >= 0.5 + delta) "up" else if (r$auc <= 0.5 - delta) "down" else "none"
        sub <- if (!is.na(r$pb_fdr)) sprintf("FDR %s, %s", fmt_p(r$pb_fdr), m$species) else sprintf("no p-value, %s", m$species)
        tile(m$label, icon, call, sprintf("AUC %.2f", r$auc), sub, flag = flag)
      })

      b <- e$bulk[e$bulk$contrast == SUMMARY_BULK_CONTRAST, ]
      bulk_tile <- if (!nrow(b)) tile("Whole liver", "flask", NA, "n/a", "not in the bulk cohorts", dim = TRUE) else
        tile("Whole liver", "flask", if (b$fdr < 0.05) (if (b$meta_log2FC > 0) "up" else "down") else "none",
             sprintf("%+.2f", b$meta_log2FC), sprintf("MASH vs control, %d cohort%s, FDR %s", b$n_studies,
                                                       if (b$n_studies > 1) "s" else "", fmt_p(b$fdr)))

      iv <- e$invitro[e$invitro$contrast == SUMMARY_INVITRO_CONTRAST, ]
      iv_tile <- if (!nrow(iv)) tile("iHeps model", "vial", NA, "n/a", "not measured", dim = TRUE) else {
        call <- if (grepl("^up", iv$call)) "up" else if (grepl("^down", iv$call)) "down" else "none"
        tile("iHeps model", "vial", call, sub(" \\(.*", "", iv$call), sprintf("%s, full model", sub(".*\\((.*)\\)", "\\1", iv$call)))
      }

      wm <- e$wgcna_module
      wt <- e$wgcna_trait
      wgcna_tile <- if (!nrow(wm)) info_tile("Co-expression", "diagram-project", "n/a", "not in WGCNA", dim = TRUE) else {
        swatch <- shiny::span(style = sprintf("display:inline-block;width:.9rem;height:.9rem;border-radius:4px;background:%s",
                                              module_hex(wm$module[1])))
        info_tile("Co-expression", "diagram-project", wm$module[1],
                  if (nrow(wt)) sprintf("module r = %+.2f with stage", wt$cor_stage[1]) else "not assigned to a module",
                  dim = wm$module[1] == "grey", swatch = swatch)
      }
      ppi_tile <- if (!nrow(e$ppi)) info_tile("Interactors", "circle-nodes", "n/a", "not in the PPI network", dim = TRUE) else
        info_tile("Interactors", "circle-nodes", format(e$ppi$degree, big.mark = ","),
                  sprintf("more partners than %.0f%% of proteins%s", e$ppi$degree_pct, if (e$key_protein) ", key protein" else ""))
      kg_tile <- if (!nrow(e$kg)) info_tile("Knowledge graph", "share-nodes", "n/a", "not a node", dim = TRUE) else
        info_tile("Knowledge graph", "share-nodes", sprintf("top %.0f%%", max(1, 100 - e$kg$composite_pct)),
                  sprintf("composite centrality, cluster %s", e$kg$cluster))
      drug_tile <- if (!nrow(e$drugs)) info_tile("Drugs", "capsules", "n/a", "no network-active drug", dim = TRUE) else
        info_tile("Drugs", "capsules", nrow(e$drugs), paste(utils::head(e$drugs$drug, 2), collapse = ", "))
      hs <- if (is.null(d$hep_specificity)) NULL else d$hep_specificity[d$hep_specificity$gene_human == g, ]
      hs_tile <- if (is.null(hs) || !nrow(hs)) NULL else if (is.na(hs$auc[1])) {
        tile("Hepatocyte-specific?", "crosshairs", NA, "n/a", "mouse NASH: not detected", dim = TRUE)
      } else {
        call <- if (hs$auc[1] >= 0.55) "up" else if (hs$auc[1] <= 0.45) "down" else "none"
        tile("Hepatocyte-specific?", "crosshairs", call, sprintf("AUC %.2f", hs$auc[1]), "hepatocytes vs other cells (mouse)")
      }

      shiny::tagList(
        shiny::div(class = "ev-group-label", shiny::icon("microscope"), "Hepatocytes (single-cell)"),
        shiny::div(class = "evidence-strip", sc_tiles),
        shiny::div(class = "ev-group-label", shiny::icon("layer-group"), "Whole liver, iHeps model, networks and drugs"),
        shiny::div(class = "evidence-strip", bulk_tile, iv_tile, wgcna_tile, ppi_tile, kg_tile, drug_tile, hs_tile))
    })

    # Charts
    output$sc_plot <- plotly::renderPlotly({
      s <- gene_sc()
      shiny::validate(shiny::need(nrow(s) > 0, "Not measured in any single-cell dataset."))
      s$label <- d$datasets_meta$label[match(s$dataset, d$datasets_meta$dataset)]
      s$label <- factor(s$label, levels = rev(meta$label))
      s$x <- ifelse(is.na(s$auc), 0.5, s$auc)
      s$col <- ifelse(is.na(s$auc), PAL$na, ifelse(s$auc >= 0.5 + delta, PAL$up, ifelse(s$auc <= 0.5 - delta, PAL$down, PAL$none)))
      s$hover <- sprintf("<b>%s</b><br>AUC %s<br>detected %.0f%% NASH / %.0f%% control<br>FDR %s",
                         s$label, ifelse(is.na(s$auc), "not tested", sprintf("%.2f", s$auc)),
                         100 * s$pct_disease, 100 * s$pct_control, fmt_p(s$pb_fdr))
      shapes <- c(list(vline(0.5)), lapply(seq_len(nrow(s)), function(i) list(
        type = "line", x0 = 0.5, x1 = s$x[i], y0 = as.character(s$label[i]), y1 = as.character(s$label[i]), yref = "y",
        line = list(color = s$col[i], width = 3))))
      plotly::plot_ly(s, x = ~x, y = ~label, type = "scatter", mode = "markers",
                      marker = list(size = 16, color = s$col, line = list(color = "#fff", width = 2)),
                      text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(range = 0.5 + c(-1, 1) * max(0.15, max(abs(s$x - 0.5)) + 0.05),
                                title = "AUC (below 0.5 lower in NASH, above 0.5 higher)", automargin = TRUE),
                   yaxis = list(title = "", automargin = TRUE), shapes = shapes, showlegend = FALSE)
    })

    output$bulk_plot <- plotly::renderPlotly({
      b <- gene_bulk()
      shiny::validate(shiny::need(nrow(b) > 0, "Not measured in the bulk dataset."))
      b <- b[order(b$contrast), ]
      b$sig <- !is.na(b$padj) & b$padj < 0.05
      b$short <- sub(" vs control", "", sub("Stage ", "S", sub(" vs 0", " vs S0", b$contrast)))
      b$short <- factor(b$short, levels = unique(b$short))
      b$hover <- sprintf("<b>%s</b><br>log2FC %+.2f<br>padj %s", b$contrast, b$log2FC, fmt_p(b$padj))
      cols <- ifelse(b$log2FC > 0, PAL$up, PAL$down)
      plotly::plot_ly(b, x = ~short, y = ~log2FC, type = "scatter", mode = "markers",
                      error_y = list(type = "data", array = 1.96 * b$lfcSE, color = "rgba(0,0,0,0.25)", thickness = 1.5, width = 0),
                      marker = list(size = 12, color = ifelse(b$sig, cols, "#FFFFFF"), line = list(color = cols, width = 2)),
                      text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "", tickangle = -30, automargin = TRUE),
                   yaxis = list(title = "log2 fold change", automargin = TRUE), shapes = list(hline(0)), showlegend = FALSE)
    })

    # Forest plot: each cohort plus the meta-analysis, per contrast
    output$forest <- plotly::renderPlotly({
      g <- gene()
      st <- d$bulk_cohorts$studies[d$bulk_cohorts$studies$gene_human == g, ]
      mt <- d$bulk_cohorts$meta[d$bulk_cohorts$meta$gene_human == g, ]
      shiny::validate(shiny::need(nrow(st) > 0, "Not measured in the bulk cohorts."))
      rows <- rbind(
        data.frame(contrast = st$contrast, label = st$study, lfc = st$log2FC, se = st$se, kind = "study",
                   p = st$pvalue, stringsAsFactors = FALSE),
        data.frame(contrast = mt$contrast, label = "Meta-analysis", lfc = mt$meta_log2FC, se = mt$meta_se, kind = "meta",
                   p = mt$fdr, stringsAsFactors = FALSE))
      rows$contrast <- factor(rows$contrast, levels = BULK_COHORT_LEVELS)
      rows <- rows[order(rows$contrast, rows$kind != "study", rows$label), ]
      rows$y <- paste0(rows$contrast, ", ", rows$label)
      rows$y <- factor(rows$y, levels = rev(unique(rows$y)))
      col <- ifelse(rows$kind != "meta", "#98A2B3",
                    ifelse(rows$p >= 0.05, "#667085", ifelse(rows$lfc > 0, PAL$up, PAL$down)))
      rows$hover <- sprintf("<b>%s</b><br>%s<br>log2FC %+.2f (95%% CI %+.2f to %+.2f)<br>%s %s", rows$label, rows$contrast,
                            rows$lfc, rows$lfc - 1.96 * rows$se, rows$lfc + 1.96 * rows$se,
                            ifelse(rows$kind == "meta", "FDR", "p"), fmt_p(rows$p))
      plotly::plot_ly(rows, x = ~lfc, y = ~y, type = "scatter", mode = "markers",
                      error_x = list(type = "data", array = 1.96 * rows$se, color = "rgba(16,24,40,0.35)", thickness = 1.5, width = 0),
                      marker = list(symbol = ifelse(rows$kind == "meta", "diamond", "square"),
                                    size = ifelse(rows$kind == "meta", 15, 9), color = col),
                      text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "log2 fold change (95% CI)", zeroline = TRUE, automargin = TRUE),
                   yaxis = list(title = "", automargin = TRUE, tickfont = list(size = 11)),
                   shapes = list(vline(0)), showlegend = FALSE)
    })

    # iHeps heatmap: lines by exposures
    output$invitro <- plotly::renderPlotly({
      iv <- d$invitro[!is.na(d$invitro$gene_human) & d$invitro$gene_human == gene(), ]
      shiny::validate(shiny::need(nrow(iv) > 0, "Not measured in the iHeps model."))
      lv <- levels(d$invitro$contrast)
      z <- t(sapply(c("1b", "5a"), function(l) vapply(lv, function(cc) {
        v <- iv$log2FC[iv$line == l & iv$contrast == cc]
        if (length(v)) v[1] else NA_real_
      }, 0)))
      sig <- t(sapply(c("1b", "5a"), function(l) vapply(lv, function(cc) {
        v <- iv$significant[iv$line == l & iv$contrast == cc]
        length(v) && isTRUE(v[1])
      }, TRUE)))
      lim <- max(1, max(abs(z), na.rm = TRUE))
      xl <- c("OA+PA", "+ resistin/myostatin", "+ PBMC")
      txt <- matrix(ifelse(is.na(z), "", sprintf("%+.2f%s", z, ifelse(sig, " ●", ""))), nrow = 2)
      plotly::plot_ly(x = xl, y = c("line 1b", "line 5a"), z = z, type = "heatmap", zmin = -lim, zmax = lim,
                      colorscale = list(c(0, PAL$down), c(0.5, "#F2F4F7"), c(1, PAL$up)), showscale = FALSE,
                      hoverinfo = "none", xgap = 4, ygap = 4) |>
        plotly::add_annotations(x = rep(xl, each = 2), y = rep(c("line 1b", "line 5a"), 3), text = as.vector(txt),
                                showarrow = FALSE, font = list(size = 15, color = PAL$ink)) |>
        plot_style(xaxis = list(showgrid = FALSE, side = "top"), yaxis = list(showgrid = FALSE, autorange = "reversed"))
    })

    output$ppi <- reactable::renderReactable({
      g <- gene()
      net <- d$ppi_network
      i <- match(g, net$nodes)
      shiny::validate(shiny::need(!is.na(i), "Not in the protein interaction network."))
      nb <- unique(c(net$to[net$from == i], net$from[net$to == i]))
      nbn <- net$nodes[nb]
      out <- data.frame(Partner = nbn, Degree = net$degree$degree[nb], Key = nbn %in% d$ppi$key_proteins,
                        Target = nbn %in% target_human)
      out <- out[order(-out$Key, -out$Degree), ]
      rt(out, page = 10, columns = list(
        Partner = reactable::colDef(style = list(fontWeight = 600)),
        Degree = col_bar("Partner's interactions", max = max(net$degree$degree), color = PAL$down, digits = 0, width = 170),
        Key = col_check("Early-MAFLD key protein"), Target = col_check("Target gene")))
    })

    output$cluster <- reactable::renderReactable({
      e <- ev()
      shiny::validate(shiny::need(nrow(e$kg) > 0, "Not a node in the MASH knowledge graph."))
      nodes <- d$kg_nodes[d$kg_nodes$cluster == e$kg$cluster & d$kg_nodes$name != e$kg$name, ]
      nodes <- nodes[order(nodes$type, -nodes$composite_pct), ]
      rt(data.frame(Node = nodes$name, Type = nodes$type, Composite = nodes$composite_pct / 100), page = 10, columns = list(
        Node = reactable::colDef(minWidth = 180, style = list(fontWeight = 600)),
        Type = reactable::colDef(html = TRUE, cell = function(v) chip_html(v, switch(v, drug = "up", disease = "warn", "muted"))),
        Composite = col_bar("Centrality percentile (within type)", digits = 2, width = 200)))
    })

    output$drugs <- reactable::renderReactable({
      x <- ev()$drugs
      shiny::validate(shiny::need(nrow(x) > 0, "No network-active drug targets this gene."))
      rt(data.frame(Drug = paste(x$drug, x$drugbank, sep = "|"), z = x$z, Mechanism = x$moa, Indication = x$indication),
         page = 8, columns = list(Drug = col_name_id("Drug", 160), z = col_num("Network z"),
           Mechanism = reactable::colDef(minWidth = 260, style = list(fontSize = "0.8rem")),
           Indication = reactable::colDef(minWidth = 260, style = list(fontSize = "0.8rem", color = PAL$muted))))
    })

    output$table <- reactable::renderReactable({
      s <- gene_sc()
      shiny::req(nrow(s) > 0)
      out <- data.frame(Dataset = d$datasets_meta$label[match(s$dataset, d$datasets_meta$dataset)],
                        Gene = s$gene, AUC = s$auc, Detected = sprintf("%f|%f", s$pct_disease, s$pct_control),
                        log2FC = s$pb_logFC, FDR = s$pb_fdr)
      rt(out, searchable = FALSE, columns = list(
        Dataset = reactable::colDef(style = list(fontWeight = 600)),
        AUC = col_effect(), Detected = col_detect(), log2FC = col_num("Pseudobulk log2FC"), FDR = col_p("FDR")))
    })

    list(gene = gene, gene_sc = gene_sc, gene_bulk = gene_bulk, ev = ev, story = story)
  })
}

BULK_COHORT_LEVELS <- c("MASLD vs control", "Early MASLD vs control", "MASH vs control", "MASH vs early MASLD")
