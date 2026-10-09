# Drugs: knowledge graph, network-active drugs and PPI proximity, joined by DrugBank ID.

drugsUI <- function(id, d) {
  ns <- shiny::NS(id)
  shiny::tagList(
    page_header("capsules", "Drugs", "Candidate drugs from the knowledge graph, the protein interaction network and the co-expression network"),
    shiny::div(class = "toolbar",
      shiny::span(class = "toolbar-label", shiny::icon("magnifying-glass"), " Drug lookup"),
      shiny::selectizeInput(ns("drug"), NULL, choices = NULL, width = "340px",
                            options = list(placeholder = "Search a drug by name or DrugBank ID\u2026"))),
    shiny::uiOutput(ns("drug_card")),
    bslib::navset_pill(
      bslib::nav_panel(shiny::span(shiny::icon("share-nodes"), " Knowledge graph"),
        shiny::div(class = "toolbar mt-3",
          pills(ns("kg_sp"), c("All drugs" = "all", "On NASH / steatosis paths" = "paths")),
          filter_popover(shiny::sliderInput(ns("kg_n"), "Top drugs by PageRank", 50, 2000, 200, step = 50),
                         title = "Ranking")),
        bslib::layout_columns(
          col_widths = c(6, 6),
          bslib::card(bslib::card_header(card_title("PageRank vs betweenness", "chart-simple",
                        "Colour = whether the drug lies on shortest paths to NASH / hepatic steatosis in the graph.")),
                      plot_output(ns("kg_plot"), height = "420px")),
          bslib::card(bslib::card_header(card_title("Ranked drugs", "ranking-star")),
                      reactable::reactableOutput(ns("kg_table"))))
      ),
      bslib::nav_panel(shiny::span(shiny::icon("diagram-project"), " Network-active drugs"),
        shiny::p(class = "page-intro small mt-3", shiny::icon("circle-info"),
                 " 132 drugs whose targets sit closer than chance to the fibrosis-associated co-expression network ",
                 "(network proximity z), with mechanism, indication and targets."),
        bslib::card(reactable::reactableOutput(ns("active_table")))
      ),
      bslib::nav_panel(shiny::span(shiny::icon("circle-nodes"), " Network proximity"),
        shiny::div(class = "toolbar mt-3",
          pills(ns("ppi_named"), c("All drugs" = "all", "Also in knowledge graph" = "kg")),
          filter_popover(shiny::sliderInput(ns("ppi_z"), "Proximity z ≤", -8, 2, -2, step = 0.5),
                         title = "Proximity")),
        bslib::layout_columns(
          col_widths = c(6, 6),
          bslib::card(bslib::card_header(card_title("Closer than chance to early-MAFLD proteins", "bullseye",
                        "z < 0 = drug targets sit closer to the key proteins than random target sets. Bubble size = key proteins hit directly.")),
                      plot_output(ns("ppi_plot"), height = "420px")),
          bslib::card(bslib::card_header(card_title("Drugs", "capsules")),
                      reactable::reactableOutput(ns("ppi_table"))))
      )
    )
  )
}

drugsServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    kg_sel <- shiny::reactive({
      x <- d$kg$drugs   # sorted by PageRank
      if (identical(input$kg_sp, "paths")) x <- x[x$in_any_shortest_paths, ]
      utils::head(x, input$kg_n %||% 200)
    })
    path_label <- function(x) with(x, ifelse(in_nash_shortest_paths & in_steatosis_shortest_paths, "NASH + steatosis",
                                     ifelse(in_nash_shortest_paths, "NASH", ifelse(in_steatosis_shortest_paths, "Steatosis", "None"))))
    path_cols <- c("NASH + steatosis" = PAL$up, "NASH" = "#E07A5F", "Steatosis" = PAL$warn, "None" = "#CBD2D9")

    output$kg_plot <- render_plot({
      x <- kg_sel()
      x <- x[is.finite(x$pagerank_score) & is.finite(x$betweenness_score), ]
      x$paths <- path_label(x)
      x$hover <- sprintf("<b>%s</b> (%s)<br>PageRank #%d<br>paths: %s<br>PPI proximity z: %s",
                         x$name, x$drugbank_accession, x$pagerank_rank, x$paths, fmt_num(x$ppi_proximity_z))
      plot_ly(x, x = ~pagerank_score, y = ~log10(betweenness_score + 1), type = "scatter", mode = "markers",
                      color = ~factor(paths, levels = names(path_cols)), colors = path_cols,
                      marker = list(size = 9, line = list(color = "#fff", width = 1)),
                      text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "PageRank"), yaxis = list(title = "log10(betweenness + 1)"))
    })

    output$kg_table <- reactable::renderReactable({
      x <- kg_sel()
      paths <- ifelse(x$in_nash_shortest_paths & x$in_steatosis_shortest_paths, "NASH|Steatosis",
               ifelse(x$in_nash_shortest_paths, "NASH", ifelse(x$in_steatosis_shortest_paths, "Steatosis", "")))
      out <- data.frame(Rank = x$pagerank_rank, Drug = paste(x$name, x$drugbank_accession, sep = "|"),
                        PageRank = x$pagerank_score, Paths = paths, z = x$ppi_proximity_z)
      rt(out, page = 12, columns = list(
        Rank = reactable::colDef(maxWidth = 60, align = "right", style = list(color = PAL$muted)),
        Drug = col_name_id("Drug"),
        PageRank = col_bar("PageRank", max = max(d$kg$drugs$pagerank_score), color = PAL$primary),
        Paths = reactable::colDef(name = "Shortest paths", minWidth = 130, html = TRUE, cell = function(v) {
          if (!nzchar(v)) return("")
          paste(vapply(strsplit(v, "|", fixed = TRUE)[[1]], function(k) chip_html(k, if (k == "NASH") "up" else "warn"), ""),
                collapse = " ")
        }),
        z = col_num("PPI z")))
    })

    ppi_sel <- shiny::reactive({
      x <- d$ppi$proximity
      x <- x[!is.na(x$z) & x$z <= (input$ppi_z %||% -2), ]
      if (identical(input$ppi_named, "kg")) x <- x[!is.na(x$drug_name), ]
      x
    })

    output$ppi_plot <- render_plot({
      x <- ppi_sel()
      shiny::validate(shiny::need(nrow(x) > 0, "No drugs pass the filter."))
      x$label <- ifelse(is.na(x$drug_name), x$drug_id, x$drug_name)
      x$hover <- sprintf("<b>%s</b> (%s)<br>z = %.2f, d = %.2f<br>%d targets, %d key proteins hit",
                         x$label, x$drug_id, x$z, x$d, x$n_targets, x$n_key_0deg)
      plot_ly(x, x = ~d, y = ~z, type = "scatter", mode = "markers",
                      marker = list(size = pmin(6 + 1.5 * x$n_key_0deg, 34), color = PAL$down, opacity = 0.55,
                                    line = list(color = "#fff", width = 1)),
                      text = ~hover, hoverinfo = "text") |>
        plot_style(xaxis = list(title = "Mean distance to key proteins (d)"), yaxis = list(title = "Proximity z"),
                   showlegend = FALSE)
    })

    output$ppi_table <- reactable::renderReactable({
      x <- ppi_sel()
      out <- data.frame(Drug = paste(ifelse(is.na(x$drug_name), x$drug_id, x$drug_name), x$drug_id, sep = "|"),
                        z = x$z, KeyHit = x$n_key_0deg, Targets = x$targets_key_proteins,
                        KGrank = x$kg_pagerank_rank)
      rt(out, page = 12, columns = list(
        Drug = col_name_id("Drug", 150), z = col_num("z"),
        KeyHit = col_bar("Key proteins hit", max = max(d$ppi$proximity$n_key_0deg, na.rm = TRUE),
                         color = PAL$down, digits = 0, width = 120),
        Targets = reactable::colDef(name = "Which key proteins", minWidth = 200,
                                    style = list(fontSize = "0.75rem", color = PAL$muted)),
        KGrank = reactable::colDef(name = "KG rank", align = "right", maxWidth = 80)))
    })

    # Network-active drugs
    output$active_table <- reactable::renderReactable({
      a <- d$active_drugs$drugs
      rt(data.frame(Drug = paste(a$drug, a$drugbank, sep = "|"), z = a$z, Distance = a$distance,
                    Mechanism = a$moa, Indication = a$indication,
                    Targets = vapply(strsplit(a$targets, ",\\s*"), function(t) sprintf("%d targets", length(t)), "")),
         page = 12, columns = list(
           Drug = col_name_id("Drug", 170), z = col_num("Proximity z"), Distance = col_num("Distance"),
           Mechanism = reactable::colDef(minWidth = 280, style = list(fontSize = "0.8rem")),
           Indication = reactable::colDef(minWidth = 240, style = list(fontSize = "0.8rem", color = PAL$muted)),
           Targets = reactable::colDef(maxWidth = 100)))
    })

    # Drug lookup
    drug_choices <- local({
      kg <- d$kg$drugs
      px <- d$ppi$proximity
      ad <- d$active_drugs$drugs
      ids <- unique(c(kg$drugbank_accession, px$drug_id[px$id_type == "DrugBank"], ad$drugbank))
      nm <- ifelse(!is.na(match(ids, kg$drugbank_accession)), kg$name[match(ids, kg$drugbank_accession)],
                   ad$drug[match(ids, ad$drugbank)])
      nm[is.na(nm)] <- ids[is.na(nm)]
      o <- order(toupper(nm), method = "radix")
      stats::setNames(ids[o], sprintf("%s (%s)", nm[o], ids[o]))
    })
    shiny::updateSelectizeInput(session, "drug", choices = drug_choices, server = TRUE, selected = character())

    output$drug_card <- shiny::renderUI({
      id <- input$drug
      if (is.null(id) || !nzchar(id)) return(NULL)
      kg <- d$kg$drugs[d$kg$drugs$drugbank_accession == id, ]
      px <- d$ppi$proximity[d$ppi$proximity$drug_id == id, ]
      ad <- d$active_drugs$drugs[d$active_drugs$drugs$drugbank == id, ]
      name <- if (nrow(kg)) kg$name[1] else if (nrow(ad)) ad$drug[1] else id
      stat <- function(icon, label, value, sub) shiny::div(class = "ev-tile info",
        shiny::div(class = "ev-source", shiny::icon(icon), label),
        shiny::div(class = "ev-main", shiny::span(class = "ev-value", value)), shiny::div(class = "ev-sub", sub))
      paths <- if (nrow(kg)) c(if (kg$in_nash_shortest_paths) "NASH", if (kg$in_steatosis_shortest_paths) "steatosis") else NULL
      targets <- if (nrow(ad)) trimws(strsplit(ad$targets[1], ",")[[1]]) else character()
      bslib::card(class = "summary-card",
        bslib::card_body(
          shiny::div(class = "gene-head", shiny::h3(class = "gene-name", name), chip(id, "capsules", "muted"),
                     if (length(paths)) chip(paste("on shortest paths to", paste(paths, collapse = " & ")), "route", "up")),
          if (nrow(ad)) shiny::p(class = "sum-text", shiny::tags$b("Mechanism: "), ad$moa[1]),
          if (nrow(ad)) shiny::p(class = "sum-text text-muted", shiny::tags$b("Indication: "), ad$indication[1]),
          shiny::div(class = "evidence-strip mt-2",
            stat("share-nodes", "Knowledge graph", if (nrow(kg)) sprintf("#%d", kg$pagerank_rank[1]) else "n/a",
                 if (nrow(kg)) sprintf("PageRank among %s drugs", format(nrow(d$kg$drugs), big.mark = ",")) else "not in the graph"),
            stat("circle-nodes", "PPI proximity", if (nrow(px) && !is.na(px$z[1])) sprintf("z %.2f", px$z[1]) else "n/a",
                 if (nrow(px)) sprintf("%d targets, %d key proteins hit", px$n_targets[1], px$n_key_0deg[1]) else "not analysed"),
            stat("diagram-project", "Co-expression network", if (nrow(ad)) sprintf("z %.2f", ad$z[1]) else "n/a",
                 if (nrow(ad)) sprintf("%d targets", length(targets)) else "not network-active")),
          if (length(targets)) shiny::div(class = "mt-2", shiny::span(class = "toolbar-label", "Targets "),
            lapply(utils::head(targets, 40), function(t) shiny::tags$a(class = "chip", style = "cursor:pointer",
              onclick = sprintf("Shiny.setInputValue('gene-pick', '%s', {priority: 'event'}); mashGo('gene');", t), t)))
        ))
    })

    list(kg_sel = kg_sel, ppi_sel = ppi_sel)
  })
}
