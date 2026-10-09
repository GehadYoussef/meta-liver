# Knowledge graph: clusters of genes, drugs and diseases, and the most central nodes.

KG_TYPE <- c("gene/protein" = "gene", drug = "drug", disease = "disease")
KG_COL <- c(gene = "#0F9F8F", drug = "#2F80C3", disease = "#E5A13A")
LIVER_TERMS <- "liver|hepat|steato|cirrho|biliar|cholest"

kg_clusters <- function(k) {
  k$kind <- KG_TYPE[k$type]
  out <- do.call(rbind, lapply(split(k, k$cluster), function(x) {
    dis <- x[x$kind == "disease", ]
    dis <- dis[order(-dis$composite_pct), ]
    data.frame(cluster = x$cluster[1], n = nrow(x), gene = sum(x$kind == "gene"), drug = sum(x$kind == "drug"),
               disease = nrow(dis), top_disease = if (nrow(dis)) dis$name[1] else "",
               liver = any(grepl(LIVER_TERMS, dis$name, ignore.case = TRUE)),
               nash = any(dis$name == "non-alcoholic steatohepatitis"))
  }))
  out[order(!out$nash, !out$liver, -out$n), ]
}

kg_link <- function(cluster, text = cluster) {
  sprintf('<a class="gene-link" onclick="Shiny.setInputValue(\'kg-cluster\', %d, {priority: \'event\'}); mashGo(\'kg\');">%s</a>',
          as.integer(cluster), text)
}

n_label <- function(n, word) sprintf("%d %s%s", n, word, if (n == 1) "" else "s")

gene_link <- function(g) {
  sprintf('<a class="gene-link" onclick="Shiny.setInputValue(\'gene-pick\', \'%s\', {priority: \'event\'}); mashGo(\'gene\');">%s</a>', g, g)
}

kgUI <- function(id, d) {
  ns <- shiny::NS(id)
  k <- d$kg_nodes
  shiny::tagList(
    page_header("share-nodes", "Knowledge graph",
                sprintf("%s genes, drugs and diseases in %d clusters",
                        format(nrow(k), big.mark = ","), length(unique(k$cluster)))),
    shiny::div(class = "toolbar",
      shiny::span(class = "toolbar-label", shiny::icon("magnifying-glass"), " Find a node"),
      shiny::selectizeInput(ns("node"), NULL, choices = NULL, width = "340px",
                            options = list(placeholder = "Gene, drug or disease…")),
      shiny::div(class = "kg-legend",
        lapply(names(KG_COL), function(t) shiny::span(shiny::span(class = "kg-dot", style = sprintf("background:%s", KG_COL[[t]])),
                                                      paste0(t, "s"))),
        shiny::span(shiny::icon("droplet", class = "kg-liver"), "liver disease in cluster"))),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(bslib::card_header(card_title("Clusters", "table-cells",
                    "Each tile is a cluster of the graph, with its mix of genes, drugs and diseases. Click a tile to open it.")),
                  shiny::uiOutput(ns("tiles"))),
      bslib::card(bslib::card_header(shiny::uiOutput(ns("cluster_title"))),
                  shiny::uiOutput(ns("diseases")),
                  bslib::navset_underline(
                    bslib::nav_panel(shiny::span(shiny::icon("dna"), " Genes"), table_output(ns("genes"))),
                    bslib::nav_panel(shiny::span(shiny::icon("capsules"), " Drugs"), table_output(ns("drugs")))))
    ),
    bslib::card(
      bslib::card_header(shiny::div(class = "d-flex justify-content-between align-items-center w-100",
        card_title("Most central nodes", "ranking-star",
                   "Composite centrality: weighted geometric mean of PageRank, betweenness and eigenvector percentiles, within node type."),
        pills(ns("type"), c("Genes" = "gene", "Drugs" = "drug", "Diseases" = "disease")))),
      table_output(ns("top")))
  )
}

kgServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    k <- d$kg_nodes
    k$kind <- KG_TYPE[k$type]
    cl <- kg_clusters(k)
    cluster <- shiny::reactiveVal(cl$cluster[1])
    shiny::observeEvent(input$cluster, cluster(as.integer(input$cluster)))

    node_choices <- stats::setNames(seq_len(nrow(k)), sprintf("%s (%s)", k$name, k$kind))
    node_choices <- node_choices[order(names(node_choices), method = "radix")]
    shiny::updateSelectizeInput(session, "node", choices = node_choices, server = TRUE, selected = character())
    shiny::observeEvent(input$node, {
      i <- suppressWarnings(as.integer(input$node))
      if (!is.na(i)) cluster(k$cluster[i])
    })

    output$tiles <- shiny::renderUI({
      sel <- cluster()
      shiny::div(class = "kg-grid", lapply(seq_len(nrow(cl)), function(i) {
        x <- cl[i, ]
        mix <- lapply(names(KG_COL), function(t) shiny::span(style = sprintf("flex:%d;background:%s", x[[t]], KG_COL[[t]])))
        shiny::div(class = paste("kg-tile", if (x$cluster == sel) "active"),
                   title = if (nzchar(x$top_disease)) paste("Top disease:", x$top_disease) else "No disease nodes",
                   onclick = sprintf("Shiny.setInputValue('%s', %d, {priority: 'event'})", session$ns("cluster"), x$cluster),
                   shiny::div(class = "kg-id", x$cluster, if (x$liver) shiny::icon("droplet", class = "kg-liver")),
                   shiny::div(class = "kg-mix", mix),
                   shiny::div(class = "kg-n", x$n))
      }))
    })

    members <- shiny::reactive(k[k$cluster == cluster(), ])

    output$cluster_title <- shiny::renderUI({
      x <- cl[cl$cluster == cluster(), ]
      shiny::div(class = "d-flex align-items-center gap-2 flex-wrap",
        card_title(sprintf("Cluster %d", x$cluster), "share-nodes"),
        chip(n_label(x$gene, "gene"), "dna"), chip(n_label(x$drug, "drug"), "capsules", "down"),
        chip(n_label(x$disease, "disease"), "virus", "warn"))
    })

    output$diseases <- shiny::renderUI({
      dis <- members()
      dis <- dis[dis$kind == "disease", ]
      dis <- dis[order(-dis$composite_pct), ]
      if (!nrow(dis)) return(shiny::p(class = "small text-muted", "No disease nodes in this cluster."))
      shown <- utils::head(dis, 12)
      shiny::div(class = "mb-2",
        lapply(shown$name, function(n) chip(n, NULL, if (grepl(LIVER_TERMS, n, ignore.case = TRUE)) "warn" else "muted")),
        if (nrow(dis) > nrow(shown)) shiny::span(class = "small text-muted", sprintf("and %d more", nrow(dis) - nrow(shown))))
    })

    output$genes <- render_table({
      g <- members()
      g <- g[g$kind == "gene", ]
      shiny::validate(shiny::need(nrow(g) > 0, "No genes in this cluster."))
      g <- g[order(-g$composite_pct), ]
      b <- d$bulk_cohorts$meta[d$bulk_cohorts$meta$contrast == SUMMARY_BULK_CONTRAST, ]
      b <- b[match(toupper(g$name), toupper(b$gene_human)), ]
      ev <- d$evidence[match(toupper(g$name), toupper(d$evidence$gene_human)), ]
      liver <- ifelse(is.na(b$fdr), NA, ifelse(b$fdr >= 0.05, "none", ifelse(b$meta_log2FC > 0, "up", "down")))
      sc <- ifelse(is.na(ev$direction) | ev$evidence == 0, NA, ifelse(ev$direction > 0, "up", ifelse(ev$direction < 0, "down", "none")))
      rt(data.frame(Gene = gene_link(g$name), Centrality = g$composite_pct / 100,
                    Liver = liver, SC = sc, stringsAsFactors = FALSE),
         page = 10, columns = list(
           Gene = col_def(html = TRUE, minWidth = 110),
           Centrality = col_bar("Centrality percentile", digits = 2, width = 170),
           Liver = utils::modifyList(col_call("Whole liver"), list(minWidth = 95)),
           SC = utils::modifyList(col_call("Hepatocytes"), list(minWidth = 110))))
    })

    output$drugs <- render_table({
      x <- members()
      x <- x[x$kind == "drug", ]
      shiny::validate(shiny::need(nrow(x) > 0, "No drugs in this cluster."))
      x <- x[order(-x$composite_pct), ]
      kd <- d$kg$drugs[match(x$drugbank_accession, d$kg$drugs$drugbank_accession), ]
      paths <- ifelse(kd$in_nash_shortest_paths & kd$in_steatosis_shortest_paths, "NASH|Steatosis",
               ifelse(kd$in_nash_shortest_paths, "NASH", ifelse(kd$in_steatosis_shortest_paths, "Steatosis", "")))
      paths[is.na(paths)] <- ""
      rt(data.frame(Drug = paste(x$name, x$drugbank_accession, sep = "|"), Centrality = x$composite_pct / 100, Paths = paths),
         page = 10, columns = list(
           Drug = col_name_id("Drug", 180),
           Centrality = col_bar("Centrality percentile", digits = 2, width = 170),
           Paths = col_def(name = "On shortest paths to", html = TRUE, minWidth = 140, cell = function(v) {
             if (!nzchar(v)) return("")
             paste(vapply(strsplit(v, "|", fixed = TRUE)[[1]], function(p) chip_html(p, if (p == "NASH") "up" else "warn"), ""),
                   collapse = "")
           })))
    })

    output$top <- render_table({
      t <- input$type %||% "gene"
      x <- k[k$kind == t, ]
      x <- utils::head(x[order(-x$composite_pct, -x$pagerank_score), ], 300)
      name <- if (t == "gene") gene_link(x$name) else sprintf("<b>%s</b>", esc(x$name))
      rt(data.frame(Name = name, Cluster = kg_link(x$cluster, paste("Cluster", x$cluster)),
                    Centrality = x$composite_pct / 100, PageRank = x$pagerank_score, stringsAsFactors = FALSE),
         page = 10, columns = list(
           Name = col_def(html = TRUE, minWidth = 220),
           Cluster = col_def(html = TRUE, minWidth = 100),
           Centrality = col_bar("Centrality percentile", digits = 2, width = 200),
           PageRank = col_num("PageRank", 3)))
    })
  })
}
