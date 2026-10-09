# Overview: headline numbers, dataset cards, agreement heatmap and key findings.

overviewUI <- function(id, d) {
  ns <- shiny::NS(id)
  meta <- d$datasets_meta
  meta <- meta[order(meta$species != "human", -meta$bulk_agreement), ]
  g <- d$consistency$genes
  nds <- length(d$consistency$datasets)
  replicated <- g[g$n_up == nds | g$n_down == nds, ]
  replicated <- replicated[order(replicated$n_down > 0, replicated$gene_human), ]
  n_tested <- length(unique(d$sc$gene_human[d$sc$tested]))

  glass <- function(icon, value, label, color) {
    shiny::div(class = "glass",
      shiny::div(class = "glass-icon", style = sprintf("--c:%s", color), shiny::icon(icon)),
      shiny::div(shiny::div(class = "glass-value", value), shiny::div(class = "glass-label", label)))
  }

  ds_card <- function(i) {
    m <- meta[i, ]
    ring_col <- if (!m$directions_reliable) PAL$na else if (m$bulk_agreement >= 0.6) PAL$primary
                else if (m$bulk_agreement >= 0.5) PAL$warn else PAL$up
    status <- if (!m$directions_reliable) {
      bslib::tooltip(shiny::span(class = "ds-status", style = sprintf("color:%s", PAL$warn), shiny::icon("triangle-exclamation")),
                     sprintf("Directions unreliable: %.0f%% of genes move one way (1 animal per group)", m$pct_up))
    } else if (!m$pvalues) {
      bslib::tooltip(shiny::span(class = "ds-status", style = sprintf("color:%s", PAL$muted), shiny::icon("circle-exclamation")),
                     "No sample-level p-values")
    } else {
      bslib::tooltip(shiny::span(class = "ds-status", style = sprintf("color:%s", PAL$primary), shiny::icon("circle-check")),
                     "Replicated design: per-donor p-values available")
    }
    dots <- c(rep("nash", m$n_disease), rep("ctrl", m$n_control))
    shiny::div(class = paste("ds-card", if (!m$directions_reliable) "flagged"),
      status,
      shiny::div(class = "ds-head",
        shiny::div(class = paste("ds-avatar", m$species), species_icon(m$species)),
        shiny::div(shiny::div(class = "ds-name", m$label),
                   shiny::div(class = "ds-meta", sprintf("%s, %s, %s", tools::toTitleCase(m$species), m$assay, m$geo)))),
      shiny::div(
        shiny::div(class = "donors", lapply(dots, function(k) shiny::span(class = paste("donor", k)))),
        shiny::div(class = "donor-legend", shiny::tags$b(class = "nash", sprintf("%d NASH", m$n_disease)), " vs ",
                   shiny::tags$b(class = "ctrl", sprintf("%d control", m$n_control)),
                   if (m$species == "human") " donors" else " animals")),
      shiny::div(class = "ds-numbers",
        shiny::div(shiny::div(class = "big-number", format(m$n_hepatocytes, big.mark = ",")),
                   shiny::div(class = "big-label", "hepatocytes")),
        shiny::div(class = "ring-wrap", shiny::div(class = "ring-caption", "agrees with bulk"),
                   ring_svg(m$bulk_agreement, ring_col)))
    )
  }

  gene_chip <- function(gene, dir) {
    shiny::tags$a(class = paste("gene-chip", dir), gene,
                  onclick = sprintf("Shiny.setInputValue('gene-pick', '%s', {priority: 'event'}); mashGo('gene');", gene))
  }

  shiny::tagList(
    shiny::div(class = "hero",
      shiny::icon("dna", class = "hero-deco"),
      shiny::div(class = "hero-eyebrow", "MASH hepatocyte omics"),
      shiny::h1("Hepatocyte gene changes in MASH"),
      shiny::p("Each gene is scored in every dataset and checked against bulk liver, an iPSC model and networks."),
      shiny::div(class = "hero-stats",
        glass("layer-group", nrow(meta), "single-cell datasets", "#5EEAD4"),
        glass("microscope", format(sum(meta$n_hepatocytes, na.rm = TRUE), big.mark = ","), "hepatocytes", "#93C5FD"),
        glass("dna", format(n_tested, big.mark = ","), "genes tested", "#FCD34D"),
        glass("circle-check", nrow(replicated), sprintf("replicated in all %d datasets", nds), "#FDA4AF"))
    ),
    shiny::div(class = "section-label", shiny::icon("layer-group"), "Datasets"),
    shiny::div(class = "ds-grid", lapply(seq_len(nrow(meta)), ds_card)),
    shiny::div(class = "section-label", shiny::icon("arrows-up-down"), "Replication"),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header(card_title("Direction agreement", "code-compare",
          "Share of genes called up or down in both datasets that go the same way. 50% is chance.")),
        plot_output(ns("heatmap"), height = "300px")
      ),
      bslib::card(
        bslib::card_header(shiny::div(class = "d-flex align-items-center gap-2 w-100",
          card_title("Replicated genes", "circle-check",
            sprintf("Same direction (AUC ≥ 0.55 or ≤ 0.45) in all %d datasets with reliable directions. Click a gene to open it.", nds)),
          chip(sprintf("%d up", sum(replicated$n_up == nds)), "arrow-up", "up"),
          chip(sprintf("%d down", sum(replicated$n_down == nds)), "arrow-down", "down"))),
        shiny::div(class = "gene-chips",
          lapply(seq_len(nrow(replicated)), function(i)
            gene_chip(replicated$gene_human[i], if (replicated$n_up[i] == nds) "up" else "down"))),
        shiny::div(class = "rep-foot",
          shiny::span(shiny::tags$b(sum(grepl("^consistent", g$consistency))), " consistent in 2+ datasets"),
          shiny::span(shiny::tags$b(sum(grepl("^conflicting", g$consistency))), " conflicting"))
      )
    )
  )
}

overviewServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    output$heatmap <- render_plot({
      m <- d$datasets_meta[order(d$datasets_meta$species != "human", -d$datasets_meta$bulk_agreement), ]
      agreement_heatmap(d$consistency$pairwise, m$dataset, m$label)
    })
  })
}
