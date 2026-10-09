# Overview: headline numbers, dataset cards, agreement heatmap and key findings.

overviewUI <- function(id, d) {
  ns <- shiny::NS(id)
  meta <- d$datasets_meta
  meta <- meta[order(meta$species != "human", -meta$bulk_agreement), ]
  cmp <- d$consistency_compare
  v2_all <- cmp[cmp$version == "v2 (fixed)" & cmp$gene_set == "all genes", ]
  best <- meta[which.max(meta$bulk_agreement), ]

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

  insight <- function(icon, color, soft, title, text) {
    shiny::div(class = "insight", style = sprintf("--c:%s;--c-soft:%s", color, soft),
               shiny::div(class = "insight-icon", shiny::icon(icon)),
               shiny::div(shiny::div(class = "insight-title", title), shiny::div(class = "insight-text", text)))
  }

  shiny::tagList(
    shiny::div(class = "hero",
      shiny::icon("dna", class = "hero-deco"),
      shiny::div(class = "hero-eyebrow", "MASH hepatocyte omics"),
      shiny::h1("What changes in liver cells as fatty liver disease progresses?"),
      shiny::p("Single-cell, bulk, in-vitro, network and knowledge-graph evidence for MASH. Every single-cell result is ",
               "computed per donor, on hepatocytes only, and checked against an independent bulk cohort."),
      shiny::div(class = "hero-stats",
        glass("layer-group", nrow(meta), "single-cell datasets (2 human, 2 mouse)", "#5EEAD4"),
        glass("microscope", format(sum(meta$n_hepatocytes, na.rm = TRUE), big.mark = ","), "hepatocytes analysed", "#93C5FD"),
        glass("arrows-up-down", v2_all$consistent_in_all, "genes consistent in every usable dataset", "#FDA4AF"),
        glass("bullseye", sprintf("%.0f%%", 100 * best$bulk_agreement), sprintf("best agreement with bulk (%s)", best$label), "#FCD34D"))
    ),
    shiny::div(class = "section-label", shiny::icon("layer-group"), "Datasets"),
    shiny::div(class = "ds-grid", lapply(seq_len(nrow(meta)), ds_card)),
    shiny::div(class = "section-label", shiny::icon("lightbulb"), "What the data say"),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header(card_title("Do datasets agree on direction?", "code-compare",
          "Share of genes called up or down in both datasets that go the same way. 50% = chance.")),
        plotly::plotlyOutput(ns("heatmap"), height = "300px")
      ),
      bslib::card(
        bslib::card_body(fillable = FALSE, class = "insights",
          insight("arrows-up-down", PAL$primary, "var(--primary-soft)", "A steatosis program replicates",
                  "SREBF1, PLIN2 and FABP1 go up and cholesterol-synthesis genes go down in every usable dataset."),
          insight("triangle-exclamation", PAL$warn, "var(--warn-soft)", "The two human cohorts disagree",
                  "Xiao and Wang mostly point in opposite directions, under every processing choice tested."),
          insight("bullseye", PAL$down, "var(--down-soft)", "Bulk agreement varies by dataset",
                  "Wang agrees best with the independent bulk cohort. A result from one cohort needs replication.")
        )
      )
    )
  )
}

overviewServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {
    output$heatmap <- plotly::renderPlotly({
      m <- d$datasets_meta[order(d$datasets_meta$species != "human", -d$datasets_meta$bulk_agreement), ]
      agreement_heatmap(d$consistency$pairwise, m$dataset, m$label)
    })
  })
}
