# Home: gene search, the data behind the app, replicated genes, citation.

CITATION_SHORT <- "Weihs J, Baldo F, Cardinali A, Youssef G, et al. Combined stem cell and predictive models reveal flavin cofactors as targets in metabolic liver dysfunction. bioRxiv 2024."
CITATION_FULL <- paste(
  "Weihs J, Baldo F, Cardinali A, Youssef G, Ludwik K, Haep N, Tang P, Kumar P, Engelmann C, Quach S,",
  "Meindl M, Kucklick M, Engelmann S, Chillian B, Rothe M, Meierhofer D, Lurje I, Hammerich L, Ramachandran P,",
  "Kendall TJ, Fallowfield JA, Stachelscheid H, Sauer I, Tacke F, Bufler P, Hudert C, Han N, Rezvani M.",
  "Combined stem cell and predictive models reveal flavin cofactors as targets in metabolic liver dysfunction.",
  "bioRxiv 2024.10.10.617610.")
DOI_URL <- "https://doi.org/10.1101/2024.10.10.617610"

doi_link <- function() shiny::tags$a(href = DOI_URL, target = "_blank", "doi:10.1101/2024.10.10.617610")

# One card per single-cell dataset: design, hepatocytes, reliability, agreement with bulk
dataset_cards <- function(d) {
  meta <- d$datasets_meta[order(d$datasets_meta$species != "human", -d$datasets_meta$bulk_agreement), ]
  card <- function(i) {
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
                   ring_svg(m$bulk_agreement, ring_col))))
  }
  shiny::div(class = "ds-grid", lapply(seq_len(nrow(meta)), card))
}

overviewUI <- function(id, d) {
  meta <- d$datasets_meta
  g <- d$consistency$genes
  nds <- length(d$consistency$datasets)
  replicated <- g[g$n_up == nds | g$n_down == nds, ]
  replicated <- replicated[order(replicated$n_down > 0, replicated$gene_human), ]
  cohorts <- sort(unique(d$bulk_cohorts$studies$study))
  lines <- sort(unique(d$invitro$line))

  layer <- function(icon, page, title, value, detail) {
    shiny::tags$button(type = "button", class = "layer-tile", onclick = sprintf("mashGo('%s')", page),
      shiny::span(class = "layer-tile-icon", shiny::icon(icon)),
      shiny::span(class = "layer-tile-body",
        shiny::span(class = "layer-tile-title", title),
        shiny::span(class = "layer-tile-value", value),
        shiny::span(class = "layer-tile-detail", detail)))
  }
  gene_chip <- function(gene, dir) {
    shiny::tags$a(class = paste("gene-chip", dir), gene,
                  onclick = sprintf("Shiny.setInputValue('gene-pick', '%s', {priority: 'event'}); mashGo('gene');", gene))
  }

  shiny::tagList(
    shiny::div(class = "home-hero",
      shiny::h1("Meta Liver"),
      shiny::p("A hypothesis engine for metabolic liver disease"),
      shiny::div(class = "home-search", shiny::icon("magnifying-glass"),
        shiny::tags$input(type = "text", placeholder = "Search a gene, e.g. SREBF1", autocomplete = "off",
          onkeydown = "if (event.key === 'Enter' && this.value.trim()) { Shiny.setInputValue('gene-pick', this.value.trim(), {priority: 'event'}); mashGo('gene'); }"))),

    shiny::div(class = "layer-grid-home",
      layer("microscope", "markers", "Single-cell", sprintf("%d datasets", nrow(meta)), paste(sort(meta$label), collapse = ", ")),
      layer("flask", "bulk", "Bulk liver", sprintf("%d cohorts", length(cohorts)), paste(cohorts, collapse = ", ")),
      layer("vial", "invitro", "In-vitro", sprintf("%d iPSC lines", length(lines)), "3 MASLD exposures"),
      layer("share-nodes", "kg", "Networks", "3 layers", "PPI, co-expression, knowledge graph")),

    bslib::card(
      bslib::card_header(shiny::div(class = "d-flex align-items-center gap-2 w-100",
        card_title(sprintf("Replicated in all %d single-cell datasets", nds), "circle-check",
          "Same direction (AUC ≥ 0.55 or ≤ 0.45) in every dataset with reliable directions. Click a gene to open it."),
        chip(sprintf("%d up", sum(replicated$n_up == nds)), "arrow-up", "up"),
        chip(sprintf("%d down", sum(replicated$n_down == nds)), "arrow-down", "down"))),
      shiny::div(class = "gene-chips",
        lapply(seq_len(nrow(replicated)), function(i)
          gene_chip(replicated$gene_human[i], if (replicated$n_up[i] == nds) "up" else "down"))),
      shiny::div(class = "rep-foot",
        shiny::span(shiny::tags$b(sum(grepl("^consistent", g$consistency))), " consistent in 2+ datasets"),
        shiny::span(shiny::tags$b(sum(grepl("^conflicting", g$consistency))), " conflicting"),
        shiny::tags$a(class = "rep-more", onclick = "mashGo('consistency')", "All datasets ", shiny::icon("arrow-right")))),

    shiny::div(class = "home-cite",
      shiny::div(shiny::tags$b("Cite: "), CITATION_SHORT, " ", doi_link()),
      shiny::div("Han lab, University of Cambridge, and Rezvani lab, Charité Berlin. ",
                 shiny::tags$a(onclick = "mashGo('about')", "Team")))
  )
}

overviewServer <- function(id, d) {
  shiny::moduleServer(id, function(input, output, session) {})
}
