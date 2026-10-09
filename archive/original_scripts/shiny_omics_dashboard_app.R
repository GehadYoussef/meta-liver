# shiny_omics_dashboard/app.R
# Run with: shiny::runApp(".")
# Data files expected in ./data/ (included in this bundle).

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(readr)
  library(readxl)
  library(ggplot2)
  library(plotly)
  library(DT)
})

DATA_DIR <- "data"

# -----------------------------
# Helpers
# -----------------------------
file_or_null <- function(path) if (file.exists(path)) path else NULL

read_auc_coassolo <- function(path) {
  if (is.null(path)) return(NULL)
  df <- readr::read_csv(path, show_col_types = FALSE)
  # normalise column names
  df <- df %>%
    rename_with(~tolower(.x)) %>%
    rename(gene = gene, auc = auc) %>%
    mutate(dataset = "Coassolo")
  df
}

read_auc_su <- function(path) {
  if (is.null(path)) return(NULL)
  df <- readr::read_csv(path, show_col_types = FALSE)
  df <- df %>%
    rename_with(~tolower(.x)) %>%
    rename(gene = gene, auc = auc) %>%
    mutate(dataset = "SU")
  df
}

read_kg_supp6 <- function(path) {
  if (is.null(path)) return(list(nash = NULL, hep = NULL, nodes = NULL))
  nash <- readxl::read_xlsx(path, sheet = "nash_shortestpaths", skip = 3) %>%
    rename(drugbank_accession = DrugBank_Accession, drug_name = `Drug Name`) %>%
    mutate(list_source = "Shortest paths from NASH")
  hep <- readxl::read_xlsx(path, sheet = "hepaticsteatosis_shortestpaths", skip = 3) %>%
    rename(drugbank_accession = DrugBank_Accession, drug_name = `Drug Name`) %>%
    mutate(list_source = "Shortest paths from Hepatic steatosis")
  nodes <- readxl::read_xlsx(path, sheet = "MASH_Subgraph_all_nodes", skip = 3) %>%
    rename_with(~tolower(gsub("\\s+", "_", .x))) %>%
    rename(
      name = name,
      drugbank_accession = drugbank_accession,
      type = type,
      pagerank_score = pagerank_score,
      betweenness_score = betweenness_score,
      eigen_score = eigen_score,
      cluster = cluster
    )
  list(nash = nash, hep = hep, nodes = nodes)
}

# -----------------------------
# Load defaults (from ./data)
# -----------------------------
default_paths <- list(
  kg_xlsx = file_or_null(file.path(DATA_DIR, "Fatima_Supp Table 6.xlsx")),
  auc_coassolo = file_or_null(file.path(DATA_DIR, "Coassolo_AUC_scores_target_genes_full_stats.csv")),
  auc_su = file_or_null(file.path(DATA_DIR, "SU_Hepatocyte_gene_AUC_direction_stats_balanced_new.csv"))
)

# -----------------------------
# UI
# -----------------------------
ui <- navbarPage(
  title = "Omics & Network Results Explorer",
  theme = bslib::bs_theme(version = 5, bootswatch = "flatly"),
  tabPanel("Gene AUC",
    fluidPage(
      fluidRow(
        column(4,
          h4("Data inputs"),
          p("If the bundled files are present in ./data, you can ignore uploads. Uploading overrides defaults."),
          fileInput("auc_coassolo_up", "Upload Coassolo AUC CSV", accept = c(".csv")),
          fileInput("auc_su_up", "Upload SU AUC CSV", accept = c(".csv")),
          hr(),
          uiOutput("dataset_ui"),
          uiOutput("gene_ui"),
          uiOutput("dir_ui")
        ),
        column(8,
          h4("AUC summary"),
          plotlyOutput("auc_plot", height = "360px"),
          hr(),
          h4("Gene statistics (selected dataset)"),
          DTOutput("gene_stats_tbl")
        )
      ),
      hr(),
      h4("Top genes"),
      fluidRow(
        column(3, numericInput("top_n", "Show top N genes", value = 50, min = 10, step = 10)),
        column(3, checkboxInput("only_sig", "Only adjusted p < 0.05 (if available)", value = FALSE)),
        column(6, textInput("gene_search", "Search gene (substring)", value = ""))
      ),
      DTOutput("top_tbl")
    )
  ),
  tabPanel("Knowledge graph drugs",
    fluidPage(
      fluidRow(
        column(4,
          h4("Knowledge graph input"),
          fileInput("kg_up", "Upload Supp Table 6 Excel", accept = c(".xlsx")),
          hr(),
          sliderInput("top_drugs", "Top drugs by PageRank", min = 50, max = 2000, value = 200, step = 50),
          checkboxInput("only_shortestpath", "Show only drugs appearing in shortest-path lists", value = FALSE)
        ),
        column(8,
          h4("Drug prioritisation (algorithms)"),
          plotlyOutput("drug_scatter", height = "360px"),
          hr(),
          DTOutput("drug_tbl")
        )
      )
    )
  ),
  tabPanel("WGCNA modules",
    fluidPage(
      h4("Upload WGCNA gene–module table"),
      p("Expected columns include gene and module (e.g., module_colour), plus optional kME or module membership scores."),
      fileInput("wgcna_up", "Upload WGCNA CSV", accept = c(".csv")),
      uiOutput("wgcna_gene_ui"),
      DTOutput("wgcna_tbl")
    )
  ),
  tabPanel("Drug results (placeholders)",
    fluidPage(
      h4("Drug-network results"),
      p("If you have a drug proximity / enrichment results table (CSV), upload it here to enable ranking and filtering."),
      fileInput("drugres_up", "Upload drug results CSV", accept = c(".csv")),
      uiOutput("drugres_ui"),
      DTOutput("drugres_tbl")
    )
  )
)

# -----------------------------
# Server
# -----------------------------
server <- function(input, output, session) {

  # ---- AUC data (reactive, uploads override defaults) ----
  auc_coassolo <- reactive({
    path <- if (!is.null(input$auc_coassolo_up)) input$auc_coassolo_up$datapath else default_paths$auc_coassolo
    read_auc_coassolo(path)
  })

  auc_su <- reactive({
    path <- if (!is.null(input$auc_su_up)) input$auc_su_up$datapath else default_paths$auc_su
    read_auc_su(path)
  })

  auc_all <- reactive({
    bind_rows(auc_coassolo(), auc_su()) %>%
      filter(!is.na(gene), !is.na(auc)) %>%
      mutate(gene = as.character(gene))
  })

  output$dataset_ui <- renderUI({
    ds <- sort(unique(auc_all()$dataset))
    selectInput("dataset_sel", "Dataset", choices = ds, selected = if (length(ds)) ds[1] else NULL)
  })

  output$gene_ui <- renderUI({
    df <- auc_all()
    req(nrow(df) > 0)
    genes <- sort(unique(df$gene))
    selectizeInput("gene_sel", "Gene", choices = genes, selected = genes[1], options = list(placeholder = "Type to search", maxOptions = 5000))
  })

  output$dir_ui <- renderUI({
    df <- auc_all()
    req(nrow(df) > 0)
    # direction column may be absent for some datasets
    has_dir <- "direction" %in% names(df)
    if (!has_dir) return(NULL)
    selectInput("dir_sel", "Direction filter (if available)", choices = c("All", "Up in disease", "Up in control"), selected = "All")
  })

  gene_row <- reactive({
    df <- auc_all()
    req(nrow(df) > 0, input$dataset_sel, input$gene_sel)
    out <- df %>% filter(dataset == input$dataset_sel, gene == input$gene_sel)
    if (nrow(out) == 0) out <- df %>% filter(gene == input$gene_sel)
    out
  })

  output$auc_plot <- renderPlotly({
    df <- auc_all()
    req(nrow(df) > 0, input$gene_sel)
    gdf <- df %>% filter(gene == input$gene_sel) %>%
      select(dataset, auc) %>%
      distinct() %>%
      arrange(desc(auc))

    p <- ggplot(gdf, aes(x = dataset, y = auc)) +
      geom_col() +
      coord_cartesian(ylim = c(0, 1)) +
      labs(x = NULL, y = "AUC", title = paste0("AUC for ", input$gene_sel, " across datasets")) +
      theme_minimal(base_size = 13)

    plotly::ggplotly(p, tooltip = c("x", "y"))
  })

  output$gene_stats_tbl <- renderDT({
    df <- gene_row()
    req(nrow(df) > 0)
    DT::datatable(
      df %>% select(any_of(c("dataset","gene","auc","avg_log2fc","avg_logfc","direction","p_val","p_val_adj","pct_nash","pct_chow","pct_control","pct_nafld"))),
      options = list(pageLength = 5, scrollX = TRUE)
    )
  })

  top_df <- reactive({
    df <- auc_all()
    req(nrow(df) > 0, input$dataset_sel)
    out <- df %>% filter(dataset == input$dataset_sel)

    # direction filtering if present
    if ("direction" %in% names(out) && !is.null(input$dir_sel) && input$dir_sel != "All") {
      if (input$dir_sel == "Up in disease") out <- out %>% filter(str_detect(tolower(direction), "disease|nash|nafld"))
      if (input$dir_sel == "Up in control") out <- out %>% filter(str_detect(tolower(direction), "control|chow"))
    }

    # significance filtering if available
    if (isTRUE(input$only_sig) && ("p_val_adj" %in% names(out))) {
      out <- out %>% filter(!is.na(p_val_adj), p_val_adj < 0.05)
    }

    # gene search
    if (!is.null(input$gene_search) && nchar(trimws(input$gene_search)) > 0) {
      out <- out %>% filter(str_detect(tolower(gene), tolower(trimws(input$gene_search))))
    }

    out %>% arrange(desc(auc)) %>% head(input$top_n)
  })

  output$top_tbl <- renderDT({
    df <- top_df()
    req(nrow(df) > 0)
    DT::datatable(
      df %>% select(any_of(c("gene","auc","direction","avg_log2fc","avg_logfc","p_val_adj","p_val"))),
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })

  # ---- Knowledge graph ----
  kg_data <- reactive({
    path <- if (!is.null(input$kg_up)) input$kg_up$datapath else default_paths$kg_xlsx
    read_kg_supp6(path)
  })

  kg_drugs_ranked <- reactive({
    kg <- kg_data()
    req(!is.null(kg$nodes))
    nodes <- kg$nodes %>% filter(tolower(type) == "drug") %>%
      mutate(drugbank_accession = as.character(drugbank_accession))
    nash_set <- if (!is.null(kg$nash)) unique(kg$nash$drugbank_accession) else character()
    hep_set  <- if (!is.null(kg$hep))  unique(kg$hep$drugbank_accession)  else character()

    nodes %>%
      mutate(
        in_nash_shortestpaths = drugbank_accession %in% nash_set,
        in_hepaticsteatosis_shortestpaths = drugbank_accession %in% hep_set,
        in_any_shortestpaths = in_nash_shortestpaths | in_hepaticsteatosis_shortestpaths
      ) %>%
      arrange(desc(pagerank_score))
  })

  output$drug_scatter <- renderPlotly({
    df <- kg_drugs_ranked()
    req(nrow(df) > 0)
    keep <- df %>% slice_head(n = input$top_drugs)
    if (isTRUE(input$only_shortestpath)) keep <- keep %>% filter(in_any_shortestpaths)

    p <- ggplot(keep, aes(x = pagerank_score, y = betweenness_score, text = paste0(name, "\n", drugbank_accession))) +
      geom_point() +
      labs(x = "PageRank score", y = "Betweenness score", title = "Top drugs by PageRank (scatter: PageRank vs Betweenness)") +
      theme_minimal(base_size = 13)

    plotly::ggplotly(p, tooltip = "text")
  })

  output$drug_tbl <- renderDT({
    df <- kg_drugs_ranked()
    req(nrow(df) > 0)
    keep <- df %>% slice_head(n = input$top_drugs)
    if (isTRUE(input$only_shortestpath)) keep <- keep %>% filter(in_any_shortestpaths)

    DT::datatable(
      keep %>%
        select(name, drugbank_accession, pagerank_score, betweenness_score, eigen_score, cluster,
               in_nash_shortestpaths, in_hepaticsteatosis_shortestpaths),
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })

  # ---- WGCNA upload ----
  wgcna <- reactive({
    if (is.null(input$wgcna_up)) return(NULL)
    df <- readr::read_csv(input$wgcna_up$datapath, show_col_types = FALSE) %>%
      rename_with(~tolower(gsub("\\s+", "_", .x)))
    df
  })

  output$wgcna_gene_ui <- renderUI({
    df <- wgcna()
    if (is.null(df) || nrow(df) == 0) return(NULL)
    gene_col <- if ("gene" %in% names(df)) "gene" else names(df)[1]
    selectizeInput("wgcna_gene", "Filter by gene", choices = sort(unique(df[[gene_col]])), multiple = TRUE,
                   options = list(placeholder = "Optional", maxOptions = 5000))
  })

  output$wgcna_tbl <- renderDT({
    df <- wgcna()
    if (is.null(df) || nrow(df) == 0) return(DT::datatable(data.frame(Message = "Upload a WGCNA CSV to view results.")))
    gene_col <- if ("gene" %in% names(df)) "gene" else names(df)[1]
    out <- df
    if (!is.null(input$wgcna_gene) && length(input$wgcna_gene) > 0) out <- out %>% filter(.data[[gene_col]] %in% input$wgcna_gene)
    DT::datatable(out, options = list(pageLength = 25, scrollX = TRUE))
  })

  # ---- Drug results upload (generic) ----
  drugres <- reactive({
    if (is.null(input$drugres_up)) return(NULL)
    readr::read_csv(input$drugres_up$datapath, show_col_types = FALSE) %>%
      rename_with(~tolower(gsub("\\s+", "_", .x)))
  })

  output$drugres_ui <- renderUI({
    df <- drugres()
    if (is.null(df) || nrow(df) == 0) return(NULL)
    cols <- names(df)
    tagList(
      selectInput("drugres_sort", "Sort by", choices = cols, selected = cols[1]),
      numericInput("drugres_n", "Show top N rows", value = 100, min = 10, step = 10)
    )
  })

  output$drugres_tbl <- renderDT({
    df <- drugres()
    if (is.null(df) || nrow(df) == 0) return(DT::datatable(data.frame(Message = "Upload a drug results CSV to view and rank candidates.")))
    req(input$drugres_sort, input$drugres_n)
    out <- df
    # try numeric sort desc if possible
    sort_col <- input$drugres_sort
    if (is.numeric(out[[sort_col]])) out <- out %>% arrange(desc(.data[[sort_col]])) else out <- out %>% arrange(.data[[sort_col]])
    out <- out %>% head(input$drugres_n)
    DT::datatable(out, options = list(pageLength = 25, scrollX = TRUE))
  })
}

shinyApp(ui, server)
