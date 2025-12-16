# Meta Liver - Interactive Shiny App for Liver Genomics Analysis
# Integrates WGCNA modules, enrichment analysis, AUC scores, drugs, and PPI networks

library(shiny)
library(shinydashboard)
library(DT)
library(ggplot2)
library(plotly)
library(dplyr)
library(tidyr)
library(igraph)
library(visNetwork)
library(readxl)

# Source data loader
source("data_loader.R")

# Load all data
cat("Loading data...\n")
data <- source("data_loader.R")$value

# Extract components
expr_data <- data$expr_data
traits_data <- data$traits_data
mes_data <- data$mes_data
mod_trait_cor <- data$mod_trait_cor
mod_trait_pval <- data$mod_trait_pval
auc_data <- data$auc_data
ppi_data <- data$ppi_data
drugs_data <- data$drugs_data
enrichment_list <- data$enrichment_list
nodes_list <- data$nodes_list
gene_mapping_list <- data$gene_mapping_list
modules <- data$modules

cat("Data loaded successfully. Starting Shiny app...\n")

# ============================================================================
# UI DEFINITION
# ============================================================================

ui <- dashboardPage(
  dashboardHeader(
    title = "Meta Liver",
    titleWidth = 250,
    tags$head(
      tags$style(HTML("
        .main-header .logo {
          font-weight: bold;
          font-size: 20px;
        }
      "))
    )
  ),
  
  dashboardSidebar(
    width = 250,
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("chart-line")),
      menuItem("Module Explorer", tabName = "modules", icon = icon("project-diagram")),
      menuItem("Gene Analysis", tabName = "genes", icon = icon("dna")),
      menuItem("Drug Discovery", tabName = "drugs", icon = icon("pills")),
      menuItem("Enrichment", tabName = "enrichment", icon = icon("list")),
      menuItem("PPI Network", tabName = "network", icon = icon("network-wired")),
      menuItem("Data Tables", tabName = "tables", icon = icon("table"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # =====================================================================
      # DASHBOARD TAB
      # =====================================================================
      tabItem(tabName = "dashboard",
        fluidRow(
          box(
            title = "Meta Liver Analysis Platform",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            HTML("
              <h4>Welcome to Meta Liver</h4>
              <p>An interactive platform for exploring liver genomics data including:</p>
              <ul>
                <li><b>WGCNA Modules:</b> Co-expression modules from weighted gene co-expression network analysis</li>
                <li><b>Gene Expression:</b> Normalized expression profiles across 201 samples</li>
                <li><b>Enrichment Analysis:</b> Functional annotation of modules (GO terms, CORUM complexes)</li>
                <li><b>AUC Scores:</b> Gene discrimination ability between disease states</li>
                <li><b>Drug Targets:</b> 135 approved drugs with PPI-based targets</li>
                <li><b>PPI Networks:</b> Protein-protein interactions for each module</li>
              </ul>
            ")
          )
        ),
        fluidRow(
          valueBox(
            value = nrow(expr_data),
            subtitle = "Samples",
            icon = icon("flask"),
            color = "blue",
            width = 3
          ),
          valueBox(
            value = ncol(expr_data),
            subtitle = "Genes",
            icon = icon("dna"),
            color = "green",
            width = 3
          ),
          valueBox(
            value = length(modules),
            subtitle = "Modules",
            icon = icon("project-diagram"),
            color = "purple",
            width = 3
          ),
          valueBox(
            value = nrow(drugs_data),
            subtitle = "Drugs",
            icon = icon("pills"),
            color = "red",
            width = 3
          )
        ),
        fluidRow(
          box(
            title = "Module-Trait Correlations",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("dashboard_mod_trait_plot", height = "400px")
          )
        )
      ),
      
      # =====================================================================
      # MODULE EXPLORER TAB
      # =====================================================================
      tabItem(tabName = "modules",
        fluidRow(
          box(
            title = "Select Module",
            status = "primary",
            solidHeader = TRUE,
            width = 3,
            selectInput("module_select", "Module:", choices = modules),
            textOutput("module_info")
          ),
          box(
            title = "Module Statistics",
            status = "info",
            solidHeader = TRUE,
            width = 9,
            tableOutput("module_stats")
          )
        ),
        fluidRow(
          box(
            title = "Module Genes",
            status = "success",
            solidHeader = TRUE,
            width = 12,
            DTOutput("module_genes_table")
          )
        )
      ),
      
      # =====================================================================
      # GENE ANALYSIS TAB
      # =====================================================================
      tabItem(tabName = "genes",
        fluidRow(
          box(
            title = "Gene Search",
            status = "primary",
            solidHeader = TRUE,
            width = 4,
            textInput("gene_search", "Search Gene (ENSG ID or name):", placeholder = "e.g., ENSG00000000003"),
            actionButton("gene_search_btn", "Search", class = "btn-primary"),
            br(), br(),
            textOutput("gene_found_msg")
          ),
          box(
            title = "Gene Information",
            status = "info",
            solidHeader = TRUE,
            width = 8,
            tableOutput("gene_info_table")
          )
        ),
        fluidRow(
          box(
            title = "Expression Profile",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("gene_expr_plot", height = "300px")
          ),
          box(
            title = "AUC Score Information",
            status = "warning",
            solidHeader = TRUE,
            width = 6,
            tableOutput("gene_auc_table")
          )
        )
      ),
      
      # =====================================================================
      # DRUG DISCOVERY TAB
      # =====================================================================
      tabItem(tabName = "drugs",
        fluidRow(
          box(
            title = "Drug Search",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            DTOutput("drugs_table")
          )
        ),
        fluidRow(
          box(
            title = "Drug Statistics",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("drug_zscore_plot", height = "400px")
          ),
          box(
            title = "Distance Distribution",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("drug_distance_plot", height = "400px")
          )
        )
      ),
      
      # =====================================================================
      # ENRICHMENT TAB
      # =====================================================================
      tabItem(tabName = "enrichment",
        fluidRow(
          box(
            title = "Select Module for Enrichment",
            status = "primary",
            solidHeader = TRUE,
            width = 3,
            selectInput("enrichment_module_select", "Module:", choices = modules)
          )
        ),
        fluidRow(
          box(
            title = "Functional Enrichment Results",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            DTOutput("enrichment_table")
          )
        )
      ),
      
      # =====================================================================
      # PPI NETWORK TAB
      # =====================================================================
      tabItem(tabName = "network",
        fluidRow(
          box(
            title = "PPI Network Overview",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            p("Protein-protein interaction network with", nrow(ppi_data), "edges"),
            p("Top interactions by frequency:"),
            DTOutput("ppi_summary_table")
          )
        ),
        fluidRow(
          box(
            title = "Network Statistics",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            tableOutput("network_stats_table")
          )
        )
      ),
      
      # =====================================================================
      # DATA TABLES TAB
      # =====================================================================
      tabItem(tabName = "tables",
        tabsetPanel(
          tabPanel("AUC Scores",
            DTOutput("auc_table")
          ),
          tabPanel("Module-Trait Correlations",
            DTOutput("mod_trait_table")
          ),
          tabPanel("Traits",
            DTOutput("traits_table")
          ),
          tabPanel("Drugs",
            DTOutput("drugs_full_table")
          )
        )
      )
    )
  )
)

# ============================================================================
# SERVER DEFINITION
# ============================================================================

server <- function(input, output, session) {
  
  # =========================================================================
  # DASHBOARD TAB
  # =========================================================================
  
  output$dashboard_mod_trait_plot <- renderPlotly({
    # Prepare data for plotting
    plot_data <- mod_trait_cor %>%
      rownames_to_column("Module") %>%
      rename(Correlation = stage) %>%
      arrange(Correlation)
    
    plot_ly(plot_data, x = ~Correlation, y = ~reorder(Module, Correlation), 
            type = "bar", orientation = "h",
            marker = list(color = ~Correlation, colorscale = "RdBu", 
                         showscale = TRUE, cmin = -1, cmax = 1)) %>%
      layout(title = "Module-Trait Correlations (Disease Stage)",
             xaxis = list(title = "Correlation Coefficient"),
             yaxis = list(title = "Module"),
             height = 400)
  })
  
  # =========================================================================
  # MODULE EXPLORER TAB
  # =========================================================================
  
  output$module_info <- renderText({
    mod <- input$module_select
    n_genes <- length(nodes_list[[mod]])
    paste("Module:", mod, "\nGenes:", n_genes)
  })
  
  output$module_stats <- renderTable({
    mod <- input$module_select
    n_genes <- length(nodes_list[[mod]])
    
    # Get enrichment info
    enrich_df <- enrichment_list[[mod]]
    n_enriched <- nrow(enrich_df)
    
    data.frame(
      Statistic = c("Number of Genes", "Number of Enrichments", 
                   "Module-Trait Correlation", "P-value"),
      Value = c(
        n_genes,
        n_enriched,
        round(mod_trait_cor[paste0("ME", mod), 1], 4),
        round(mod_trait_pval[paste0("ME", mod), 1], 4)
      )
    )
  })
  
  output$module_genes_table <- renderDT({
    mod <- input$module_select
    genes <- nodes_list[[mod]]
    
    # Try to get gene mapping
    gene_map <- gene_mapping_list[[mod]]
    
    if (!is.null(gene_map) && nrow(gene_map) > 0) {
      datatable(gene_map, options = list(pageLength = 10, scrollX = TRUE),
                rownames = FALSE)
    } else {
      datatable(data.frame(Gene = genes), options = list(pageLength = 10),
                rownames = FALSE)
    }
  })
  
  # =========================================================================
  # GENE ANALYSIS TAB
  # =========================================================================
  
  gene_search_result <- reactiveVal(NULL)
  
  observeEvent(input$gene_search_btn, {
    search_term <- input$gene_search
    
    # Search in expression data
    if (search_term %in% rownames(expr_data)) {
      gene_search_result(search_term)
    } else if (search_term %in% colnames(expr_data)) {
      gene_search_result(search_term)
    } else {
      gene_search_result(NULL)
    }
  })
  
  output$gene_found_msg <- renderText({
    if (is.null(gene_search_result())) {
      "Gene not found"
    } else {
      paste("Gene found:", gene_search_result())
    }
  })
  
  output$gene_info_table <- renderTable({
    gene <- gene_search_result()
    if (is.null(gene)) return(NULL)
    
    # Get AUC info if available
    auc_info <- auc_data %>% filter(Gene == gene)
    
    if (nrow(auc_info) > 0) {
      data.frame(
        Property = c("Gene ID", "AUC Score", "% Chow", "% NASH"),
        Value = c(gene, 
                 round(auc_info$AUC[1], 3),
                 round(auc_info$pct_Chow[1], 3),
                 round(auc_info$pct_NASH[1], 3))
      )
    } else {
      data.frame(Property = "Gene ID", Value = gene)
    }
  })
  
  output$gene_expr_plot <- renderPlotly({
    gene <- gene_search_result()
    if (is.null(gene)) return(NULL)
    
    if (gene %in% colnames(expr_data)) {
      expr_vals <- expr_data[, gene]
    } else if (gene %in% rownames(expr_data)) {
      expr_vals <- as.numeric(expr_data[gene, ])
    } else {
      return(NULL)
    }
    
    plot_data <- data.frame(
      Sample = 1:length(expr_vals),
      Expression = expr_vals
    )
    
    plot_ly(plot_data, x = ~Sample, y = ~Expression, type = "scatter", mode = "markers") %>%
      layout(title = paste("Expression Profile:", gene),
             xaxis = list(title = "Sample"),
             yaxis = list(title = "Normalized Expression"))
  })
  
  output$gene_auc_table <- renderTable({
    gene <- gene_search_result()
    if (is.null(gene)) return(NULL)
    
    auc_info <- auc_data %>% filter(Gene == gene)
    if (nrow(auc_info) > 0) {
      auc_info[, 1:5]
    } else {
      data.frame(Message = "No AUC data available for this gene")
    }
  })
  
  # =========================================================================
  # DRUG DISCOVERY TAB
  # =========================================================================
  
  output$drugs_table <- renderDT({
    drugs_clean <- drugs_data %>%
      select(starts_with("Drug") | contains("distance") | contains("z-score") | contains("Target")) %>%
      head(50)
    
    datatable(drugs_clean, options = list(pageLength = 10, scrollX = TRUE),
              rownames = FALSE)
  })
  
  output$drug_zscore_plot <- renderPlotly({
    drugs_plot <- drugs_data %>%
      filter(!is.na(`z-score`)) %>%
      arrange(`z-score`) %>%
      head(20)
    
    plot_ly(drugs_plot, x = ~`z-score`, y = ~`Drug Name`, type = "bar", 
            orientation = "h") %>%
      layout(title = "Top 20 Drugs by Z-Score",
             xaxis = list(title = "Z-Score"),
             yaxis = list(title = "Drug"))
  })
  
  output$drug_distance_plot <- renderPlotly({
    drugs_plot <- drugs_data %>%
      filter(!is.na(distance)) %>%
      arrange(distance) %>%
      head(20)
    
    plot_ly(drugs_plot, x = ~distance, y = ~`Drug Name`, type = "bar",
            orientation = "h", marker = list(color = "steelblue")) %>%
      layout(title = "Top 20 Drugs by Distance",
             xaxis = list(title = "Distance"),
             yaxis = list(title = "Drug"))
  })
  
  # =========================================================================
  # ENRICHMENT TAB
  # =========================================================================
  
  output$enrichment_table <- renderDT({
    mod <- input$enrichment_module_select
    enrich_df <- enrichment_list[[mod]]
    
    if (!is.null(enrich_df) && nrow(enrich_df) > 0) {
      datatable(enrich_df %>% head(100),
                options = list(pageLength = 10, scrollX = TRUE),
                rownames = FALSE)
    } else {
      datatable(data.frame(Message = "No enrichment data available"),
                rownames = FALSE)
    }
  })
  
  # =========================================================================
  # PPI NETWORK TAB
  # =========================================================================
  
  output$ppi_summary_table <- renderDT({
    ppi_summary <- ppi_data %>%
      group_by(prot_pair) %>%
      summarise(Sources = n(), .groups = "drop") %>%
      arrange(desc(Sources)) %>%
      head(20)
    
    datatable(ppi_summary, options = list(pageLength = 10),
              rownames = FALSE)
  })
  
  output$network_stats_table <- renderTable({
    data.frame(
      Statistic = c("Total Interactions", "Unique Proteins (approx)", 
                   "Average Interactions per Protein"),
      Value = c(
        nrow(ppi_data),
        length(unique(c(ppi_data$prot1_hgnc_id, ppi_data$prot2_hgnc_id))),
        round(nrow(ppi_data) / length(unique(c(ppi_data$prot1_hgnc_id, ppi_data$prot2_hgnc_id))), 2)
      )
    )
  })
  
  # =========================================================================
  # DATA TABLES TAB
  # =========================================================================
  
  output$auc_table <- renderDT({
    datatable(auc_data, options = list(pageLength = 10, scrollX = TRUE),
              rownames = FALSE)
  })
  
  output$mod_trait_table <- renderDT({
    mod_trait_combined <- cbind(
      Module = rownames(mod_trait_cor),
      Correlation = round(mod_trait_cor$stage, 4),
      PValue = round(mod_trait_pval$stage, 4)
    ) %>% as.data.frame()
    
    datatable(mod_trait_combined, options = list(pageLength = 10),
              rownames = FALSE)
  })
  
  output$traits_table <- renderDT({
    traits_display <- cbind(
      Sample = rownames(traits_data),
      traits_data
    ) %>% as.data.frame()
    
    datatable(traits_display, options = list(pageLength = 10),
              rownames = FALSE)
  })
  
  output$drugs_full_table <- renderDT({
    datatable(drugs_data, options = list(pageLength = 10, scrollX = TRUE),
              rownames = FALSE)
  })
}

# ============================================================================
# RUN THE APP
# ============================================================================

shinyApp(ui, server)
