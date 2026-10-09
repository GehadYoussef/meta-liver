# Server-logic tests for the Shiny app modules.
app_env <- function() {
  env <- new.env()
  suppressWarnings(suppressPackageStartupMessages({
    for (f in list.files(here::here("app", "R"), pattern = "[.]R$", full.names = TRUE)) sys.source(f, envir = env)
  }))
  env$d <- readRDS(here::here("app", "data", "app_data.rds"))
  env
}
env <- app_env()
d <- env$d

test_that("every page UI builds without error", {
  for (f in c("overviewUI", "markersUI", "consistencyUI", "bulkUI", "drugsUI", "modulesUI", "aboutUI")) {
    expect_s3_class(env[[f]]("x", d), "shiny.tag.list")
  }
  expect_s3_class(env$geneUI("x"), "shiny.tag.list")
})

test_that("direction filter returns genes for every dataset", {
  for (ds in d$datasets_meta$dataset) {
    shiny::testServer(env$markersServer, args = list(d = d), {
      session$setInputs(dataset = ds, dir = "up_in_disease", min_pct = 0.1, fdr = FALSE,
                        targets = FALSE, search = "", top_n = 50)
      expect_gt(nrow(top()), 0, label = paste(ds, "up"))
      session$setInputs(dir = "down_in_disease")
      expect_gt(nrow(top()), 0, label = paste(ds, "down"))
    })
  }
})

test_that("down-regulated genes are ranked strongest first", {
  shiny::testServer(env$markersServer, args = list(d = d), {
    session$setInputs(dataset = "su_GSE166504", dir = "down_in_disease", min_pct = 0.1,
                      fdr = FALSE, targets = FALSE, search = "", top_n = 20)
    x <- top()
    expect_true(all(x$auc < 0.5))
    expect_equal(x$auc, sort(x$auc))
  })
})

test_that("no untested gene is shown with AUC 0", {
  expect_false(any(d$sc$auc == 0 & !d$sc$tested, na.rm = TRUE))
  expect_true(all(is.na(d$sc$auc[!d$sc$tested])))
})

test_that("shortest-path filter is applied before top N", {
  shiny::testServer(env$drugsServer, args = list(d = d), {
    session$setInputs(kg_n = 200, kg_sp = "paths")
    expect_equal(nrow(kg_sel()), 200)
    expect_true(all(kg_sel()$in_any_shortest_paths))
    session$setInputs(ppi_z = -2, ppi_named = "all")
    expect_false(anyNA(ppi_sel()$drug_id))   # drugs without a z-score must not become empty rows
  })
})

test_that("gene page gathers evidence for a gene present in all sources", {
  shiny::testServer(env$geneServer, args = list(d = d), {
    session$setInputs(gene = "CYP2E1")
    expect_gt(nrow(gene_sc()), 0)
    expect_gt(nrow(gene_bulk()), 0)
    expect_type(output$strip, "list")
    session$setInputs(pick = "SREBF1")   # quick-pick chip
  })
})

test_that("direction consistency filters work", {
  shiny::testServer(env$consistencyServer, args = list(d = d), {
    k <- length(d$consistency$datasets)
    session$setInputs(set = "all", cat = "consistent", min = k, species = FALSE, bulk = FALSE, version = "v2")
    expect_true(all(genes()$n_up == k | genes()$n_down == k))
    session$setInputs(cat = "conflicting", min = 2)
    expect_true(all(genes()$n_up > 0 & genes()$n_down > 0))
    session$setInputs(cat = "consistent", species = TRUE)
    w <- genes()
    expect_true(all(w$mouse == w$human))
  })
})

test_that("bulk and modules pages respond to their controls", {
  shiny::testServer(env$bulkServer, args = list(d = d), {
    session$setInputs(contrast = "Stage 2 vs 0", lfc = 1)
    expect_true(all(rows()$contrast == "Stage 2 vs 0"))
  })
  shiny::testServer(env$modulesServer, args = list(d = d), {
    expect_true(module() %in% d$wgcna$trait$module)
  })
})

test_that("every plot and table renders without error", {
  render <- function(server, inputs, outputs) {
    shiny::testServer(server, args = list(d = d), {
      do.call(session$setInputs, inputs)
      # validate()/need() empty states are expected, other errors are bugs
      for (o in outputs) expect_error(tryCatch(output[[o]], shiny.silent.error = function(e) NULL), NA, label = o)
    })
  }
  render(env$overviewServer, list(), "heatmap")
  render(env$geneServer, list(gene = "SREBF1"), c("header", "strip", "sc_plot", "bulk_plot", "table"))
  render(env$markersServer, list(dataset = "wang_human_GSE212837", dir = "both", min_pct = 0.1, fdr = FALSE,
                                 targets = FALSE, search = "", top_n = 50), c("note", "bars", "plot", "table"))
  render(env$consistencyServer, list(set = "all", cat = "consistent", min = 2, species = FALSE, bulk = FALSE,
                                     version = "v2"), c("bulk_bars", "heatmap", "count", "matrix", "table", "compare"))
  render(env$bulkServer, list(contrast = "NASH F3 vs control", lfc = 1), c("title", "table_title", "volcano", "table"))
  render(env$drugsServer, list(kg_n = 200, kg_sp = "all", ppi_z = -2, ppi_named = "all"),
         c("kg_plot", "kg_table", "ppi_plot", "ppi_table"))
  render(env$modulesServer, list(), c("bars", "title", "enrich", "genes"))
  render(env$aboutServer, list(), c("datasets", "comparison", "sensitivity"))
})

test_that("table cell helpers produce valid HTML", {
  expect_match(env$bar_html(0.37, 1, "#000", "0.37"), "width:37%;", fixed = TRUE)
  expect_match(env$col_effect()$cell(0.72), "\u25B2 0.72", fixed = TRUE)
  expect_match(env$col_effect()$cell(0.30), "\u25BC 0.30", fixed = TRUE)
  expect_equal(env$col_call("x")$cell(NA), "")                       # not tested = blank
  expect_match(env$col_name_id("x")$cell("A&B|ID1"), "A&amp;B", fixed = TRUE)   # escaped
})

# Features ported from Meta Liver

test_that("new pages and outputs render", {
  render <- function(server, inputs, outputs) {
    shiny::testServer(server, args = list(d = d), {
      do.call(session$setInputs, inputs)
      # validate()/need() empty states are expected, other errors are bugs
      for (o in outputs) expect_error(tryCatch(output[[o]], shiny.silent.error = function(e) NULL), NA, label = o)
    })
  }
  for (f in c("screenerUI", "invitroUI")) expect_s3_class(env[[f]]("x", d), "shiny.tag.list")
  render(env$geneServer, list(gene = "PLIN2"),
         c("summary", "strip", "forest", "invitro", "ppi", "cluster", "drugs", "table"))
  render(env$screenerServer, list(direction = "any", same_direction = TRUE, use_sc = TRUE, sc_min_evidence = 0.3,
                                  sc_min_n = 2, use_bulk = TRUE, bulk_contrast = "MASH vs control", bulk_fdr = 0.05,
                                  bulk_min_studies = 2, bulk_consistent = TRUE, max_n = 50), c("count", "table"))
  render(env$invitroServer, list(contrast = "OA+PA + resistin/myostatin + PBMC", line = "1b", lfc = 1),
         c("steps", "agree", "title", "volcano", "table"))
  render(env$bulkServer, list(contrast = "NASH F3 vs control", meta_contrast = "MASH vs control",
                              meta_min_studies = 2, meta_consistent = TRUE), c("meta_volcano", "meta_title", "meta_table"))
  render(env$drugsServer, list(drug = "DB00157", kg_n = 200, kg_sp = "all", ppi_z = -2, ppi_named = "all"),
         c("drug_card", "active_table"))
})

test_that("screener ranks before cutting and enforces direction agreement", {
  m <- env$gene_master(d)
  crit <- list(direction = "any", same_direction = TRUE, sc = list(min_evidence = 0.3, min_n = 2, require_pb = FALSE),
               bulk = list(contrast = "MASH vs control", max_fdr = 0.05, min_studies = 2, consistent = TRUE))
  r <- env$run_screener(m, d, crit, 10)
  expect_equal(nrow(r$table), 10)
  expect_gt(r$n_total, 10)
  score <- r$table$evidence + pmin(1, -log10(pmax(r$table$bulk_fdr, 1e-300)) / 10)
  expect_equal(score, sort(score, decreasing = TRUE))                     # strongest first
  expect_true(all(r$table$sc_dir == sign(r$table$bulk_log2FC)))          # layers agree
  crit$direction <- "down"
  expect_true(all(env$run_screener(m, d, crit, 500)$table$sc_dir == -1))
})

test_that("gene narrative handles conflicting and missing evidence", {
  saa2 <- env$gene_narrative(env$gene_evidence("SAA2", d))
  expect_match(saa2$rows$single_cell$text, "Datasets disagree", fixed = TRUE)
  expect_equal(saa2$rows$single_cell$strength, "very low")
  none <- env$gene_narrative(env$gene_evidence("NOT_A_GENE", d))
  expect_match(none$headline, "not clearly changed", fixed = TRUE)
  expect_type(none$copy, "character")
})

test_that("plot_ly builds one trace per colour group, keeps factor order and single points as arrays", {
  df <- data.frame(a = c(1, 2, 3), b = factor(c("z", "y", "z"), levels = c("z", "y")), g = c("p", "q", "p"))
  fig <- env$plot_ly(df, x = ~a, y = ~b, color = ~g, colors = c(p = "red", q = "blue"), type = "bar",
                     marker = list(size = c(5, 6, 7)))
  expect_length(fig$data, 2)
  expect_equal(fig$data[[2]]$marker$color, "blue")
  expect_equal(as.vector(fig$data[[1]]$marker$size), c(5, 7))
  expect_equal(as.vector(fig$layout$yaxis$categoryarray), c("z", "y"))
  json <- shiny:::toJSON(env$plot_ly(data.frame(a = 1), x = ~a, y = ~a)$data)
  expect_match(json, '"x":[1]', fixed = TRUE)
  named <- env$plot_ly(data.frame(a = 1:2), x = ~a, y = ~a, marker = list(color = c(p = "red", q = "blue")))
  expect_match(shiny:::toJSON(named$data), '"color":["red","blue"]', fixed = TRUE)
})
