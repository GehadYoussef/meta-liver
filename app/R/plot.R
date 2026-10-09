# Small plotly.js wrapper. Replaces the plotly R package, whose ~40 dependencies made the
# browser (Shinylive) version slow to start. Same call style: plot_ly(data, x = ~col, ...).

plotly_dep <- function() {
  htmltools::htmlDependency("plotly.js", "2.34.0",
    src = c(href = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0"), script = "plotly.min.js")
}

plot_output <- function(id, height = "400px") {
  htmltools::tagList(plotly_dep(), shiny::div(id = id, class = "mash-plot", style = sprintf("height:%s", height)))
}

render_plot <- function(expr, env = parent.frame(), quoted = FALSE) {
  func <- shiny::exprToFunction(expr, env, quoted)
  shiny::createRenderFunction(func, function(fig, session, name, ...) fig_json(fig), plot_output)
}

plot_ly <- function(data = NULL, ..., color = NULL, colors = NULL, height = NULL, source = NULL) {
  ev <- function(a) if (inherits(a, "formula")) eval(a[[2]], data, environment(a)) else a
  args <- lapply(list(...), ev)
  color <- ev(color)
  n <- max(0, vapply(args[intersect(names(args), c("x", "y"))], length, 1L))
  layout <- list()
  for (ax in c("x", "y")) if (is.factor(args[[ax]])) {
    layout[[paste0(ax, "axis")]] <- list(categoryorder = "array", categoryarray = I(levels(args[[ax]])))
  }
  if (!is.null(height)) layout$height <- height
  groups <- if (is.null(color)) list(seq_len(n)) else split(seq_len(n), factor(color))
  traces <- lapply(names(groups) %||% "", function(g) {
    i <- if (is.null(color)) seq_len(n) else groups[[g]]
    tr <- lapply(args, function(a) subset_arg(a, i, n))
    for (k in intersect(names(tr), c("marker", "line", "error_x", "error_y"))) {
      tr[[k]] <- lapply(tr[[k]], function(a) subset_arg(a, i, n, sub = TRUE))
    }
    if (!is.null(color)) {
      col <- if (!is.null(names(colors))) colors[[g]] else colors[[match(g, names(groups))]]
      tr$name <- g
      tr$marker <- utils::modifyList(tr$marker %||% list(), list(color = col))
    }
    tr
  })
  structure(list(data = traces, layout = layout, config = list()), class = "mash_plot")
}

# Per-point vectors are subset to the trace and wrapped in I() so they stay JSON arrays
subset_arg <- function(a, i, n, sub = FALSE) {
  if (is.list(a) || is.null(a) || is.matrix(a)) return(a)
  if (length(a) == n && n > 0) return(I(as_json_vec(a[i])))
  if (!sub && length(a) > 1) return(I(as_json_vec(a)))
  a
}

as_json_vec <- function(a) unname(if (is.factor(a)) as.character(a) else a)

plot_layout <- function(fig, ...) {
  fig$layout <- utils::modifyList(fig$layout, list(...))
  fig
}

plot_config <- function(fig, ...) {
  fig$config <- utils::modifyList(fig$config, list(...))
  fig
}

add_annotations <- function(fig, x, y, text, ...) {
  extra <- list(...)
  new <- lapply(seq_along(text), function(k) c(list(x = x[k], y = y[k], text = text[k]), extra))
  fig$layout$annotations <- c(fig$layout$annotations, new)
  fig
}

fig_json <- function(fig) unclass(fig)
