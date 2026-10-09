# Small HTML table output with search, sorting and pages. Replaces reactable, whose
# htmlwidgets dependencies (rmarkdown, knitr, ...) slowed the browser version's start-up.

col_def <- function(name = NULL, cell = NULL, html = FALSE, align = NULL, minWidth = NULL, maxWidth = NULL,
                    style = NULL, format = NULL) {
  structure(list(name = name, cell = cell, html = html, align = align, minWidth = minWidth,
                 maxWidth = maxWidth, style = style, format = format), class = "col_def")
}

col_format <- function(digits = NULL, separators = FALSE) list(digits = digits, separators = separators)

table_output <- function(id) shiny::div(id = id, class = "mash-table")

render_table <- function(expr, env = parent.frame(), quoted = FALSE) {
  func <- shiny::exprToFunction(expr, env, quoted)
  shiny::createRenderFunction(func, function(x, session, name, ...) x, table_output)
}

css_style <- function(style) {
  if (is.null(style)) return("")
  paste0(gsub("([A-Z])", "-\\L\\1", names(style), perl = TRUE), ":", unlist(style), collapse = ";")
}

format_values <- function(v, def) {
  if (!is.null(def$cell)) {
    out <- vapply(v, function(x) as.character(def$cell(x)), "")
    return(if (isTRUE(def$html)) out else esc(out))
  }
  if (is.numeric(v)) {
    f <- def$format
    out <- if (!is.null(f$digits)) formatC(v, format = "f", digits = f$digits, big.mark = if (isTRUE(f$separators)) "," else "")
           else if (isTRUE(f$separators)) formatC(v, format = "d", big.mark = ",")
           else ifelse(!is.na(v) & v == round(v), formatC(v, format = "d", big.mark = ""), formatC(v, digits = 3, format = "g"))
    out[is.na(v)] <- ""
    return(out)
  }
  out <- ifelse(is.na(v), "", as.character(v))
  if (isTRUE(def$html)) out else esc(out)
}

# Columns of cell HTML are built here, so the browser only pages, sorts and filters
rt <- function(df, columns = list(), page = 10, searchable = TRUE) {
  cols <- lapply(names(df), function(n) {
    def <- columns[[n]] %||% col_def()
    v <- df[[n]]
    num <- is.numeric(v) || is.logical(v)
    list(name = def$name %||% n, align = def$align %||% if (num) "right" else "left",
         style = paste0(css_style(def$style),
                        sprintf(";min-width:%dpx", def$minWidth %||% 70),
                        if (!is.null(def$maxWidth)) sprintf(";max-width:%dpx", def$maxWidth) else ""),
         cells = I(unname(format_values(v, def))),
         sort = I(unname(if (num) as.numeric(v) else tolower(gsub("<[^>]+>", "", as.character(v))))))
  })
  text <- if (nrow(df)) do.call(paste, lapply(cols, function(c) tolower(gsub("<[^>]+>", "", c$cells)))) else character()
  list(cols = cols, text = I(text), page = page, searchable = searchable)
}
