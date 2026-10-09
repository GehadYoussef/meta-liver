# Palette, theme and shared UI, plot and table helpers.

PAL <- list(
  primary = "#0F9F8F",
  up      = "#E05263",   # higher in NASH
  down    = "#2F80C3",   # lower in NASH
  none    = "#98A2B3",   # no change
  na      = "#D0D5DD",   # not tested
  warn    = "#E5A13A",
  ink     = "#101828",
  muted   = "#667085"
)

DIR_INFO <- list(
  up_in_disease   = list(label = "Up in NASH",   color = PAL$up,   icon = "arrow-up"),
  down_in_disease = list(label = "Down in NASH", color = PAL$down, icon = "arrow-down"),
  no_change       = list(label = "No change",    color = PAL$none, icon = "minus"),
  not_tested      = list(label = "Not detected", color = PAL$na,   icon = "circle")
)

app_theme <- function() {
  bslib::bs_theme(
    version = 5,
    primary = PAL$primary, bg = "#FFFFFF", fg = PAL$ink,
    base_font = bslib::font_collection("Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"),
    heading_font = bslib::font_collection("Inter", "Segoe UI", "system-ui", "sans-serif"),
    "border-radius" = "0.75rem", "font-size-base" = "0.95rem"
  )
}

page_header <- function(icon, title, subtitle = NULL) {
  shiny::div(class = "page-header",
             shiny::div(class = "ph-icon", shiny::icon(icon)),
             shiny::div(shiny::h1(title), if (!is.null(subtitle)) shiny::p(subtitle)))
}

# Ring gauge for a fraction in [0, 1]
ring_svg <- function(frac, color = PAL$primary, size = 58, label = NULL) {
  r <- 22
  circ <- 2 * pi * r
  f <- if (is.na(frac)) 0 else max(0, min(1, frac))
  label <- label %||% (if (is.na(frac)) "n/a" else sprintf("%.0f%%", 100 * frac))
  htmltools::HTML(sprintf(
    '<svg width="%d" height="%d" viewBox="0 0 56 56" aria-label="%s">
       <circle cx="28" cy="28" r="%d" fill="none" stroke="rgba(16,24,40,.08)" stroke-width="6"/>
       <circle cx="28" cy="28" r="%d" fill="none" stroke="%s" stroke-width="6" stroke-linecap="round"
               stroke-dasharray="%.1f %.1f" transform="rotate(-90 28 28)"/>
       <text x="28" y="32.5" text-anchor="middle" font-size="12.5" font-weight="800" fill="currentColor">%s</text>
     </svg>', size, size, label, r, r, color, f * circ, circ, label))
}

# Components

chip <- function(text, icon = NULL, class = "") {
  shiny::span(class = paste("chip", class), if (!is.null(icon)) shiny::icon(icon), text)
}

info_tip <- function(text) {
  bslib::tooltip(shiny::span(class = "info-tip", shiny::icon("circle-info")), text)
}

card_title <- function(title, icon = NULL, tip = NULL) {
  shiny::div(class = "card-title-row",
             if (!is.null(icon)) shiny::icon(icon, class = "card-title-icon"),
             shiny::span(title), if (!is.null(tip)) info_tip(tip))
}

filter_popover <- function(..., title = "Filters") {
  bslib::popover(
    shiny::tags$button(type = "button", class = "btn btn-gear", title = title,
                       shiny::icon("sliders"), shiny::span(class = "ms-1", "More filters")),
    ..., title = title, placement = "bottom"
  )
}

# Radio buttons styled as pills in styles.css
pills <- function(id, choices, selected = NULL) {
  shiny::div(class = "pills", shiny::radioButtons(id, NULL, choices = choices, selected = selected, inline = TRUE))
}

species_icon <- function(species) shiny::icon(if (identical(species, "human")) "user" else "paw")

# Plots

plot_style <- function(p, xaxis = list(), yaxis = list(), legend = list(), ...) {
  axis_default <- list(gridcolor = "rgba(0,0,0,0.06)", zerolinecolor = "rgba(0,0,0,0.15)", automargin = TRUE)
  p |>
    plot_layout(
      font = list(family = "Inter, Segoe UI, system-ui, sans-serif", size = 12, color = PAL$muted),
      paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
      xaxis = utils::modifyList(axis_default, xaxis),
      yaxis = utils::modifyList(axis_default, yaxis),
      legend = utils::modifyList(list(orientation = "h", y = -0.2, font = list(size = 11)), legend),
      margin = list(l = 10, r = 10, t = 10, b = 10),
      hoverlabel = list(bgcolor = "#FFFFFF", bordercolor = "rgba(0,0,0,0.1)", font = list(color = PAL$ink)),
      ...
    ) |>
    plot_config(displayModeBar = FALSE)
}

vline <- function(x) list(type = "line", x0 = x, x1 = x, y0 = 0, y1 = 1, yref = "paper",
                          line = list(dash = "dot", color = "rgba(0,0,0,0.35)", width = 1))
hline <- function(y) list(type = "line", y0 = y, y1 = y, x0 = 0, x1 = 1, xref = "paper",
                          line = list(dash = "dot", color = "rgba(0,0,0,0.35)", width = 1))

fmt_p <- function(p) ifelse(is.na(p), "n/a", formatC(p, format = "g", digits = 2))
fmt_num <- function(x, d = 2) ifelse(is.na(x), "n/a", formatC(x, format = "f", digits = d))

module_hex <- function(m) {
  vapply(m, function(x) tryCatch(grDevices::rgb(t(grDevices::col2rgb(x)), maxColorValue = 255),
                                 error = function(e) "#999999"), "")
}

# Tables

rt <- function(df, columns = list(), page = 10, searchable = TRUE, ...) {
  reactable::reactable(
    df, columns = columns, defaultPageSize = page, searchable = searchable,
    compact = TRUE, highlight = TRUE, borderless = TRUE, striped = FALSE,
    showPageSizeOptions = FALSE, paginationType = "simple",
    defaultColDef = reactable::colDef(headerClass = "rt-head", minWidth = 70),
    theme = reactable::reactableTheme(
      headerStyle = list(color = PAL$muted, fontWeight = 600, fontSize = "0.78rem",
                         textTransform = "uppercase", letterSpacing = "0.03em",
                         borderBottom = "1px solid rgba(0,0,0,0.08)"),
      rowHighlightStyle = list(background = "rgba(15,118,110,0.06)"),
      searchInputStyle = list(borderRadius = "999px", width = "220px", fontSize = "0.85rem")
    ),
    ...
  )
}

# Cells are plain HTML strings because building tags per row makes large tables slow.
esc <-function(x) htmltools::htmlEscape(as.character(x))
ARROW <- c(up_in_disease = "\u25B2", down_in_disease = "\u25BC", no_change = "\u2013", not_tested = "")

pill_html <- function(direction, text) {
  info <- DIR_INFO[[direction]] %||% DIR_INFO$not_tested
  sprintf('<span class="dir-pill" style="--c:%s">%s %s</span>', info$color, ARROW[[direction]] %||% "", esc(text))
}

bar_html <- function(value, max = 1, color = PAL$primary, label = NULL, thin = FALSE) {
  if (is.na(value)) return("")
  w <- max(0, min(100, 100 * value / max))
  sprintf('<div class="bar-cell"><div class="bar-track%s"><div class="bar-fill" style="width:%.0f%%;background:%s"></div></div><span class="bar-num">%s</span></div>',
          if (thin) " thin" else "", w, color, label)
}

col_bar <- function(name, max = 1, color = PAL$primary, digits = 2, width = 130) {
  reactable::colDef(name = name, minWidth = width, align = "left", html = TRUE, cell = function(value) {
    bar_html(value, max, color, if (is.na(value)) "" else formatC(value, format = "f", digits = digits))
  })
}

col_dir <- function(name = "Direction") {
  reactable::colDef(name = name, minWidth = 120, html = TRUE, cell = function(value) {
    pill_html(value, (DIR_INFO[[value]] %||% DIR_INFO$not_tested)$label)
  })
}

col_call <- function(name) {
  reactable::colDef(name = name, align = "center", minWidth = 70, html = TRUE, cell = function(value) {
    if (is.na(value)) return('')
    key <- switch(value, up = "up_in_disease", down = "down_in_disease", none = "no_change", "no_change")
    sprintf('<span style="color:%s;font-weight:700">%s</span>', DIR_INFO[[key]]$color, ARROW[[key]])
  })
}

col_num <- function(name, digits = 2) {
  reactable::colDef(name = name, format = reactable::colFormat(digits = digits), align = "right")
}
col_p <- function(name) {
  reactable::colDef(name = name, align = "right", cell = function(value) fmt_p(value))
}
col_check <- function(name) {
  reactable::colDef(name = name, align = "center", html = TRUE, cell = function(value) {
    if (isTRUE(value)) sprintf('<span style="color:%s;font-weight:700">\u2713</span>', PAL$primary) else ""
  })
}

# AUC in a pill coloured by its direction
col_effect <- function(name = "Effect (AUC)", delta = 0.05) {
  reactable::colDef(name = name, minWidth = 105, align = "left", html = TRUE, cell = function(value) {
    if (is.na(value)) return('<span class="call-na">n/a</span>')
    dir <- if (value >= 0.5 + delta) "up_in_disease" else if (value <= 0.5 - delta) "down_in_disease" else "no_change"
    pill_html(dir, formatC(value, format = "f", digits = 2))
  })
}

# Value is "nash|control" detection
col_detect <- function(name = "Detected (NASH / control)") {
  reactable::colDef(name = name, minWidth = 150, html = TRUE, cell = function(value) {
    v <- as.numeric(strsplit(value, "|", fixed = TRUE)[[1]])
    sprintf('<div class="dual-bar">%s%s</div>',
            bar_html(v[1], 1, PAL$up, sprintf("%.0f%%", 100 * v[1]), thin = TRUE),
            bar_html(v[2], 1, PAL$down, sprintf("%.0f%%", 100 * v[2]), thin = TRUE))
  })
}

# Lower triangle only, so each pair of datasets shows once
agreement_heatmap <- function(pairwise, ds, labels, hover = TRUE) {
  z <- matrix(NA_real_, length(ds), length(ds), dimnames = list(ds, ds))
  n <- z
  for (k in seq_len(nrow(pairwise))) {
    a <- pairwise$dataset_a[k]
    b <- pairwise$dataset_b[k]
    if (a %in% ds && b %in% ds) {
      z[a, b] <- pairwise$agreement[k]
      n[a, b] <- pairwise$n_genes[k]
    }
  }
  z[upper.tri(z, diag = TRUE)] <- NA
  rows <- -1
  cols <- -length(ds)
  z <- z[rows, cols, drop = FALSE]
  n <- n[rows, cols, drop = FALSE]
  yl <- labels[rows]
  xl <- labels[cols]
  txt <- matrix(sprintf("%s vs %s<br>%s go the same way<br>%s genes", rep(yl, length(xl)), rep(xl, each = length(yl)),
                        ifelse(is.na(z), "n/a", sprintf("%.0f%%", 100 * z)), n), length(yl))
  plot_ly(x = xl, y = yl, z = z, type = "heatmap", zmin = 0, zmax = 1, showscale = FALSE,
                  # amber = disagree, grey = chance, teal = agree
                  colorscale = list(c(0, "#E5A13A"), c(0.35, "#F6DDB4"), c(0.5, "#EEF1F4"),
                                    c(0.7, "#9FDCD3"), c(1, "#0F9F8F")),
                  text = txt, hoverinfo = if (hover) "text" else "none", xgap = 4, ygap = 4) |>
    add_annotations(x = rep(xl, each = length(yl)), y = rep(yl, length(xl)),
                            text = ifelse(is.na(as.vector(z)), "", sprintf("%.0f%%", 100 * as.vector(z))),
                            showarrow = FALSE, font = list(size = 15, color = PAL$ink, family = "Inter")) |>
    plot_style(xaxis = list(showgrid = FALSE, fixedrange = TRUE, automargin = TRUE),
               yaxis = list(showgrid = FALSE, autorange = "reversed", fixedrange = TRUE, automargin = TRUE))
}

# Value is "name|id"
col_name_id <- function(name, min_width = 170) {
  reactable::colDef(name = name, minWidth = min_width, html = TRUE, cell = function(value) {
    v <- strsplit(value, "|", fixed = TRUE)[[1]]
    id <- if (length(v) > 1 && nzchar(v[2]) && v[2] != v[1]) sprintf('<div class="cell-id">%s</div>', esc(v[2])) else ""
    sprintf('<div class="cell-name">%s</div>%s', esc(v[1]), id)
  })
}

chip_html <- function(text, class = "") sprintf('<span class="chip %s">%s</span>', class, esc(text))
