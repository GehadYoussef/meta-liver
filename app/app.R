# MASH Omics Explorer. Run with shiny::runApp("app"). Pages open from ?page=gene etc.

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
})

d <- readRDS(file.path("data", "app_data.rds"))

PAGES <- data.frame(
  id      = c("overview", "gene", "screener", "markers", "consistency", "bulk", "invitro", "drugs", "modules", "about"),
  label   = c("Overview", "Gene lookup", "Gene screener", "Markers", "Consistency", "Whole liver", "In-vitro model",
              "Drugs", "Modules", "About"),
  icon    = c("house", "magnifying-glass", "filter", "ranking-star", "arrows-up-down", "flask", "vial", "capsules",
              "diagram-project", "circle-info"),
  section = c("Explore", "Explore", "Explore", "Explore", "Evidence", "Evidence", "Evidence", "Evidence", "Evidence", "Reference")
)

rail <- function() {
  items <- lapply(unique(PAGES$section), function(sec) {
    p <- PAGES[PAGES$section == sec, ]
    tagList(div(class = "rail-section", sec),
            lapply(seq_len(nrow(p)), function(i) tags$button(
              type = "button", class = "rail-item", `data-page` = p$id[i],
              onclick = "mashGo(this.dataset.page)",
              icon(p$icon[i]), span(class = "rail-label", p$label[i]))))
  })
  tags$aside(class = "rail",
    div(class = "rail-brand", div(class = "rail-logo", icon("dna")),
        div(class = "rail-name", "MASH Omics", tags$small("Explorer"))),
    items,
    div(class = "rail-spacer"),
    div(class = "rail-foot", span(sprintf("Data %s", substr(d$built_at, 1, 10))),
        input_dark_mode(id = "dark_mode", mode = "light")))
}

ui <- page(
  title = "MASH Omics Explorer",
  theme = app_theme(),
  tags$head(
    tags$link(rel = "preconnect", href = "https://fonts.googleapis.com"),
    tags$link(rel = "stylesheet",
              href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"),
    tags$link(rel = "stylesheet", href = "styles.css"),
    tags$script(HTML("
      window.mashGo = function(p) {
        if (!document.querySelector('.rail-item[data-page=\"' + p + '\"]')) p = 'overview';
        document.querySelectorAll('.rail-item').forEach(function(b) { b.classList.toggle('active', b.dataset.page === p); });
        if (window.Shiny && Shiny.setInputValue) Shiny.setInputValue('page', p, {priority: 'event'});
        history.replaceState(null, '', '?page=' + p);
        window.scrollTo({top: 0});
      };
      $(document).on('shiny:connected', function() {
        var p = (new URLSearchParams(location.search).get('page') || 'overview').toLowerCase();
        mashGo(p);
      });"))
  ),
  div(class = "app-shell",
    rail(),
    tags$main(class = "main",
      navset_hidden(
        id = "pages",
        nav_panel_hidden("overview", overviewUI("overview", d)),
        nav_panel_hidden("gene", geneUI("gene")),
        nav_panel_hidden("screener", screenerUI("screener", d)),
        nav_panel_hidden("markers", markersUI("markers", d)),
        nav_panel_hidden("consistency", consistencyUI("consistency", d)),
        nav_panel_hidden("bulk", bulkUI("bulk", d)),
        nav_panel_hidden("invitro", invitroUI("invitro", d)),
        nav_panel_hidden("drugs", drugsUI("drugs", d)),
        nav_panel_hidden("modules", modulesUI("modules", d)),
        nav_panel_hidden("about", aboutUI("about", d))
      )
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$page, nav_select("pages", input$page))
  overviewServer("overview", d)
  geneServer("gene", d)
  screenerServer("screener", d)
  markersServer("markers", d)
  consistencyServer("consistency", d)
  bulkServer("bulk", d)
  invitroServer("invitro", d)
  drugsServer("drugs", d)
  modulesServer("modules", d)
  aboutServer("about", d)
}

shinyApp(ui, server)
