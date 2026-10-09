# Shared helpers: logging, config, paths.

log_step <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")
}

timed <- function(label, expr) {
  log_step(label, " ...")
  t0 <- Sys.time()
  res <- force(expr)
  log_step(label, " done (", format(round(difftime(Sys.time(), t0), 1)), ")")
  invisible(res)
}

project_path <- function(...) here::here(...)

# Config paths are relative to the project root unless absolute.
resolve_path <- function(path) {
  if (is.null(path) || !nzchar(path)) return(path)
  if (grepl("^(/|[A-Za-z]:[/\\\\]|~)", path)) return(path.expand(path))
  project_path(path)
}

load_config <- function(file = project_path("config", "config.yml")) {
  yaml::read_yaml(file)
}

# Global defaults overridden by the dataset's own entries.
dataset_config <- function(cfg, name) {
  ds <- cfg$datasets[[name]]
  if (is.null(ds)) stop("Dataset '", name, "' not found in config.yml")
  modifyList(cfg$defaults, ds)
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

write_run_info <- function(out_dir, ds_cfg, extra = list()) {
  info <- c(
    list(run_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), config = ds_cfg),
    extra
  )
  yaml::write_yaml(info, file.path(out_dir, "run_info.yml"))
  writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
}
