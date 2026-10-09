# Build the browser version of the app (Shinylive) into site/, for the gh-pages branch.
shinylive::export(here::here("app"), here::here("site"))
file.create(here::here("site", ".nojekyll"))
