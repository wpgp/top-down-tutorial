# loads package dependencies and functions

# install packages
pkgs <- c('sf','raster','fasterize', 'tictoc', 'tidyverse', 'exactextractr', 'data.table')
for(pkg in pkgs[!pkgs %in% installed.packages()]){
  install.packages(pkg)
}
rm(pkg, pkgs)

# functions
for(fun in list.files(file.path(dirname(rstudioapi::getSourceEditorContext()$path),'R'))){
  source(file.path(dirname(rstudioapi::getSourceEditorContext()$path),'R',fun))
}
rm(fun)

