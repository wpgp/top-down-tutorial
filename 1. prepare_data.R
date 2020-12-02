# cleanup
rm(list=ls()); gc(); cat("\014"); try(dev.off(), silent=T)

try(source(file.path(dirname(rstudioapi::getSourceEditorContext()$path),'0. setup.R')))

# 1. Set-up ---------------------------------------------------------------

# working directory
local <- T
if(local){
  setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path),'wd'))
} else {
  setwd("//worldpop.files.soton.ac.uk/worldpop/Projects/WP517763_GRID3/Working/git/top-down-tutorial")
}
dir.create('in', showWarnings=F)
dir.create('out', showWarnings=F)

data_path <- file.path(getwd(), "in")
output_path <- file.path(getwd(), "out")

# copy source data
copyWP(srcdir = 'E:/WorldPop/Projects/WP517763_GRID3/Working/git/top-down-tutorial/in',
       outdir = data_path,
       local=T)

# load packages
library(sf) # manipulating vector GIS file
library(raster) # manipulating raster GIS file
library(fasterize) # fast rasterization
library(tictoc) # compute running time
library(tidyverse) # manipulating dataframes
library(exactextractr) # fast zonal statistics
library(data.table) # fast dataframe writing

##-- load data --##

# Boundaries from IGBE 2019
admin3_poly <- st_read(file.path(data_path, 'adminBoundaries/admin_municipality.gpkg'), stringsAsFactors = F)

# Mastergrid from WorldPop
masterGrid <- raster(file.path(data_path, "masterGrid.tif"))

# Population data from IGBE projections 2020
# https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?view=detalhes&id=293420
pop_admin3 <- read.csv(file.path(data_path, 'population/pop_projections_2020_municipality.csv'))

# EA boundaries


# IBGE census EAs(public)
# http://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/2019/Malha_de_setores_(shp)_Brasil/
EA_poly <- st_read(file.path(data_path, 'censusEAs/BR_Setores_2019.shp'), stringsAsFactors = F)



# 2. Create covariates dataset  ---------------------------------------------------


# Create a raster stack of all covariates
raster_names <- list.files(file.path(data_path, "covariates/"), pattern="tif$", full.names = T)
raster_stack <- stack(raster_names) # 10 rasters


# Extract zonal statistics for every EA 
tic() # 1h 
cov_EA <- exact_extract(raster_stack, EA_poly, fun='mean', progress=T, force_df=T, stack_apply=T)
toc() 

cov_EA$EA_id <- EA_poly$CD_SETOR

# Extract zonal statistics for every municipality 
tic() # 22 min
cov_admin3 <- exact_extract(raster_stack, admin3_poly, fun='mean', progress=T, force_df=T, stack_apply=T)
toc()
cov_admin3$geo_code <- admin3_poly$CD_GEOCODI


# 3. Create training dataset ----------------------------------------------

# area of each municipality (admin3)
admin3_poly$area <- st_area(admin3_poly) # m2

# create data.frame
master_train <- cov_admin3 %>% 
  left_join(
    admin3_poly %>% 
      st_drop_geometry() %>% 
      rename(geo_code = CD_GEOCODI) %>% 
      select(geo_code, area)
  ) %>% 
  right_join(pop_admin3 %>% 
               mutate(
                 geo_code = as.character(geo_code)
               )) 


# 4. Create predicting dataset --------------------------------------------

# Find corresponding admin 3 
admin3_poly <- admin3_poly %>% 
  mutate(
    CD_GEOCODI = as.integer(CD_GEOCODI)
  )
admin3_raster <- fasterize(admin3_poly, masterGrid, field= 'CD_GEOCODI')

tic() # 6 min
EA_admin3 <- exact_extract(admin3_raster, EA_poly, fun='mode', force_df=T)
toc() 

# Join covariates value
EA_admin3$EA_id <- EA_poly$CD_SETOR

master_predict <- EA_admin3 %>% 
  mutate(geo_code = as.character(mode)) %>% 
  select(-mode) %>% 
  right_join(
    cov_EA
  ) 

apply(master_predict,2, function(x) sum(is.na(x)))
# 46 EAs are not assigned to an admin3
# This is due to:
# 1. tiny islands that were in the gridEZ baseline and not on the NSo boundaries dataset
# 2. Tiny outputs from gridEZ algorithm

# only keep rows with no NAs
master_predict <- master_predict[apply(master_predict,1,function(x) sum(is.na(x)) == 0), ]


# Overcome issues in admin3 not present in EA dataset
admin3_withPop <- master_train %>% 
  select(geo_code) %>%
  mutate(
    admin3_municip =T
  ) %>% 
  left_join(
    master_predict %>% 
      group_by(geo_code) %>% 
      summarise(admin3_ea=T)
  )

master_train <- master_train %>% 
  left_join(
    admin3_withPop
  ) %>%
  filter(!is.na(admin3_municip)&!is.na(admin3_ea)) %>% 
  select(-starts_with('admin'))
  
# 5. Save outputs ---------------------------------------------------------

# fwrite(cov_EA, file.path(output_path, "cov_EA.csv"))
# write.csv(cov_admin3, file.path(output_path, "cov_admin3.csv"))

x <- c('name_muni','geo_code','pop','area')
x <- c(x, sort(names(master_train)[!names(master_train) %in% x], decreasing=T))
fwrite(master_train[,x], file.path(output_path, "master_train.csv"))

x <- c('EA_id','geo_code')
x <- c(x, sort(names(master_predict)[!names(master_predict) %in% x], decreasing=T))
fwrite(master_predict[,x], file.path(output_path, "master_predict.csv"))


# 6. Copy over to tutorial folder -----------------------------------------

if(local){
  input_path <- '../wd/out/'
} else {
  input_path <- "//worldpop.files.soton.ac.uk/worldpop/Projects/WP517763_GRID3/Working/git/top-down-tutorial/out"
}

file.copy(from = c(file.path(input_path, 'master_train.csv'), file.path(input_path, 'master_predict.csv')),
          to = c('../dat/top-down-tutorial/master_train.csv','../dat/top-down-tutorial/master_predict.csv'),
          overwrite = T)


