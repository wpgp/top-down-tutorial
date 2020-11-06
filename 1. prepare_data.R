

# 1. Set-up ---------------------------------------------------------------

library(sf) # manipulating vector GIS file
library(raster) # manipulating raster GIS file
library(fasterize) # fast rasterization
library(tictoc) # compute running time
library(tidyverse) # manipulating dataframes
library(exactextractr) # fast zonal statistics
library(data.table) # fast dataframe writing


drive_path <- "//worldpop.files.soton.ac.uk/worldpop/Projects/WP517763_GRID3/Working/git/top-down-tutorial/"
data_path <- paste0(drive_path, "in/")
output_path <- paste0(drive_path, "out/")


# Boundaries from IGBE 2019
# https://www.ibge.gov.br/en/geosciences/territorial-organization/regional-division/23708-brazilian-territorial-division.html?=&t=o-que-e
admin3_poly <- st_read(paste0(data_path, 'adminBoundaries/admin_municipality.gpkg'), stringsAsFactors = F)

# Mastergrid from WorldPop
masterGrid <- raster(paste0(data_path, "masterGrid.tif"))

# Population data from IGBE census 2007
# https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?view=detalhes&id=293420
pop_admin3 <- read.csv(paste0(data_path, 'population/pop_2007_municipality.csv'))

# EA from gridEZ large sccenario and converted to polygons
EA_poly <- st_read(paste0(data_path, 'gridEZ/EZ_poly.gpkg'))
EA_poly$EZ_id <- 1:nrow(EA_poly) # ids were not unique in gridEZ output



# 2. Create covariates dataset  ---------------------------------------------------


# Create a raster stack of all covariates
raster_names <- list.files(paste0(data_path, "covariates/"), pattern="tif$", full.names = T)
raster_stack <- stack(raster_names) # 10 rasters


# Extract zonal statistics for every EA
tic()
cov_EA <- exact_extract(raster_stack, EA_poly, fun='mean', progress=T, force_df=T, stack_apply=T)
toc() #30min

cov_EA$EA_id <- EA_poly$EZ_id

# Extract zonal statistics for every admin3


tic()
cov_admin3 <- exact_extract(raster_stack, admin3_poly, fun='mean', progress=T, force_df=T, stack_apply=T)
toc() #27min
cov_admin3$geo_code <- admin3_poly$CD_GEOCODI

write.csv(cov_EA, paste0(output_path, "cov_EA.csv"))
write.csv(cov_admin3, paste0(output_path, "cov_admin3.csv"))

# 3. Create training dataset ----------------------------------------------

master_train <- cov_admin3 %>% 
  right_join(pop_admin3 %>% 
               mutate(
                 geo_code = as.character(geo_code)
               )) # six municipalities are not in the 2007 partition


# 4. Create predicting dataset --------------------------------------------

# Find corresponding admin 3
admin3_poly <- admin3_poly %>% 
  mutate(
    CD_GEOCODI = as.integer(CD_GEOCODI)
  )
admin3_raster <- fasterize(admin3_poly, masterGrid, field= 'CD_GEOCODI')

tic()
EA_admin3 <- exact_extract(admin3_raster, EA_poly, fun='mode', force_df=T)
toc() #4min

# Join covariates value
EA_admin3$EA_id <- EA_poly$EZ_id

master_predict <- EA_admin3 %>% 
  mutate(geo_code = as.integer(mode)) %>% 
  select(-mode) %>% 
  right_join(
    cov_EA # add covariates
  ) 

apply(master_predict,2, function(x) sum(is.na(x)))
# 719 EAs are not assigned to an admin3
# This is due to:
# 1. tiny islands that were in the gridEZ baseline and not on the NSo boundaries dataset
# 2. Tiny outputs from gridEZ algorithm


master_predict <- master_predict %>% 
  filter(!is.na(geo_code))


# Overcome issues in admin3 not present in 2007
admin3_withPop <- master_train %>% 
  select(geo_code) %>%
  mutate(
    admin3_withPop =T
  )

master_predict <- master_predict %>% 
  left_join(
    admin3_withPop
  ) %>%
  filter(!is.na(admin3_withPop)) %>% 
  select(-admin3_withPop)
  
# 5. Save outputs ---------------------------------------------------------

write.csv(master_train, paste0(output_path, "master_train.csv"))
fwrite(master_predict, paste0(output_path, "master_predict.csv"))





