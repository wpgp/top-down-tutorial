
#-----------------------------------#

# packages
library(randomForest) # estimating random forest model

# working directory
local <- T
if(local){
  setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path),'dat/top-down-tutorial'))
} else {
  setwd("//worldpop.files.soton.ac.uk/worldpop/Projects/WP517763_GRID3/Working/git/top-down-tutorial")
}

#--

# training data from municipalities
master_train <- read.csv("master_train.csv")

head(master_train[,1:5]) # only showing first five columns

# covariates from enumeration areas
master_predict <- read.csv("master_predict.csv")

head(master_predict[,1:4]) # only showing first four columns

#--

# response
y_data <- master_train$pop / master_train$area

y_data <- log(y_data)

#--

# histogram of response
hist(y_data, main=NA, xlab="log(population_density)")

#--

# covariate names (i.e. column names)
cols <- colnames(master_train)
print(cols)

# select column names that contain the word 'mean'
cov_names <- cols[grepl('mean', cols)] 
print(cov_names)

# subset the data.frame to only these columns
x_data <- master_train[,cov_names]
head(x_data[,1:2]) # only showing first two columns

#--

# model fitting
popfit <- tuneRF(x=x_data, 
                 y=y_data, 
                 plot=TRUE, 
                 mtryStart=length(x_data)/3, 
                 ntreeTry=500, 
                 improve=0.0001, # threshold on the OOB error to continue the search
                 stepFactor=1.20, # incremental improvement of mtry
                 trace=TRUE, 
                 doBest=TRUE, # last model trained with the best mtry
                 nodesize=length(y_data)/1000, 
                 na.action=na.omit, 
                 importance=TRUE, 
                 sampsize=length(y_data), # size of the sample to draw for OOB
                 replace=TRUE) # sample with replacement

#--

names(popfit)

#--

popfit$mtry

#--

# save model
save(popfit, file='popfit.Rdata')

#--

# load model
load('popfit.Rdata')

#--

# random forest predictions
master_predict$predicted <- predict(popfit, 
                                    newdata = master_predict)

#--

# back-transform predictions to natural scale
master_predict$predicted_exp <- exp(master_predict$predicted)

#--

# sum exponentiated predictions among EAs in each municipality
predicted_exp_sum <- aggregate(master_predict$predicted_exp, 
                               by = list(geo_code=master_predict$geo_code), 
                               FUN = sum)

# modify column names
names(predicted_exp_sum) <- c('geo_code','predicted_exp_sum')

# merge predicted_exp_sum into master_train based on geo_code
master_predict <- merge(master_predict, 
                        predicted_exp_sum, 
                        by='geo_code')

#--

# merge municipality total populations from master_train into master_predict
master_predict <- merge(master_predict,
                        master_train[,c('geo_code','pop')],
                        by = 'geo_code')

# modify column name
names(master_predict)[ncol(master_predict)] <- 'pop_municipality'

#--

# calculate EA-level population estimates
master_predict$predicted_pop <- with(master_predict, predicted_exp / predicted_exp_sum * pop_municipality)

#--

# sum EA population estimates within each municipality
test <- aggregate(master_predict$predicted_pop, 
                  by = list(geo_code=master_predict$geo_code), 
                  FUN = sum)

# modify column names
names(test) <- c('geo_code','predicted_pop')

# merge municipality population totals
test <- merge(test,
              master_train[,c('geo_code','pop')],
              by = 'geo_code')

# test if estimates match muncipality population totals
all(round(test$pop) == round(test$predicted_pop))

#--

# goodness-of-fit metrics
print(popfit) 

#--

# plot observed vs predicted (out-of-bag)
plot(x = y_data,
     y = predict(popfit), 
     main = 'Observed vs Predicted log-Densities')

# 1:1 line
abline(a=0, b=1, col='red')

#--

# plot residuals (out-of-bag)
plot(x = predict(popfit), 
     y = y_data - predict(popfit), 
     main = 'Residuals vs Predicted',
     ylab = 'Out-of-bag residuals',
     xlab = 'Out-of-bag prediction')

# horizontal line at zero
abline(h=0, col='red')

#--

layout(matrix(1:2, nrow=1))

for(cov_name in c('mean.bra_srtm_slope_100m', 'mean.bra_viirs_100m_2016')){
  
  # combine EA-level and municipality-level values into a single vector
  y <- c(master_predict[,cov_name], master_train[,cov_name])
  
  # create corresponding vector identifying spatial scale
  x <- c(rep('enumeration_area', nrow(master_predict)),
         rep('municipality', nrow(master_train)))
  
  # create boxplot
  boxplot(y~x, xlab='Spatial Scale', ylab=cov_name)
}


#----------------- figures -------------------#

if(F){
  
  # enumeration areas
  ea <- sf::st_read(file.path(rd,'wd/in/censusEAs/BR_Setores_2019.shp'))
  ea$EA_id <- 1:nrow(ea)
  
  ea <- merge(ea,
              master_predict[,c('EA_id','predicted_pop')],
              by = 'EA_id')
  
  sf::st_write(ea, file.path(rd,'wd/out/enumeration_areas.gpkg'))
  
  
  # municipalities
  municipality <- sf::st_read(file.path(rd,'wd/in/adminBoundaries/admin_municipality.gpkg'))
  municipality$geo_code <- municipality$CD_GEOCODI
  
  municipality <- merge(municipality,
                        master_train[,c('geo_code','pop')],
                        by = 'geo_code')
  
  sf::st_write(municipality, file.path(rd,'wd/out/municipality.gpkg'))
}

#------- tips and tricks ------#
if(F){
  
  
  # results to EA polygons
  
  library('sf')
  
  sf_polygons <- st_read('../../wd/in/censusEAs/BR_Setores_2019.shp')

  sf_polygons$EA_id <- 1:nrow(sf_polygons)

  sf_polygons <- merge(sf_polygons,
                       master_predict,
                       by='EA_id')
  
  st_write(sf_polygons,
           'master_predict_polygons.shp')
  
  st_write(sf_polygons,
           'master_predict_polygons.gpkg')
  
  
  
  # zonal statistics
  
  library(raster)
  library(exactextractr)
  
  raster_covariate <- raster('../../wd/in/covariates/bra_viirs_100m_2016.tif')
  
  sf_polygons$mean.bra_viirs_100m_2016 <- exact_extract(x = raster_covariate,
                                                        y = sf_polygons,
                                                        fun = 'mean')
  
  write.csv(st_drop_geometry(sf_polygons), 
            file = 'EA_covariates.csv',
            row.names = FALSE)
  
  st_write(sf_polygons, 'EA_covariates.shp')
  
  
  
  # gridded population estimates
  
  mastergrid <- raster('../../wd/in/bra_level0_100m_2000_2020.tif')
  
  cells <- which(!is.na(mastergrid[1:1e7]))
  
  mastergrid_predict <- data.frame(row.names = cells)
  
  
  raster_covariate <- raster('../../wd/in/covariates/bra_viirs_100m_2016.tif')
  
  mastergrid_predict[cells, 'bra_viirs_100m_2016'] <- raster_covariate[cells]
  
  
  xy <- xyFromCell(mastergrid, cells)
  
  mastergrid_predict[cells, 'bra_viirs_100m_2016_alt'] <- extract(raster_covariate, xy)
  
  write.csv(mastergrid_predict,
            file = 'mastergrid_predict.csv',
            row.names = FALSE)
  
  
  mastergrid_predict$predicted_pop <- runif(nrow(mastergrid_predict), 0, 1000)
  
  raster_predict <- raster(mastergrid)
  
  raster_predict[cells] <- mastergrid_predict[cells, 'predicted_pop']
  
  writeRaster(raster_predict, 
              file = 'raster_predict.tif')
  
  
  
  # parallel processing
  
  
  library(doParallel)
  
  predict_pop <- function(df, model=popfit){
    
    # EA-level predictions
    prediction <- predict(model, newdata = df)
    
    # back-transform to population density
    density <- exp(prediction)
    
    # calculate weights
    weights <- density / sum(density) 
    
    # disaggregate municipality total to EA-level
    pop <- weights * df$pop_municipality[1]
    
    # result to data.frame
    result <- data.frame(id=df$id, predicted_pop_parallel=pop)
    
    # return result
    return(result)
  }
  
  master_predict$id <- 1:nrow(master_predict)
  
  list_master_predict <- split(x = master_predict,
                               f = master_predict$geo_code)
  
  cores <- detectCores()
  
  cluster <- makeCluster(cores)
  
  registerDoParallel(cluster)
  
  predicted <- foreach(i = 1:length(list_master_predict), 
                       .combine = 'rbind',
                       .packages = c("randomForest")) %dopar% 
    predict_pop(df = list_master_predict[[i]])
  
  stopCluster(cluster)
  
  master_predict <- merge(master_predict,
                          predicted,
                          by = 'id')
  
}
#---------------------------------------------#









