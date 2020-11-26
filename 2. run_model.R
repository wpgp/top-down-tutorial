# cleanup
rm(list=ls()); gc(); cat("\014"); try(dev.off(), silent=T)

try(source(file.path(dirname(rstudioapi::getSourceEditorContext()$path),'0_setup.R')))

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
copyWP(srcdir = 'Projects/WP517763_GRID3/Working/git/top-down-tutorial/in',
       outdir = data_path)

# load packages
library(tictoc) # compute running time
library(tidyverse) # manipulating dataframes
library(randomForest) # estimating randomo forest model
library(data.table) # fast dataframe writing
library(doParallel) # processing in parallel
library(sf) # manipulating vector GIS file

##-- load data --## 

# Previously built datasets
master_train <- fread(file.path(output_path, "master_train.csv"), data.table = F)
master_predict <-fread(file.path(output_path, "master_predict.csv"), data.table = F)

# IBGE census EAs (public)
# http://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/2019/Malha_de_setores_(shp)_Brasil/
EA_poly <- st_read(file.path(data_path, 'censusEAs/BR_Setores_2019.shp'))
EA_poly$EZ_id <- 1:nrow(EA_poly) # ids were not unique in gridEZ output


# 2. Train the model ------------------------------------------------------

# calculate population density
y_data <- master_train$pop/master_train$area

# log-transform population density
y_data <- log(y_data)

# prepare covariates
names <- colnames(master_train)
cov_names <- names[grepl('mean', names)] # here we select all preprocessed covariates as predictors
x_data <- master_train %>% 
  select(all_of(cov_names))

# fit random forest model
tic() # 90 sec
popfit <- tuneRF(x=x_data, 
                     y=y_data, 
                     plot=TRUE, 
                     mtryStart=length(x_data)/3, 
                     ntreeTry=500, 
                     improve=0.0001, 
                     stepFactor=1.20, 
                     trace=TRUE, 
                     doBest=TRUE, 
                     nodesize=length(y_data)/1000, 
                     na.action=na.omit, 
                     importance=TRUE, 
                     replace=TRUE) 
toc()

# save model
save(popfit, file= file.path(output_path, 'popfit.Rdata')) 

# goodness-of-fit metrics
print(popfit) 

# variable importance plot
jpeg('out/var_importance.jpg')
varImpPlot(popfit, type=1) 
dev.off()

# out-of-bag residuals plot
jpeg('out/residuals.jpg')
plot(y_data, (y_data - predict(popfit)), main='Residuals vs Observed')
abline(a=0, b=0, lty=2)
dev.off()

# out-of-bag observed vs predicted plot
jpeg('out/obs_pred.jpg')
plot(y_data,predict(popfit), main='Predicted vs Observed')
abline(a=0, b=1, lty=2, col='red')
dev.off()


# 3. Predict Population ------------------------------------------------------
dir.create('out/predictions', recursive=T, showWarnings=F)

# prediction function
predict_pop <- function(df, census, model=popfit){
  
  # Apply model on EA dataset
  prediction_set <- predict(model, 
                            newdata=df, 
                            predict.all=TRUE)
  
  # Predict density
  output <- data.frame(density= exp(apply(prediction_set$individual, MARGIN=1, mean)))
  
  # Disaggregate admin3 totals
  output$pop <- (output$density/sum(output$density))*census
  output$density <- NULL
  
  # Add ids
  output$geo_code <- df$geo_code
  output$EA_id <- df$EA_id
  
  #Write output
  fwrite(output, file.path(output_path, 
                           paste0("predictions/predictions_",df$geo_code[1], ".csv")))
}

# Create a list of EAs per admin3 for parallel processing
predict_admin3 <- master_predict %>% 
  arrange(geo_code) %>% 
  group_by(geo_code) %>% # order the list by geo_code
  group_split() 

# Create vector of admin 3 pop sorted by admin3 code and present in master_predict
admin3_pop <- master_train %>% 
  select(geo_code, pop) %>% 
  arrange(geo_code) %>% 
  select(pop) %>% 
  unlist()

# parallel processing for random forest predictions
tic() # 109 sec
co <- detectCores()-2
predicted <- NULL
cl <- makeCluster(co)
registerDoParallel(cl)
predicted <- foreach(i = 1:length(predict_admin3),
                     .packages = c("tidyverse", "data.table", "randomForest")) %dopar% {
    predict_pop(df = predict_admin3[[i]],
                census = admin3_pop[[i]])
  } 

stopCluster(cl)
toc() 


# 4. Map predictions ------------------------------------------------------

# Combine predictions
predictions_list <- list.files(
  file.path(output_path, "predictions"), pattern = ".csv", full.names = T) 
print(length(predictions_list)) #should match nb of admin3 

tic() #130 sec
predictions <- bind_rows(lapply(predictions_list, fread))
toc() 

# Join predictions to EA vector dataset
EA_poly <- EA_poly %>% 
  left_join(predictions, by=c('EZ_id'='EA_id'))

# save predictions
st_write(EA_poly, file.path(output_path, 'EA_predict_poly.gpkg'), append = F) 


# 5. Double check totals --------------------------------------------------

predictions_admin3 <- predictions %>% 
  group_by(geo_code) %>% 
  summarise(
    pop_predicted= sum(pop)
  ) %>% 
  left_join(
    master_train %>% 
      select(geo_code, pop)
  ) %>% 
  mutate(
    diff = pop-round(pop_predicted)
  )
summary(predictions_admin3$diff)

