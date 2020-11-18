
# 1. Set-up ---------------------------------------------------------------

library(tictoc) # compute running time
library(tidyverse) # manipulating dataframes
library(randomForest) # estimating randomo forest model
library(data.table) # fast dataframe writing
library(doParallel) # processing in parallel
library(sf) # manipulating vector GIS file


drive_path <- "//worldpop.files.soton.ac.uk/worldpop/Projects/WP517763_GRID3/Working/git/top-down-tutorial/"
data_path <- paste0(drive_path, "in/")
output_path <- paste0(drive_path, "out/")

# Previously built datasets
master_train <- fread(paste0(output_path, "master_train.csv"), data.table = F)
master_predict <-fread(paste0(output_path, "master_predict.csv"), data.table = F)

# EA from gridEZ large sccenario and converted to polygons
EA_poly <- st_read(paste0(data_path, 'gridEZ/EZ_poly.gpkg'))
EA_poly$EZ_id <- 1:nrow(EA_poly) # ids were not unique in gridEZ output

# 2. Train the model ------------------------------------------------------

y_data <- master_train$pop/master_train$area
y_data <- log(y_data)

names <- colnames(master_train)
cov_names <- names[grepl('mean', names)] # here we select all preprocessed covariates as predictors
x_data <- master_train %>% 
  select(all_of(cov_names))

tic()
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
toc()#90sec
print(popfit) # for goodness-of-fit metrics

save(popfit, file= paste0(output_path, 'popfit.Rdata')) # save model

varImpPlot(popfit, type=1) # for variable importance

# OOB goodness-of-fit plot
plot(y_data, (y_data - predict(popfit)), main='Residuals vs Observed')
abline(a=0, b=0, lty=2)

plot(y_data,predict(popfit), main='Predicted vs Observed')
abline(a=0, b=1, lty=2)



# 3. Predict Population ------------------------------------------------------

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
  fwrite(output, paste0(output_path, "/predictions/predictions_",
                        df$geo_code[1], ".csv"))
}

# Create a list of EAs per admin3 for parallel processing
predict_admin3 <- master_predict %>% 
  arrange(geo_code) %>% 
  group_by(geo_code) %>% # order the list by geo_code
  group_split() 

# Create vector of admin 3 pop sorted by admin3 code and present in master_predict
admin3_pop <- master_train %>% 
  select(
    geo_code, pop
  ) %>% 
  arrange(geo_code) %>% 
  select(pop) %>% unlist()

# Run predictions in parallel
co <- detectCores()-2
tic()
cl <- makeCluster(co)
registerDoParallel(cl)
predicted <- NULL
predicted <- foreach(
  i=1:length(predict_admin3), 
  .packages=c("tidyverse", "data.table", "randomForest")) %dopar% {
    predict_pop(
      predict_admin3[[i]],
      admin3_pop[[i]]
    )
  } 

stopCluster(cl)
toc() #109sec


# 4. Map predictions ------------------------------------------------------

# Combine predictions
predictions_list <- list.files(
  paste0(output_path, "/predictions/"), pattern = ".csv", full.names = T) 
print(length(predictions_list)) #should match nb of admin3 

tic()
predictions <- bind_rows(lapply(predictions_list, fread))
toc() #130 sec

# Join predictions to EA vector dataset
EA_poly <- EA_poly %>% 
  left_join(predictions, by=c('EZ_id'='EA_id'))

st_write(EA_poly, paste0(output_path, 'EA_predict_poly.gpkg'), append = F)


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
