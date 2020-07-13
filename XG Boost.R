if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071","xgboost")  #Check, and if needed install the necessary packages
library(readxl)
library(tidyverse) # Load Libraries
library(caret)
library(gbm)
library(dplyr)
library(lars)
library(moments)
library(caret)
library(knitr)
library(corrplot)
library(gbm)
library(glmnet)
library(mice)
library(ggplot2)
library(Amelia)
library(readr)
library(dplyr)
library(xgboost)

new_credit_data <- read_excel(file.choose(), sheet=1) # Import new applications file
new_credit_data$default_0<- as.factor(0)

credit_data <- read_excel(file.choose(), sheet=1) # Import credit data file
combined_files <- rbind(credit_data, new_credit_data) ### Combine files
str(combined_files)
tail(combined_files)

combined_files$SEX <- as.factor(combined_files$SEX)  #### Ajust the variables accordingly
combined_files$EDUCATION <- as.factor(combined_files$EDUCATION)
combined_files$MARRIAGE <- as.factor(combined_files$MARRIAGE)
distinct(combined_files, AGE) 
ages1 <- c(paste(seq(20, 95, by = 5), seq(20 + 5 - 1, 100 - 1, by = 5),
                 sep = "-"), paste(100, "+", sep = ""))
combined_files$AGE <- cut(combined_files$AGE, breaks = c(seq(20, 100, by = 5), Inf), labels = ages1, right = FALSE)
combined_files$AGE <- as.factor(combined_files$AGE)
combined_files$PAY_1 <- as.factor(combined_files$PAY_1)
combined_files$PAY_2 <- as.factor(combined_files$PAY_2)
combined_files$PAY_3 <- as.factor(combined_files$PAY_3)
combined_files$PAY_4 <- as.factor(combined_files$PAY_4)
combined_files$PAY_5 <- as.factor(combined_files$PAY_5)
combined_files$PAY_6 <- as.factor(combined_files$PAY_6)
combined_files$default_0 <- as.factor(combined_files$default_0)
combined_files$FPD<- as.factor(combined_files$FPD)  ### FPD= First payment default. If client didnt pay anything in Pay amount then =1
combined_files$SPD<- as.factor(combined_files$SPD)  ## SPD= Second payment default and so on 
combined_files$T3PD<- as.factor(combined_files$T3PD)
combined_files$F4PD<- as.factor(combined_files$F4PD)
combined_files$F5TH<- as.factor(combined_files$F5TH)
combined_files$S6th<- as.factor(combined_files$S6th)

tail(combined_files)
# Create a custom function to fix missing values ("NAs") and preserve the NA info as surrogate variables
fixNAs<-function(data_frame){
  # Define reactions to NAs
  integer_reac<-0
  factor_reac<-"FIXED_NA"
  character_reac<-"FIXED_NA"
  date_reac<-as.Date("1900-01-01")
  # Loop through columns in the data frame and depending on which class the variable is, apply the defined reaction and create a surrogate
  
  for (i in 1 : ncol(data_frame)){
    if (class(data_frame[,i]) %in% c("numeric","integer")) {
      if (any(is.na(data_frame[,i]))){
        data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
          as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
        data_frame[is.na(data_frame[,i]),i]<-integer_reac
      }
    } else
      if (class(data_frame[,i]) %in% c("factor")) {
        if (any(is.na(data_frame[,i]))){
          data_frame[,i]<-as.character(data_frame[,i])
          data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
            as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
          data_frame[is.na(data_frame[,i]),i]<-factor_reac
          data_frame[,i]<-as.factor(data_frame[,i])
          
        } 
      } else {
        if (class(data_frame[,i]) %in% c("character")) {
          if (any(is.na(data_frame[,i]))){
            data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
              as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
            data_frame[is.na(data_frame[,i]),i]<-character_reac
          }  
        } else {
          if (class(data_frame[,i]) %in% c("Date")) {
            if (any(is.na(data_frame[,i]))){
              data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
                as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
              data_frame[is.na(data_frame[,i]),i]<-date_reac
            }
          }  
        }       
      }
  } 
  return(data_frame) 
  
}

combined_files<-fixNAs(combined_files) #Apply fixNAs function to the data to fix missing values

combined_files<- combined_files[-c(1473,6647,6969),]  ### Remove outliers
#write.csv(combined_files, file = "combined_files.csv", row.names=F)

credit_data_matrix <- model.matrix( default_0 ~ ., data = combined_files)[1:23997,-1]

##to split the credit card file** into the training and testing sets for model training and model testing:
x_train <- credit_data_matrix[ inTrain,]  
x_test <- credit_data_matrix[ -inTrain,]
#write.csv(inTrain, file = "inTrain.csv", row.names=F)

y_train <-as.factor(training$default_0)
y_test <-testing$default_0


model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 11,  # hyperparameter: size of a tree in each boosting iteration
                       nrounds =30,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic",
                       lambda=1,
                       gamma=8,
                       min_child_weight=16,
                       subsample=0.80,
                       colsample_bytree=0.5
)

XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction>0.221083,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)


#####################################   Prediction    #############################################

#### Split dataset
training<- combined_files[1:23997,]  #### here are all records from the credit card file that we already have default or not info
#prediction<- combined_files[23998:24997,]   ###here are all records from new applications that we need to predict

combined_files_matrix <- model.matrix( default_0 ~ ., data = combined_files)[,-1]

x_train <- combined_files_matrix[1:23997,]
y_train <-training$default_0

x_prediction <- combined_files_matrix[23998:24997,]
#y_prediction <- prediction$default_0

##now use the entire credit dataset to build the model:
final_model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 11,  # hyperparameter: size of a tree in each boosting iteration
                       nrounds =30,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic",
                       lambda=1,
                       gamma=8,
                       min_child_weight=16,
                       subsample=0.80,
                       colsample_bytree=0.5
)


##to predict the new application dataset:
XGboost_prediction<-predict(model_XGboost,newdata=x_prediction, type="response") #The following lines of code make the predictions and export the csv 

XGboost_classification<-rep("1",1000)
XGboost_classification[XGboost_prediction<0.221083]="0"
XGboost_classification<-as.factor(XGboost_classification)
predictions<- ifelse(XGboost_classification==0,1,0)
file1<-data.frame(default_0 = predictions)
write.csv(file1, file = " Q1 XGBoost.csv", row.names=F)


