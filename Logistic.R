if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071") #Check, and if needed install the necessary packages
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
library(plyr)

credit_data <- read_excel(file.choose(), sheet=2)


str(credit_data)

distinct(credit_data, ID )
credit_data$SEX <- as.factor(credit_data$SEX)
credit_data$EDUCATION <- as.factor(credit_data$EDUCATION)
credit_data$MARRIAGE <- as.factor(credit_data$MARRIAGE)
distinct(credit_data, AGE)  ### Maybe group them?

ages <- c(paste(seq(20, 95, by = 5), seq(20 + 5 - 1, 100 - 1, by = 5),
                sep = "-"), paste(100, "+", sep = ""))
ages

credit_data$AGE <- cut(credit_data$AGE, breaks = c(seq(20, 100, by = 5), Inf), labels = ages, right = FALSE)
credit_data$AGE <- as.factor(credit_data$AGE)

credit_data$PAY_1 <- as.factor(credit_data$PAY_1)
distinct(credit_data, PAY_1)
credit_data$PAY_2 <- as.factor(credit_data$PAY_2)
credit_data$PAY_3 <- as.factor(credit_data$PAY_3)
credit_data$PAY_4 <- as.factor(credit_data$PAY_4)
credit_data$PAY_5 <- as.factor(credit_data$PAY_5)
credit_data$PAY_6 <- as.factor(credit_data$PAY_6)
credit_data$default_0 <- as.factor(credit_data$default_0)
credit_data$FPD<- as.factor(credit_data$FPD)
credit_data$SPD<- as.factor(credit_data$SPD)
credit_data$T3PD<- as.factor(credit_data$T3PD)
credit_data$F4PD<- as.factor(credit_data$F4PD)
credit_data$F5TH<- as.factor(credit_data$F5TH)
credit_data$S6th<- as.factor(credit_data$S6th)
###credit_data$Analisis1<- as.factor(credit_data$Analisis1)
###credit_data$Analisis2<- as.factor(credit_data$Analisis2)
###credit_data$Analisis3<- as.factor(credit_data$Analisis3)
###credit_data$Analisis4<- as.factor(credit_data$Analisis4)
###credit_data$Analisis5<- as.factor(credit_data$Analisis5)
###credit_data$Analisis6<- as.factor(credit_data$Analisis6)
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

credit_data<-fixNAs(credit_data) #Apply fixNAs function to the data to fix missing values

credit_data<- credit_data[-c(1473,6647,6969),]




set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = credit_data$default_0,
                               p = 22996/23997, list = FALSE)
training <- credit_data[ inTrain,]
testing <- credit_data[ -inTrain,]




model_logistic<-glm(default_0~ . , data=training, family="binomial"(link="logit"))

summary(model_logistic) 


model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 

par(mfrow=c(1,4))
plot(model_logistic_stepwiseAIC) #Error plots: similar nature to lm plots
par(mfrow=c(1,1))



###Finding predicitons: probabilities and classification for testing
logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=testing,type="response") #Predict probabilities
logistic_classification<-rep("1",1000)
logistic_classification[logistic_probabilities<0.221083]="0" 
logistic_classification<-as.factor(logistic_classification)

###Confusion matrix  
confusionMatrix(logistic_classification,testing$default_0 ,positive = "1") #Display confusion matrix  #### no funciona

####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, testing$default_0)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") #Create AUC data
logistic_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
logistic_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value


new_credit_data <- read_excel(file.choose(), sheet=1)
new_credit_data$default_0<- as.factor(0)
new_credit_data$SEX <- as.factor(new_credit_data$SEX)
new_credit_data$EDUCATION <- as.factor(new_credit_data$EDUCATION)
new_credit_data$MARRIAGE <- as.factor(new_credit_data$MARRIAGE)
distinct(new_credit_data, AGE) 
ages1 <- c(paste(seq(20, 95, by = 5), seq(20 + 5 - 1, 100 - 1, by = 5),
                 sep = "-"), paste(100, "+", sep = ""))
new_credit_data$AGE <- cut(new_credit_data$AGE, breaks = c(seq(20, 100, by = 5), Inf), labels = ages1, right = FALSE)
new_credit_data$AGE <- as.factor(new_credit_data$AGE)
new_credit_data$PAY_1 <- as.factor(new_credit_data$PAY_1)
distinct(new_credit_data, PAY_1)
new_credit_data$PAY_2 <- as.factor(new_credit_data$PAY_2)
new_credit_data$PAY_3 <- as.factor(new_credit_data$PAY_3)
new_credit_data$PAY_4 <- as.factor(new_credit_data$PAY_4)
new_credit_data$PAY_5 <- as.factor(new_credit_data$PAY_5)
new_credit_data$PAY_6 <- as.factor(new_credit_data$PAY_6)
new_credit_data$default_0 <- as.factor(new_credit_data$default_0)
new_credit_data$FPD<- as.factor(new_credit_data$FPD)
new_credit_data$SPD<- as.factor(new_credit_data$SPD)
new_credit_data$T3PD<- as.factor(new_credit_data$T3PD)
new_credit_data$F4PD<- as.factor(new_credit_data$F4PD)
new_credit_data$F5TH<- as.factor(new_credit_data$F5TH)
new_credit_data$S6th<- as.factor(new_credit_data$S6th)



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

new_credit_data<-fixNAs(new_credit_data) #Apply fixNAs function to the data to fix missing values


logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=new_credit_data,type="response") 
logistic_classification<-rep("1",1000)
logistic_classification[logistic_probabilities<0.221083]="0"
logistic_classification<-as.factor(logistic_classification)
predictions<- ifelse(logistic_classification==0,1,0)
file1<-data.frame(default_0 = predictions)
write.csv(file1, file = " Q1 Logistic new.csv", row.names=F)


#### Lift chart
plotLift(logistic_probabilities, testing$default_0, cumulative = TRUE, n.buckets = 10) # Plot Lift chart


  