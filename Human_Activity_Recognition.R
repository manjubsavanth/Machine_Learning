# Project on Human Activity Recognition
#Using devices such as JawboneUp, NikeFuelBand, and Fitbitit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

#In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#Source for Train and Test is below, 
#The training data for this project are available here:
  
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

#The test data are available here:
  
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#Clear the environment.
rm(list=ls(all=TRUE))

#Load the libraries

library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
library(corrplot)
library(doSNOW)

#Set seed inorder to reproduce the the results
set.seed(12345)

#Load the train and test data
train_Data <- read.csv("pml-training.csv")
test_Data <- read.csv("pml-testing.csv")

str(train_Data)
summary(train_Data)
dim(train_Data)

#From the train data we can see that many of the features are zero and near zero variables. 
#Also most of them have missing values. 
#removing the zero and near zero variables and NA values 

NZV <- (nearZeroVar(train_Data))      #Find the coulmns with near zero variables
NZV
train_Data <- train_Data[, -NZV]      #Remove the near zero variables from train data 
dim(train_Data)

NAdata   <- sapply(train_Data, function(x) mean(is.na(x))) > 0.95   #Use sapply on the columns having NA values 95% or more
train_Data <- train_Data[, NAdata==FALSE]                           #obtain the train data by filtering NA values. 
dim(train_Data)

# remove identification only variables (columns 1 to 5) from train data
train_Data <- train_Data[, -(1:5)]
dim(train_Data)

#Partitioning Trianing data in to train(60%) and validation set(40%). 

train_part <- createDataPartition(train_Data$classe, p=0.6, list=FALSE)
trainD <- train_Data[train_part, ]
validate <- train_Data[-train_part, ]
dim(trainD)
dim(validate)


#Using the correlation function to find the correation between the data variables. 

cor_Matrix <- cor(trainD[,-54])        #Correlation excluding the target variable. 
corrplot(cor_Matrix, order = "FPC", method = "color", type = "lower",tl.cex = 0.6, tl.col = rgb(0, 0, 0))             #Correlationplot using plot

#Highly correlated variables are shown in dark colors in the graph. 
#From the graph it is visible that there are very less correlated variables. Hence proceeding without removal f any correlated variables. 

#Model Building
#I have decided to go with the popular bagging techinque Random Forest with cross validation technique to see how it performes on the data. 

# instruct train to use 3-fold CV to select optimal tuning parameters
modelControl <- trainControl(method="cv", number=3, verboseIter=F)

# Set up doSNOW package for multi-core training. This is helpful as we're going to be training a lot of trees.
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(1234)

# fit model on train data
model <- train(classe ~ ., data=trainD, method="rf", trControl=modelControl)

#print final model details
model$finalModel

#Shutdown cluster
stopCluster(cl)

# use model to predict classe in validation set 
preds <- predict(model, newdata=validate)

# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(validate$classe, preds)

#From the validation set, Model accuracy was found to be 99.69% with 500 as number of trees. Since this is a far superior accuracy, I will stick with this model. 
#Now, I will retrain the model with comeplete available train data. 

#Re-training the Selected Model with complete available training data. 

# instruct train to use 3-fold CV to select optimal tuning parameters
modelControl <- trainControl(method="cv", number=3, verboseIter=F)

# Set up doSNOW package for multi-core training. This is helpful as we're going to be training a lot of trees.
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(1234)

# fit model on train data
model <- train(classe ~ ., data=train_Data, method="rf", trControl=modelControl)

#print final model details
model$finalModel

#Shutdown cluster
stopCluster(cl)

#Making Predictions on Test Set 

# use model to predict classe in test set 
preds <- predict(model, newdata=test_Data)
preds
confusionMatrix(test_Data$classe, preds)

# Build a table of prediction
table(preds)

# Write out a CSV file for submission of test results

submit_pred <- data.frame(problem_id = rep(1:20), classe = preds)

write.csv(submit_pred, file = "HAR_preds_classe.CSV", row.names = FALSE)


