#library(mice)
#library('car')
library(nnet)
library(caret)
library(plyr)
library(gbm)
library(parallel)
library(e1071)

#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}


SummaryGINI <- function(data, lev = NULL, model = NULL){
  out = (NormalizedGini(data$obs, data$pred))
  names(out) <- "GINI"
  out
}




setwd('~/Desktop/BootCamp/kaggle')
data = read.csv('train.csv')
set.seed(3)
train=data
len= nrow(train)
train$Id = NULL
#train$Hazard[which(train$Hazard>15)]=16


pred = data.frame(row.names=c(1:len))
number=5
repeats=1
gbmGrid <-  expand.grid(interaction.depth = c(5, 6, 7), n.trees = (18:22)*50,
                        shrinkage = c(0.05), n.minobsinnode = 20)

ctrl = trainControl(method = "repeatedcv",number = number,
                    repeats = repeats, summaryFunction = SummaryGINI)

for(i in 1:1){
  print(i)
  tmp = train
  #tmp$Hazard = as.factor(tmp$Hazard)
  m_gbm = train(Hazard~., data=tmp, method = "gbm", metric = "GINI", 
                trControl = ctrl, tuneGrid = gbmGrid)
  tmp$Hazard= NULL
  pred=predict(m_gbm, newdata =tmp , type = "raw")
}


G= NormalizedGini(data$Hazard, pred)
print(G)