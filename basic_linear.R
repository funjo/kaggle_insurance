setwd('/Users/fangzhoucheng/Documents/Workspace/kaggle_insurance/')
train = read.csv('train.csv')
test = read.csv('test.csv')

# Basic inspection
# No na's
length(train[is.na(train)])
length(test[is.na(test)])
dim(train)
dim(test)
names(train)
names(test)
str(train)
str(test)

# get rid of id in both data sets
train$Id <- NULL
test.id = test$Id
test$Id <- NULL

# factorize Hazard
train$Hazard <- as.factor(train$Hazard)

# get rid of Hazard in training set
y <- train$Hazard
train$Hazard <- NULL

# response distribution
library(vioplot)
vioplot(y, col="gold")
summary(y)
y <- as.factor(y)
summary(as.factor(y))
length(levels(as.factor(y)))
tbl <- table(as.factor(y))
#cbind(Count=tbl,Perc=round(prop.table(tbl)*100,2))
#addmargins(round(prop.table(tbl)*100,2))
round(prop.table(tbl)*100,2)

# Find near zero values
library(caret)
nzv <- nearZeroVar(train, saveMetrics= FALSE)
train <- train[, -nzv]
dim(train)
str(train)
nzv_test <- nearZeroVar(test, saveMetrics= FALSE)
test <- test[, -nzv_test]
dim(test)
str(test)

# naive Bayes
library(e1071)
head(train)
model = naiveBayes(Hazard~., train); summary(model)
pred = predict(model, test)
# pred = predict(model, train[1:100,])
# table(pred,train$Hazard[1:100])
# library(vioplot)
# vioplot(as.numeric(pred),as.numeric(train$Hazard[1:100]), col="gold")
pred_test = as.numeric(pred)
a=length(pred_test)
result= data.frame(row.names= c(1:a))
result$Id = test.id
result$Hazard = pred_test
write.table(result, 'naivebayes.csv', sep=',', row.names= FALSE)
# kaggle NormalizedGini: 0.069877

# multiple linear regression
library(car)
par(mfrow=c(1,1))
model = lm(Hazard~., train); summary(model)
boxCox(model)
# log transformation of y
model = lm(log(Hazard)~., train); summary(model)
prep2submit <- function(model){
  test = read.csv('test.csv')
  test=test
  iid = test$Id
  test$Id = NULL
  pred_test = predict(model, newdata =test , type = "response")
  a=length(pred_test)
  result= data.frame(row.names= c(1:a))
  result$Id = iid
  result$Hazard = round(pred_test)
  write.table(result, 'submit.csv', sep=',', row.names= FALSE)
  return(NULL)
}
prep2submit(model)
# kaggle NormalizedGini: 0.024539

# knn calssification
extractFeatures2 <- function(data) {
  character_cols <- names(Filter(function(x) x=="factor", sapply(data, class)))
  for (col in character_cols) {
    data[,col] <- ordered(data[,col])
    data[,col] <- as.integer(data[,col])
  }
  return(data)
}
train_numeric <- extractFeatures2(train)
test_numeric  <- extractFeatures2(test)
str(train_numeric)
str(test_numeric)
library(class)
knn.pred = knn(train_numeric, test_numeric, y, k=10)
a=length(knn.pred)
result= data.frame(row.names= c(1:a))
result$Id = test.id
result$Hazard = knn.pred
write.table(result, 'submit.csv', sep=',', row.names= FALSE)
# Kaggle NormalizedGini: 0.066800

# Linear Discriminant Analysis
extractFeatures2 <- function(data) {
  character_cols <- names(Filter(function(x) x=="factor", sapply(data, class)))
  for (col in character_cols) {
    data[,col] <- ordered(data[,col])
    data[,col] <- as.integer(data[,col])
  }
  return(data)
}
train_numeric <- extractFeatures2(train)
test_numeric  <- extractFeatures2(test)
str(train_numeric)
str(test_numeric)
train_numeric$Hazard = as.factor(train_numeric$Hazard)
library(MASS)
model = lda(Hazard~., data = train_numeric)
pred_test = predict(model, newdata =test_numeric , type = "response")
pred_test = as.numeric(data.frame(pred_test)$class)
# summary(as.factor(pred_test))
# library(vioplot)
# vioplot(pred_test, col="gold")
a=length(pred_test)
result= data.frame(row.names= c(1:a))
result$Id = test.id
result$Hazard = pred_test
write.table(result, 'submit.csv', sep=',', row.names= FALSE)
# Kaggle NormalizedGini: 0.000339

# stepwise subset regression
library(leaps)
start <- lm(Hazard ~ T1_V1 + T1_V11 + T1_V2 + T1_V3 + T1_V3 + T1_V4 + T2_V2
            + T2_V9 + T2_V13 + T2_V15, data = train)
full <- lm(Hazard ~ ., data = train)
empty <- lm(Hazard ~ 1, data = train)
bounds <- list(upper = full, lower = empty)
step(start, bounds, direction = "both")

# glmnet
library(boot)
model = glm(Hazard~., train, family = binomial)
cv.glm(train,model,K=10)

#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df
  df$random = (1:nrow(df))/nrow(df)
  df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  print(df)
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

NormalizedGini(test$Hazard, pred)











