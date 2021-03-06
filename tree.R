setwd('/Users/fangzhoucheng/Documents/Workspace/kaggle_insurance/')
library(doMC)
registerDoMC(cores = 5)
library(xgboost)
library(caret)
library(randomForest)
library(readr)

cat("Reading data\n")
train = read.csv('train.csv')
test = read.csv('test.csv')

# We'll convert all the characters to factors so we can train a randomForest model on them
extractFeatures <- function(data) {
  character_cols <- names(Filter(function(x) x=="character", sapply(data, class)))
  for (col in character_cols) {
    data[,col] <- as.factor(data[,col])
  }
  return(data)
}

trainFea <- extractFeatures(train)
testFea  <- extractFeatures(test)

cat("Training model - random forest\n")
set.seed(1)
rf <- randomForest(trainFea[,3:34], trainFea$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)
pred_rf <- predict(rf, testFea[,2:33])

# Create the response variable
y = train$Hazard

# Create the predictor data set and encode categorical variables using caret library.
mtrain = train[,-c(1,2)]
mtest = test[,-c(1)]
dummies <- dummyVars(~ ., data = mtrain)
mtrain = predict(dummies, newdata = mtrain)
mtest = predict(dummies, newdata = mtest)

# Set necessary parameters and use parallel threads
param <- list("objective" = "reg:linear",
              "nthread" = 8,
              "verbose"= 0,
              "silent" = 1,
              "eta" = .01,
              "max.depth" = 7,
              "min_child_weight" = 5,
              "scale_pos_weight" = 1.0,
              "subsample" = 0.8)

cat("Training model - Xgboost\n") 
# Fit the model
# xgb.fit = xgboost(param=param, data = mtrain, label = y, nrounds=2000) 
# pred_xgboost <- predict(xgb.fit, mtest)

# early stopping
dtrain <- xgb.DMatrix(mtrain[-c(1:5000),], label = y[-c(1:5000)])
dtest <- xgb.DMatrix(mtrain[1:5000,], label = y[1:5000])
watchlist <- list(eval = dtest, train = dtrain)
xgb.fit <- xgb.train(param = param, data = dtrain, label = y, nround = 2000,
                     watchlist = watchlist, early.stop.round = 200, maximize = FALSE)
pred_xgboost <- predict(xgb.fit, mtest)

cat("Training model - Xgboost no.2\n") 
dtrain <- xgb.DMatrix(mtrain[-c(nrow(mtrain)-4999:nrow(mtrain)),],
                      label = y[-c(length(y)-4999:length(y))])
dtest <- xgb.DMatrix(mtrain[nrow(mtrain)-4999:nrow(mtrain),],
                     label = y[length(y)-4999:length(y)])
watchlist <- list(eval = dtest, train = dtrain)
xgb.fit2 <- xgb.train(param = param, data = dtrain, label = y, nround = 2000,
                     watchlist = watchlist, early.stop.round = 200, maximize = FALSE)
pred_xgboost2 <- predict(xgb.fit, mtest)

cat("Predicting Hazard...\n")
# Predict Hazard for the test set
submission <- data.frame(Id=test$Id)
submission$Hazard <- (pred_rf + pred_xgboost + pred_xgboost2)/3
write_csv(submission, "submit_rf+xgboost+stop100+cross.csv")
