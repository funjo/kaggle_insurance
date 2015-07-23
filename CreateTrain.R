# GOAL: Predict a transformed count of hazards or pre-existing 
# damages using a dataset of property information.
library(caret)
setwd('~/Documents/__NYC_DSA/Dev/kaggle_insurance/')
train = read.csv('train.csv')

# Take out id column and separate y from x
train = train[-1]
y = train$Hazard

# Convert all the characters to factors for training
extractFeatures <- function(data) {
  character_cols <- names(Filter(function(x) x=="character", sapply(data, class)))
  for (col in character_cols) {
    data[,col] <- as.factor(data[,col])
    data[,col] <- ordered(data[,col])
    # if needed, conversion to numeric: data[,col] <- as.numeric(data[,col])
  }
  return(data)
}
train <- extractFeatures(train)

# no NAs
sum(is.na(train)) 

# Create 80% of train into training set for our models. 
# Try to keep same proportions as original.
train_index <- createDataPartition(y, p = 0.8,
                                   list = FALSE,
                                   times = 1)

# Are there any left out of the training data set? Yes.
orig = unique(y)
new = unique(y[train_index])
orig[!(orig %in% new)]

# Find near zero values
nzv <- nearZeroVar(train, saveMetrics = FALSE)
fil_train <- train[, -nzv]
write.csv(fil_train[train_index, ], file='train80.csv')
write.csv(fil_train[-train_index, ], file='train20.csv')

nrow(fil_train[-train_index, ])/nrow(train) * 100

# The optimal power transformation is found via the Box-Cox Test where
# -1. is a reciprocal
# -.5 is a recriprocal square root
# 0.0 is a log transformation
# .5 is a square toot transform and
# 1.0 is no transform.
# library(MASS)
# boxcox(train$Hazard ~., data = train)
# boxcox(train$Hazard ~ vars, data = train)
