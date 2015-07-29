# Adaboosting
setwd('~/Documents/__NYC_DSA/Dev/kaggle_insurance/')
library(rpart)
library(mlbench)
library(caret)
library(lattice)
library(ggplot2)
library(adabag)
library(MASS)
library(gbm)
train <- read.csv('train.csv')
test <- read.csv('test.csv')
train <- train[ ,-1]
#train$Hazard[train$Hazard>1] = 0
#test$Hazard[test$Hazard>1] = 0

train$Hazard = factor(train$Hazard)

ptm <- proc.time()
model <- boosting(Hazard ~ ., data = train, mfinal = 100, # increasing the iterations reduces the prediction error
                           control = rpart.control(cp = -1), # http://stackoverflow.com/questions/16135708/adabag-boosting-function-throws-error-when-giving-mfinal10
                          coeflearn = 'Breiman')
proc.time() - ptm
model$importance
test$Hazard = as.factor(rep(2,nrow(test))) # need to have predictions so function runs.
model.pred <- predict.boosting(model,newdata=test)
model.pred$confusion
model.pred$error

#comparing error evolution in training and test set
errorevol(model,train)->evol.train
errorevol(model,test)->evol.test
plot.errorevol(evol.test, evol.train)
# Seems after 20 iterations, no significant improvement in error.
str(test)
submission <- data.frame(Id=test$Id)
submission$Hazard <- model.pred$class
summary(submission)
write.csv(submission, "ABoost_hazard.csv")

####======================###
####RESULTS
####======================###
# Using all levels of Hazard and mfinal = 5.
#> proc.time() - ptm
#user  system elapsed 
#152.047  32.267 184.389 
#> model.pred$error
#[1] 0.6752623

# Using all levels of Hazard and mfinal = 100.
#> proc.time() - ptm
#user   system  elapsed 
#2118.389  293.829 2412.225 
#> model.pred$error
#[1] 0.6388862

# Using levels 1 through 20 and all rest 20 
# train$Hazard[train$Hazard>20] = 20; test$Hazard[test$Hazard>20] = 20 or
# train$Hazard[train$Hazard>15] = 15; test$Hazard[test$Hazard>15] = 15 with
# log transforming the Y's 0.6395725
# results in:
#> proc.time() - ptm
#user   system  elapsed 
#1637.829  375.473 2014.395
#> model.pred$error
#[1] 0.6388862

# Using 1 and rest (train$Hazard[train$Hazard>1] = 0; test$Hazard[test$Hazard>1] = 0)
# and mfinal = 100
# > proc.time() - ptm
# user   system  elapsed 
# 841.460  293.384 1135.056 
#> model.pred$confusion
# Observed Class
# Predicted Class    0    1
# 0 5199 2546
# 1 1278 1176
# > model.pred$error
# [1] 0.3749387

# with only 20 iterations and coeflearn = Zhu, error = 0.4151387. 
# with only 20 iterations and coeflearn = Freund, error = 0.421708. 
# with only 20 iterations and coeflearn = Breiman, error = 0.390973.

# Using boosting.cv(v=10) and train$Hazard[train$Hazard>30] = 30; test$Hazard[test$Hazard>30] = 30
#> proc.time() - ptm
#user   system  elapsed 
#8273.164 1602.128 9880.937 
#> model.pred$confusion
# Observed Class
#Predicted Class    
#  0    1
#0 4904 2426
#1 1573 1296
#> model.pred$error
#[1] 0.3920973

