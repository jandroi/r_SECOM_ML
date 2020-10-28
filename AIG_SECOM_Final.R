
library(foreign)
secom<- read.spss("secom_mod.SAV", to.data.frame = TRUE)


library(caret)
set.seed(12345)
trainIndex <- createDataPartition(secom$class, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)


train <- secom[trainIndex,]
test <- secom[-trainIndex,]

dim(train)
# Near Zero Value + 55% Missing Values + Timestamp 
train <- train[ , -nearZeroVar(train)]
train <- train[ ,-which(colMeans(is.na(train)) > 0.55)]
train <- subset( train, select = -timestamp)

# OTONA
OTONA <- function(x){
  x = scale(x, center=TRUE, scale=TRUE)
  x[x>=3]<-NA
  x[x<=-3]<-NA
  x <- x * attr(x, 'scaled:scale') + attr(x, 'scaled:center')
  x
}

train[,3:ncol(train)] <- data.frame(sapply(train[,3:ncol(train)],OTONA))


# Knn Imputation
library(DMwR)

train <- knnImputation(train, distData = NULL )


# Boruta Selection
library(Boruta)
boruta_train4 <- Boruta(class~.,data = train[,-1], doTrace=2)

getSelectedAttributes(boruta_train)
TentativeRoughFix(boruta_train)

train1 <- as.data.frame(train[,c("class",getSelectedAttributes(boruta_train, withTentative = TRUE))])
train2 <- as.data.frame(train[,c("class",getSelectedAttributes(boruta_train2, withTentative = TRUE))])
train3 <- as.data.frame(train[,c("class",getSelectedAttributes(boruta_train3, withTentative = TRUE))])
train4 <- as.data.frame(train[,c("class",getSelectedAttributes(boruta_train4, withTentative = TRUE))])

dim(train1)
dim(train2)
dim(train3)
dim(train4)


# Rose Balance
library(ROSE)

trainR1 <- ROSE(class ~ ., data = train1)$data
trainR2 <- ROSE(class ~ ., data = train2, p = 0.2)$data
trainR3 <- ROSE(class ~ ., data = train3)$data
trainR4 <- ROSE(class ~ ., data = train4)$data

table(trainR1$class)
table(trainR2$class)
table(trainR3$class)
table(trainR4$class)


# Test Treatment
test <- test[ , -nearZeroVar(test)]
test <- test[ ,-which(colMeans(is.na(test)) > 0.55)]
test <- subset( test, select = -timestamp)
test <- knnImputation(test, distData = NULL )
test$class <- as.factor(test$class)


trainR1$class <- as.factor(trainR1$class)
trainR2$class <- as.factor(trainR2$class)
trainR3$class <- as.factor(trainR3$class)
trainR4$class <- as.factor(trainR4$class)


# Model

library(randomForest)
model1 <- randomForest(class ~ ., data =trainR1)
model2 <- randomForest(class ~ ., data =trainR2)
model3 <- randomForest(class ~ ., data =trainR3)
model4 <- randomForest(class ~ ., data =trainR4)

pred1 <- predict(model1,test)
pred2 <- predict(model2,test)
pred3 <- predict(model3,test)
pred4 <- predict(model4,test)

library(MLmetrics)
cat("1 F1 score: ",F1_Score(test$class, pred1, positive='1'),"\n")
cat("2 F1 score: ",F1_Score(test$class, pred2, positive='1'),"\n")
cat("3 F1 score: ",F1_Score(test$class, pred3, positive='1'),"\n")
cat("4 F1 score: ",F1_Score(test$class, pred4, positive='1'),"\n")


confusionMatrix(table(pred1, test$class), positive="1")
confusionMatrix(table(pred2, test$class), positive="1")
confusionMatrix(table(pred3, test$class), positive="1")
confusionMatrix(table(pred4, test$class), positive="1")


roc_over1 <- roc.curve(test$class, pred1, plotit = T, main="Pred1")
roc_over1

roc_over2 <- roc.curve(test$class, pred2, plotit = T, main="Pred2")
roc_over2

roc_over3 <- roc.curve(test$class, pred3, plotit = T, main="Pred3")
roc_over3

roc_over4 <- roc.curve(test$class, pred4, plotit = T, main="Pred4")
roc_over4

RMSE(pred1,test$class)


fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  classProbs = FALSE
  ## repeated ten times
  #repeats = 30
  
      )

model5<- train(class~., data=trainR4, method='rf')

pred5 <- predict(model5,test)
cat("5 F1 score: ",F1_Score(test$class, pred5, positive='1'),"\n")
confusionMatrix(table(pred5, test$class), positive="1")
roc_over5 <- roc.curve(test$class, pred5, plotit = T, main="Pred4")
roc_over5
model5



