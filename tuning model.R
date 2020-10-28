library(foreign)
secom<- read.spss("secom_mod.SAV", to.data.frame = TRUE)


library(caret)
set.seed(500)
trainIndex <- createDataPartition(secom$class, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- secom[trainIndex,]
test <- secom[-trainIndex,]

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

train <- knnImputation(train, k=10)

library(Boruta)
boruta_train <- Boruta(class~.,data = train[,-1], doTrace=2)

# train1 <- 10k knn
train <- as.data.frame(train[,c("class",getSelectedAttributes(boruta_train, withTentative = TRUE))])

# Rose
trainR <- ROSE(class ~ ., data = train)$data

# Test Treatment
test <- test[ , -nearZeroVar(test)]
test <- test[ ,-which(colMeans(is.na(test)) > 0.55)]
test <- subset( test, select = -timestamp)
test <- knnImputation(test, distData = NULL )
test$class <- as.factor(test$class)


trainR$class <- as.factor(trainR$class)


# Model
control <- trainControl(method="repeatedcv", number=10, repeats=5)
metric <- "Accuracy"
mtry <- sqrt(ncol(trainR))
tunegrid <- expand.grid(.mtry=mtry)
model1 <- train(class~., data=trainR, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
model1

library(MLmetrics)
pred1 <- predict(model1, test)
cat("1 F1 score: ",F1_Score(test$class, pred1, positive='1'),"\n")
confusionMatrix(table(pred1, test$class), positive="1")
roc.curve(test$class, pred1, plotit = T, main="Pred4")

control2 <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
mtry <- sqrt(ncol(trainR))
model2 <- train(class~., data=trainR, method="rf", metric=metric, tuneLength=15, trControl=control2)
model2
plot(model2)


pred2 <- predict(model2, test)
cat("1 F1 score: ",F1_Score(test$class, pred2, positive='1'),"\n")
confusionMatrix(table(pred2, test$class), positive="1")
roc.curve(test$class, pred2, plotit = T, main="Pred4")


tunegrid <- expand.grid(.mtry=c(1:15))
model3 <- train(class~., data=trainR, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(model3)
plot(model3)

pred3 <- predict(model3, test)
cat("1 F1 score: ",F1_Score(test$class, pred3, positive='1'),"\n")
confusionMatrix(table(pred3, test$class), positive="1")
roc.curve(test$class, pred3, plotit = T, main="Pred4")


