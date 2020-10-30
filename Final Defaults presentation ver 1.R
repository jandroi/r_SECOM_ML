
# ====================================== 7. Balancing ===========================================
training_balanced <- ROSE(class ~ ., data = training_selected, hmult.majo = 1, hmult.mino = 1 )$data
training_balanced$class <- as.factor(training_balanced$class)
table(training_balanced$class)
# ====================================== 8. Modeling ============================================
# Test_set preparation: -ID-timestamp, Zero Variance, >55% NA's, Outliers, Imputation and Target as factor
test_set <- subset( test_set, select = -timestamp)
test_set <- subset( test_set, select = -ID)
test_set <- test_set[, -nearZeroVar(test_set)]
test_set <- test_set[, -which(colMeans(is.na(test_set)) > 0.55)]
test_set[,2:ncol(test_set)] <- data.frame(sapply(test_set[,2:ncol(test_set)],OTONA))
test_set <- knnImputation(test_set, k=5, scale = T, meth = "weighAvg", distData = NULL)
test_set$class <- as.factor(test_set$class)

# Random forest
set.seed(500)
RF_model <- randomForest(class ~ ., data = training_balanced, ntree= 500, mtry = 5)

#==================================== 9. Evaluation ==========================================================

RF_rose_pred <- predict(RF_model, test_set,type="response")
confusionMatrix(RF_rose_pred, as.factor(test_set$class), positive = "1")
roc.curve(test_set$class, RF_rose_pred, plotit = T, main ="ROC curve / Rose")
cat("F1 score: ",F1_Score(test_set$class, RF_rose_pred, positive = "1"),"\n")








