library(pacman)
p_load(foreign)       # SPSS
p_load(caTools)       # Split
p_load(caret)         # NearzeroVariance
p_load(DMwR)          # Knn
p_load(Boruta)        # Boruta
p_load(ROSE)          # Rose
p_load(e1071)         # Required for Random Forest
p_load(randomForest)  #Random Forest Model
p_load(MLmetrics)     # F1 score

# ====================================== 1. Data Loading  ========================================
# Load SPSS data file as data frame
secom_1<- read.spss("C:/Users/sszfo/Desktop/MPMD 2nd Semester/Data Mining/A03_Case_Study_SEMICONDUCTOR/secom_mod.SAV", to.data.frame = TRUE)

# ====================================== 2. Split ================================================
set.seed(500)

split = sample.split(secom_1$class, SplitRatio =0.8)
training_set = subset(secom_1, split == TRUE)
test_set= subset(secom_1, split == FALSE)

# ====================================== 3. Dimension Reduction ==================================
# Zero Variance, Missing Values > 55%, Irrelevant Features
training_set <- training_set[, -nearZeroVar(training_set)]
training_set <- training_set[, -which(colMeans(is.na(training_set)) > 0.55)]
training_set <- subset( training_set, select = -timestamp)
training_set <- subset( training_set, select = -ID)

# ====================================== 4. Outlier Treatment ====================================
OTONA <- function(x){
  x = scale(x, center=TRUE, scale=TRUE)
  x[x>=3]<-NA
  x[x<=-3]<-NA
  x <- x * attr(x, 'scaled:scale') + attr(x, 'scaled:center')
  x
  }
training_set[,2:ncol(training_set)] <- data.frame(sapply(training_set[,2:ncol(training_set)],OTONA))

# ====================================== 5. Missing Values Imputation ===========================
set.seed(500)
training_set <- knnImputation(training_set, k=8, scale = T, meth = "weighAvg", distData = NULL)

# ====================================== 6. Feature Selection ===================================
# Boruta
set.seed(500)
boruta_train <- Boruta(class ~ ., 
                       data = training_set, 
                       doTrace=2)
# Selected Features
training_selected <- as.data.frame(training_set[,c("class",getSelectedAttributes(boruta_train, withTentative = TRUE))])

# ====================================== 7. Balancing ===========================================
set.seed(500)
training_balanced <- ROSE(class ~ ., data = training_selected, N=2500, hmult.majo = 0.5, hmult.mino = 1 )$data
training_balanced$class <- as.factor(training_balanced$class)

# ====================================== 8. Modeling ============================================
# Test_set preparation: -ID-timestamp, Zero Variance, >55% NA's, Outliers, Imputation and Target as factor
test_set <- subset( test_set, select = -timestamp)
test_set <- subset( test_set, select = -ID)
test_set <- test_set[, -nearZeroVar(test_set)]
test_set <- test_set[, -which(colMeans(is.na(test_set)) > 0.55)]
test_set[,2:ncol(test_set)] <- data.frame(sapply(test_set[,2:ncol(test_set)],OTONA))
set.seed(500)
test_set <- knnImputation(test_set, k=8, scale = T, meth = "weighAvg", distData = NULL)
test_set$class <- as.factor(test_set$class)

# Random forest
set.seed(500)
RF_model <- randomForest(class ~ ., data = training_balanced, ntree= 6000, mtry = 2)

#==================================== 9. Evaluation ==========================================================

RF_rose_pred <- predict(RF_model, test_set,type="response")
confusionMatrix(RF_rose_pred, as.factor(test_set$class), positive = "1")
roc.curve(test_set$class, RF_rose_pred, plotit = T, main ="ROC curve / Rose")
cat("F1 score: ",F1_Score(test_set$class, RF_rose_pred, positive = "1"),"\n")

