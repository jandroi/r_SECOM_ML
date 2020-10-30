# === PREPARE SYSTEM
# set language to English
Sys.setenv(LANG = "en")
library(pacman)
p_load(data.table)
p_load(curl)
p_load(foreign)
p_load(caTools)
p_load(caret)
p_load(DMwR)
p_load(Boruta)
p_load(ROSE)
p_load(e1071)
p_load(randomForest)
p_load(MLmetrics)


#====================================== 1. Data Loading ========================================================
# Read the SPSS data file and setting it up as a data frame
secom_1<- read.spss("data/secom_mod.SAV", to.data.frame = TRUE)

# split data to training and test sets
set.seed(500)

split = sample.split(secom_1$class, SplitRatio =0.8)
training_set = subset(secom_1, split == TRUE)
test_set= subset(secom_1, split == FALSE)

# ============================== 3. Reducing dimensionality of data by feature removal  ==================================

training_set <- training_set[, -nearZeroVar(training_set)]
training_set <- training_set[, -which(colMeans(is.na(training_set)) > 0.55)]
training_set <- subset( training_set, select = -timestamp)
training_set <- subset( training_set, select = -ID)

# ============================== 4. Oulier Identification and Treatment ==================================
# Number of outlier in training set
OTONA <- function(x){
  x = scale(x, center=TRUE, scale=TRUE)
  x[x>=3]<-NA
  x[x<=-3]<-NA
  x <- x * attr(x, 'scaled:scale') + attr(x, 'scaled:center')
  x
}

training_set[,2:ncol(training_set)] <- data.frame(sapply(training_set[,2:ncol(training_set)],OTONA))

# ============================== 5. Mising Value Imputation  ==================================

# checking for number of missing values
# if there is missing value, we have to impute the missing values
sum(is.na(training_set))
sum(is.na(test_set))

# -> As a result, we found out 16152 NA in our training dataset and 1780 in our test dataset
set.seed(500)
training_set <- knnImputation(training_set, k=8, scale = T, meth = "weighAvg", distData = NULL)

# checking if there are missing values after knn imputation
sum(is.na(training_set))

# ============================== 6. Important Features selection  ==================================
#Boruta
set.seed(500)
boruta_train <- Boruta(class ~ .,
                       data = training_set,
                       doTrace=2, maxRuns=500)

# print names of important features
print(paste("feature selected: ",getSelectedAttributes(boruta_train)))
print(paste("feature nonrejected:",getNonRejectedFormula(boruta_train)))

# create importance chart
plotImpHistory(boruta_train)

# Selected Features
training_selected <- as.data.frame(training_set[,c("class","feature081",
                     getSelectedAttributes(boruta_train))])

#ROSE balanced
training_balanced <- ROSE(class ~ ., data = training_selected, N=2500, hmult.majo = 0.5, hmult.mino = 1 )$data
training_balanced$class <- as.factor(training_balanced$class)



table(training_selected$class)
table(training_balanced$class)


# ======================================= 7.1 Data Preparation for Test set ============================================
## Data cleansing for test set
# Logic here is any column with constant values will have Zero Variance, delete that column for test set
test_set <- subset( test_set, select = -timestamp)
test_set <- subset( test_set, select = -ID)

test_set <- test_set[, -nearZeroVar(test_set)]

#Remove columns with missing values more than 55%
test_set <- test_set[, -which(colMeans(is.na(test_set)) > 0.55)]

test_set$class <- as.factor(test_set$class)

# Outlier removal from test dataset
test_set[,2:ncol(test_set)] <- data.frame(sapply(test_set[,2:ncol(test_set)],OTONA))

# Impute missing value using KNN
set.seed(500)
test_set <- knnImputation(test_set, k=8, scale = T, meth = "weighAvg", distData = NULL)
sum(is.na(test_set))



#============================================= 8. Random forest =============================================================
# Random forest with Balancing
set.seed(500)
RF_model <- randomForest(class ~ ., data = training_balanced, ntree= 6000, mtry = 2)



#==================================== 9. confusion Matrix ==========================================================

RF_rose_pred <- predict(RF_model, test_set,type="response")

confusionMatrix(RF_rose_pred, as.factor(test_set$class), positive = "1")

roc.curve(test_set$class, RF_rose_pred, plotit = T, main ="ROC curve / Rose")

cat("F1 score: ",F1_Score(test_set$class, RF_rose_pred, positive = "1"),"\n")
