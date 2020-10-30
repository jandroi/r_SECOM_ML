# === PREPARE SYSTEM
# set language to English
Sys.setenv(LANG = "en")
if (!require("pacman")) install.packages("pacman")
if (!require("lattice")) install.packages("lattice")
if (!require("ggplot2")) install.packages("ggplot2")
# Step 1: Install and load R Packages
library(pacman)
p_load(data.table)
p_load(curl)

# ====================================== 1. Data Loading  ========================================
p_load(foreign)
secom<- read.spss("secom_mod.SAV", to.data.frame = TRUE)

# ====================================== 2. Split ================================================
p_load(caTools)
set.seed(42)
split = sample.split(secom$class, SplitRatio =0.8)
train = subset(secom, split == TRUE)
test = subset(secom, split == FALSE)

# ====================================== 3. Dimension Reduction ==================================

train <- train[ , -nearZeroVar(train)]
train <- train[ ,-which(colMeans(is.na(train)) > 0.55)]
train <- subset( train, select = -timestamp)
train <- subset( train, select = -ID)

# ====================================== 4. Outlier Treatment ====================================

OTONA <- function(x){
  x = scale(x, center=TRUE, scale=TRUE)
  x[x>=3]<-NA
  x[x<=-3]<-NA
  x <- x * attr(x, 'scaled:scale') + attr(x, 'scaled:center')
  x
}

train[,3:ncol(train)] <- data.frame(sapply(train[,3:ncol(train)],OTONA))

# ====================================== 5. Missing Values Imputation ===========================
p_load(DMwR)
train <- knnImputation(train, k=5, scale = T, meth = "weighAvg", distData = NULL)

# ====================================== 6. Feature Selection ===================================
p_load(Boruta)
boruta_train2 <- Boruta(class ~ .,
                       data = train,
                       doTrace=2)

train_selected <- as.data.frame(train[,c("class",getSelectedAttributes(boruta_train2, withTentative = TRUE))])

# ====================================== 7. Balancing ===========================================
p_load(ROSE)

train_balanced <- ROSE(class ~ ., data = train_selected)$data
table(train_selected$class)
table(train_balanced$class)

# ====================================== 8. Modeling ============================================
# Test Treatment
test <- test[ , -nearZeroVar(test)]
test <- test[ ,-which(colMeans(is.na(test)) > 0.55)]
test <- subset( test, select = -timestamp)
test <- knnImputation(test, distData = NULL )

# Converting $Class to factor types
train_balanced$class <- as.factor(train_balanced$class)
test$class <- as.factor(test$class)

# Model Random Forest
p_load(e1071)
p_load(randomForest)

RF_model <- randomForest(class ~ ., data = train_balanced)
model_pred <- predict(RF_model, test, positive="1")

confusionMatrix(table(model_pred, test$class), positive = "1")

cat("F1 score: ",F1_Score(test$class, model_pred, positive='1'),"\n")
confusionMatrix(table(model_pred, test$class), positive="1")
roc.curve(test$class, model_pred, plotit = T, main="RF_Model")
