library(AppliedPredictiveModeling)
library(caret)
#library(rattle)
library(rpart.plot)
library(randomForest)

## importing the data
df_training <- read.csv("pml-training.csv", na.string=c("NA", ""), header=TRUE)
colnames_train <- colnames(df_training)
df_testing <- read.csv("pml-testing.csv",na.string=c("NA", ""), header=TRUE)
colnames_test <- colnames(df_testing)

## compare the colnames in train and test df. 
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_test)-1])

## count the number of non-NAs in each col
nonNAs <- function(x) {
  as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

## Build vector of missing data or NA cols to drop
colcnts <- nonNAs(df_training)
drops <- c()
for (cnt in 1:length(colcnts)) {
  if (colcnts[cnt] < nrow(df_training)) {
    drops <- c(drops, colnames_train[cnt])
  }
}

## drop NAs and the first 7 columns
df_training <- df_training[,!names(df_training) %in% drops]
df_training <- df_training[,8:length(colnames(df_training))]

df_testing <- df_testing[,!names(df_testing) %in% drops]
df_testing <- df_testing[,8:length(colnames(df_testing))]

## remaining columns
colnames(df_training)
colnames(df_testing)
print(dim(df_training))
print(dim(df_testing))

## check for covariates that have virtually no variablility.
nsv <- nearZeroVar(df_training, saveMetrics=TRUE)
nsv

## divide the training set into four roughly equal sets
set.seed(666)
ids_small <- createDataPartition(y=df_training$classe, p=0.25, list=FALSE)
df_small1 <- df_training[ids_small,]
df_remainder <- df_training[-ids_small,]

set.seed(666)
ids_small <- createDataPartition(y=df_remainder$classe, p=0.33, list=FALSE)
df_small2 <- df_remainder[ids_small,]
df_remainder <- df_remainder[-ids_small,]

set.seed(666)
ids_small <- createDataPartition(y=df_remainder$classe, p=0.5, list=FALSE)
df_small3 <- df_remainder[ids_small,]
df_small4 <- df_remainder[-ids_small,]

## split each set into a training set 60% and a testing set 40%
set.seed(666)
inTrain <- createDataPartition(y=df_small1$classe, p=0.6, list=FALSE)
df_small_training1 <- df_small1[inTrain,]
df_small_testing1 <- df_small1[-inTrain,]

set.seed(666)
inTrain <- createDataPartition(y=df_small2$classe, p=0.6, list=FALSE)
df_small_training2 <- df_small2[inTrain,]
df_small_testing2 <- df_small2[-inTrain,]

set.seed(666)
inTrain <- createDataPartition(y=df_small3$classe, p=0.6, list=FALSE)
df_small_training3 <- df_small3[inTrain,]
df_small_testing3 <- df_small3[-inTrain,]

set.seed(666)
inTrain <- createDataPartition(y=df_small4$classe, p=0.6, list=FALSE)
df_small_training4 <- df_small4[inTrain,]
df_small_testing4 <- df_small4[-inTrain,]


## apply the Classification Tree
set.seed(666)
modFit <- train(df_small_training1$classe ~ ., data = df_small_training1, method="rpart")
print(modFit, digits=3)

print(modFit$finalModel, digits=3)

## run on the df_small_testing1
predictions <- predict(modFit, newdata=df_small_testing1)

print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)

# Train on training set 1 of 4 with both preprocessing and cross validation.
set.seed(666)
modFit <- train(df_small_training1$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = df_small_training1, method="rpart")
print(modFit, digits=3)

# Run against testing set 1 of 4 with both preprocessing and cross validation.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)


## Random Forest
# Train on training set 1 of 4 with only cross validation.
set.seed(666)
modFit <- train(df_small_training1$classe ~ ., method="rf", trControl=trainControl(method = "cv", 
                number = 4), data=df_small_training1)
print(modFit, digits=3)

# Run against testing set 1 of 4.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)

# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))

set.seed(666)
modFit <- train(df_small_training1$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training1)
print(modFit, digits=3)

# Run against testing set 1 of 4.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)

# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))

# Train on training set 2 of 4 
set.seed(666)
modFit <- train(df_small_training2$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training2)
print(modFit, digits=3)

# Run against testing set 2 of 4.
predictions <- predict(modFit, newdata=df_small_testing2)
print(confusionMatrix(predictions, df_small_testing2$classe), digits=4)

# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))

# Train on training set 3 of 4.
set.seed(666)
modFit <- train(df_small_training3$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training3)
print(modFit, digits=3)

# Run against testing set 3 of 4.
predictions <- predict(modFit, newdata=df_small_testing3)
print(confusionMatrix(predictions, df_small_testing3$classe), digits=4)

# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))

# Train on training set 4 of 4.
set.seed(666)
modFit <- train(df_small_training4$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training4)
print(modFit, digits=3)

# Run against testing set 4 of 4.
predictions <- predict(modFit, newdata=df_small_testing4)
print(confusionMatrix(predictions, df_small_testing4$classe), digits=4)

# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))



