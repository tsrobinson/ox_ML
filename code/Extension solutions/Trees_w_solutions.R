################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 3/5                                 ##
##                                                                            ##
##                        Random Forest and BART Predictions                  ##
##                                  Walkthrough                               ##
##                                                                            ##
################################################################################

# These extensions use the Cooperative Congressional Election Survey 2018 data
# Surveys US voters across all US states + D.C.
# Full data and codebook are available at https://doi.org/10.7910/DVN/ZSBZ7K

# install.packages("randomForest")
# install.packages("BART")
# install.packages("dplyr")

library(randomForest)
library(BART)
library(dplyr)

#### Extension task ####

## Can you come up with a more accurate estimator?

## Some options to consider:

# a. Add more variables
# b. Alter the hyperparameters

## I'll focus on random forest here.

#### 1. Load the data ####
set.seed(89)
cces <- read.csv("data/cces_formatted_oxml.csv")

# Convert predictors to factors
for (v in 1:ncol(cces)) {
  
  if (is.character(cces[[v]])) {
    
    cces[[v]] <- as.factor(cces[[v]])
    
  }
  
}

## How complex is this data?

# How many columns?
ncol(cces)

# How about the effective number of columns?
unique_cols <- apply(cces,2, function (x) ifelse(is.numeric(x),1,length(unique(x))))
sum(unique_cols)

#### 2. Setup a prediction problem ####

# Simplify outcome
cces$votetrump <- ifelse(cces$vote2016 == "Donald Trump", "Trump","Other")

cces$votetrump <- as.factor(cces$votetrump)

# Split the data into test and train sets

train_indices <- sample(1:nrow(cces), 30000)
test_indices <-  setdiff(1:nrow(cces), train_indices)

#### 2. Run a random forest model ####

## First solution -- add state of residence:

# To make this easier on RStudio Cloud, let's only consider a few variables
train_vars <- c("birthyr", "gender", "sexuality", "trans", "educ", "votereg", "race",
                "inputstate")

# Combine all the above steps
X_train <- cces[train_indices, c("votetrump",train_vars)]
X_test <- cces[test_indices, train_vars]

# Extract Y values from test dataset so we can validate the results
Y_val <- cces$votetrump[test_indices]

# Check the proportions of outcomes across sets of data
prop.table(table(X_train$votetrump))
prop.table(table(Y_val))

## Train
rf_w_state <- randomForest(votetrump ~ ., data = X_train, 
                         mtry =2, ntree = 500,
                         importance = TRUE)

# Make predictions on new data
rf_state_pred <- predict(rf_w_state, X_test, type = "class")

# Get the accuracy of predictions on the test data
rf_state_acc <- mean(rf_state_pred == Y_val)

## A modest improvement to 67.3% accuracy

#### 3. Alter number of tries ####

rf_mtry <- randomForest(votetrump ~ ., data = X_train, 
                           mtry =4, ntree = 500,
                           importance = TRUE)

# Make predictions on new data
rf_mtry_pred <- predict(rf_mtry, X_test, type = "class")

# Get the accuracy of predictions on the test data
rf_mtry_acc <- mean(rf_mtry_pred == Y_val)

# 65.7%... Worse performance!!
# This might be because we have too few variables in the model -- increasing mtry
# overfits, as it effectively allows the model to choose a large proportion of the 
# possible variables at each split

#### 4. Add in the full vector of variables (but not the original vote var.) ####

X_train <- cces[train_indices,]
X_train$vote2016 <- NULL

X_test <- cces[test_indices,]
X_test$vote2016 <- NULL
X_test$votetrump <- NULL

# Extract Y values from test dataset so we can validate the results
Y_val <- cces$votetrump[test_indices]

## Train
rf_all_var <- randomForest(votetrump ~ ., data = X_train, 
                           mtry =4, ntree = 500,
                           importance = TRUE)

# Make predictions on new data
rf_all_pred <- predict(rf_all_var, X_test, type = "class")

# Get the accuracy of predictions on the test data
rf_all_acc <- mean(rf_all_pred == Y_val)

# 88.1% accuracy!!

#### 5. One final check: let's increase the number of trees

f_all_1000 <- randomForest(votetrump ~ ., data = X_train, 
                          mtry =4, ntree = 1000,
                          importance = TRUE)

# Make predictions on new data
rf_all_1000_pred <- predict(f_all_1000, X_test, type = "class")

# Get the accuracy of predictions on the test data
rf_all_1000_acc <- mean(rf_all_1000_pred == Y_val)

## A minor improvement to 88.3% -- probably not worth the additional computation
# time




