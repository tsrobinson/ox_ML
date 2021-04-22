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

# This walkthrough uses the Cooperative Congressional Election Survey 2018 data
# Surveys US voters across all US states + D.C.
# Full data and codebook are available at https://doi.org/10.7910/DVN/ZSBZ7K

# install.packages("randomForest")
# install.packages("BART")
# install.packages("dplyr")

library(randomForest)
library(BART)
library(dplyr)

################################################################################
#### 0. Clean the raw data ####
#
# library(tidyverse)
# library(haven)
#
# # !!! NB: This file is 800MB
# cces <- read_dta("https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/ZSBZ7K/H5IDTA")
# 
# cces_format <- cces %>%
#   select(inputstate, birthyr, gender, sexuality, trans, educ, votereg, race, employ,
#          internethome, internetwork, marstat, pid3, ideo5, pew_bornagain,
#          pew_religimp, pew_churatd, pew_prayer, religpew, child18num, newsint, faminc_new,
#          ownhome, urbancity, immstat, cit1, unionhh, investor, CC18_317) %>%
#   mutate(child18num = ifelse(is.na(child18num),0,child18num)) %>% 
#   as_factor() %>% 
#   rename(vote2016 = CC18_317) %>% 
#   na.omit() # Note, I'm having to remove lots of missing data!
# 
# apply(cces_format, 2, function (x) sum(is.na(x)))
# nrow(na.omit(cces_format))
# 
# write_csv(cces_format, "data/cces_formatted_oxml.csv")
################################################################################

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

# To make this easier on RStudio Cloud, let's only consider a few variables
train_vars <- c("birthyr", "gender", "sexuality", "trans", "educ", "votereg", "race")

# Combine all the above steps
X_train <- cces[train_indices, c("votetrump",train_vars)]
X_test <- cces[test_indices, train_vars]

# Extract Y values from test dataset so we can validate the results
Y_val <- cces$votetrump[test_indices]

# Check the proportions of outcomes across sets of data
prop.table(table(X_train$votetrump))
prop.table(table(Y_val))

#### 3. Run a random forest model ####

## Train
rf_model <- randomForest(votetrump ~ ., data = X_train, 
                         mtry = 2, ntree = 500,
                         importance = TRUE)

## Inspect the mechanics of the model
importance(rf_model)
varImpPlot(rf_model)

# Make predictions on new data
rf_predictions <- predict(rf_model, X_test, type = "class")

# Get the accuracy of predictions on the test data
rf_acc <- mean(rf_predictions == Y_val)

# Visualise this by merging the predictions and true Y into a data.frame
rf_comp <- data.frame(y_pred = rf_predictions,
                      y_true = Y_val)  


#### 4. Run a BART model

# Just to clear out some memory we can run a "garbage collection"
gc()

# Train the model
# NB: The BART package requires a numeric outcome variable
bart_model <- pbart(x.train = X_train[,-1],
                    y.train = ifelse(X_train$votetrump == "Trump",1,0),
                    ntree = 50L, numcut = 100L)

# Predict the out-of-sample outcomes
bart_pred_probs <- predict(bart_model, newdata = bartModelMatrix(X_test))

# This yields a probability, so let's round to either 0 or 1
bart_pred_bin <- round(bart_pred_probs$prob.test.mean)

# And relabel
bart_predictions <- ifelse(bart_pred_bin == 1, "Trump","Other")

# Then calculate as with RF
bart_acc <- mean(bart_predictions == Y_val)

# Compare against actual and RF
full_comp <- data.frame(y_true = Y_val,
                        y_rf = rf_predictions,
                        y_bart = bart_predictions)

#### Extension task ####

## Can you come up with a more accurate estimator?

## Some options to consider:

# a. Add more variables
# b. Alter the hyperparameters

## For an interesting application, read the following short article:
# Bisbee, James (2019) “BARP: Improving Mister P Using Bayesian Additive 
# Regression Trees,” American Political Science Review. Cambridge University 
# Press, 113(4), pp. 1060–1065. doi: 10.1017/S0003055419000480.

# Available online here: https://cup.org/3gzOt9o 

# Why is BART so useful for this specific application? Think about inference vs.
# prediction issues




