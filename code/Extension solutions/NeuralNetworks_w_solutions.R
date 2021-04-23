################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 4/5                                 ##
##                                                                            ##
##                              Neural Networks in R                          ##
##                                  Walkthrough                               ##
##                                                                            ##
################################################################################

# The data used today is called the "Adult Data Set" and is census data from the US
# It is typically used in machine learning benchmarking applications
# The original data can be accessed from the UCI Machine Learning Repository:
# https://archive.ics.uci.edu/ml/datasets/adult

#### Extension exercises ####

## Train a neural network on the same data we worked on yesterday to solve the 
# same prediction problem -- i.e. predicting presidential vote choice
#
# Tasks:
# 1. Construct a neural network fit to the CCES data
#
# 2. Refine your prediction accuracy by varying the network structure/
# hyperparameters
# 
# 3.  Extend the network to a multi-class prediction problem? I.e. alter
# the network so that you predict "Trump","Clinton", or "Other"

#### 1 + 2. Build a neural network on CCES data ####
## I won't focus too much on tuning, and leave that to you
# Here I'll just demonstrate how to build the basic neural network approach

library(tensorflow)
library(keras)
library(tidyverse)
library(recipes)

set.seed(89)

## Read in the data
cces <- read_csv("data/cces_formatted_oxml.csv")

# Let's break out our main outcome
vote2016 <- cces$vote2016
cces$vote2016 <- NULL

# For extension 1:
cces$votetrump <- ifelse(vote2016 == "Donald Trump", 1, 0)
train_index <- sample(c(T,F), nrow(cces), replace = TRUE)

# Break up our data into train and test
cces_train <- cces[train_index,]
cces_test <- cces[!train_index,]

y_train <- cces_train$votetrump
y_test <- cces_test$votetrump

# Construct a "recipe"
rec_obj <- recipe(votetrump ~ ., data = cces) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% # One-hot encode columns
  step_center(all_predictors(), -all_outcomes()) %>% # Centre all predictors on 0
  step_scale(all_predictors(), -all_outcomes()) %>% # Scale all predictors with sd=1
  prep(data = cces)

x_train <- bake(rec_obj, new_data = cces_train) %>% select(-votetrump)
x_test  <- bake(rec_obj, new_data = cces_test) %>% select(-votetrump)

## Construct a neural network (I'm just going to bump up the number of nodes)
model <- keras_model_sequential() %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  
  compile(
    optimizer = 'sgd',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

history <- fit(
  object = model,
  x = as.matrix(x_train), 
  y = y_train, 
  batch_size = 50,
  epochs = 30,
  validation_split = 0.30
)

# Notice that the validation accuracy starts to decline towards the end of training, 
# while the training loss continues to decrease
# A sure sign of overfitting!

#### 3. Multiclass outcome

## The main difference here is that we need a new outcome

## Let's define y again as a three-class variable
# For some reason keras wants it numeric and indexed from 0
# This is likely for Python compatibility 
cces$votetrump <- NULL
cces$votechoice <- case_when(vote2016 == "Donald Trump" ~ 0,
                             vote2016 == "Hillary Clinton" ~ 1,
                             TRUE ~ 2) 

cces_train <- cces[train_index,]
cces_test <- cces[!train_index,]

# We'll separately transform the outcome using the to_categorical command
y_train <- to_categorical(cces_train$votechoice, num_classes = 3)
y_test <- to_categorical(cces_test$votechoice, num_classes = 3)

rec_obj <- recipe(votechoice ~ ., data = cces) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data = cces)

x_train <- bake(rec_obj, new_data = cces_train) %>% select(-votechoice)
x_test  <- bake(rec_obj, new_data = cces_test) %>% select(-votechoice)

model2 <- keras_model_sequential() %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax') %>% # Changed this to capture 3 class output 
  
  compile(
    optimizer = 'sgd',
    loss      = 'categorical_crossentropy', # Notice we alter the loss function
    metrics   = c('accuracy')
  )

history2 <- fit(
  object = model2,
  x = as.matrix(x_train), 
  y = y_train, 
  batch_size = 50,
  epochs = 30,
  validation_split = 0.30
)

