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

#### 0. System setup ####

## Set up commands -- follow this to set up on your own machine
# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")
# install.packages("reticulate")
# tensorflow::install_tensorflow(version = "2.4", 
#                                method = "virtualenv", 
#                                envname = "r-reticulate")
# install.packages("keras")
# keras::install_keras()
# install.packages("recipes")

#### 1. Load packages and data ####
library(tensorflow)
library(keras)
library(tidyverse)
library(recipes)

set.seed(89)

## Read in the data
adult <- read_csv("https://raw.githubusercontent.com/MIDASverse/MIDASpy/master/Examples/adult_data.csv") %>% 
  drop_na() %>% # This is not good practise (we're just doing it for the sake of demonstration!)
  select(-X1)

#### 2. Setup the prediction problem ####

# Break up our data into train and test
train_index <- sample(c(T,F), nrow(adult), replace = TRUE)
adult_train <- adult[train_index,]
adult_test <- adult[!train_index,]

y_train <- ifelse(adult_train$class_labels == ">50K",1,0)
y_test <- ifelse(adult_test$class_labels == ">50K",1,0)

#### 3. Build a network ####

# Construct a "recipe"
rec_obj <- recipe(class_labels ~ ., data = adult) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% # One-hot encode columns
  step_center(all_predictors(), -all_outcomes()) %>% # Centre all predictors on 0
  step_scale(all_predictors(), -all_outcomes()) %>% # Scale all predictors with sd=1
  prep(data = adult)

x_train <- bake(rec_obj, new_data = adult_train) %>% select(-class_labels)
x_test  <- bake(rec_obj, new_data = adult_test) %>% select(-class_labels)

## Construct a neural network
model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  
  compile(
    optimizer = 'sgd', # Stochastic gradient descent -- what we hand-coded on Monday!
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy') # Determines what is plotted while training occurs
  )

#### 4. Train the network ####

history <- fit(
  object = model,
  x = as.matrix(x_train), 
  y = y_train, 
  batch_size = 50,
  epochs = 30,
  validation_split = 0.30
)

#### 5. Dropout ####

## Add dropout
model_w_dropout <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  
  compile(
    optimizer = 'sgd', # Stochastic gradient descent -- a variation of what we hand-coded on Monday!
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy') # Determines what is plotted while training occurs
  )

history2 <- fit(
  object = model_w_dropout,
  x = as.matrix(x_train), 
  y = y_train, 
  batch_size = 50,
  epochs = 30,
  validation_split = 0.30
)

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