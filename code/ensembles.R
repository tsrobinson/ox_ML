################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 4/5                                 ##
##                                                                            ##
##                              Ensemble Methods                              ##
##                                  Walkthrough                               ##
##                                                                            ##
################################################################################

# This walkthrough uses the Cooperative Congressional Election Survey 2018 data
# Surveys US voters across all US states + D.C.
# Full data and codebook are available at https://doi.org/10.7910/DVN/ZSBZ7K

#### 0. System setup ####

## Set up commands -- follow this to set up on your own machine
# install.packages("SuperLearner")
# install.packages(c("caret", "glmnet", "randomForest", "ggplot2", "RhpcBLASctl"))

#### 1. Load packages and data ####
library(SuperLearner)

set.seed(89)

#NB THIS IS A PLACEHOLDER FOR THE FINAL CODE