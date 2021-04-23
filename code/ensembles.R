################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 5/5                                 ##
##                                                                            ##
##                                Ensemble Methods                            ##
##                                  Walkthrough                               ##
##                                                                            ##
################################################################################

# This walkthrough uses the Cooperative Congressional Election Survey 2018 data
# Surveys US voters across all US states + D.C.
# Full data and codebook are available at https://doi.org/10.7910/DVN/ZSBZ7K

#### 0. System setup ####

## Set up commands -- follow this to set up on your own machine
# install.packages("glmnet")
# install.packages("randomForest")
# install.packages("nlmrt")

#### 1. Load packages and data ####
library(glmnet)
library(randomForest)
library(nlmrt)

set.seed(89)

cces <- read.csv("data/cces_formatted_oxml.csv")

# Convert predictors to factors
for (v in 1:ncol(cces)) {
  if (is.character(cces[[v]])) {
    cces[[v]] <- as.factor(cces[[v]])
  }
}

# Recode outcome variable
cces$votetrump <- ifelse(cces$vote2016 == "Donald Trump", 1, 0)

# Get an approximate 2:1 split of train and test data
train_indices <- sample(1:nrow(cces), 30000)
test_indices <-  setdiff(1:nrow(cces), train_indices)

train_vars <- c("birthyr", "gender", "sexuality", "trans", "educ", "votereg", "race")

x_train <- cces[train_indices, train_vars]
x_test <- cces[test_indices, train_vars]

y_train <- cces$votetrump[train_indices]
y_test <- cces$votetrump[test_indices]

#### 2. Estimate individual models ####
## NB. we won't cross-validate our results here (except choosing LASSO lambda)

## Estimate some models

# Logistic

logit_model <- glm(paste0("votetrump ~ ",paste0(train_vars, collapse = " + ")),
                   data = cbind(votetrump = y_train, x_train),
                   family = binomial(link="logit"))

# LASSO

# Generate dummy variables (because glmnet doesn't like factors)
lasso_format <- function(X) {
  
  cat_vars <- c("gender","sexuality","trans","educ","votereg","race")
  contr.list <- lapply(1:length(cat_vars), function (x) contr.sum)
  names(contr.list) <- paste0("factor(",cat_vars,")")
  fac_mod_mat <- model.matrix(as.formula(paste0("~", paste0(names(contr.list), collapse = " + "))),
                              data=X[,cat_vars],contrasts.arg=contr.list)[,-1]
  
  mod_mat <- cbind(X$birthyr, fac_mod_mat)
  
  return(mod_mat)
  
}

x_train_lasso <- lasso_format(x_train)

cv_lambda <- cv.glmnet(x = x_train_lasso, y = y_train, alpha = 1)$lambda.min

lasso_mod <- glmnet(x = x_train_lasso, y = y_train, alpha = 1, lambda = cv_lambda)

## Random Forest

rf_model <- randomForest(votetrump ~ ., 
                         data = cbind(votetrump = as.factor(y_train), x_train))


#### 3. Generate stacking model ####

# Get predictions on training data
logit_yhat_train <- predict(logit_model, type = "response")

lasso_yhat_train <- predict(lasso_mod, newx = x_train_lasso, type = "response")

rf_yhat_train <- predict(rf_model, type = "prob")[,2]

train_preds <- data.frame(Y = y_train,
                          logit = as.numeric(logit_yhat_train),
                          lasso = as.numeric(lasso_yhat_train),
                          rf = as.numeric(rf_yhat_train))

## Estimate the weights using nonlinear least squares estimator

stack_model <- nlxb(Y ~ (b1 * logit + b2 * lasso + (1-b1-b2) * rf),
                    data = train_preds,
                    lower = numeric(2),
                    start = list(b1 = 1/3, b2 = 1/3))


# Inspect the coefficients
train_wgts <- c(stack_model$coefficients, 1 - sum(stack_model$coefficients))

#### 4. Generate a stacked predictor function

yhat_test_logit <- predict(logit_model, newdata = x_test, type = "response")

yhat_test_lasso <-  predict(lasso_mod, newx = lasso_format(x_test), type = "response")

yhat_test_rf <- predict(rf_model, newdata = x_test, type = "prob")[,2]

stacked_pred <- cbind(as.numeric(yhat_test_logit),
                      as.numeric(yhat_test_lasso),
                      as.numeric(yhat_test_rf))

yhat_test <- stacked_pred %*% train_wgts

# Accuracy

mean((yhat_test - y_test)^2)

#### Extension exercise ####

## 1. Using the various scripts we have gone through this week, can you compare
# the mean squared error of the logit, LASSO, and RF probabilities to the mean
# squared error of the stacked estimator?

## 2. Can you recode section 4 into a function that would allow you to make new
# predictions given an input dataset X_test, logit, lasso, and RF models, and a
# corresponding vector of weights recovered from the stacked estimator?

## 3. Can you perform similar analysis to the above using the SuperLearner
# package.
#
## Further information for task 3:
# The above code is unwieldy and we have packages that bundle the various
# operations into one or two function calls.
#
# A particularly good R package in this regard is SuperLearner which is developed
# by Eric Polley, Erin LeDell, Chris Kennedy, Sam Lendle, and Mark van der Laan
#
# These authors have also put together an excellent guide to the package which is
# available here:
# https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html
#
## Using the SuperLearner package and workflow in the above guide, can you repeat
# the above analysis? 
# 
# NB 1: Given the use of cross-validation etc. don't expect to find identical 
# results between our simplistic estimator and the SuperLearner equivalent.
#
# NB 2: You may find that this won't run on RStudio cloud for memory reasons,
# so if possible try to complete this exercise on your own machine.
