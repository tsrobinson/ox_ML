################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 2/5                                 ##
##                                                                            ##
##                          LASSO Models and Cross Validation                 ##
##                                  Walkthrough                               ##
##                                                                            ##
################################################################################

## NB: This code adapts the replication materials contained in Blackwell and 
# Olson (2021), ``Reducing Model Misspecification and Bias in the Estimation of
# Interactions'', available at: https://doi.org/10.7910/DVN/HZYFRI

## NB: In turn, this paper replicates original findings presented in:
# Escribà‐Folch, A., Meseguer, C. and Wright, J. (2018), Remittances and Protest
# in Dictatorships. American Journal of Political Science, 62: 889-904. 
# https://doi.org/10.1111/ajps.12382

## NB: For the purpose of demonstration, we'll make several simplifying
# assumptions about both the post-double selection procedure and the inference 
# model -- you can review the two papers to compare how our strategy differs

# install.packages("haven")
# install.packages("lfe")
# install.packages("glmnet")

library(haven)
library(lfe)
library(glmnet)

set.seed(89)

emw <- read_dta("data/efmw_replication.dta")

# Create variables included in original paper

emw$dist <- log(1 + (1 / (emw$dist_coast)))
emw$distwremit <- log(1 + ( (emw$richremit / 1000000) * (emw$dist)))

# limit the data to relevant columns and complete cases of those variables

emw <- emw[, c("Protest", "remit", "dict", "l1gdp", "l1pop", "l1nbr5", "l12gr",
               "l1migr", "elec3", "cowcode", "period", "distwremit",
               "caseid", "year")]

emw <- na.omit(emw)

controls <- c("l1gdp", "l1pop", "l1nbr5", "l12gr", "l1migr", "elec3")

# Generate fixed effects (don't worry too much about this)
contr.list <- list(contr.sum, contr.sum)
names(contr.list) <- c("factor(period)","factor(cowcode)")
mod_mat <- model.matrix(~factor(period)+factor(cowcode),data=emw,contrasts.arg=contr.list)[,-1]

# Matrix of controls incl. fixed effects
X <- as.matrix(cbind(emw[,controls],mod_mat))

# Moderator of interest
V <- emw$dict

# Interacted version
VX <- as.matrix(V*X)

# Treatment
D <- emw$remit

# Interacted treatment
DV <- D*V

# Outcome
Y <- emw$Protest

#### Stage 1. Estimate LASSO models ####

lasso_selector <- function(LHS, RHS, folds) {
  
  # Get best lambda using cross-fold validation
  cv_lambda <- cv.glmnet(x = RHS, y = LHS, alpha = 1, nfolds = folds)$lambda.min
  
  # Estimate final model
  lasso <- glmnet(x=RHS, y=LHS, alpha=1, lambda = cv_lambda)
  
  # Find non-zero coefficients
  coef_index <- which(coef(lasso) != 0) - 1 
  
  return(coef_index)
  
}

## Define RHS matrix

RHS_matrix <- as.matrix(cbind(V = V,X,VX))

# Optional but useful to keep track of names
colnames(RHS_matrix) <- c("V", colnames(X), 
                          paste0("V_",colnames(X)))

Y_lasso <-  lasso_selector(LHS = Y, RHS = RHS_matrix, folds = 10)
D_lasso <-  lasso_selector(LHS = D, RHS = RHS_matrix, folds = 10)
DV_lasso <- lasso_selector(LHS = DV, RHS = RHS_matrix, folds = 10)

selected_columns <- unique(c(Y_lasso, D_lasso, DV_lasso))

#### Stage 2. Estimate inference model

ds_matrix <- as.data.frame(cbind(Protest=Y,
                                 remit=D,
                                 remit_dict=DV,
                                 RHS_matrix[,selected_columns]))

ds_model <- glm("Protest~.", data = ds_matrix)

naive_model <- glm(paste0(c("Protest ~ remit*dict",
                            controls,
                            "as.factor(period) + as.factor(cowcode)"), 
                          collapse = " + "),
                   data = emw)

fmod_model <- glm("Protest~.", 
                  data = as.data.frame(cbind(Protest=Y,
                                             remit=D,
                                             remit_dict=DV,
                                             RHS_matrix)))

summary(naive_model)
summary(ds_model)

# Compare against fully moderated model
summary(fmod_model)

#### Extension task 1 ####

# 1. Generate a training dataset X with 100 variables and 2000 observations,
# where each observation is a draw from a random uniform distribution between -5
# and 5.

# 2. Generate an outcome vector Y that has the following features
#   a. Linear in relation to all 100 variables
#   b. As X1 increases by 1, Y increases by 10
#   c. As X2 increases by 1, Y decreases by 5
#   d. X3-X10 do not affect Y
#   e. X11-X100 have coefficients drawn from a random normal distribution with
# mean = 0, sd = 0.05

# 3. Estimate a cross-validated LASSO model to find lambda

# 4. Estimate a final LASSO model using the results from 3.

# Qs. What are the sizes of coefficient X1 X2? Do X3-X10 have non-zero
# coefficients? What about X11-X100?

#### Extension task 2 ####

## Blackwell and Olson propose a slightly more complicated regularization
# procedure in the post-double selection, where they apply separate 
# regularisation to each coefficient.

# Further information can be found on pages 14-15 of the article, available here:
# https://mattblackwell.org/files/papers/lasso-inters.pdf

## To see this in action, look at the rlasso_cluster function from line 221 in 
# the following file:
# https://github.com/mattblackwell/inters/blob/master/R/lasso_interactions.R

# 1. Using the help file for glmnet, i.e. type `?glmnet` into the console, and
# looking at lines 252-254 in the github file above, what argument is passed to
# glmnet to penalize individual coefficient values?

# 2. Without worrying too much about the surrounding code, describe what 
# lines 250-261 are doing algorithmically. 

