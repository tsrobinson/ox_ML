################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 2/5                                 ##
##                                                                            ##
##                          LASSO Models and Cross Validation                 ##
##                                  Extensions                                ##
##                                                                            ##
################################################################################

# install.packages("haven")
# install.packages("lfe")
# install.packages("glmnet")

library(lfe)
library(glmnet)



#### Extension task 1 ####

# Lots of randomisation going on, so let's set a seed value:
set.seed(89)

# 1. Generate a training dataset X with 100 variables and 2000 observations,
# where each observation is a draw from a random uniform distribution between -5
# and 5.

# Create an empty data.frame
X <- matrix(nrow=2000, ncol=100)

# For every column in our dataframe, replace empty values with draws from random uniform
X <- apply(X, 2, function (x) runif(length(x),-5,5))

# Optional, but let's add some column names to make things clearer:

colnames(X) <- paste0("X",1:100)


# 2. Generate an outcome vector Y that has the following features
#   a. Linear in relation to all 100 variables
#   b. As X1 increases by 1, Y increases by 10
#   c. As X2 increases by 1, Y decreases by 5
#   d. X3-X10 do not affect Y
#   e. X11-X100 have coefficients drawn from a random normal distribution with
# mean = 0, sd = 0.05

## I will show two ways of constructing Y. The first is to iteratively add elements
# to the outcome. This is fine because by constraint a, we know these are linear
# terms, so we can add them sequentially:

# Address steps b and c first
Y <- 10*X[,1] -5*X[,2]

# Step d just means we can ignore these 10 variables

# Step e requires us to draw the coefficients, so:
norm_coefs <- rnorm(n = 90, mean = 0, sd = 0.05)

# Next, let's loop through these coefficients and add them to Y

for (i in 1:length(norm_coefs)) { # Notice our loop doesn't need to start at 1
  
  # notice because we want to start with X11, we have to adjust the index for X
  Y <- Y + norm_coefs[i]*X[,i+10] 
  
}

# Now we can check Y
Y

## The alternative approach is to first construct a full vector of coefficients,
# then use matrix multiplication to generate a n x 1 vector of outputs:
# i.e. X = n x 100 matrix, beta = 100 x 1 vector, therefore Xbeta = n x 1


# I'll use the norm_coefs vector above to make sure we get the same result

beta <- c(10,-5,rep(0,8),norm_coefs)
# NB: I use rep(0,8) to add 8 zeros to the coefs corresponding to X3-X10 

Y_alt <- X %*% beta

# To prove this is equivalent:

sum(Y == Y_alt) # should equal 2000 i.e. every y value is the same

# 3. Estimate a cross-validated LASSO model to find lambda

cv_lambda <- cv.glmnet(x = X, y = Y, alpha = 1, nfolds = 10)$lambda.min

# 4. Estimate a final LASSO model using the results from 3.

final_model <- glmnet(x = X, y = Y, alpha = 1, lambda = cv_lambda)

# Qs. What are the sizes of coefficient X1 X2? Do X3-X10 have non-zero
# coefficients? What about X11-X100?

# Inspect coefficients:

coef(final_model)

# Notice that X1 and X2 have large coefficients -- not quite as large as their
# true value, but not far off. We can see why they are close by looking at lambda
# which is quite a small value (0.16) and therefore it is only adding a small
# bit of bias to the model.

# Fortunately (!) X3-X10 all have exactly missing values -- this is good, because
# we know a priori they have no effect

# For $X11-x100 we see that some are present, and some are missing. If we compare
# these values against the original coefs in norm_coefs you'll see that the 
# non-zero coefficients in teh final model match the larger variables within 

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

# We can call the help file by running:
?glmnet
# The argument is `penalty.factor` which allows you to further penalise
# coefficients. Notice that penalty.factor is already being used in 
# "standard" lasso coefficients, but since penalty.factor = 1, it has no effect.

# What makes penalty.factor so useful is that it can be a vector, i.e. we can 
# pass to the this argument a list of different scalars for each coefficient, and
# thus diferentially penalize each coefficient separately i.e. we scale the 
# hyperparameter lambda on a coefficient by coefficient basis.

# 2. Without worrying too much about the surrounding code, describe what 
# lines 250-261 are doing algorithmically. 

# For reference, the code is:

while (count <= num_iter & (max(abs(phi0 - phi1)) > tol)) {
  if (count == 1) {
    cf_tmp <- glmnet::glmnet(x = x, y = y, lambda = pen_norm * lambda0/(4 * n),
                             penalty.factor = phi0, intercept = FALSE,
                             standardize = FALSE)$beta
  } else {
    phi0 <- phi1
    pen_norm <- sum(phi1) / p
    cf_tmp <- glmnet::glmnet(x = x, y = y, lambda = pen_norm * lambda0/(2 * n),
                             penalty.factor = phi1, intercept = FALSE,
                             standardize = FALSE)$beta
  }
  ...
}
  
# The first thing to note here is that they use a while loop, that means the code
# will automatically stop once a certain condition is met. This is used instead 
# of a for loop like we covered in Monday's walkthrough. We could adapt 
# yesterday's extension code to do something similar.

# This code will keep looping until either we hit the maximum number of iterations
# or the change in the adjustment parameter phi is less than (or equal to) some 
# specified tolerance level.

# What we can see here is that, iteratively, the code is running a lasso model,
# passing in a penalty factor parameter phi. 

# Again, like with our, extension exercise for Monday, the model is remembering 
# both the old penalty factor phi0 (i.e. the value from the previous iteration) 
# and the current value phi1. 

# The if condition here just helps the model start where there isn't a phi1 
# value yet. After the first iteration, the code will always run the "else" 
# portion of this code. 

# Essentially, the authors are refining their estimates of the penalty factors
# by running models sequentially and passing in a refined parameter each time.
# If you scroll down beyond line 261, you will see that they calculate an estimated
# error, ehat, and then recalculate the penalty factors as:

sqrt(1 / n * t(t(ehat ^ 2) %*% x ^ 2))

# We don't need to worry too much about this, but essentially the Blackwell and
# Olson model uses more sophisticated regularisation that enables differential
# penalisation of individual factors using the penalty.factors argument and a
# sequential tuning algorithm to set the values of this parameter.




