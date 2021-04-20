################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 1/5                                 ##
##                                                                            ##
##                    Constructing a logistic regression estimator            ##
##                            Walkthrough of extensions                       ##
##                                                                            ##
################################################################################

#### 1. Define the data as in the main code ####

# Generate some random covariates
set.seed(89)

X_data <- data.frame(X0 = 1, # Include a constant to model the intercept
                     X1 = runif(100,-5,5),
                     X2 = runif(100,-2,2))

# Calculate Y based on known functional form (incl. random error term)
Y <- 1/(1+exp(-(3 + X_data$X1 - 2*X_data$X2 + rnorm(100,0,0.05))))

# Custom function to get logistic yhat predictions
predict <- function(row, coefficients) {
  pred_terms <- row*coefficients
  yhat <- sum(pred_terms)
  return(1/(1+exp(-yhat)))
}

###############################################################################

## Extensions to consider

# * Initialise random coefficient values
# * Implement a stopping rule

#   Below I create two new training functions that implement the suggested 
# extensions separately. Then I merge both extensions together into a v2 of the
# estimator.

#### S1. Initialising random coefficient values ####

#   Using random values is straightforward. Let's redefine the beginning of our
# training function, such that the coefficients are drawn from a random normal
# distribution centred around 0 (rather than being zero exactly).

train_rand <- function(X,y,l_rate, reps, init_sd) {
  
  coefs <- rnorm(ncol(X), 0, init_sd) # NEW CODE
  
  # Everything else remains the same
  for (rep in 1:reps) {
    total_error <- 0
    for (i in 1:nrow(X)) {
      row_vec <- as.numeric(X[i,])
      yhat_i <- predict(row_vec, coefficients = coefs)
      error <- y[i] - yhat_i
      total_error <- total_error + error^2
      coefs <- sapply(1:length(coefs), function (i) coefs[i] + l_rate*error*yhat_i*(1-yhat_i)*row_vec[i])
    }
    MSE <- total_error/nrow(X)
    message(paste0("Iteration ",rep,"/",reps," -- MSE = ",MSE))
  }
  return(coefs)
}

# Try it out:
train_rand(X_data, Y, l_rate = 0.3, reps = 1000, init_sd = 1)

# In this instance, there's not much difference between the two algorithms

#### S2. Implementing a stopping rule

#   My hint in class was to add a hyperparameter and to give the model a
# memory.

# Let's think intuitively about what we want to do:
# 1. When updating the parameters yields big changes in the MSE, we are clearly
# still on the steeper portion of the loss function. Therefore, we want the
# algorithm to keep refining our estimates.

# 2. When we start randomly fluctuating around the bottom of the loss function,
# we want the model to stop.

#   We can measure both 1 and 2 by considering the *change* in the total loss. 
# The general intuition is to BREAK out of the model loop once the change in the
# total loss across repetitions becomes smaller than some value (the 
# hyperparameter).

#   Our loss values will differ dependent on the range of the outcome, and so 
# it is better to consider the relative change in the loss function:

# I.e. stop when abs(MSE - prev_MSE)/prev_MSE < some constant (a hyperparameter)

#   Finally, we should insure ourselves against the case where the model never
# converges sufficiently and so the code keeps running forever! Let's rename the
# reps arguments to max_reps, and keep the outer for loop so that, even if it 
# does not converge it will stop after max_reps iterations.

train_stop <- function(X,y,l_rate, max_reps, stop_criteria) {
  
  coefs <- rep(0, ncol(X))
  MSE_prev <- 2*sum(y) # NEW CODE
  
  # This chunk is as before
  for (rep in 1:max_reps) {
    total_error <- 0
    for (i in 1:nrow(X)) {
      row_vec <- as.numeric(X[i,])
      yhat_i <- predict(row_vec, coefficients = coefs)
      error <- y[i] - yhat_i
      total_error <- total_error + error^2
      coefs <- sapply(1:length(coefs), function (i) coefs[i] + l_rate*error*yhat_i*(1-yhat_i)*row_vec[i])
    }
    
    MSE <- total_error/nrow(X)
    
    # NEW CODE:
    delta_MSE <- abs(MSE-MSE_prev)/MSE_prev # Calculate relative change in MSE
    
    if (delta_MSE < stop_criteria) {
      
      message(paste0("Model converged on iteration ",rep,". Final MSE = ",MSE))
      
      break # This command exits the outer for loop and returns the coefficients
      
    } else {
      
      MSE_prev <- MSE # Update the previous MSE value for the next iteration
      
      message(paste0("Iteration ",rep," -- MSE = ",MSE))
      
    }
  }
  return(coefs)
}

# Try it out:
train_stop(X_data, Y, max_reps = 1000, l_rate = 0.3, stop_criteria = 0.0001)

# Model converges after 154 iterations!

#### S3. Putting it all together ####

train_v2 <- function(X, y, l_rate, max_reps, stop_criteria, init_sd) {
  
  coefs <- rnorm(ncol(X),0,init_sd)
  
  MSE_prev <- 2*sum(y)
  
  for (rep in 1:max_reps) {
    
    total_error <- 0
    
    for (i in 1:nrow(X)) {
      
      row_vec <- as.numeric(X[i,])
      
      yhat_i <- predict(row_vec, coefficients = coefs)
      error <- y[i] - yhat_i
      total_error <- total_error + error^2
      
      coefs <- sapply(1:length(coefs), function (i) coefs[i] + l_rate*error*yhat_i*(1-yhat_i)*row_vec[i])
      
    }
    
    MSE <- total_error/nrow(X) 
    
    delta_MSE <- abs(MSE-MSE_prev)/MSE_prev 
    
    if (delta_MSE < stop_criteria) {
      
      message(paste0("Model converged on iteration ",rep,". Final MSE = ",MSE))
      
      break
      
    } else {
      
      MSE_prev <- MSE 
      
      message(paste0("Iteration ",rep," -- MSE = ",MSE))
      
    }
  }
  
  return(coefs)
}

# What value do we want stop_criteria to be? Again, it's up to the researcher.
# Since this is the relative change, we can interpret these numbers as a
# proportion. So I chose 0.0001 which corresponds to a change less than 0.01% --
# that seems sufficient to get a good estimate of the parameters. You might want
# a smaller value.

train_v2(X_data, Y, max_reps = 1000, l_rate = 0.3, init_sd = 1, stop_criteria = 0.0001)

# Model converges after 152 iterations.
