################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 1/5                                 ##
##                                                                            ##
##                    Constructing a logistic regression estimator            ##
##                                                                            ##
################################################################################

#### 1. Define data and a prediction function ####

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

#### 2. Comparing a guess to real values

# Take a guess at the coefficients
coef_guess <- c(0,0.5,1)
Yhat_guess <- apply(X_data, 1, predict, coefficients = coef_guess)

# Calculate MSE
MSE_guess <- mean((Y - Yhat_guess)^2)

# Generate "true" values for test data 
coef_actual <- c(3,1,-2)
Yhat_actual <- apply(X_data, 1, predict, coefficients = coef_actual)

# Calculate MSE
MSE_actual <- mean((Y - Yhat_actual)^2)

#### 3. Define a logistic training algorithm ####

train <- function(X,y,l_rate, reps) {
  
  # Instantiate model with basic guess of 0 for all coefs
  coefs <- rep(0, ncol(X))
  
  for (rep in 1:reps) {
    
    total_error <- 0
    
    # Iterate through each row in the dataset
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

#### 4. Apply our algorithm ####

coefs_logit <- train(X = X_data, y = Y, l_rate = 0.3, reps = 1000)

## Extensions to consider

# * Initialise random coefficient values
# * Implement a stopping rule
