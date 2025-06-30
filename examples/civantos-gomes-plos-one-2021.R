
# Civantos-Gómez et al. (PloS ONE, 2021)

# load relevant libraries
library(rlang)
library(xgboost)
library(randomForest)

# simulate a minimal example of the data and framework

# n_plots
n_plots <- 1000

# n_sp
n_sp <- 5

# species names
sp_names <- paste0("species_", seq_len(n_sp))

# environmental variable
env_var <- rnorm(n = n_plots, mean = 10, sd = 5)

# loop over each species
plot_list <- list()
for (j in seq_len(n_sp)) {

  # draw an alpha value
  alpha <- runif(n = 1, min = 1, max = 20)
  
  # draw an environmental relationship
  beta <- runif(n = 1, min = -5, max = 5)
  
  # get lambda values across plots
  lambda <- alpha + (beta * env_var)
  
  # truncate at 0.1
  lambda <- ifelse(lambda < 1, 1, lambda)
  
  # draw plot values
  abund_vals <- rpois(n = length(lambda), lambda = lambda)
  
  # pull into a data.frame
  df <-
    dplyr::tibble(plot_id = seq_len(n_plots),
                  env_var = env_var,
                  species = sp_names[j],
                  env_alpha = alpha,
                  env_beta = beta,
                  lambda_vals = lambda,
                  individuals = abund_vals)
  
  # add to a list
  plot_list[[j]] <- df
  
}

# bind into a data.frame
df_out <- 
  dplyr::bind_rows(plot_list)
head(df_out)

# bind the species variables
df_out <-
  dplyr::full_join(df_out, 
                   df_out |>
                     dplyr::select(plot_id, species, individuals) |>
                     tidyr::pivot_wider(names_from = "species",
                                        values_from = "individuals"),
                   by = "plot_id")

# set the sp variable to 0 if species is focal
for (sp in sp_names) {
  df_out[[sp]] <- ifelse(df_out[["species"]] == sp, 0, df_out[[sp]])
}


## Modelling

# convert species to a factor
df_out$species <- as.factor(df_out$species)

# target variable
y <- df_out$individuals

# features (environment + competitors)
X <- 
  df_out |>
  dplyr::select(env_var, dplyr::all_of(sp_names))  # only predictors, not response

# optionally scale predictors
X_scaled <- scale(X)

# set-up the train-test splits
set.seed(123)
n <- nrow(X_scaled)
train_indices <- sample(seq_len(n), size = 0.75 * n)

X_train <- X_scaled[train_indices, ]
X_test <- X_scaled[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# linear model
lm_model <- lm(y_train ~ ., data = as.data.frame(X_train))
pred_lm <- predict(lm_model, newdata = as.data.frame(X_test))

rmse_lm <- sqrt(mean((y_test - pred_lm)^2))

plot(y_test, pred_lm)
abline(0, 1)

# random forest
rf_model <- randomForest(x = X_train, y = y_train)
pred_rf <- predict(rf_model, X_test)

rmse_rf <- sqrt(mean((y_test - pred_rf)^2))

plot(y_test, pred_rf)
abline(0, 1)

# xgboost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

xgb_model <- xgboost(data = dtrain, nrounds = 50, objective = "reg:squarederror", verbose = 0)
pred_xgb <- predict(xgb_model, dtest)

rmse_xgb <- sqrt(mean((y_test - pred_xgb)^2))

plot(y_test, pred_xgb)
abline(0, 1)

# compare the rmse
dplyr::tibble(Model = c("Linear model", "Random forest", "XGboost"),
              RMSE = c(rmse_lm, rmse_rf, rmse_xgb))












