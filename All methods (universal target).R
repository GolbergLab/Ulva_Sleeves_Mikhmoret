
## original variables + subsets + output
datadf <- seaweedCSV[, c("Duration", "Age","Density_IN",
                         "sumT_Daily", "sumT12_4PM_Daily", 
                         "sumT2_6AM_Daily","ND17C", "ND17_25C","ND25_28C", "ND28_30C", "ND30C",
                         "avgL_24hr", "avgL_Day", "maxL_EXP","avgL_ALG","minT_EXP", "maxT_EXP", "avgT_EXP", "minT12_4PM", 
                         "maxT12_4PM","avgT12_4PM", "minT2_6AM", "maxT2_6AM", "avgT2_6AM",
                         "Target")] ## 
datadf<- na.omit(datadf) # clean dataframe
attach(datadf)

orgdata <- datadf[,c("Duration", "Age","Density_IN",
                     "sumT_Daily", "sumT12_4PM_Daily", 
                     "sumT2_6AM_Daily","ND17C", "ND17_25C","ND25_28C", "ND28_30C", "ND30C",
                     "avgL_24hr", "avgL_Day", "maxL_EXP","avgL_ALG","Target")]

## out minT_EXP", "maxT_EXP", "avgT_EXP", "minT12_4PM", "maxT12_4PM","avgT12_4PM", "minT2_6AM", "maxT2_6AM", "avgT2_6AM"

##engeneered features = (measured temperatures minus average)
engdata<- data.frame(
  MEminT_EXP=((minT_EXP-mean(minT_EXP))),
  MEmaxT_EXP=((maxT_EXP-mean(maxT_EXP))),
  MEavgT_EXP=((avgT_EXP-mean(avgT_EXP))),
  MEminT12_4PM=((minT12_4PM-mean(minT12_4PM))),
  MEmaxT12_4PM=((maxT12_4PM-mean(maxT12_4PM))),
  MEavgT12_4PM=((avgT12_4PM-mean(avgT12_4PM))),
  MEminT2_6AM=((minT2_6AM-mean(minT2_6AM))),
  MEmaxT2_6AM=((maxT2_6AM-mean(maxT2_6AM))),
  MEavgT2_6AM=((avgT2_6AM-mean(avgT2_6AM)))
)

##marge dataframes of original and engeneered variables
mydata <- data.frame(orgdata, engdata)

## LIBRARIES
library(tidyverse)          # Pipe operator (%>%) and other commands
library(caret)              # Random split of data/cross validation
library(olsrr)              # Heteroscedasticity Testing (ols_test_score)
library(car)                # Muticolinearity detection (vif)
library(broom)              # Diagnostic Metric Table (augment)
library(tidyr)
library(dplyr)              # Normalization
library(flextable)
library(performance)        # Model comparisons
library(glmnet)             # lASSO
library(ggplot2)
library(ggpubr)
library(linex)              # Disjointification

# Load data for analysis
data <- mydata

# Split data into training and test sets (3:1 ratio)
set.seed(123)
split <- simple.split(data$target_variable, train = 0.75, ratio = 1)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)




# METHOD 1 - FFS + OLS

# Compute Spearman correlation and rank variables
cor_matrix <- cor(train[, 1:24], method = "spearman")
cor_df <- data.frame(variable = names(cor_matrix), corr_with_target = cor_matrix[, "target_variable"])
cor_df <- cor_df[order(-abs(cor_df$corr_with_target)), ]

# Disjointify training data to obtain orthogonal linear system
dj <- disjointify(train[, 1:24], target_variable = "target_variable", rho = 0.75)
selected_vars <- dj$selected_variables

# Build linear models using selected variables
models <- list()
for (i in seq_along(selected_vars)) {
  formula <- as.formula(paste("target_variable ~", paste(selected_vars[1:i], collapse = "+")))
  if (all(abs(cor(train[, selected_vars[1:i]], method = "spearman")) < 0.75)) {
    models[[i]] <- lm(formula, data = train)
  }
}

  # Training set performance
  train_preds <- predict(models[[i]], newdata = train)
  results[i, "RMSE_train"] <- RMSE(train_preds, train$target_variable)
  results[i, "MAPE_train"] <- MAPE(train_preds, train$target_variable)
  results[i, "R2_train"] <- summary(models[[i]])$r.squared
}

# Select most effective explanatory variables based on similarity of performance metrics
best_models <- results[which.max(abs(results$RMSE_train)),]
best_vars <- names(coef(models[[best_models$model]])[-1])


## Part 2 - OLS

# Run best subset selection based on adjusted R-squared
model_ar2 <- ols_best_subset_selection(best_models, criterion = "adjr2")
summary(model_ar2)

# Run best subset selection based on mean squared error
model_mse <- ols_best_subset_selection(response, predictors, criterion = "mse")
summary(model_mse)

# Run best subset selection based on Akaike information criterion
model_aic <- ols_best_subset_selection(response, predictors, criterion = "aic")
summary(model_aic)

bestmodel <- lm(target ~  "input varables selected on previous step", data = train)
bestmd1 <- best_model

####### Predictions for TEST

predictions1 <- bestmd1 %>% predict(test)
data.frame(R2 = R2(predictions, test$Target),
           RMSE = RMSE(predictions, test$Target),
           MAPE= get_MAPE(model = bestmd1, data = test, response = "Target"))# train MAPE

## Scatter Plot - Actual VS Predicted
values1 <- data.frame(Actual=test$Target, Predicted = predictions1)

met1<-ggplot(values1,
             aes(x = Predicted,
                 y = Actual)) +
  geom_point() +
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2) +
  labs(x='Predicted Values', y='Actual Values', title='Fit Graph for Target_variable')
met1


## validation Random Shuffling 
nreps <- 10000
randR2 <- numeric(nreps)
set.seed(123)
for (i in 1:nreps)
{
  randR2[i] <- summary(lm(sample(Target) ~ "input varables selected for best model", data = test))$r.square ## best model in this part - TEST set only
}
actR2 <- R2(predictions1, test$Target)
## append the observed value to the list of results
r.square <- c(randR2,actR2)

hist(r.square,col="gray",las=1,main="R-square for randomized Target_variable - Test set")
abline(v=actR2,col="red")

## METHOD 2  -   L A S S O


Y <- train$Target
X <- model.matrix(Target~.,train)[,-1]
lassocv <- cv.glmnet(X, Y, alpha=1, nfolds=10)
lassomdl <- glmnet(X,Y,alpha=1, nlambda=100)
lassocoef <- coef(lassomdl, s=lassocv$lambda.min)
plot(lassomdl, xvar="lambda", lwd=2)
abline(v=log(lassocv$lambda.min), col="black",lty=2, lwd=2)

lassocoef

model <- lm(Target ~  "input variables selected by lasso", data=train)

summary(model)

predictions <- model %>% predict(train)
data.frame( R2 = R2(predictions, train$Target),
            RMSE = RMSE(predictions, train$Target),
            MAPE= get_MAPE(model = model, data = train, response = "Target"))# train MAPE

predictions3 <- model %>% predict(test)
data.frame( R2 = R2(predictions, test$Target),
            RMSE = RMSE(predictions, test$Target),
            MAPE= get_MAPE(model = model, data = test, response = "Target"))# test MAPE

## Scatter Plot - Actual VS Predicted
values3 <- data.frame(Actual=test$Target, Predicted = predictions3)

met2<-ggplot(values3,
             aes(x = Predicted,
                 y = Actual)) +
  geom_point() +
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2) +
  labs(x='Predicted Values', y='Actual Values', title='Fit Graph for Target_variable')
met2

## validation Random Shuffling 
nreps <- 10000
randR2 <- numeric(nreps)
set.seed(123)
for (i in 1:nreps)
{
  randR2[i] <- summary(lm(sample(Target) ~ "input variables selected by lasso", data = test))$r.square ## best model in this part - TEST set only
}
actR2 <- R2(predictions3, test$Target)
## append the observed value to the list of results
r.square <- c(randR2,actR2)

hist(r.square,col="gray",las=1,main="R-square for randomized Target_variable - Test set")
abline(v=actR2,col="red")



## Follow up - Common set of variables (So called method 3)

bestmd4 <- lm(Target ~  "set of common variables selected by used methods", data=train)

summary(bestmd4)


predictions4 <- bestmd4 %>% predict(test)
data.frame( R2 = R2(predictions, test$Target),
            RMSE = RMSE(predictions, test$Target),
            MAPE= get_MAPE(model = model, data = test, response = "Target"))# test MAPE

## Scatter Plot - Actual VS Predicted
values4 <- data.frame(Actual=test$Target, Predicted = predictions4)


## built figure for scatter plots
met3<-ggplot(values4,
             aes(x = Predicted,
                 y = Actual)) +
  geom_point() +
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2) +
  labs(x='Predicted Values', y='Actual Values', title='Fit Graph for Target_variable')
met3

## validation Random Shuffling 
nreps <- 10000
randR2 <- numeric(nreps)
set.seed(123)
for (i in 1:nreps)
{
  randR2[i] <- summary(lm(sample(Target) ~ "set of common variables selected by used methods", data = test))$r.square ## best model in this part - TEST set only
}
actR2 <- R2(predictions4, test$Target)
## append the observed value to the list of results
r.square <- c(randR2,actR2)

hist(r.square,col="gray",las=1,main="R-square for randomized Target_variable - Test set")
abline(v=actR2,col="red")



## Combine fit graphs

figure <- ggarrange(met1, met1, met2,
                    labels = c("A", "B", "C"),
                    ncol = 2, nrow = 2)
figure

