rm(list=ls())
library(leaps)
library(caret)
library(pls)
library(dplyr) 
library(ggplot2)
library(kknn)
library(rpart)





# Data Analysis 
data = read.csv("HDB_data_2021_sample.csv")
data$resale_price = data$resale_price / 1000
data <- data %>%
  mutate(
    flat_type = case_when(
      `flat_type_1.ROOM` == 1 ~ "1 Room",
      `flat_type_2.ROOM` == 1 ~ "2 Room",
      `flat_type_3.ROOM` == 1 ~ "3 Room",
      `flat_type_4.ROOM` == 1 ~ "4 Room",
      `flat_type_5.ROOM` == 1 ~ "5 Room",
      `flat_type_EXECUTIVE` == 1 ~ "Executive",
      `flat_type_MULTI.GENERATION` == 1 ~ "Multi Generation",
      TRUE ~ "Other"
    )
  )

data = data %>%
  mutate(
    floor_category = case_when(
      `storey_range_01.TO.03` == 1 | `storey_range_01.TO.05` == 1 | 
        `storey_range_04.TO.06` == 1 | `storey_range_06.TO.10` == 1 |
        `storey_range_07.TO.09` == 1 ~ "Low Floor",
      
      `storey_range_10.TO.12` == 1 | `storey_range_11.TO.15` == 1 | 
        `storey_range_13.TO.15` == 1 | `storey_range_16.TO.18` == 1 | 
        `storey_range_16.TO.20` == 1 ~ "Mid Floor",
      
      `storey_range_19.TO.21` == 1 | `storey_range_21.TO.25` == 1 | 
        `storey_range_22.TO.24` == 1 | `storey_range_25.TO.27` == 1 | 
        `storey_range_26.TO.30` == 1 | `storey_range_28.TO.30` == 1 |
        `storey_range_31.TO.33` == 1 | `storey_range_31.TO.35` == 1 |
        `storey_range_34.TO.36` == 1 | `storey_range_36.TO.40` == 1 |
        `storey_range_37.TO.39` == 1 | `storey_range_40.TO.42` == 1 |
        `storey_range_43.TO.45` == 1 | `storey_range_46.TO.48` == 1 |
        `storey_range_49.TO.51` == 1 ~ "High Floor",
      
      TRUE ~ NA_character_  
    )
  )

price_by_floor_category = data %>%
  group_by(floor_category) %>%
  summarise(avg_resale_price = mean(resale_price, na.rm = TRUE))

ggplot(price_by_floor_category, aes(x = floor_category, y = avg_resale_price)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(
    x = "Floor Category",
    y = "Average Resale Price (Thousand Dollars)",
    title = "Average Resale Price by Floor Category"
  )


summary_by_flat_type = data %>%
  group_by(flat_type) %>%
  summarise(
    avg_resale_price = mean(resale_price, na.rm = TRUE),
    avg_floor_area = mean(floor_area_sqm, na.rm = TRUE),
  )

print(summary_by_flat_type)





# Data Pre-processing step
# "HDB_data_2021_sample.xlms" was converted to a csv file "HDB_data_2021_sample.csv" to be read
HDB = read.csv("HDB_data_2021_sample.csv")
# Normalizing the resale price by dividing it by $1,000
HDB$resale_price = HDB$resale_price / 1000
# Finding any collinear variables and removing them
linear_combos = findLinearCombos(HDB)
if (!is.null(linear_combos$remove)) {
  HDB = HDB[, -linear_combos$remove]
}
# Finding any variables with near zero variance and removing them 
nzv = nearZeroVar(HDB)
if (length(nzv) > 0) {
  HDB = HDB[, -nzv]
}
# Setting of seed to make results replicable
set.seed(100)
# Applying 80/20 training and test set
ntrain = 4800
tr = sample(1:nrow(HDB), ntrain)
train = HDB[tr,]
test = HDB[-tr,]

# Fitting the base model using backward propagation
nvmax_val = ncol(HDB)-1
lm_fit = regsubsets(resale_price~., data = train, method = "backward", nvmax = nvmax_val)
best_model_index = which.max(summary(lm_fit)$adjr2)
selected_coefficients = coef(lm_fit, best_model_index)
# Extracting the predictors used in the best model
selected_predictors = names(selected_coefficients)[-1]
lm_formula = as.formula(paste("resale_price ~", paste(selected_predictors, collapse = " + ")))
# Fitting the model
final_lmfit = lm(lm_formula, data = train)
summary(final_lmfit)$adj.r.squared
# Adjusted R squared value is 0.9006071
# Predict with the linear model
prediction_fit = predict(final_lmfit, newdata = test)
mse_fit = mean((test$resale_price - prediction_fit)^2)
rmse_fit = sqrt(mse_fit)
# MSE for the linear model is 3173.5
# RMSE for the linear model is 56.3




# Predicting with knn
knn_formula = as.formula(paste("resale_price ~", paste(selected_predictors, collapse = " + ")))
df.loocv=train.kknn(knn_formula, data=train, kmax=100, kernel = "rectangular")
# Determine the best k value
kbest=df.loocv$best.parameters$k
# Fit the model and predict with the best k value
knnpredcv=kknn(knn_formula, train, test, k=kbest, kernel="rectangular")
mse_knn = mean((test$resale_price - knnpredcv$fitted.values)^2)
rmse_knn = sqrt(mse_knn)
# MSE for the knn model is 7251.3
# RMSE for the knn model is 85.2





# Predicting with decision tree
tree_model = as.formula(paste("resale_price ~", paste(selected_predictors, collapse = " + ")))
# Fit the entire tree
big.tree = rpart(tree_model, method="anova", data=train, cp=0.001)
# Determine the best cp value 
bestxerr = big.tree$cptable[which.min(big.tree$cptable['xerror'])]
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]
# Use best cp value to prune the tree
best.tree = prune(big.tree, cp=bestcp)
# Predict with the best tree
treefit = predict(best.tree, newdata = test, type = "vector")
mse_tree = mean((test$resale_price-treefit)^2)
rmse_tree = sqrt(mse_tree)
# MSE for the tree model is 4292.2
# RMSE for the tree model is 65.5
# Plot the best tree
plot(best.tree,uniform=TRUE)
text(best.tree,digits=4,use.n=TRUE,fancy=,bg='lightblue')