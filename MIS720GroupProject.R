install.packages("dplyr")
install.packages("corrplot")
install.packages("pheatmap")
library(pheatmap)
library(corrplot)
library(ggplot2)
library(dplyr)
library(pROC)
library(readxl)
library(e1071)
library(caret)
telecom_data <- read_excel("C:/Users/kpasalkar7588/Downloads/churn-bigml-80.xlsx")

# Rename variables in telecom_data
telecom_data <- telecom_data %>%
  rename(
    Account_length = "Account length",
    International_plan = "International plan",
    Voice_mail_plan = "Voice mail plan",
    Number_vmail_messages = "Number vmail messages",
    Total_day_minutes = "Total day minutes",
    Total_day_calls = "Total day calls",
    Total_day_charge = "Total day charge",
    Total_eve_minutes = "Total eve minutes",
    Total_eve_calls = "Total eve calls",
    Total_eve_charge = "Total eve charge",
    Total_night_minutes = "Total night minutes",
    Total_night_calls = "Total night calls",
    Total_night_charge = "Total night charge",
    Total_intl_minutes = "Total intl minutes",
    Total_intl_calls = "Total intl calls",
    Total_intl_charge = "Total intl charge",
    Customer_service_calls = "Customer service calls"
  )

# Convert 'International plan' and 'Voice mail plan' from 'No', 'Yes' to 0, 1
telecom_data$International_plan <- ifelse(telecom_data$International_plan == "No", 0, 1)
telecom_data$Voice_mail_plan <- ifelse(telecom_data$Voice_mail_plan == "No", 0, 1)


# Function to plot histogram
plot_histogram <- function(data, variable) {
  ggplot(data, aes(x = !!as.name(variable), fill = Churn)) +
    geom_histogram(binwidth = 10, color = "black", alpha = 0.7) +
    labs(title = paste("Histogram of", variable, "with respect to Churn"),
         x = variable, y = "Frequency") +
    theme_minimal() +
    theme(legend.position = "top")
}

# Plotting histograms for all quantitative variables
quantitative_vars <- c("Account_length", "Number_vmail_messages", "Total_day_minutes", 
                       "Total_day_calls", "Total_day_charge", "Total_eve_minutes", 
                       "Total_eve_calls", "Total_eve_charge", "Total_night_minutes", 
                       "Total_night_calls", "Total_night_charge", "Total_intl_minutes", 
                       "Total_intl_calls", "Total_intl_charge", "Customer_service_calls")


for (var in quantitative_vars) {
  print(plot_histogram(telecom_data, var))
}

# Defining function to plot pi charts
plot_pie_chart <- function(data, variable) {
  counts <- table(data[[variable]])
  pie(counts, labels = paste(names(counts), ": ", counts), main = variable)
}

# Plotting pie charts for each qualitative variable
qualitative_vars <- c("State", "International_plan", "Voice_mail_plan", "Churn")

for (var in qualitative_vars) {
  plot_pie_chart(telecom_data, var)
}

# List of numeric variables of interest
numeric_vars <- c("Account_length", "International_plan", "Voice_mail_plan", "Number_vmail_messages", "Total_day_minutes", 
                  "Total_day_calls", "Total_day_charge", "Total_eve_minutes", 
                  "Total_eve_calls", "Total_eve_charge", "Total_night_minutes", 
                  "Total_night_calls", "Total_night_charge", "Total_intl_minutes", 
                  "Total_intl_calls", "Total_intl_charge", "Customer_service_calls", "Churn")

cor_matrix <- cor(telecom_data[numeric_vars], use = "complete.obs")
cor_matrix_rounded <- round(cor_matrix, 2)


pheatmap(cor_matrix, 
         color = colorRampPalette(c("yellowgreen", "white", "tomato"))(100),
         main = "Correlation Matrix Heatmap", 
         display_numbers = TRUE,
         clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean",
         clustering_method = "complete")


#Summary of dataset
summary(telecom_data)

# Creating a box plot of each quantitative variable against "Churn"
for (var in quantitative_vars) {
  boxplot(formula(paste(var, "~ Churn")), data = telecom_data, main = paste("Box plot of", var, "by Churn"))
}

# Creating a box plot of each qualitative variable against "Churn"
for (var in qualitative_vars) {
  table_data <- table(telecom_data[[var]], telecom_data$Churn)
  barplot(table_data, beside = TRUE, legend.text = TRUE,
          main = paste("Bar plot of", var, "by Churn"),
          xlab = var, ylab = "Frequency", col = c("blue", "red"))
}

# Split the data into training and testing sets
set.seed(123)
index <- createDataPartition(telecom_data$Churn, p = 0.8, list = FALSE)
train_data <- telecom_data[index, ]
test_data <- telecom_data[-index, ]

# Extracting the predictor variables and the target variable
predictors <- train_data[, names(train_data) != 'Churn']
target <- train_data$Churn


# Training the SVM model with a linear kernel
svm_linear_model <- svm(Churn ~ ., data = train_data, type = 'C-classification', kernel = 'linear', probability = TRUE)

# Training the SVM model with a polynomial kernel
svm_poly_model <- svm(Churn ~ ., data = train_data, type = 'C-classification', kernel = 'polynomial', degree = 3, probability = TRUE)

# Training the SVM model with a radial kernel
svm_radial_model <- svm(Churn ~ ., data = train_data, type = 'C-classification', kernel = 'radial', probability = TRUE)



# Predicting on the test data using the linear kernel model
predictions_linear <- predict(svm_linear_model, test_data[, names(test_data) != 'Churn'])

# Predicting on the test data using the polynomial kernel model
predictions_poly <- predict(svm_poly_model, test_data[, names(test_data) != 'Churn'])

# Predicting on the test data using the radial kernel model
predictions_radial <- predict(svm_radial_model, test_data[, names(test_data) != 'Churn'])



# Evaluating the model performance of linear kernel
confusion_matrix_linear <- table(Predicted = predictions_linear, Actual = test_data$Churn)
print(confusion_matrix_linear)

# Evaluating the model performance of polynomial kernel
confusion_matrix_poly <- table(Predicted = predictions_poly, Actual = test_data$Churn)
print(confusion_matrix_poly)

# Evaluating the model performance of radial kernel
confusion_matrix_radial <- table(Predicted = predictions_radial, Actual = test_data$Churn)
print(confusion_matrix_radial)


# Calculating performance metrics of linear kernel
accuracy_linear <- sum(diag(confusion_matrix_linear)) / sum(confusion_matrix_linear)

# Calculating performance metrics of polynomial kernel
accuracy_poly <- sum(diag(confusion_matrix_poly)) / sum(confusion_matrix_poly)

# Calculating performance metrics of radial kernel
accuracy_radial <- sum(diag(confusion_matrix_radial)) / sum(confusion_matrix_radial)


print(paste('Accuracy with linear kernel:', accuracy_linear))
print(paste('Accuracy with polynomial kernel:', accuracy_poly))
print(paste('Accuracy with Radial kernel:', accuracy_radial))


# Calculating Misclassification Rate for Linear Kernel SVM
misclassification_rate_linear <- mean(predictions_linear != test_data$Churn)
cat("Misclassification Rate for Linear Kernel SVM: ", misclassification_rate_linear, "\n")

# Calculating Misclassification Rate for Polynomial Kernel SVM
misclassification_rate_poly <- mean(predictions_poly != test_data$Churn)
cat("Misclassification Rate for Polynomial Kernel SVM: ", misclassification_rate_poly, "\n")

# Calculating Misclassification Rate for Radial Kernel SVM
misclassification_rate_radial <- mean(predictions_radial != test_data$Churn)
cat("Misclassification Rate for Radial Kernel SVM: ", misclassification_rate_radial, "\n")


computeAndPlotROC <- function(model, test_data, model_name) {
  predictions <- predict(model, test_data[, names(test_data) != 'Churn'], probability = TRUE)
  
  prob_churn <- attr(predictions, "probabilities")[, 2]

  actual_factor <- factor(test_data$Churn, levels = c(0, 1), labels = c("No", "Yes"))
  
  # Computing ROC curve 
  roc_values <- roc(actual_factor, prob_churn)
  
  # Plotting ROC curve
  plot(roc_values, main = paste("ROC Curve for", model_name), col = "#1c61b6")
  
  # Printing AUC
  cat(paste("AUC for", model_name, ":", auc(roc_values), "\n"))
  
  return(auc(roc_values))
}

auc_linear <- computeAndPlotROC(svm_linear_model, test_data, "Linear SVM")
auc_poly <- computeAndPlotROC(svm_poly_model, test_data, "Polynomial SVM")
auc_radial <- computeAndPlotROC(svm_radial_model, test_data, "Radial SVM")


# Calculating Misclassification Rate for Linear Kernel SVM
misclassification_rate_linear <- mean(predictions_linear != test_data$Churn)
cat("Misclassification Rate for Linear Kernel SVM: ", misclassification_rate_linear, "\n")

# Calculating Misclassification Rate for Polynomial Kernel SVM
misclassification_rate_poly <- mean(predictions_poly != test_data$Churn)
cat("Misclassification Rate for Polynomial Kernel SVM: ", misclassification_rate_poly, "\n")

# Calculating Misclassification Rate for Radial Kernel SVM
misclassification_rate_radial <- mean(predictions_radial != test_data$Churn)
cat("Misclassification Rate for Radial Kernel SVM: ", misclassification_rate_radial, "\n")

