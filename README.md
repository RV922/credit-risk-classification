# credit-risk-classification

# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis:
    The analysis aims to evaluate the logistic regression model's effectiveness in predicting credit risk and offer insights for decision-making purposes.

* Explain what financial information the data was on, and what you needed to predict:
    The data used in this analysis pertains to lending and credit risk assessment. Specifically, the dataset contains financial information related to loans, including details about the borrowers, loan amounts, repayment terms, and the loan status, indicating whether a loan is healthy or has a high risk of defaulting. The goal is to use the features in the dataset to build a model that can accurately predict whether a loan is likely to be healthy or if it poses a high risk of defaulting.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`):
    The "loan_status" column (Target Variable "y") indicates if a loan is healthy (0) or at high risk of defaulting (1). The remaining columns (The Features "X") are used to predict the loan status, including financial and personal information related to the applicants.

* Describe the stages of the machine learning process you went through as part of this analysis:
    Data Preprocessing
    Making the labels set (y) from the target variable "loan_status" and the features set (X) from the remaining columns.
    Splitting the data into training and testing sets using train_test_split.

    Model Training
    Fitting a logistic regression model using the training data (X_train and y_train).
    Training the model on the training data to learn the patterns in the features and their relationship the loan_status.

    Model Evaluation
    Making predictions on the testing data using the trained logistic regression model. Evaluating the model's performance by developing a confusion matrix to comprehend the model's predictions and printing a classification report to get the precision, recall, and F1-score for each class.

    Performance Analysis
    Analyzing the confusion matrix and classification report to assess how nicely the logistic regression model predicts both the healthy (0) and high-risk (1) loan labels.

    Recommendation
    Based on the model performance, providing a recommendation on whether the logistic regression model is suitable for predicting credit risk or an a different approach is needed.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
    The logistic regression algorithm was used to create a binary classification model.

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
    Accuracy score: 0.99
    Precision: Class 0: 1.00, Class 1: 0.87
    Recall: Class 0: 1.00, Class 1: 0.91

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
Class 0 performs better than Class 1, since its precision and recall values are 1.00

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
If you do not recommend any of the models, please justify your reasoning.

This performance tell us that for Class 1 even though the values are not low they are not as close to 1.00 as Class 0 is meaning that is performance is more compromise.





Instructions
The instructions for this Challenge are divided into the following subsections:

Split the Data into Training and Testing Sets
Create a Logistic Regression Model with the Original Data
Write a Credit Risk Analysis Report


Split the Data into Training and Testing Sets
Open the starter code notebook and use it to complete the following steps:

1. Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

2. Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

    NOTE
    A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

3. Split the data into training and testing datasets by using train_test_split.


Create a Logistic Regression Model with the Original Data
Use your knowledge of logistic regression to complete the following steps:

1. Fit a logistic regression model by using the training data (X_train and y_train).

2. Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

3. Evaluate the model’s performance by doing the following:

    Generate a confusion matrix.

    Print the classification report.

4. Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?


Write a Credit Risk Analysis Report
Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:

1. An overview of the analysis: Explain the purpose of this analysis.

2. The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.

3. A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.



Requirements
Split the Data into Training and Testing Sets (30 points)

    To receive all points, you must:

    Read the lending_data.csv data from the Resources folder into a Pandas DataFrame. (5 points)

    Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns. (10 points)

    Split the data into training and testing datasets by using train_test_split. (15 points)

Create a Logistic Regression Model (30 points)

    To receive all points, you must:

    Fit a logistic regression model by using the training data (X_train and y_train). (10 points)

    Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model. (5 points)

    Evaluate the model’s performance by doing the following:

    Generate a confusion matrix. (5 points)

    Generate a classification report. (5 points)

    Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels? (5 points)

Write a Credit Risk Analysis Report (20 points)

    To receive all points, you must:

    Provide an overview that explains the purpose of this analysis. (5 points)

    Using a bulleted list, describe the accuracy, precision, and recall scores of the machine learning model. (5 points)

    Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning. (10 points)

Coding Conventions and Formatting (10 points)

    To receive all points, you must:

    Place imports at the top of the file, just after any module comments and docstrings and before module globals and constants. (3 points)

    Name functions and variables with lowercase characters, with words separated by underscores. (2 points)

    Follow DRY (Don’t Repeat Yourself) principles, creating maintainable and reusable code. (3 points)

    Use concise logic and creative engineering where possible. (2 points)

Code Comments (10 points)

    To receive all points, your code must:

    Be well commented with concise, relevant notes that other developers can understand. (10 points)