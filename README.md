# End-to-End-Ann-project

Customer Churn Prediction Using Artificial Neural Network (ANN)
This project is an end-to-end machine learning solution for predicting customer churn based on demographic and account data. Using an Artificial Neural Network (ANN) model, the solution evaluates customer behavior and determines the likelihood of customer churn. This model leverages a grid search mechanism to identify the best parameter combinations for accurate prediction.


Project Overview
The goal of this project is to build a model capable of predicting customer churn for a business using customer data. It allows companies to proactively manage retention efforts by identifying at-risk customers. The solution includes:

Data Processing: Handling missing values, encoding categorical variables, and standardizing numerical features.
Model Training with ANN: Building an ANN model for binary classification.
Hyperparameter Tuning with Grid Search: Using grid search to find the optimal combination of hyperparameters to maximize model accuracy.
Deployment: The model is deployed via a web interface where users can input customer data and get real-time churn predictions.
Dataset
The dataset includes various features such as:

Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Has Credit Card
Is Active Member
Estimated Salary
These features are used to predict whether a customer will churn (leave the company) or stay.

Tech Stack
Programming Language: Python
Libraries: TensorFlow, Keras, Scikit-learn, Pandas, NumPy
Model: Artificial Neural Network (ANN)
Hyperparameter Tuning: Grid Search
Deployment: Streamlit (or Flask)