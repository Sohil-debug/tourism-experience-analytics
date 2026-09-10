# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

## Project Overview

This project analyzes tourism experience data using data analytics, SQL, machine learning, and recommendation techniques.

The project includes:
- Data Cleaning and Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Data Visualization
- SQL Analytics
- Regression
- Classification
- Collaborative Filtering Recommendation System
- Streamlit Deployment

## Project Objectives

### 1. Regression
Predict attraction ratings based on tourism, attraction, visit, and user-related features.

### 2. Classification
Predict the user's likely visit mode such as Business, Couples, Family, Friends, or Solo.

### 3. Recommendation System
Generate personalized attraction recommendations based on historical user-attraction ratings.

## Dataset

The project uses tourism datasets containing information about:
- Transactions
- Users
- Cities
- Attractions
- Attraction Types
- Visit Modes
- Countries
- Regions
- Continents

The final feature-engineered dataset contains approximately 52,930 records and 27 features.

## Data Preparation

The following steps were performed:
1. Dataset extraction
2. Dataset integration
3. Missing-value handling
4. Date transformation
5. Feature engineering
6. Categorical encoding
7. Numerical scaling
8. Creation of user-level and attraction-level features

Important engineered features include:
- VisitDate
- VisitQuarter
- VisitSeason
- RatingCategory
- UserVisitCount
- AttractionPopularity
- AttractionAverageRating
- UserAverageRating

## Exploratory Data Analysis

EDA was performed to understand:
- Rating distributions
- Visit modes
- Tourism seasons
- Popular attractions
- Attraction types
- City-level tourism activity
- User behavior patterns

## SQL Analysis

SQLite was used to perform analytical queries on the integrated tourism dataset.

SQL analyses include:
1. Rating analysis by visit mode
2. Top 10 most visited attractions
3. Top 10 cities by tourism visits
4. Attraction type analysis
5. Seasonal tourism analysis

SQL result files are stored inside the sql_results folder.

## Regression - Attraction Rating Prediction

Target: Rating

Models used:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Best model: Random Forest Regressor

Results:
- MAE: 0.7074
- MSE: 0.8173
- RMSE: 0.9041
- R2: 0.1322

The Random Forest model performed better than the Linear Regression and Decision Tree models based on the evaluated metrics.

## Classification - Visit Mode Prediction

Target: VisitModeName

Models used:
- Random Forest Classifier
- Logistic Regression

Best overall model: Logistic Regression

Results:
- Accuracy: 0.4849
- Precision: 0.4573
- Recall: 0.4849
- F1 Score: 0.4422

## Recommendation System

An item-based collaborative filtering approach was implemented.

A User x Attraction rating matrix was created and item similarity was calculated using cosine similarity.

The system generates a ranked list of personalized attraction recommendations based on historical user ratings.

Recommendation evaluation uses MAP@5 with one held-out interaction per eligible user.

## Streamlit Application

The project includes an interactive Streamlit dashboard with the following pages:

1. Overview
2. Data Analysis
3. SQL Insights
4. Rating Prediction
5. Visit Mode Prediction
6. Recommendations
7. Model Performance
8. Feature Importance
9. Dataset Explorer

## Project Structure

tourism_project/
    app.py
    requirements.txt
    tourism_feature_engineered.csv
    tourism_analytics.db
    random_forest_model.pkl
    preprocessor.pkl
    logistic_regression_classifier.pkl
    classification_preprocessor.pkl
    train_user_item_matrix.pkl
    item_similarity_matrix.pkl
    recommendation_evaluation.csv
    model_comparison.csv
    feature_importance.csv
    classification_model_comparison.csv
    app_backup.py
    sql_results/

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SQLite
- Streamlit
- OpenPyXL

## Project Outcome

The project provides an end-to-end tourism analytics solution combining descriptive analytics, SQL, machine learning, and recommendation techniques.

The final Streamlit application provides an interactive interface for exploring tourism trends, predicting attraction ratings, predicting visit modes, and generating personalized attraction recommendations.