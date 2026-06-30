# **Used-Car-Market-Analysis-and-prediction**
This is my first project.
<br>
Author: Ashish Sharma
Repository: Used-Car-Market-Analysis-and-prediction
<br>
**Overview**
What this project does  
This repository analyzes the used car market and builds predictive models to estimate resale prices. It combines exploratory data analysis, feature engineering, and supervised learning to produce interpretable and performant price predictions.

Key goals
Understand market drivers that affect used car prices.
Build and compare multiple regression models.
Provide a reproducible pipeline from raw data to deployed model.

Data and Preprocessing
Dataset

Source: (describe dataset origin here; add link in repository)

Typical fields: make, model, year, mileage, fuel_type, transmission, owner_count, location, price

Preprocessing steps

Cleaning: remove duplicates, handle missing values, normalize text fields.

Feature engineering: age calculation from year, mileage buckets, one-hot encoding for categorical features, target log-transform when appropriate.

Splitting: stratified or time-aware split depending on dataset characteristics.

Files

data/raw/ raw CSV files.

data/processed/ cleaned and feature-engineered CSVs.

notebooks/ exploratory notebooks for EDA and feature selection.

Modeling and Evaluation

Models included
Baseline: Linear Regression, Ridge, Lasso.
Tree-based: Random Forest, XGBoost, LightGBM.
Ensemble: Stacked or blended models.

Training pipeline
Use src/train.py to run experiments.
Hyperparameter tuning via cross-validation and grid/random search.
Save best model artifacts to models/ with metadata.

Evaluation metrics

Primary: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
Secondary: 𝑅2, Mean Absolute Percentage Error (MAPE).

Reproducibility
Set random seeds in training scripts.
Log experiment parameters and results to experiments/ or an ML tracking tool.
