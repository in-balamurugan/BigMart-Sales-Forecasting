# BigMart Sales Prediction Pipeline: One-Page Summary

## Overview
This project aims to predict sales for BigMart outlets using historical sales data.

## Data Preprocessing & Feature Engineering
- **Missing Value Imputation:** Item weights, visibility, and outlet sizes are imputed using group values.
- **Feature Creation:** New features include item categories, outlet age, price/visibility bins, and interaction items 
- **Aggregated Features:** Outlet and item-type level statistics are merged to enrich the dataset.
- **Target Encoding:** KFold target encoding is applied to product and store identifier
- **Categorical Encoding:** Ordinal and one-hot encoding are used for categorical variables.

## Modeling Approach
- **Models Used:** LightGBM, XGBoost, and Random Forest are trained using KFold cross-validation.
- **Meta-Model:** Out-of-fold predictions from base models are stacked and used to train a  regression model, which produces the final predictions.
- **Evaluation:** RMSE is used to assess model performance across folds.
- **Hyperparameter Tuning:** Optuna used for hyperparameter tuning

## Pipeline Flow
1. **Data Loading:** Reads train and test datasets from Google Drive
2. **Preprocessing:** Applies feature engineering and encoding to both train and test sets.
3. **Model Training:** Trains three base models with cross-validation and generates predictions.
4. **Stacking:** Combines base model outputs using a meta-model for improved accuracy.
5. **Submission Files:** Saves predictions for each model, the meta-model, and a simple average as CSV files for competition submission.

## Pipeline Flow Diagram
```mermaid
flowchart TD
	A[Store Data Loading] --> B[Preprocessing]
	B --> C[Feature Engineering]
	C --> D[Model Training]
	D --> E[LightGBM]
	D --> F[XGBoost]
	D --> G[Random Forest]
	E & F & G --> H[Stacking ]
	H --> I[Submission Files]
```

## Things Tried That Did Not Help
1. Target encoding of item/store identifiers gave similar performance to label encoding.
2. Hyperparameter tuning after 200 trial didnt significantly help
3. Averaging submissions did not help so moved to meta model

