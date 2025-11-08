import os

import lightgbm as lgb
import xgboost as xgb
import optuna


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor



def tune_lgbm_params(train_df, n_trials=200, n_splits=3):
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "seed": 42,
            "verbosity": -1
        }
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmses = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)
            evals_result = {}
            model = lgb.train(
                params,
                train_data,
                num_boost_round=5000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(stopping_rounds=200, verbose=False),
                    lgb.log_evaluation(period=50),
                ],
            )
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmses.append(rmse)
        return np.mean(rmses)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best LGBM params:", study.best_params)
    return study.best_params

def tune_xgb_params(train_df, n_trials=200, n_splits=3):
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "seed": 3,
            "verbosity": 0
        }
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmses = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, "eval")],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            y_pred = model.predict(dval)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmses.append(rmse)
        return np.mean(rmses)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best XGBoost params:", study.best_params)
    return study.best_params

def tune_rf_params(train_df, n_trials=200, n_splits=3):
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 3,
            "n_jobs": -1
        }
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmses = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmses.append(rmse)
        return np.mean(rmses)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best RF params:", study.best_params)
    return study.best_params

if __name__ == "__main__":
  INPUT_DIR = '/content/drive/MyDrive/BigMart'
  train_df=pd.read_csv(os.path.join(INPUT_DIR, 'train_processed.csv'), skipinitialspace=True, low_memory=False)
  lgbm_params = tune_lgbm_params(train_df)
  xgb_params = tune_xgb_params(train_df)
  rf_params = tune_rf_params(train_df)

