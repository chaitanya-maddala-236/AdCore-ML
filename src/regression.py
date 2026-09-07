"""
AdCore-ML :: Model A - Revenue Forecasting
---------------------------------------------
Baseline: Linear Regression
Model 2 : Random Forest Regressor
Model 3 : XGBoost Regressor
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

NUMERIC_FEATURES = [
    "age", "content_duration", "bid_amount", "impressions", "session_duration",
    "pages_viewed", "cpm", "cpc", "ctr", "cvr", "engagement_rate",
    "user_engagement_score", "ad_frequency", "content_affinity",
    "rolling_7d_revenue", "rolling_7d_ctr", "avg_session_duration_user",
    "is_weekend", "hour",
]
CATEGORICAL_FEATURES = [
    "device_type", "subscription_type", "content_genre", "ad_type",
    "ad_placement", "advertiser_category", "time_of_day",
]
TARGET = "revenue"


def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def build_preprocessor():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    models = {}

    # ---- Baseline: Linear Regression --------------------------------
    lin_pipe = Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())])
    lin_pipe.fit(X_train, y_train)
    pred = lin_pipe.predict(X_test)
    results["Linear Regression"] = _score(y_test, pred)
    models["Linear Regression"] = lin_pipe

    # ---- Random Forest -------------------------------------------------
    rf_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)
    pred = rf_pipe.predict(X_test)
    results["Random Forest"] = _score(y_test, pred)
    models["Random Forest"] = rf_pipe

    # ---- XGBoost -----------------------------------------------------
    xgb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    pred = xgb_pipe.predict(X_test)
    results["XGBoost"] = _score(y_test, pred)
    models["XGBoost"] = xgb_pipe

    best_model_name = min(results, key=lambda k: results[k]["MAE"])

    output = {
        "comparison": results,
        "best_model": best_model_name,
        "explanation": (
            f"{best_model_name} wins on MAE/RMSE because it captures non-linear "
            "interactions between engagement, device, and campaign features that a "
            "purely linear model cannot represent, while controlling variance better "
            "than a single-tree model would. We select on MAE (not just R^2) because "
            "revenue is right-skewed and MAE is more robust to the few very large "
            "spend campaigns, giving a truer picture of typical forecast error."
        ),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    return output, models


def _score(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/adcore_features.csv", parse_dates=["timestamp"])
    out, models = train_and_evaluate(df)
    with open("results/regression_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    import joblib
    joblib.dump(models[out["best_model"]], "results/best_regression_model.pkl")
