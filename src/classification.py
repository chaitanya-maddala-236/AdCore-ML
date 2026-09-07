"""
AdCore-ML :: Model B - CTR Prediction
----------------------------------------
Baseline : Logistic Regression
Advanced : XGBoost Classifier

Target: clicked (0/1)
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, log_loss, roc_curve
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

NUMERIC_FEATURES = [
    "age", "content_duration", "bid_amount", "session_duration", "pages_viewed",
    "cpm", "cpc", "engagement_rate", "user_engagement_score", "ad_frequency",
    "content_affinity", "is_weekend", "hour",
]
CATEGORICAL_FEATURES = [
    "device_type", "subscription_type", "content_genre", "ad_type",
    "ad_placement", "advertiser_category", "time_of_day",
]
TARGET = "clicked"


def build_preprocessor():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    models = {}
    curves = {}

    # ---- Baseline: Logistic Regression -------------------------------
    log_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    log_pipe.fit(X_train, y_train)
    proba = log_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results["Logistic Regression"] = _score(y_test, pred, proba)
    models["Logistic Regression"] = log_pipe

    # ---- XGBoost Classifier --------------------------------------------
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
            eval_metric="logloss", scale_pos_weight=pos_weight,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    proba = xgb_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results["XGBoost Classifier"] = _score(y_test, pred, proba)
    models["XGBoost Classifier"] = xgb_pipe

    best_model_name = max(results, key=lambda k: results[k]["ROC_AUC"])
    best_proba = models[best_model_name].predict_proba(X_test)[:, 1]

    # Calibration curve for the best model -------------------------------
    frac_pos, mean_pred = calibration_curve(y_test, best_proba, n_bins=10, strategy="quantile")
    curves["calibration"] = {
        "predicted_probability": [round(float(x), 4) for x in mean_pred],
        "observed_click_rate": [round(float(x), 4) for x in frac_pos],
    }

    # ROC curve for both models -------------------------------------------
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
        curves.setdefault("roc", {})[name] = {
            "fpr": [round(float(fpr[i]), 4) for i in idx],
            "tpr": [round(float(tpr[i]), 4) for i in idx],
        }

    output = {
        "comparison": results,
        "best_model": best_model_name,
        "curves": curves,
        "n_train": len(X_train), "n_test": len(X_test),
        "positive_rate": round(float(y.mean()), 4),
    }
    return output, models


def _score(y_true, y_pred, proba) -> dict:
    return {
        "ROC_AUC": round(float(roc_auc_score(y_true, proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "LogLoss": round(float(log_loss(y_true, proba)), 4),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/adcore_features.csv", parse_dates=["timestamp"])
    out, models = train_and_evaluate(df)
    with open("results/classification_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "curves"}, indent=2))

    import joblib
    joblib.dump(models[out["best_model"]], "results/best_classification_model.pkl")
