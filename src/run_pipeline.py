"""
AdCore-ML :: End-to-End Pipeline Runner
-------------------------------------------
Synthetic Data -> Data Quality -> Feature Engineering -> 3 Models -> SHAP
-> combined results/dashboard_data.json (consumed by the dashboard UI)

Run:
    python src/run_pipeline.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import joblib

from data_generator import generate_dataset
from preprocessing import run_quality_pipeline
from features import engineer_features
import regression
import classification
import clustering
import explainability

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    t0 = time.time()

    log("1/6 Generating synthetic dataset ...")
    raw = generate_dataset(n_rows=150_000)
    raw.to_csv(DATA / "adcore_raw.csv", index=False)
    log(f"    {len(raw):,} raw rows generated")

    log("2/6 Running data quality pipeline ...")
    clean, quality_report = run_quality_pipeline(raw)
    clean.to_csv(DATA / "adcore_clean.csv", index=False)
    log(f"    {quality_report['rows_removed']} rows removed "
        f"({quality_report['rows_removed_pct']}%) -> {len(clean):,} clean rows")

    log("3/6 Engineering features ...")
    featured = engineer_features(clean)
    featured.to_csv(DATA / "adcore_features.csv", index=False)
    log(f"    Feature matrix: {featured.shape[0]:,} rows x {featured.shape[1]} cols")

    log("4/6 Training Model A: Revenue Forecasting (Linear / RF / XGBoost) ...")
    reg_results, reg_models = regression.train_and_evaluate(featured)
    joblib.dump(reg_models[reg_results["best_model"]], RESULTS / "best_regression_model.pkl")
    log(f"    Best: {reg_results['best_model']} "
        f"(MAE={reg_results['comparison'][reg_results['best_model']]['MAE']})")

    log("5/6 Training Model B: CTR Prediction (Logistic / XGBoost) ...")
    clf_results, clf_models = classification.train_and_evaluate(featured)
    log(f"    Best: {clf_results['best_model']} "
        f"(ROC-AUC={clf_results['comparison'][clf_results['best_model']]['ROC_AUC']})")

    log("6/6 Training Model C: Audience Segmentation (K-Means) ...")
    cluster_results = clustering.run_clustering(featured, k=4)
    log(f"    K={cluster_results['k_selected']}, "
        f"silhouette={cluster_results['final_silhouette_score']}")

    log("Running SHAP explainability on best regression model ...")
    shap_results = explainability.run_shap(featured)

    # ------------------------------------------------------------------
    # Persist individual + combined results
    # ------------------------------------------------------------------
    with open(RESULTS / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2, default=str)
    with open(RESULTS / "regression_results.json", "w") as f:
        json.dump(reg_results, f, indent=2)
    with open(RESULTS / "classification_results.json", "w") as f:
        json.dump(clf_results, f, indent=2)
    with open(RESULTS / "clustering_results.json", "w") as f:
        json.dump(cluster_results, f, indent=2)
    with open(RESULTS / "shap_results.json", "w") as f:
        json.dump(shap_results, f, indent=2)

    overview = {
        "dataset": {
            "raw_rows": int(quality_report["raw_row_count"]),
            "clean_rows": int(quality_report["clean_row_count"]),
            "rows_removed_pct": quality_report["rows_removed_pct"],
            "features": int(featured.shape[1]),
            "total_revenue": round(float(featured["revenue"].sum()), 2),
            "total_impressions": int(featured["impressions"].sum()),
            "total_clicks": int(featured["clicks"].sum()),
            "avg_ctr": round(float(featured["ctr"].mean()), 4),
            "countries": int(featured["country"].nunique()),
            "date_range": [str(featured["timestamp"].min()), str(featured["timestamp"].max())],
        },
    }

    combined = {
        "overview": overview,
        "data_quality": quality_report,
        "regression": reg_results,
        "classification": clf_results,
        "clustering": cluster_results,
        "shap": shap_results,
    }
    with open(RESULTS / "dashboard_data.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    log(f"Done in {time.time() - t0:.1f}s. Combined results -> results/dashboard_data.json")


if __name__ == "__main__":
    main()
