"""Combine already-computed result JSON files into results/dashboard_data.json
without re-running the (expensive) model training steps."""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def main():
    quality = load("data_quality_report.json")
    reg = load("regression_results.json")
    clf = load("classification_results.json")
    clus = load("clustering_results.json")
    shap_ = load("shap_results.json")

    featured = pd.read_csv(DATA / "adcore_features.csv", usecols=[
        "revenue", "impressions", "clicks", "ctr", "country", "timestamp"
    ], parse_dates=["timestamp"])

    overview = {
        "raw_rows": int(quality["raw_row_count"]),
        "clean_rows": int(quality["clean_row_count"]),
        "rows_removed_pct": quality["rows_removed_pct"],
        "total_revenue": round(float(featured["revenue"].sum()), 2),
        "total_impressions": int(featured["impressions"].sum()),
        "total_clicks": int(featured["clicks"].sum()),
        "avg_ctr": round(float(featured["ctr"].mean()), 4),
        "countries": int(featured["country"].nunique()),
        "date_range": [str(featured["timestamp"].min()), str(featured["timestamp"].max())],
    }

    combined = {
        "overview": overview,
        "data_quality": quality,
        "regression": reg,
        "classification": clf,
        "clustering": clus,
        "shap": shap_,
    }
    with open(RESULTS / "dashboard_data.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print("Wrote results/dashboard_data.json")
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
