"""
AdCore-ML :: Data Quality Pipeline
-----------------------------------
Raw Data -> Schema Validation -> Missing Value Detection -> Duplicate
Detection -> Outlier Detection -> Feature Validation -> Clean Dataset

Produces a JSON quality report alongside the cleaned dataframe so the
issue can be shown on the dashboard.
"""

import numpy as np
import pandas as pd

EXPECTED_SCHEMA = {
    "user_id": "int", "age": "float", "gender": "object", "country": "object",
    "city": "object", "device_type": "object", "subscription_type": "object",
    "content_id": "int", "content_genre": "object", "content_duration": "float",
    "content_rating": "object", "campaign_id": "int", "ad_type": "object",
    "ad_placement": "object", "advertiser_category": "object", "bid_amount": "float",
    "impressions": "int", "clicks": "int", "conversions": "int",
    "session_duration": "float", "pages_viewed": "int", "cpm": "float",
    "cpc": "float", "revenue": "float", "cost": "float", "timestamp": "datetime",
}


def schema_validation(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    return {"expected_columns": len(EXPECTED_SCHEMA), "missing_columns": missing_cols,
            "actual_columns": df.shape[1]}


def missing_value_report(df: pd.DataFrame) -> dict:
    pct = (df.isna().mean() * 100).round(3)
    return {col: float(v) for col, v in pct[pct > 0].sort_values(ascending=False).items()}


def duplicate_report(df: pd.DataFrame) -> dict:
    dupe_mask = df.duplicated(subset=[c for c in df.columns if c != "timestamp"], keep="first")
    return {"duplicate_rows": int(dupe_mask.sum()),
            "duplicate_pct": round(float(dupe_mask.mean() * 100), 3)}


def outlier_report(df: pd.DataFrame) -> dict:
    report = {}
    for col in ["revenue", "session_duration", "cost", "bid_amount"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        report[col] = {"n_outliers": n_out, "pct": round(n_out / len(df) * 100, 3),
                        "lower_bound": round(float(lower), 2), "upper_bound": round(float(upper), 2)}
    return report


def feature_validation(df: pd.DataFrame) -> dict:
    ctr = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)
    issues = {
        "negative_revenue": int((df["revenue"] < 0).sum()),
        "negative_session_duration": int((df["session_duration"] < 0).sum()),
        "negative_cost": int((df["cost"] < 0).sum()),
        "ctr_over_100pct": int((ctr > 1).sum()),
        "conversions_exceed_clicks": int((df["conversions"] > df["clicks"]).sum()),
        "invalid_timestamps": int(df["timestamp"].isna().sum()),
    }
    return issues


def run_quality_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Runs the full Raw -> Clean pipeline and returns (clean_df, report)."""
    report = {
        "raw_row_count": len(df),
        "schema_validation": schema_validation(df),
        "missing_values_pct": missing_value_report(df),
        "duplicates": duplicate_report(df),
        "outliers": outlier_report(df),
        "feature_validation_issues": feature_validation(df),
    }

    clean = df.copy()

    # 1. Drop exact/near-duplicates
    clean = clean.drop_duplicates(subset=[c for c in clean.columns if c != "timestamp"], keep="first")

    # 2. Fix impossible funnel values: clicks cannot exceed impressions,
    #    conversions cannot exceed clicks
    clean["clicks"] = np.minimum(clean["clicks"], clean["impressions"])
    clean["conversions"] = np.minimum(clean["conversions"], clean["clicks"])

    # 3. Fix negative sign errors (billing glitches) via absolute value
    clean["cost"] = clean["cost"].abs()
    clean["session_duration"] = clean["session_duration"].abs()

    # 4. Cap extreme outliers (winsorize) rather than deleting rows,
    #    to preserve data volume while removing distortion
    for col in ["revenue", "session_duration"]:
        upper = clean[col].quantile(0.995)
        clean[col] = np.minimum(clean[col], upper)

    # 5. Impute missing values
    clean["age"] = clean["age"].fillna(clean["age"].median())
    clean["gender"] = clean["gender"].fillna("Unknown")
    clean["city"] = clean["city"].fillna("Unknown")
    clean["content_rating"] = clean["content_rating"].fillna(clean["content_rating"].mode()[0])
    clean["bid_amount"] = clean["bid_amount"].fillna(clean["bid_amount"].median())
    clean["session_duration"] = clean["session_duration"].fillna(clean["session_duration"].median())

    # 6. Drop invalid timestamps (none expected, safety net)
    clean = clean.dropna(subset=["timestamp"])

    report["clean_row_count"] = len(clean)
    report["rows_removed"] = len(df) - len(clean)
    report["rows_removed_pct"] = round((len(df) - len(clean)) / len(df) * 100, 3)

    return clean.reset_index(drop=True), report


if __name__ == "__main__":
    import json
    raw = pd.read_csv("data/adcore_raw.csv", parse_dates=["timestamp"])
    clean_df, rpt = run_quality_pipeline(raw)
    clean_df.to_csv("data/adcore_clean.csv", index=False)
    with open("results/data_quality_report.json", "w") as f:
        json.dump(rpt, f, indent=2, default=str)
    print(json.dumps(rpt, indent=2, default=str))
