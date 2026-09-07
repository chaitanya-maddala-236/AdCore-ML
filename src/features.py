"""
AdCore-ML :: Feature Engineering
----------------------------------
Builds the shared feature set used by all three downstream models
(revenue forecasting, CTR prediction, audience segmentation).
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    # Core ratios -----------------------------------------------------
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0.0)
    df["cvr"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], 0.0)
    df["engagement_rate"] = np.where(
        df["impressions"] > 0,
        (df["clicks"] + df["conversions"] + df["pages_viewed"]) / df["impressions"],
        0.0,
    )

    # User engagement score (composite, min-max scaled 0-1) -----------
    raw_score = (
        0.4 * df["session_duration"].rank(pct=True)
        + 0.3 * df["pages_viewed"].rank(pct=True)
        + 0.3 * df["ctr"].rank(pct=True)
    )
    df["user_engagement_score"] = raw_score

    # Ad frequency: how often a user sees a given campaign ------------
    freq = df.groupby(["user_id", "campaign_id"])["impressions"].transform("sum")
    df["ad_frequency"] = freq

    # Content affinity: user's historical CTR within a genre ----------
    df["content_affinity"] = df.groupby(["user_id", "content_genre"])["ctr"].transform("mean")

    # Device-category interaction --------------------------------------
    df["device_category_interaction"] = df["device_type"] + "_" + df["advertiser_category"]

    # Time-of-day buckets ------------------------------------------------
    df["time_of_day"] = pd.cut(
        df["hour"], bins=[-1, 5, 11, 17, 21, 24],
        labels=["late_night", "morning", "afternoon", "evening", "night"],
    )
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rolling revenue / CTR (7-day, by date) ----------------------------
    daily = df.groupby(df["timestamp"].dt.date).agg(
        revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum")
    ).sort_index()
    daily["rolling_7d_revenue"] = daily["revenue"].rolling(7, min_periods=1).mean()
    daily["rolling_7d_ctr"] = (daily["clicks"] / daily["impressions"]).rolling(7, min_periods=1).mean()
    daily_map_rev = daily["rolling_7d_revenue"].to_dict()
    daily_map_ctr = daily["rolling_7d_ctr"].to_dict()
    df["rolling_7d_revenue"] = df["timestamp"].dt.date.map(daily_map_rev)
    df["rolling_7d_ctr"] = df["timestamp"].dt.date.map(daily_map_ctr)

    # Average session duration per user ----------------------------------
    df["avg_session_duration_user"] = df.groupby("user_id")["session_duration"].transform("mean")

    return df


if __name__ == "__main__":
    clean = pd.read_csv("data/adcore_clean.csv", parse_dates=["timestamp"])
    featured = engineer_features(clean)
    featured.to_csv("data/adcore_features.csv", index=False)
    print(f"Feature matrix shape: {featured.shape}")
    print(featured[[
        "ctr", "cvr", "engagement_rate", "user_engagement_score",
        "ad_frequency", "content_affinity", "rolling_7d_revenue", "rolling_7d_ctr",
    ]].describe())
