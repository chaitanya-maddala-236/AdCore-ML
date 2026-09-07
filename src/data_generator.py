"""
AdCore-ML :: Synthetic Data Generator
--------------------------------------
Generates a realistic AdTech revenue-intelligence dataset with intentional
data-quality problems (missing values, outliers, duplicates) and genuine
causal structure:

    engagement -> click probability -> conversion probability -> revenue

Run:
    python src/data_generator.py --rows 150000 --out data/adcore_raw.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42


def _choice(rng, values, size, p=None):
    return rng.choice(values, size=size, p=p)


def generate_dataset(n_rows: int = 150_000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # Dimension pools
    # ---------------------------------------------------------------
    countries = ["US", "UK", "IN", "DE", "BR", "CA", "AU", "FR", "JP", "MX"]
    country_p = [0.28, 0.10, 0.16, 0.08, 0.10, 0.06, 0.05, 0.06, 0.06, 0.05]

    cities_by_country = {
        "US": ["New York", "Chicago", "Austin", "Seattle"],
        "UK": ["London", "Manchester"],
        "IN": ["Hyderabad", "Mumbai", "Bengaluru"],
        "DE": ["Berlin", "Munich"],
        "BR": ["Sao Paulo", "Rio de Janeiro"],
        "CA": ["Toronto", "Vancouver"],
        "AU": ["Sydney", "Melbourne"],
        "FR": ["Paris", "Lyon"],
        "JP": ["Tokyo", "Osaka"],
        "MX": ["Mexico City", "Guadalajara"],
    }

    device_types = ["mobile", "desktop", "tablet", "smart_tv", "console"]
    device_p = [0.48, 0.30, 0.12, 0.08, 0.02]
    # click-through multiplier per device (mobile scrolls fast -> lower CTR)
    device_ctr_mult = {"mobile": 0.85, "desktop": 1.25, "tablet": 1.0, "smart_tv": 0.6, "console": 0.7}

    subscription_types = ["free", "basic", "premium", "family"]
    subscription_p = [0.45, 0.25, 0.20, 0.10]

    genres = ["sports", "news", "drama", "comedy", "documentary", "kids", "music", "gaming"]
    genre_engagement_bias = {  # baseline engagement multiplier
        "sports": 1.35, "news": 0.9, "drama": 1.1, "comedy": 1.05,
        "documentary": 0.8, "kids": 0.7, "music": 1.15, "gaming": 1.4,
    }

    content_ratings = ["G", "PG", "PG-13", "R"]

    ad_types = ["video", "banner", "native", "interstitial", "audio"]
    ad_placements = ["pre_roll", "mid_roll", "post_roll", "sidebar", "in_feed"]
    advertiser_categories = [
        "retail", "auto", "finance", "cpg", "tech", "travel", "gaming", "healthcare"
    ]

    n = n_rows

    # ---------------------------------------------------------------
    # Temporal backbone (18 months of activity, hourly seasonality)
    # ---------------------------------------------------------------
    start = datetime(2024, 3, 1)
    span_hours = 18 * 30 * 24
    hour_offsets = rng.integers(0, span_hours, size=n)
    timestamps = [start + timedelta(hours=int(h)) for h in hour_offsets]
    hours = np.array([t.hour for t in timestamps])
    dow = np.array([t.weekday() for t in timestamps])
    months = np.array([t.month for t in timestamps])

    # seasonality: evening + weekend lift, Nov/Dec holiday lift
    hour_lift = 1.0 + 0.35 * np.exp(-((hours - 20) ** 2) / 18.0)
    weekend_lift = np.where(dow >= 5, 1.15, 1.0)
    holiday_lift = np.where(np.isin(months, [11, 12]), 1.25, 1.0)
    seasonality = hour_lift * weekend_lift * holiday_lift

    # ---------------------------------------------------------------
    # User dimension
    # ---------------------------------------------------------------
    user_id = rng.integers(100000, 999999, size=n)
    age = rng.integers(13, 75, size=n)
    gender = _choice(rng, ["M", "F", "Other"], n, p=[0.48, 0.48, 0.04])
    country = _choice(rng, countries, n, p=country_p)
    city = np.array([rng.choice(cities_by_country[c]) for c in country])
    device_type = _choice(rng, device_types, n, p=device_p)
    subscription_type = _choice(rng, subscription_types, n, p=subscription_p)

    # ---------------------------------------------------------------
    # Content dimension
    # ---------------------------------------------------------------
    content_id = rng.integers(1000, 9999, size=n)
    content_genre = _choice(rng, genres, n)
    content_duration = np.clip(rng.normal(35, 20, n), 2, 180).round(1)  # minutes
    content_rating = _choice(rng, content_ratings, n, p=[0.15, 0.30, 0.35, 0.20])

    # ---------------------------------------------------------------
    # Advertising dimension
    # ---------------------------------------------------------------
    campaign_id = rng.integers(5000, 5999, size=n)
    ad_type = _choice(rng, ad_types, n)
    ad_placement = _choice(rng, ad_placements, n)
    advertiser_category = _choice(rng, advertiser_categories, n)
    bid_amount = np.clip(rng.gamma(4.0, 1.4, n), 0.2, 60).round(2)

    # ---------------------------------------------------------------
    # Engagement backbone (drives everything downstream)
    # ---------------------------------------------------------------
    subscription_engagement_bonus = pd.Series(subscription_type).map(
        {"free": 0.85, "basic": 1.0, "premium": 1.25, "family": 1.15}
    ).to_numpy()
    genre_bonus = pd.Series(content_genre).map(genre_engagement_bias).to_numpy()

    base_engagement = (
        0.5
        + 0.35 * (content_duration / 60.0)
        + 0.25 * subscription_engagement_bonus
        + 0.3 * genre_bonus
        + rng.normal(0, 0.25, n)
    )
    engagement_score = np.clip(base_engagement * seasonality, 0.05, None)

    session_duration = np.clip(
        (content_duration * 0.6) * (engagement_score / engagement_score.mean())
        + rng.normal(0, 8, n),
        0.5, None,
    ).round(1)  # minutes

    pages_viewed = np.clip(
        rng.poisson(lam=np.clip(engagement_score * 2.2, 0.1, None)), 0, None
    )

    # ---------------------------------------------------------------
    # Funnel: impressions -> clicks -> conversions -> revenue
    # ---------------------------------------------------------------
    impressions = rng.integers(1, 12, size=n)

    device_mult = pd.Series(device_type).map(device_ctr_mult).to_numpy()
    ctr_base = 0.03 + 0.05 * (engagement_score / (engagement_score.max()))
    ctr_prob = np.clip(ctr_base * device_mult, 0.005, 0.65)
    clicks = rng.binomial(impressions, ctr_prob)

    cvr_base = 0.08 + 0.20 * (engagement_score / engagement_score.max())
    cvr_prob = np.clip(cvr_base, 0.01, 0.75)
    conversions = rng.binomial(np.maximum(clicks, 0), cvr_prob)

    cpm = np.clip(rng.normal(8.5, 3.0, n) + bid_amount * 0.3, 0.5, None).round(2)
    cpc = np.clip(rng.normal(0.9, 0.4, n) + bid_amount * 0.05, 0.05, None).round(2)

    ad_cost = (impressions / 1000.0) * cpm + clicks * cpc * 0.15
    revenue_per_conversion = np.clip(rng.normal(22, 9, n), 3, None)
    revenue = (
        (impressions / 1000.0) * cpm * 0.4
        + clicks * cpc * 0.6
        + conversions * revenue_per_conversion
    ).round(2)

    clicked_flag = (clicks > 0).astype(int)

    df = pd.DataFrame({
        "user_id": user_id, "age": age, "gender": gender, "country": country,
        "city": city, "device_type": device_type, "subscription_type": subscription_type,
        "content_id": content_id, "content_genre": content_genre,
        "content_duration": content_duration, "content_rating": content_rating,
        "campaign_id": campaign_id, "ad_type": ad_type, "ad_placement": ad_placement,
        "advertiser_category": advertiser_category, "bid_amount": bid_amount,
        "impressions": impressions, "clicks": clicks, "conversions": conversions,
        "clicked": clicked_flag,
        "session_duration": session_duration, "pages_viewed": pages_viewed,
        "cpm": cpm, "cpc": cpc, "revenue": revenue, "cost": ad_cost.round(2),
        "timestamp": timestamps, "hour": hours, "day_of_week": dow, "month": months,
    })

    # ---------------------------------------------------------------
    # Inject realistic data-quality problems
    # ---------------------------------------------------------------
    df = _inject_missing_values(df, rng)
    df = _inject_outliers(df, rng)
    df = _inject_duplicates(df, rng)

    return df


def _inject_missing_values(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.copy()
    missing_cols = {
        "age": 0.02, "gender": 0.015, "session_duration": 0.03,
        "content_rating": 0.02, "bid_amount": 0.01, "city": 0.02,
    }
    for col, frac in missing_cols.items():
        idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.copy()
    n_outliers = int(len(df) * 0.005)

    idx = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[idx, "revenue"] = df.loc[idx, "revenue"] * rng.uniform(15, 40, n_outliers)

    idx2 = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[idx2, "session_duration"] = df.loc[idx2, "session_duration"] * rng.uniform(10, 25, n_outliers)

    # a few negative-cost / negative-duration glitches (billing system bug simulation)
    idx3 = rng.choice(df.index, size=int(len(df) * 0.001), replace=False)
    df.loc[idx3, "cost"] = -df.loc[idx3, "cost"]

    idx4 = rng.choice(df.index, size=int(len(df) * 0.001), replace=False)
    df.loc[idx4, "session_duration"] = -df.loc[idx4, "session_duration"].abs()

    # a few impossible CTR rows: clicks > impressions (tracking pixel bug)
    idx5 = rng.choice(df.index, size=int(len(df) * 0.0015), replace=False)
    df.loc[idx5, "clicks"] = df.loc[idx5, "impressions"] + rng.integers(1, 5, len(idx5))

    return df


def _inject_duplicates(df: pd.DataFrame, rng) -> pd.DataFrame:
    n_dupes = int(len(df) * 0.004)
    dupe_rows = df.sample(n=n_dupes, random_state=1)
    return pd.concat([df, dupe_rows], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=150_000)
    parser.add_argument("--out", type=str, default="data/adcore_raw.csv")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    data = generate_dataset(args.rows, args.seed)
    data.to_csv(args.out, index=False)
    print(f"Generated {len(data):,} rows -> {args.out}")
    print(data.dtypes)
