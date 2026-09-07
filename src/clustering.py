"""
AdCore-ML :: Model C - Audience Segmentation
------------------------------------------------
K-Means clustering over behavioural features, with K chosen via the
Elbow Method + Silhouette Score, followed by business-language profiling.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = [
    "session_duration", "ctr", "cvr", "revenue", "content_affinity",
    "pages_viewed",  # proxy for device_usage intensity
]


def choose_k(X_scaled, k_range=range(2, 9)) -> dict:
    inertias, silhouettes = [], []
    sample_idx = np.random.default_rng(42).choice(
        len(X_scaled), size=min(20000, len(X_scaled)), replace=False
    )
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
        silhouettes.append(float(sil))
    return {"k_values": list(k_range), "inertia": inertias, "silhouette": silhouettes}


def profile_clusters(cluster_means: pd.DataFrame) -> list[str]:
    """Labels each cluster relative to the OTHER clusters (not the global
    median), using rank position across clusters so labels stay distinct."""
    n = len(cluster_means)
    rev_rank = cluster_means["revenue"].rank(ascending=False)      # 1 = highest revenue
    dur_rank = cluster_means["session_duration"].rank(ascending=False)
    ctr_rank = cluster_means["ctr"].rank(ascending=False)

    labels = []
    for idx in cluster_means.index:
        value_tag = "High-value" if rev_rank[idx] <= n / 2 else "Low-value"
        engage_tag = "highly engaged" if dur_rank[idx] <= n / 2 else "casual"
        click_tag = "click-responsive" if ctr_rank[idx] == 1 else None
        parts = [value_tag, engage_tag]
        if click_tag:
            parts.append(click_tag)
        labels.append(" ".join(parts) + " audience")
    return labels


def run_clustering(df: pd.DataFrame, k: int = 4) -> dict:
    data = df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    elbow = choose_k(X_scaled)

    km = KMeans(n_clusters=k, n_init=15, random_state=42)
    labels = km.fit_predict(X_scaled)
    sample_idx = np.random.default_rng(1).choice(len(X_scaled), size=min(20000, len(X_scaled)), replace=False)
    final_silhouette = float(silhouette_score(X_scaled[sample_idx], labels[sample_idx]))

    data = data.copy()
    data["cluster"] = labels

    sizes = data["cluster"].value_counts().sort_index()
    cluster_ids = sorted(data["cluster"].unique())
    cluster_means = data.groupby("cluster")[FEATURES].mean()
    labels_by_cluster = dict(zip(cluster_means.index, profile_clusters(cluster_means)))

    profiles = []
    for c in cluster_ids:
        means = cluster_means.loc[c]
        profiles.append({
            "cluster": int(c),
            "size": int(sizes[c]),
            "pct_of_users": round(float(sizes[c] / len(data) * 100), 2),
            "label": labels_by_cluster[c].capitalize(),
            "avg_session_duration": round(float(means["session_duration"]), 2),
            "avg_ctr": round(float(means["ctr"]), 4),
            "avg_cvr": round(float(means["cvr"]), 4),
            "avg_revenue": round(float(means["revenue"]), 2),
            "avg_content_affinity": round(float(means["content_affinity"]), 4),
            "avg_pages_viewed": round(float(means["pages_viewed"]), 2),
        })

    # 2D projection for scatter plot (first two scaled features: session_duration, revenue)
    scatter_sample_idx = np.random.default_rng(2).choice(len(data), size=min(3000, len(data)), replace=False)
    scatter = data.iloc[scatter_sample_idx][["session_duration", "revenue", "cluster"]].to_dict("records")

    output = {
        "k_selected": k,
        "elbow_silhouette_search": elbow,
        "final_silhouette_score": round(final_silhouette, 4),
        "cluster_profiles": sorted(profiles, key=lambda p: -p["avg_revenue"]),
        "scatter_sample": scatter,
    }
    return output


if __name__ == "__main__":
    df = pd.read_csv("data/adcore_features.csv", parse_dates=["timestamp"])
    out = run_clustering(df, k=4)
    with open("results/clustering_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("scatter_sample", "elbow_silhouette_search")}, indent=2))
