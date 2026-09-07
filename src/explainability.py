"""
AdCore-ML :: Explainability
------------------------------
Runs SHAP on the best revenue-forecasting model to surface which
features drive predictions, in a business-readable format.
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap

from regression import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def run_shap(df: pd.DataFrame, model_path: str = "results/best_regression_model.pkl",
             sample_size: int = 2000) -> dict:
    pipe = joblib.load(model_path)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].sample(
        n=min(sample_size, len(df)), random_state=42
    )

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    X_trans = prep.transform(X)
    feature_names = prep.get_feature_names_out()

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Roll one-hot-encoded columns back up to their original business feature
    grouped = {}
    for fname, val in zip(feature_names, mean_abs_shap):
        clean_name = fname.replace("num__", "").replace("cat__", "")
        base = clean_name.split("_")[0] if "__" not in clean_name else clean_name
        # find which original feature this belongs to
        original = None
        for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if clean_name == f or clean_name.startswith(f + "_"):
                original = f
                break
        original = original or clean_name
        grouped[original] = grouped.get(original, 0) + float(val)

    total = sum(grouped.values())
    importance = {k: round(v / total, 4) for k, v in grouped.items()}
    importance_sorted = dict(sorted(importance.items(), key=lambda x: -x[1])[:12])

    return {"feature_importance": importance_sorted, "sample_size": len(X)}


if __name__ == "__main__":
    df = pd.read_csv("data/adcore_features.csv", parse_dates=["timestamp"])
    out = run_shap(df)
    with open("results/shap_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
