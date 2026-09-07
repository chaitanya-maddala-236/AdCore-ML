# AdCore-ML — AdTech Revenue Intelligence

**An end-to-end machine learning system that forecasts advertising revenue, predicts
which ad impressions will get clicked, and automatically segments audiences —
built on one shared, production-style data pipeline.**

[Open the interactive dashboard](dashboard/index.html) — no install required, just open the file in a browser.

---

## What this project is (for non-technical readers)

Imagine an advertising platform (think: a streaming app or a news site that
shows ads) that wants to answer three questions every revenue and marketing
team asks:

1. **How much money will we make next month?** → *Revenue forecasting*
2. **Which ads are worth showing to which people?** → *Click prediction*
3. **What kinds of users do we actually have?** → *Audience segmentation*

This project builds all three as a working system, on top of a realistic
150,000-row synthetic dataset that includes the same messiness real ad data
has — missing fields, duplicate records, and tracking bugs — and shows exactly
how that mess gets cleaned up before any model sees it.

The result is a single interactive dashboard (`dashboard/index.html`) that
tells the full story: the data problems found and fixed, how three different
forecasting models compare, how well the click-prediction model can be
trusted, what the four audience segments look like, and which factors most
influence predicted revenue.

**Why this matters as a portfolio piece:** it demonstrates the full life cycle
of a real data science project — not just "train a model and report accuracy,"
but data generation with intentional flaws, a documented cleaning pipeline,
multiple candidate models compared honestly (including explaining *why* the
winner wins), model calibration (a step many portfolio projects skip),
unsupervised segmentation with a justified choice of cluster count, and
explainability (SHAP) tying the model back to business drivers.

---

## Quick look

| | |
|---|---|
| **Dataset** | 150,600 synthetic rows → 150,000 after cleaning, 42 engineered features |
| **Revenue forecasting** | Random Forest wins (MAE 0.31) over Linear Regression and XGBoost |
| **Click prediction** | XGBoost, ROC-AUC 1.00, well-calibrated probabilities |
| **Audience segments** | 4 segments found via K-Means (elbow + silhouette = 0.34) |
| **Top revenue driver** | Conversion rate (73% of SHAP importance) |

Full metrics and business commentary: [`reports/results.md`](reports/results.md)

---

## How to view the results

**Option A — just look (recommended):**
Open [`dashboard/index.html`](dashboard/index.html) directly in any browser.
It's a self-contained page with all charts, tables, and metrics already
computed and embedded — nothing to install or run.

**Option B — run the full pipeline yourself:**
```bash
pip install -r requirements.txt
python src/run_pipeline.py
```
This regenerates the synthetic data, re-runs the data-quality pipeline,
re-trains all three models, re-runs SHAP, and writes fresh results to
`results/dashboard_data.json` (which the dashboard reads).

**Option C — explore step by step in notebooks:**
The `notebooks/` folder walks through the same pipeline interactively:
data generation → EDA → revenue forecasting → CTR prediction → audience
clustering.

---

## Project structure

```
adcore-ml/
├── dashboard/
│   ├── index.html          # the interactive results dashboard (open this)
│   └── data.js              # pre-computed results embedded for the dashboard
│
├── data/
│   ├── adcore_raw.csv        # generated synthetic data (with intentional flaws)
│   ├── adcore_clean.csv      # after the data-quality pipeline
│   └── adcore_features.csv   # after feature engineering
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_revenue_forecasting.ipynb
│   ├── 04_ctr_prediction.ipynb
│   └── 05_audience_clustering.ipynb
│
├── src/
│   ├── data_generator.py     # synthetic AdTech data generator
│   ├── preprocessing.py      # data-quality / cleaning pipeline
│   ├── features.py           # shared feature engineering
│   ├── regression.py         # Model A: revenue forecasting
│   ├── classification.py     # Model B: CTR prediction
│   ├── clustering.py         # Model C: audience segmentation
│   ├── explainability.py     # SHAP feature importance
│   ├── run_pipeline.py       # runs everything end-to-end
│   └── combine_results.py    # merges results into the dashboard's data file
│
├── results/                  # JSON outputs from each stage (metrics, profiles, SHAP)
├── reports/
│   └── results.md            # full written report with business recommendations
├── requirements.txt
└── README.md
```

---

## The pipeline, end to end

```
Synthetic AdTech Data (150k rows, intentional flaws)
          │
          ▼
   Data Quality Pipeline
   (schema check → missing values → duplicates → outliers → funnel validation)
          │
          ▼
   Shared Feature Engineering
   (CTR, CVR, engagement score, rolling 7-day revenue, content affinity, ...)
          │
   ┌──────┼────────────────┐
   ▼      ▼                ▼
Revenue  CTR              Audience
Forecast Prediction        Segmentation
(Linear/RF/XGBoost)  (Logistic/XGBoost)   (K-Means, elbow + silhouette)
   │      │                │
   └──────┼────────────────┘
          ▼
   SHAP Explainability
          ▼
   Business Recommendations + Dashboard
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · Chart.js (dashboard charts)

## Notes on the data

The dataset is **entirely synthetic**, generated with intentional causal
structure (higher engagement → higher click probability → higher conversion
probability → higher revenue) plus injected missing values, outliers,
duplicates, and a simulated tracking bug — so the project can honestly
demonstrate data-cleaning and validation work rather than starting from a
pre-cleaned Kaggle CSV. No real user or company data is used anywhere.
"# AdCore-ML" 
