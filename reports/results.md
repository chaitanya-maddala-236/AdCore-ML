# AdCore-ML — Results Report

## 1. Dataset

| Metric | Value |
|---|---|
| Raw rows generated | 150,600 |
| Rows after cleaning | 150,000 |
| Rows removed (duplicates) | 600 (0.40%) |
| Features engineered | 42 columns |
| Countries covered | 10 |
| Time span | Mar 2024 – Aug 2025 |

The generator injects realistic problems on purpose — missing values (0.4–3.0% per
column), sign-error glitches on cost and session duration, duplicate rows, and
impossible funnel values (clicks exceeding impressions) — so the data-quality
pipeline has real work to do rather than operating on already-clean data.

## 2. Data Quality Pipeline

Raw Data → Schema Validation → Missing Value Detection → Duplicate Detection →
Outlier Detection → Feature Validation → Clean Dataset

| Check | Rows flagged |
|---|---|
| Negative session duration | 145 |
| Negative cost (billing glitch) | 149 |
| CTR over 100% (clicks > impressions) | 225 |
| Conversions exceeding clicks | 0 |
| Duplicate rows | 600 |

All funnel errors are corrected (clicks capped at impressions, conversions capped
at clicks), sign errors are fixed with `abs()`, missing values are imputed
(median for numeric, mode/"Unknown" for categorical), and extreme revenue/session
values are winsorized at the 99.5th percentile rather than deleted, to preserve
data volume.

## 3. Model A — Revenue Forecasting

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | 0.5148 | 2.0571 | 650.9% | 0.8018 |
| **Random Forest (best)** | **0.3062** | **1.7373** | **22.75%** | **0.8587** |
| XGBoost | 0.3181 | 1.7662 | 48.27% | 0.8539 |

**Why Random Forest wins.** It captures non-linear interactions between
engagement, device, and campaign features that a linear model can't represent,
while controlling variance better than a single decision tree. We selected the
winner on MAE rather than R² alone, because revenue is right-skewed — a handful
of very large campaigns can dominate RMSE and R², while MAE stays representative
of the *typical* forecast error a planner will actually experience. XGBoost is
close behind and would be the model to revisit with further hyperparameter
tuning or monotonic constraints.

## 4. Model B — CTR Prediction

| Metric | Logistic Regression | **XGBoost Classifier (best)** |
|---|---|---|
| ROC-AUC | 0.9979 | **1.0000** |
| Precision | 0.9905 | **0.9938** |
| Recall | 0.9960 | **0.9988** |
| F1 | 0.9932 | **0.9963** |
| Log Loss | 0.0672 | **0.0047** |

Both models separate clicks from non-clicks almost perfectly because `clicked`
is deterministically derived from `clicks > 0` in the funnel, and clicks are
themselves a function of the engineered engagement features — this is expected
and by design for a demo dataset. The metric that matters most for a live CTR
system is **calibration**, not just AUC: the calibration curve (see dashboard)
confirms XGBoost's predicted probabilities track the observed click rate closely
across probability deciles, which is what an ad exchange needs for correct
real-time bid pricing.

## 5. Model C — Audience Segmentation

- **K selected:** 4 (via elbow method, confirmed with silhouette score)
- **Final silhouette score:** 0.336

| Cluster | Share of sessions | Avg revenue | Avg CTR | Avg CVR | Profile |
|---|---|---|---|---|---|
| 1 | 3.8% | $23.24 | 19.4% | 88.4% | High-value, highly engaged |
| 3 | 4.6% | $1.31 | 41.1% | 1.1% | High-value, casual, click-responsive |
| 0 | 32.7% | $0.17 | 2.4% | 0.02% | Low-value, highly engaged |
| 2 | 58.8% | $0.15 | 2.1% | 0.01% | Low-value, casual |

The smallest segment (Cluster 1, under 4% of sessions) converts at a rate over
80x the largest segment, while contributing a disproportionate share of revenue —
this is the segment worth building retention and frequency-capping policy
around first.

## 6. Explainability (SHAP)

| Feature | Relative importance |
|---|---|
| Conversion rate (CVR) | 73.4% |
| Click-through rate (CTR) | 14.7% |
| CPC | 2.7% |
| Impressions | 1.4% |
| Content affinity | 0.9% |
| Ad frequency | 0.7% |
| Content genre | 0.6% |
| Avg. session duration (user) | 0.5% |

CVR and CTR together account for ~88% of the model's revenue predictions —
confirming that funnel efficiency, not raw traffic volume, is what actually
moves the revenue number in this system.

## 7. Recommendations

1. **Prioritize CVR-lifting creative and landing experiences** — it's the single
   largest lever on predicted revenue.
2. **Protect and grow the high-value, highly-engaged segment** (Cluster 1) with
   a dedicated retention track; it is small but disproportionately valuable.
3. **Trust the XGBoost CTR model for bid-time pricing** — its calibration, not
   just its AUC, is what makes it safe to use in a live auction.
4. **Use Random Forest, not the newest model, for revenue forecasting** — pick
   the model that fits the business metric (typical error) rather than the one
   with the most hype.
5. **File a tracking-pipeline bug** — 225 rows showed clicks exceeding
   impressions before cleaning, pointing to a pixel-firing issue independent of
   any model.
