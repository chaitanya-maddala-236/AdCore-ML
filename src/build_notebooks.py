import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


# =====================================================================
# 01 — Data generation
# =====================================================================
nb1 = make_nb([
    md("# 01 · Data Generation\n"
       "Generates the synthetic AdCore-ML dataset: ~150k rows spanning users, "
       "content, campaigns, engagement and financial fields, with intentionally "
       "injected missing values, outliers, and duplicates so the rest of the "
       "pipeline has real data-quality work to do."),
    code("import sys\n"
         "sys.path.append('../src')\n"
         "import pandas as pd\n"
         "from data_generator import generate_dataset\n"
         "pd.set_option('display.max_columns', 40)"),
    code("df = generate_dataset(n_rows=150_000, seed=42)\n"
         "print(f'Shape: {df.shape}')\n"
         "df.head()"),
    md("## Schema"),
    code("df.dtypes"),
    md("## A quick look at the intentional data-quality issues"),
    code("print('Missing values per column (top 10):')\n"
         "print(df.isna().sum().sort_values(ascending=False).head(10))\n"
         "print()\n"
         "print('Rows where clicks > impressions (tracking bug):',\n"
         "      (df['clicks'] > df['impressions']).sum())\n"
         "print('Rows with negative cost (billing glitch):', (df['cost'] < 0).sum())\n"
         "print('Duplicate rows:', df.duplicated().sum())"),
    md("## Save to disk\n"
       "This is the same file used by the rest of the pipeline "
       "(`data/adcore_raw.csv`)."),
    code("df.to_csv('../data/adcore_raw.csv', index=False)\n"
         "print('Saved', len(df), 'rows to data/adcore_raw.csv')"),
])

# =====================================================================
# 02 — EDA
# =====================================================================
nb2 = make_nb([
    md("# 02 · Exploratory Data Analysis\n"
       "A first look at the cleaned dataset — funnel shape, seasonality, and "
       "the relationships the downstream models will need to learn."),
    code("import sys\n"
         "sys.path.append('../src')\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "df = pd.read_csv('../data/adcore_clean.csv', parse_dates=['timestamp'])\n"
         "df.describe(include='number').T"),
    md("## Revenue by device type"),
    code("rev_by_device = df.groupby('device_type')['revenue'].mean().sort_values(ascending=False)\n"
         "fig, ax = plt.subplots(figsize=(7,4))\n"
         "rev_by_device.plot(kind='bar', ax=ax, color='#33D6C0')\n"
         "ax.set_ylabel('Avg revenue per impression-row')\n"
         "ax.set_title('Average revenue by device type')\n"
         "plt.tight_layout(); plt.show()"),
    md("## Click-through rate by content genre"),
    code("ctr_by_genre = (df.groupby('content_genre').apply(\n"
         "    lambda g: g['clicks'].sum() / g['impressions'].sum(), include_groups=False)\n"
         "    .sort_values(ascending=False))\n"
         "fig, ax = plt.subplots(figsize=(7,4))\n"
         "ctr_by_genre.plot(kind='bar', ax=ax, color='#FFB800')\n"
         "ax.set_ylabel('CTR')\n"
         "ax.set_title('CTR by content genre')\n"
         "plt.tight_layout(); plt.show()"),
    md("## Daily revenue trend with 7-day rolling average"),
    code("daily = df.groupby(df['timestamp'].dt.date)['revenue'].sum()\n"
         "rolling = daily.rolling(7, min_periods=1).mean()\n"
         "fig, ax = plt.subplots(figsize=(9,4))\n"
         "daily.plot(ax=ax, alpha=0.3, color='#8C96AF', label='Daily revenue')\n"
         "rolling.plot(ax=ax, color='#FFB800', linewidth=2, label='7-day rolling avg')\n"
         "ax.legend(); ax.set_title('Daily revenue with seasonality')\n"
         "plt.tight_layout(); plt.show()"),
    md("## Engagement funnel: impressions → clicks → conversions"),
    code("funnel = pd.Series({\n"
         "    'Impressions': df['impressions'].sum(),\n"
         "    'Clicks': df['clicks'].sum(),\n"
         "    'Conversions': df['conversions'].sum(),\n"
         "})\n"
         "print(funnel)\n"
         "print(f\"\\nOverall CTR: {funnel['Clicks']/funnel['Impressions']:.2%}\")\n"
         "print(f\"Overall CVR: {funnel['Conversions']/funnel['Clicks']:.2%}\")"),
])

# =====================================================================
# 03 — Revenue forecasting
# =====================================================================
nb3 = make_nb([
    md("# 03 · Revenue Forecasting (Model A)\n"
       "Trains and compares Linear Regression, Random Forest, and XGBoost on "
       "the full feature set. The official, full-scale (150k-row) results used "
       "in the dashboard are produced by `src/regression.py`; this notebook "
       "re-runs the same code path so the methodology is inspectable end to end."),
    code("import sys\n"
         "sys.path.append('../src')\n"
         "import pandas as pd, json\n"
         "from regression import train_and_evaluate\n"
         "df = pd.read_csv('../data/adcore_features.csv', parse_dates=['timestamp'])\n"
         "df.shape"),
    code("results, models = train_and_evaluate(df)\n"
         "pd.DataFrame(results['comparison']).T"),
    md("## Why the best model wins"),
    code("print('Best model:', results['best_model'])\n"
         "print()\n"
         "print(results['explanation'])"),
    md("## Visual comparison"),
    code("import matplotlib.pyplot as plt\n"
         "comp = pd.DataFrame(results['comparison']).T\n"
         "fig, ax = plt.subplots(figsize=(7,4))\n"
         "comp[['MAE','RMSE']].plot(kind='bar', ax=ax, color=['#33D6C0','#FFB800'])\n"
         "ax.set_title('Revenue forecasting: MAE / RMSE by model')\n"
         "plt.xticks(rotation=15); plt.tight_layout(); plt.show()"),
])

# =====================================================================
# 04 — CTR prediction
# =====================================================================
nb4 = make_nb([
    md("# 04 · CTR Prediction (Model B)\n"
       "Trains Logistic Regression and XGBoost classifiers to predict whether "
       "an impression results in a click, and checks probability calibration — "
       "not just discrimination (AUC) — since CTR scores feed directly into "
       "auction pricing."),
    code("import sys\n"
         "sys.path.append('../src')\n"
         "import pandas as pd\n"
         "from classification import train_and_evaluate\n"
         "df = pd.read_csv('../data/adcore_features.csv', parse_dates=['timestamp'])\n"
         "results, models = train_and_evaluate(df)\n"
         "pd.DataFrame(results['comparison']).T"),
    md("## Calibration curve for the best model\n"
       "A well-calibrated model's predicted probabilities should track the "
       "observed click rate closely — points near the diagonal are good."),
    code("import matplotlib.pyplot as plt\n"
         "cal = results['curves']['calibration']\n"
         "fig, ax = plt.subplots(figsize=(5.5,5.5))\n"
         "ax.plot(cal['predicted_probability'], cal['observed_click_rate'],\n"
         "        marker='o', color='#33D6C0', label=results['best_model'])\n"
         "ax.plot([0,1],[0,1], linestyle='--', color='#8C96AF', label='Perfect calibration')\n"
         "ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed click rate')\n"
         "ax.set_title('Calibration curve'); ax.legend()\n"
         "plt.tight_layout(); plt.show()"),
    md("## ROC curve"),
    code("fig, ax = plt.subplots(figsize=(5.5,5.5))\n"
         "for name, curve in results['curves']['roc'].items():\n"
         "    ax.plot(curve['fpr'], curve['tpr'], label=name)\n"
         "ax.plot([0,1],[0,1], linestyle='--', color='#8C96AF', label='Random')\n"
         "ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')\n"
         "ax.set_title('ROC curve'); ax.legend()\n"
         "plt.tight_layout(); plt.show()"),
])

# =====================================================================
# 05 — Audience clustering
# =====================================================================
nb5 = make_nb([
    md("# 05 · Audience Segmentation (Model C)\n"
       "K-Means clustering over behavioural features. K is chosen with the "
       "elbow method and confirmed with silhouette score, then each cluster is "
       "profiled in business language."),
    code("import sys\n"
         "sys.path.append('../src')\n"
         "import pandas as pd\n"
         "from clustering import run_clustering\n"
         "df = pd.read_csv('../data/adcore_features.csv', parse_dates=['timestamp'])\n"
         "results = run_clustering(df, k=4)\n"
         "print('K selected:', results['k_selected'])\n"
         "print('Silhouette score:', results['final_silhouette_score'])"),
    md("## Elbow method: inertia & silhouette vs K"),
    code("import matplotlib.pyplot as plt\n"
         "search = results['elbow_silhouette_search']\n"
         "fig, ax1 = plt.subplots(figsize=(7,4))\n"
         "ax1.plot(search['k_values'], search['inertia'], marker='o', color='#FFB800')\n"
         "ax1.set_xlabel('K'); ax1.set_ylabel('Inertia', color='#FFB800')\n"
         "ax2 = ax1.twinx()\n"
         "ax2.plot(search['k_values'], search['silhouette'], marker='s', color='#33D6C0')\n"
         "ax2.set_ylabel('Silhouette score', color='#33D6C0')\n"
         "ax1.set_title('Choosing K'); plt.tight_layout(); plt.show()"),
    md("## Cluster profiles (business language)"),
    code("pd.DataFrame(results['cluster_profiles']).set_index('cluster')"),
    md("## Segment scatter: session duration vs. revenue"),
    code("import pandas as pd\n"
         "scatter = pd.DataFrame(results['scatter_sample'])\n"
         "fig, ax = plt.subplots(figsize=(7,5))\n"
         "colors = ['#FFB800','#33D6C0','#FF6B6B','#9C8CFF']\n"
         "for c, sub in scatter.groupby('cluster'):\n"
         "    ax.scatter(sub['session_duration'], sub['revenue'], s=8,\n"
         "               color=colors[c % len(colors)], label=f'Cluster {c}', alpha=0.6)\n"
         "ax.set_xlabel('Session duration (min)'); ax.set_ylabel('Revenue')\n"
         "ax.legend(); ax.set_title('Audience segments')\n"
         "plt.tight_layout(); plt.show()"),
])

notebooks = {
    "01_data_generation.ipynb": nb1,
    "02_eda.ipynb": nb2,
    "03_revenue_forecasting.ipynb": nb3,
    "04_ctr_prediction.ipynb": nb4,
    "05_audience_clustering.ipynb": nb5,
}

for name, nb in notebooks.items():
    path = NB_DIR / name
    nbf.write(nb, path)
    print("wrote", path)
