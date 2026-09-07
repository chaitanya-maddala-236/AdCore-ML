# Data folder

To keep this repository lightweight, only a **3,000-row sample** of each stage
is included here:

- `adcore_raw_sample.csv` — synthetic data straight out of the generator (with intentional flaws)
- `adcore_clean_sample.csv` — after the data-quality pipeline
- `adcore_features_sample.csv` — after feature engineering

The full results in the dashboard and `reports/results.md` were computed on
the **complete 150,000-row dataset**. To regenerate the full dataset and
re-run everything end to end:

```bash
pip install -r ../requirements.txt
python ../src/run_pipeline.py
```

This writes the full-size `adcore_raw.csv`, `adcore_clean.csv`, and
`adcore_features.csv` into this folder, retrains all models, and refreshes
`results/dashboard_data.json`.
