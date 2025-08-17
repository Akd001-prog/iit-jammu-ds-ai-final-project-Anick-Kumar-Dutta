# Predictive Maintenance for Machinery

This repo contains a **ready-to-run DS/AI project** for predictive maintenance using a small **synthetic sensor dataset**. It demonstrates the full workflow: EDA, feature engineering, model training, evaluation, interpretability, and inference artifact export.

## Contents
- `Predictive_Maintenance.ipynb` — main notebook with end-to-end pipeline
- `data/sample_sensor_data.csv` — synthetic dataset (hourly readings for 20 machines over 30 days)
- `models/` — folder where trained model + metadata will be saved
- `src/inference.py` — CLI for batch inference on a CSV
- `requirements.txt` — Python dependencies

## Quickstart
```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run the notebook in Jupyter / VSCode
jupyter notebook Predictive_Maintenance.ipynb

# Or run CLI inference on the provided data (after training once via the notebook)
python src/inference.py --model models/pd_maint_model.joblib --meta models/pd_maint_model.meta.json --input data/sample_sensor_data.csv --output predictions.csv
```

## What it shows
- Time-ordered **train/test split by machine** to reduce leakage
- Baselines: **Logistic Regression** and **Random Forest**
- **ROC-AUC/PR-AUC**, threshold selection by **F1**
- **Permutation importance** for interpretability
- Exports a **Joblib model** and **JSON metadata** (metrics + chosen threshold)

## Replace with your real data
- Put your CSV in `data/` with at least these columns:
  - `timestamp`, `machine_id` and several numeric sensor/operating columns
  - a binary label column `failure` (1=failed at this time, 0=normal)
- Update `DATA_PATH` and `feat_cols` in the notebook if your column names differ.

## Citation / Datasets
- This example uses synthetic data to be self-contained. For real experiments, consider NASA CMAPSS or other predictive maintenance datasets.
