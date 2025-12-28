# Titanic - Machine Learning from Disaster

This directory contains a small end-to-end pipeline for the Kaggle **Titanic: Machine Learning from Disaster** competition.

## Contents
- `titanic_ml.py` – main script: loads data, performs EDA, preprocesses features, evaluates models (logistic regression, random forest), trains the best model, and generates `submission.csv`.
- `test_titanic_ml.py` – unit tests for model-selection and training helpers (pytest).
- `data/` – input CSVs from the Kaggle competition (`train.csv`, `test.csv`, etc.).

## Setup
Install dependencies (example with `pip`):

```bash
pip install -r requirements.txt  # or install pandas, numpy, matplotlib, seaborn, scikit-learn, pytest
```

## Usage
Run the full pipeline (EDA + model selection + submission generation):

```bash
python titanic_ml.py
```

This will:
- Save EDA plots to `eda_plots.png`.
- Write predictions for the Kaggle test set to `submission.csv`.

To run tests:

```bash
pytest
```