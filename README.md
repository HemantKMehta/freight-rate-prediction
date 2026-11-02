# Freight Rate Prediction Challenge

## Overview

This project predicts freight rates using historical shipment data. It uses a scikit-learn `HistGradientBoostingRegressor` model to train on labelled data, evaluate performance on a chronological holdout set, and generate predictions for the validation and December scenario datasets.

## Project Structure

```text
freight-rate-prediction/
│
├── freight-rate-complete.py
├── requirements.txt
├── README.md
└── validation_predictions.csv

```

## Requirements

- Python 3.10 or later
- pip
- Dependencies listed in `requirements.txt`

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd freight-rate-prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Linux or macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Input Data

Place the assessment input files inside the `data/` directory.

The  `freight-rate-complete.py` expects these exact filenames:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`


## Train, Validate and Generate Predictions

Run:

```bash
python freight-rate-complete.py --data-dir data --output-dir outputs
```

The freight-rate-complete:

1. Loads the historical labelled data.
2. Creates a chronological training and holdout split.
3. Preprocesses numerical and categorical features.
4. Trains a HistGradientBoostingRegressor model.
5. Evaluates performance using MAE, RMSE and R².
6. Retrains the final model using all labelled training rows.
7. Generates predictions for all validation loads.
8. Generates predictions for the December scenario.
9. Saves the trained model and evaluation metrics.

## Validation Strategy

A chronological holdout strategy is used to simulate predicting future freight rates from historical data.

- Training period: January 1 – August 31, 2025.
- Holdout period: September 1 – October 31, 2025.

The final model is subsequently retrained on all labelled training data before generating the requested predictions.

## Run the Assessment Scorer

The freight-rate-complete generates:

- `outputs/validation_predictions.csv`
- `outputs/december_chart_inputs.csv`

Copy the generated December predictions into the scorer's expected input location:

```bash
cp outputs/december_chart_inputs.csv data/december_chart_inputs.csv
```

Run the scorer:

```bash
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

The scorer validates the prediction files and generates:

```text
scorer_results/candidate_december.png
```

## Model Evaluation

The chronological holdout evaluation produced:

| Metric | Result |
|---|---:|
| MAE | $131.03 |
| RMSE | $635.72 |
| R² | 0.8265 |

These metrics describe performance on the historical holdout set, not the final hidden validation score.

## Notes

- Unknown categorical values are handled by the preprocessing.
- Missing numerical values are imputed using training-set medians.
- Non-positive shipment weights are treated as missing.
- Predictions are constrained to a minimum rate of $1.
- The December scenario contains limited market information, so its predictions should be interpreted with that limitation in mind.

## Deliverables

- `validation_predictions.csv` – predicted freight rates for validation loads.
- `validation_metrics.json` – holdout evaluation metrics.
