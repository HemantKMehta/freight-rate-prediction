from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", "year", "month", "day",
    "dayofweek", "dayofyear",
]
FEATURES = CATEGORICAL + NUMERIC


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only prediction-time inputs and create a few basic date features."""
    x = frame.drop(columns=["load_id", "posted_rate"], errors="ignore").copy()
    dates = pd.to_datetime(x.pop("date"), errors="coerce")
    x["year"] = dates.dt.year
    x["month"] = dates.dt.month
    x["day"] = dates.dt.day
    x["dayofweek"] = dates.dt.dayofweek
    x["dayofyear"] = dates.dt.dayofyear

    for col in CATEGORICAL:
        if col not in x:
            x[col] = "Unknown"
        x[col] = x[col].fillna("Unknown").astype(str)

    for col in NUMERIC:
        if col not in x:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if "weight" in x:
        x.loc[x["weight"] <= 0, "weight"] = np.nan
    return x[FEATURES]


def build_model() -> Pipeline:
    """A compact, reproducible scikit-learn histogram gradient boosting model."""
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train-test.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation-predictions-template.csv")
    december = pd.read_csv(data_dir / "december-chart-inputs.csv")
    train["date"] = pd.to_datetime(train["date"], errors="coerce")

    # Use the final two months as a forward-looking holdout.
    cutoff = pd.Timestamp("2025-09-01")
    dev = train.loc[train["date"] < cutoff].copy()
    holdout = train.loc[train["date"] >= cutoff].copy()

    validation_model = build_model()
    validation_model.fit(make_features(dev), dev["posted_rate"])
    holdout_pred = np.maximum(validation_model.predict(make_features(holdout)), 1.0)
    metrics = {
        "model": "HistGradientBoostingRegressor",
        "split": "Train: 2025-01-01 through 2025-08-31; holdout: 2025-09-01 through 2025-10-31",
        "train_rows": int(len(dev)),
        "holdout_rows": int(len(holdout)),
        "mae": float(mean_absolute_error(holdout["posted_rate"], holdout_pred)),
        "rmse": float(mean_squared_error(holdout["posted_rate"], holdout_pred) ** 0.5),
        "r2": float(r2_score(holdout["posted_rate"], holdout_pred)),
    }
    (out_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Refit on every labelled row before making the requested predictions.
    final_model = build_model()
    final_model.fit(make_features(train), train["posted_rate"])
    joblib.dump(final_model, out_dir / "freight_rate_model.joblib")

    if len(template) != len(validation):
        raise ValueError("Template row count does not match validation.csv")
    validation_pred = np.maximum(final_model.predict(make_features(validation)), 1.0)
    pred = template[["load_id"]].copy()
    pred["predicted_rate"] = validation_pred
    pred.to_csv(out_dir / "validation_predictions.csv", index=False)
    pred.to_csv("validation_predictions.csv", index=False)

    december["predicted_rate"] = np.maximum(final_model.predict(make_features(december)), 1.0)
    december.to_csv(out_dir / "december_chart_inputs.csv", index=False)
    december.to_csv("december_chart_inputs.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir / 'validation_predictions.csv'}")
    print(f"Wrote {out_dir / 'december_chart_inputs.csv'}")


if __name__ == "__main__":
    main()
