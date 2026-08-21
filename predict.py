"""Predict Mercedes-Benz test-bench time for new car configurations.

Loads the trained pipeline bundle, applies the same cleaning, label
encoding, and binary_sum feature used in training, then writes
predictions to a CSV.

Usage:
    uv run predict.py --input data/test.csv --output data/predictions.csv
"""

import argparse
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

BUNDLE_PATH = "models/mercedes_pipeline_bundle.pkl"
HIGH_Y_THRESHOLD = 130

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def clean_input(df, bundle):
    """Apply the training-time cleaning and feature steps to new data."""
    clean_df = df.drop(columns=bundle["constant_cols"], errors="ignore").copy()

    unseen_count = 0
    for col in bundle["categorical_cols"]:
        encoder = bundle["encoders"][col]
        known = set(encoder.classes_) - {"unseen"}
        unseen_mask = ~clean_df[col].isin(known)
        unseen_count += int(unseen_mask.sum())
        clean_df[col] = clean_df[col].where(~unseen_mask, "unseen")
        clean_df[col] = encoder.transform(clean_df[col])

    clean_df["binary_sum"] = clean_df[bundle["binary_cols"]].sum(axis=1)

    return clean_df, unseen_count


def predict(df, bundle):
    """Clean, encode, and predict on new car configurations."""
    clean_df, unseen_count = clean_input(df, bundle)

    missing_cols = set(bundle["feature_cols"]) - set(clean_df.columns)
    if missing_cols:
        raise ValueError(f"input is missing expected columns: {sorted(missing_cols)}")

    if unseen_count:
        logger.info("mapped %d category values to 'unseen'", unseen_count)

    y_pred = bundle["pipeline"].predict(clean_df[bundle["feature_cols"]])

    high_y_count = int((y_pred > HIGH_Y_THRESHOLD).sum())
    if high_y_count:
        logger.info(
            "%d predictions above %s (model's known low-accuracy range)",
            high_y_count,
            HIGH_Y_THRESHOLD,
        )

    return y_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of car configurations to predict")
    parser.add_argument("--output", required=True, help="path to write predictions CSV")
    parser.add_argument("--bundle", default=BUNDLE_PATH, help="path to the model bundle")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    input_df = pd.read_csv(args.input)

    y_pred = predict(input_df, bundle)

    output_df = pd.DataFrame({
        "ID": input_df["ID"],
        "y_pred": y_pred,
        "model_version": bundle["model_version"],
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    })
    output_df.to_csv(args.output, index=False)
    logger.info("wrote %d predictions to %s", len(output_df), args.output)


if __name__ == "__main__":
    main()
