# Mercedes Benz Greener Manufacturing

## Overview

Based on the [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing) Kaggle competition. The data (`train.csv`/`test.csv`) has one row per car configuration: 8 categorical features (`X0`-`X8`) and ~360 binary features describing installed options, plus `y`, the time (in seconds) a car spends on the test bench. Goal: predict `y` for new configurations to cut testing time and speed up the manufacturing line.

## What's in the Notebook

The notebook (`Mercedes_Benz_Greener_Manufacturing.ipynb`) covers:

- Cleaning: drop constant columns, drop one outlier row.
- Two encodings compared for the categorical columns: one-hot vs. label encoding, plus an engineered `binary_sum` feature (count of active binary flags per row).
- Model comparison (Linear Regression, Decision Tree, Random Forest, KNN, XGBoost) via `GridSearchCV`, with the label-encoded XGBoost pipeline as the best performer.
- Error analysis: residual plot, mean absolute error by `y` range. Error concentrates in `y > 130` (rare configurations, few training examples).
- Predictions on the competition test set, written to `data/submission.csv`.
- The trained pipeline, encoders, and column lists are bundled and saved to `models/mercedes_pipeline_bundle.pkl`.

`predict.py` wraps that bundle in a standalone script so predictions don't require rerunning the notebook: it loads the bundle, applies the same cleaning/encoding/`binary_sum` steps to new data, and writes predictions to CSV. See the file table below for usage.

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | Where your datasets go. |
| [**Models**](models/) | Where trained models are saved. |
| [**predict.py**](predict.py) | Loads the saved model bundle and predicts on new car configurations. Run with `uv run predict.py --input <csv> --output <csv>`. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

### Move into the Project Folder and Install Dependencies
This installs all dependencies and creates a virtual environment in (`.venv/`).

```bash
uv sync
```

### Open the Notebook
Launch VS Code in the project root folder:

```bash
code .
```

Then open `Mercedes_Benz_Greener_Manufacturing.ipynb` and select the Python environment created by `uv sync` as the kernel.