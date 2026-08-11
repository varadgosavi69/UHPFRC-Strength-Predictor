import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/uhpfrc_mix_design_master.csv")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET_COLUMN = "compressive_strength_MPa"
KNOWN_CATEGORICAL_COLUMNS = ["fiber_type", "specimen_type", "scm_type"]
RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    known_categorical = [
        column for column in KNOWN_CATEGORICAL_COLUMNS if column in features.columns
    ]
    inferred_categorical = [
        column
        for column in features.select_dtypes(include=["object", "category", "bool"]).columns
        if column not in known_categorical
    ]
    categorical_features = known_categorical + inferred_categorical
    numeric_features = [
        column for column in features.columns if column not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def load_clean_split_data():
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    data = data.dropna(subset=[TARGET_COLUMN])
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    return x_train, x_test, y_train, y_test, list(x.columns)


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    return {
        "R2": r2_score(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MAE": mean_absolute_error(y_test, predictions),
    }


def load_metrics() -> list[dict]:
    if not METRICS_PATH.exists():
        return []

    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    if not isinstance(metrics, list):
        raise ValueError(f"Expected {METRICS_PATH} to contain a list of metric records")
    return metrics


def save_metrics(new_metrics: list[dict]) -> list[dict]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics() + new_metrics
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    return metrics


def latest_metrics_by_model(results: list[dict]) -> list[dict]:
    metrics_by_model = {}
    for record in results:
        metrics_by_model[record["Model"]] = record
    return list(metrics_by_model.values())


def make_results_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(latest_metrics_by_model(results)).sort_values(
        by=["R2", "RMSE"],
        ascending=[False, True],
    )


def print_model_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{model_name}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
