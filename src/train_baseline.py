from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/uhpfrc_mix_design_master.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "baseline_model.pkl"
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


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    return {
        "R2": r2_score(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MAE": mean_absolute_error(y_test, predictions),
    }


def main() -> None:
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

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
    }

    results = []
    fitted_models = {}
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        fitted_models[model_name] = pipeline

        print(f"\n{model_name}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")

        results.append({"Model": model_name, **metrics})

    results_table = pd.DataFrame(results).sort_values(
        by=["R2", "RMSE"],
        ascending=[False, True],
    )
    best_model_name = results_table.iloc[0]["Model"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": fitted_models[best_model_name],
            "metrics": results_table.to_dict(orient="records"),
            "target_column": TARGET_COLUMN,
            "feature_columns": list(x.columns),
        },
        MODEL_PATH,
    )

    print("\nBaseline Model Comparison")
    print(results_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved best model ({best_model_name}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
