from shutil import copyfile

import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from data_utils import (
    MODEL_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_preprocessor,
    evaluate_model,
    load_clean_split_data,
    load_metrics,
    make_results_table,
    print_model_metrics,
    save_metrics,
)


ADVANCED_MODEL_PATH = MODEL_DIR / "advanced_model.pkl"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"


def load_baseline_model_name() -> str | None:
    if not BASELINE_MODEL_PATH.exists():
        return None

    baseline_artifact = joblib.load(BASELINE_MODEL_PATH)
    return baseline_artifact.get("model_name")


def save_best_model(
    best_model_name: str,
    fitted_models: dict[str, Pipeline],
    feature_columns: list[str],
    all_metrics: list[dict],
) -> None:
    if best_model_name in fitted_models:
        joblib.dump(
            {
                "model_name": best_model_name,
                "pipeline": fitted_models[best_model_name],
                "metrics": all_metrics,
                "target_column": TARGET_COLUMN,
                "feature_columns": feature_columns,
            },
            BEST_MODEL_PATH,
        )
        return

    baseline_model_name = load_baseline_model_name()
    if best_model_name == baseline_model_name:
        copyfile(BASELINE_MODEL_PATH, BEST_MODEL_PATH)
        return

    raise ValueError(
        f"Best model is {best_model_name}, but no fitted artifact is available. "
        "Run src/train_baseline.py before src/train_advanced.py."
    )


def main() -> None:
    x_train, x_test, y_train, y_test, feature_columns = load_clean_split_data()

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "ANN": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
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

        print_model_metrics(model_name, metrics)
        results.append({"Model": model_name, **metrics})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = save_metrics(results)
    combined_table = make_results_table(load_metrics())
    best_model_name = combined_table.iloc[0]["Model"]

    joblib.dump(
        {
            "models": fitted_models,
            "metrics": results,
            "target_column": TARGET_COLUMN,
            "feature_columns": feature_columns,
        },
        ADVANCED_MODEL_PATH,
    )
    save_best_model(
        best_model_name,
        fitted_models,
        feature_columns,
        all_metrics,
    )

    print("\nCombined Model Comparison")
    print(combined_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved best model ({best_model_name}) to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
