import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from data_utils import (
    MODEL_DIR,
    TARGET_COLUMN,
    build_preprocessor,
    evaluate_model,
    load_clean_split_data,
    make_results_table,
    print_model_metrics,
    save_metrics,
)


MODEL_PATH = MODEL_DIR / "baseline_model.pkl"


def main() -> None:
    x_train, x_test, y_train, y_test, feature_columns = load_clean_split_data()

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

        print_model_metrics(model_name, metrics)

        results.append({"Model": model_name, **metrics})

    all_metrics = save_metrics(results)
    results_table = make_results_table(results)
    best_model_name = results_table.iloc[0]["Model"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": fitted_models[best_model_name],
            "metrics": results_table.to_dict(orient="records"),
            "target_column": TARGET_COLUMN,
            "feature_columns": feature_columns,
        },
        MODEL_PATH,
    )

    print("\nBaseline Model Comparison")
    print(results_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAll Saved Model Metrics")
    print(
        make_results_table(all_metrics).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved best model ({best_model_name}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
