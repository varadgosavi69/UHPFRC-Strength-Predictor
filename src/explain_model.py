from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from data_utils import MODEL_DIR, load_clean_split_data


import matplotlib.pyplot as plt  # noqa: E402


BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
DOCS_DIR = Path("docs")
SHAP_BAR_PATH = DOCS_DIR / "shap_summary_bar.png"
SHAP_BEESWARM_PATH = DOCS_DIR / "shap_summary_beeswarm.png"
BACKGROUND_SAMPLE_SIZE = 100


def load_best_model_artifact() -> dict:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_MODEL_PATH}. Run src/train_advanced.py first."
        )
    return joblib.load(BEST_MODEL_PATH)


@lru_cache(maxsize=1)
def get_model_context() -> dict:
    artifact = load_best_model_artifact()
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    x_train, x_test, _, _, feature_columns = load_clean_split_data()
    transformed_feature_names, original_feature_map = get_transformed_feature_metadata(
        preprocessor
    )

    return {
        "artifact": artifact,
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "x_train": x_train,
        "x_test": x_test,
        "feature_columns": feature_columns,
        "transformed_feature_names": transformed_feature_names,
        "original_feature_map": original_feature_map,
    }


def get_transformed_feature_metadata(preprocessor) -> tuple[list[str], list[str]]:
    feature_names = []
    original_feature_map = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or len(columns) == 0:
            continue

        if transformer == "passthrough":
            names = list(columns)
            originals = list(columns)
        else:
            names = list(transformer.get_feature_names_out(columns))
            originals = infer_original_features(transformer, columns, names)

        feature_names.extend(f"{transformer_name}__{name}" for name in names)
        original_feature_map.extend(originals)

    return feature_names, original_feature_map


def infer_original_features(transformer, columns, transformed_names: list[str]) -> list[str]:
    columns = list(columns)
    if "encoder" not in getattr(transformer, "named_steps", {}):
        return columns

    originals = []
    sorted_columns = sorted(columns, key=len, reverse=True)
    for transformed_name in transformed_names:
        original = next(
            (
                column
                for column in sorted_columns
                if transformed_name == column or transformed_name.startswith(f"{column}_")
            ),
            transformed_name,
        )
        originals.append(original)
    return originals


def to_transformed_frame(preprocessor, features: pd.DataFrame) -> pd.DataFrame:
    context = get_model_context()
    transformed = preprocessor.transform(features)
    return pd.DataFrame(
        transformed,
        columns=context["transformed_feature_names"],
        index=features.index,
    )


def build_explainer(estimator, background: pd.DataFrame):
    if isinstance(estimator, (RandomForestRegressor, XGBRegressor)):
        return shap.TreeExplainer(estimator)

    if isinstance(estimator, LinearRegression):
        return shap.LinearExplainer(estimator, background)

    if isinstance(estimator, MLPRegressor):
        return shap.KernelExplainer(estimator.predict, background)

    return shap.Explainer(estimator.predict, background)


def compute_shap_values(explainer, features: pd.DataFrame, estimator):
    if isinstance(estimator, MLPRegressor):
        shap_values = explainer.shap_values(features, silent=True)
    else:
        shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    return shap_values


def aggregate_contributions_by_input_feature(
    shap_values,
    original_feature_map: list[str],
) -> dict[str, float]:
    contributions = {}
    for original_feature, contribution in zip(original_feature_map, shap_values):
        contributions[original_feature] = (
            contributions.get(original_feature, 0.0) + float(contribution)
        )
    return contributions


def explain_single_prediction(input_dict: dict) -> dict:
    context = get_model_context()
    feature_columns = context["feature_columns"]
    missing_features = sorted(set(feature_columns) - set(input_dict))
    if missing_features:
        raise ValueError(f"Missing required input features: {missing_features}")

    input_frame = pd.DataFrame(
        [{column: input_dict[column] for column in feature_columns}]
    )
    transformed_input = to_transformed_frame(context["preprocessor"], input_frame)
    transformed_background = to_transformed_frame(
        context["preprocessor"],
        context["x_train"].sample(
            min(BACKGROUND_SAMPLE_SIZE, len(context["x_train"])),
            random_state=42,
        ),
    )

    explainer = build_explainer(context["estimator"], transformed_background)
    shap_values = compute_shap_values(
        explainer,
        transformed_input,
        context["estimator"],
    )[0]

    return {
        "predicted_compressive_strength_MPa": float(
            context["pipeline"].predict(input_frame)[0]
        ),
        "shap_contributions": aggregate_contributions_by_input_feature(
            shap_values,
            context["original_feature_map"],
        ),
    }


def save_summary_plots(shap_values, transformed_x_test: pd.DataFrame) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values,
        transformed_x_test,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, transformed_x_test, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def print_top_features(shap_values, transformed_feature_names: list[str]) -> None:
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=transformed_feature_names,
    )

    print("\nTop 5 features by mean absolute SHAP value")
    top_features = mean_abs_shap.sort_values(ascending=False).head(5)
    for rank, (feature, value) in enumerate(top_features.items(), start=1):
        print(f"{rank}. {feature}: {value:.4f}")


def main() -> None:
    context = get_model_context()
    transformed_x_train = to_transformed_frame(
        context["preprocessor"],
        context["x_train"].sample(
            min(BACKGROUND_SAMPLE_SIZE, len(context["x_train"])),
            random_state=42,
        ),
    )
    transformed_x_test = to_transformed_frame(context["preprocessor"], context["x_test"])

    explainer = build_explainer(context["estimator"], transformed_x_train)
    shap_values = compute_shap_values(
        explainer,
        transformed_x_test,
        context["estimator"],
    )

    save_summary_plots(shap_values, transformed_x_test)
    print_top_features(shap_values, context["transformed_feature_names"])


if __name__ == "__main__":
    main()
