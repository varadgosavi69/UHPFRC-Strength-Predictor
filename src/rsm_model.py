"""Quadratic response-surface modelling utilities for mix-design data."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from data_utils import build_preprocessor


def build_quadratic_response_surface(
    features: pd.DataFrame,
    degree: int = 2,
) -> Pipeline:
    """Build an RSM pipeline for the same input columns used by the ML models."""
    if features.empty:
        raise ValueError("At least one feature row is required to build an RSM pipeline")
    if degree < 1:
        raise ValueError("degree must be at least 1")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("quadratic_terms", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )


def fit_quadratic_response_surface(
    features: pd.DataFrame,
    target: pd.Series,
    degree: int = 2,
) -> Pipeline:
    """Fit a quadratic response-surface model and return the fitted pipeline."""
    if len(features) != len(target):
        raise ValueError("features and target must contain the same number of rows")
    if len(features) == 0:
        raise ValueError("At least one training row is required to fit an RSM model")

    response_surface = build_quadratic_response_surface(features, degree=degree)
    response_surface.fit(features, target)
    return response_surface


def predict_response_surface(
    model: Pipeline,
    features: pd.DataFrame,
) -> pd.Series:
    """Predict compressive strength from a fitted response-surface pipeline."""
    predictions = model.predict(features)
    return pd.Series(predictions, index=features.index, name="predicted_compressive_strength_MPa")


def get_response_surface_terms(
    model: Pipeline,
) -> Sequence[str]:
    """Return transformed linear and interaction terms from a fitted RSM model."""
    polynomial_terms = model.named_steps["quadratic_terms"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return polynomial_terms.get_feature_names_out(feature_names)
