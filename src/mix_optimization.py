"""Constraint and objective utilities for mix-design optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariableConstraint:
    """Bounds and optional categorical choices for one model input variable."""

    lower: float | None = None
    upper: float | None = None
    choices: tuple[str, ...] | None = None

    def validate(self, value: object) -> bool:
        if self.choices is not None:
            return value in self.choices
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        return (self.lower is None or value >= self.lower) and (
            self.upper is None or value <= self.upper
        )


DEFAULT_CONSTRAINTS: dict[str, VariableConstraint] = {
    "cement": VariableConstraint(200, 1500),
    "silica_fume": VariableConstraint(0, 400),
    "fly_ash": VariableConstraint(0, 500),
    "sand": VariableConstraint(0, 1800),
    "coarse_aggregate": VariableConstraint(0, 1800),
    "water": VariableConstraint(80, 400),
    "superplasticizer": VariableConstraint(0, 100),
    "fiber_type": VariableConstraint(choices=("polypropylene",)),
    "fiber_content_percent": VariableConstraint(0, 5),
    "water_binder_ratio": VariableConstraint(0.10, 0.60),
    "curing_age_days": VariableConstraint(1, 365),
    "curing_temp_celsius": VariableConstraint(0, 100),
    "specimen_type": VariableConstraint(choices=("cube", "cylinder", "prism")),
}


def validate_mix_design(
    mix_design: Mapping[str, object],
    constraints: Mapping[str, VariableConstraint] = DEFAULT_CONSTRAINTS,
) -> None:
    """Raise ValueError when a candidate violates a configured constraint."""
    missing = sorted(set(constraints) - set(mix_design))
    if missing:
        raise ValueError(f"Missing constrained variables: {missing}")

    violations = [
        name for name, constraint in constraints.items()
        if not constraint.validate(mix_design[name])
    ]
    if violations:
        raise ValueError(f"Mix design violates constraints: {violations}")


def predicted_strength_objective(
    mix_design: Mapping[str, object],
    model: Callable[[pd.DataFrame], Sequence[float]],
) -> float:
    """Return negative predicted strength so minimizers maximize compressive strength."""
    prediction = float(model(pd.DataFrame([dict(mix_design)]))[0])
    return -prediction


def select_best_candidate(
    candidates: pd.DataFrame,
    model: Callable[[pd.DataFrame], Sequence[float]],
    constraints: Mapping[str, VariableConstraint] = DEFAULT_CONSTRAINTS,
) -> tuple[dict[str, object], float]:
    """Select the highest-predicted feasible candidate from an input candidate table."""
    if candidates.empty:
        raise ValueError("At least one candidate mix design is required")

    feasible_candidates = []
    for record in candidates.to_dict(orient="records"):
        try:
            validate_mix_design(record, constraints)
        except ValueError:
            continue
        feasible_candidates.append(record)

    if not feasible_candidates:
        raise ValueError("No candidate mix design satisfies the configured constraints")

    feasible_frame = pd.DataFrame(feasible_candidates)
    predictions = np.asarray(model(feasible_frame), dtype=float)
    best_index = int(np.argmax(predictions))
    return feasible_candidates[best_index], float(predictions[best_index])
