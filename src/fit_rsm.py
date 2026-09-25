"""
Fit RSM quadratic model on the validated dataset.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from rsm_model import fit_quadratic_response_surface  # noqa: E402

DATA_FILE       = ROOT / "data" / "raw" / "master_rac_fiber_dataset.csv"
RSM_MODEL_OUT   = ROOT / "models" / "rsm_model.pkl"
TARGET          = "compressive_strength_MPa"
SOURCE_COL      = "source_paper"
DROP_META       = ["mix_id", "data_notes", SOURCE_COL]
EXCLUDE_PATTERNS = ["RSM-PREDICTED", "NOT an experimental"]

# RSM input factors (numeric mix-design parameters)
RSM_FACTORS = [
    "rca_replacement_percent",
    "pp_fiber_percent",
    "fly_ash",
    "silica_fume",
    "ggbs",
    "water_cement_ratio",
    "curing_age_days",
]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    
    if "data_notes" in df.columns:
        excl = df["data_notes"].fillna("").str.contains(
            "|".join(EXCLUDE_PATTERNS), case=False, na=False
        )
        df = df[~excl].copy()
    
    df = df.dropna(subset=[TARGET])
    
    if "rca_replacement_percent" in df.columns:
        df["rca_replacement_percent"] = (
            df["rca_replacement_percent"]
              .astype(str)
              .str.replace("~", "", regex=False)
              .str.strip()
        )
        df["rca_replacement_percent"] = pd.to_numeric(
            df["rca_replacement_percent"], errors="coerce"
        )
    
    for col in df.columns:
        if col in [TARGET, SOURCE_COL] + DROP_META:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].astype(object).fillna("unspecified")
    
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        if col in [TARGET, SOURCE_COL] + DROP_META:
            continue
        df[col] = df[col].astype(str)
    
    return df


def main():
    print("=" * 70)
    print("  RSM Quadratic Model Fitting")
    print("=" * 70)
    
    df = load_and_clean()
    print(f"\nDataset: {len(df)} rows")
    
    # Select only RSM factors
    X = df[RSM_FACTORS].copy()
    y = df[TARGET]
    
    print(f"RSM factors: {', '.join(RSM_FACTORS)}")
    print(f"\nFitting quadratic response surface (degree=2)...")
    
    # Fit the model
    rsm_pipeline = fit_quadratic_response_surface(X, y, degree=2)
    
    # Get predictions and R²
    y_pred = rsm_pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    print(f"\n✓ Fitted")
    print(f"  R² (training set) = {r2:.4f}")
    print(f"  RMSE (training set) = {rmse:.4f} MPa")
    
    # Extract coefficients and feature names
    linear_model = rsm_pipeline.named_steps["model"]
    coefficients = linear_model.coef_
    intercept = linear_model.intercept_
    
    # Get transformed feature names after preprocessing and polynomial expansion
    from rsm_model import get_response_surface_terms
    term_names = get_response_surface_terms(rsm_pipeline)
    
    print(f"\n{'─' * 70}")
    print("Quadratic Equation")
    print(f"{'─' * 70}")
    print(f"\nCS = {intercept:.4f}")
    
    # Group terms by type
    linear_terms = []
    interaction_terms = []
    quadratic_terms = []
    
    for name, coef in zip(term_names, coefficients):
        if "^2" in name:
            quadratic_terms.append((name, coef))
        elif " " in name:  # interaction (e.g., "x0 x1")
            interaction_terms.append((name, coef))
        else:
            linear_terms.append((name, coef))
    
    if linear_terms:
        print("\n  Linear terms:")
        for name, coef in linear_terms:
            sign = "+" if coef >= 0 else ""
            print(f"    {sign}{coef:+.4f} × {name}")
    
    if interaction_terms:
        print("\n  Interaction terms:")
        for name, coef in interaction_terms:
            sign = "+" if coef >= 0 else ""
            print(f"    {sign}{coef:+.4f} × {name}")
    
    if quadratic_terms:
        print("\n  Quadratic terms:")
        for name, coef in quadratic_terms:
            sign = "+" if coef >= 0 else ""
            print(f"    {sign}{coef:+.4f} × {name}")
    
    print(f"\n{'─' * 70}")
    print(f"Total terms: {len(term_names)} (excluding intercept)")
    print(f"{'─' * 70}")
    
    # Save the fitted model
    artifact = {
        "pipeline": rsm_pipeline,
        "factors": RSM_FACTORS,
        "target": TARGET,
        "r2_training": r2,
        "rmse_training": rmse,
        "n_training": len(df),
        "degree": 2,
    }
    
    joblib.dump(artifact, RSM_MODEL_OUT)
    print(f"\n✓ Saved: {RSM_MODEL_OUT.relative_to(ROOT)}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()