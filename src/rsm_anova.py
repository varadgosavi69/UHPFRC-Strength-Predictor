"""
ANOVA significance analysis for the fitted RSM model.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

DATA_FILE       = ROOT / "data" / "raw" / "master_rac_fiber_dataset.csv"
RSM_MODEL_IN    = ROOT / "models" / "rsm_model.pkl"
ANOVA_OUT       = ROOT / "models" / "rsm_anova.json"
TARGET          = "compressive_strength_MPa"
SOURCE_COL      = "source_paper"
DROP_META       = ["mix_id", "data_notes", SOURCE_COL]
EXCLUDE_PATTERNS = ["RSM-PREDICTED", "NOT an experimental"]


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


def compute_anova(X, y, pipeline):
    """
    Compute ANOVA F-statistics and p-values for RSM model.
    
    For a linear model with polynomial features, we compute:
    - Overall model F-test (model vs. intercept-only)
    - Individual term t-tests and p-values
    """
    # Get transformed features
    X_transformed = pipeline.named_steps["preprocessor"].transform(X)
    X_poly = pipeline.named_steps["quadratic_terms"].transform(X_transformed)
    
    # Get model parameters
    linear_model = pipeline.named_steps["model"]
    y_pred = pipeline.predict(X)
    residuals = y - y_pred
    
    n = len(y)
    k = X_poly.shape[1]  # number of predictors (excluding intercept)
    df_model = k
    df_resid = n - k - 1
    
    # Sum of squares
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_resid = np.sum(residuals ** 2)
    ss_model = ss_total - ss_resid
    
    # Mean squares
    ms_model = ss_model / df_model
    ms_resid = ss_resid / df_resid
    
    # Overall F-statistic
    f_statistic = ms_model / ms_resid
    p_value_overall = 1 - stats.f.cdf(f_statistic, df_model, df_resid)
    
    # R²
    r_squared = ss_model / ss_total
    
    # Standard errors for coefficients
    # Var(beta) = sigma^2 * (X'X)^-1
    XtX = X_poly.T @ X_poly
    try:
        XtX_inv = np.linalg.inv(XtX)
        se_coef = np.sqrt(ms_resid * np.diag(XtX_inv))
        
        # t-statistics and p-values for each coefficient
        t_stats = linear_model.coef_ / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
    except np.linalg.LinAlgError:
        # Singular matrix - use pseudoinverse
        XtX_pinv = np.linalg.pinv(XtX)
        se_coef = np.sqrt(ms_resid * np.diag(XtX_pinv))
        t_stats = linear_model.coef_ / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
    
    return {
        "overall": {
            "f_statistic": float(f_statistic),
            "p_value": float(p_value_overall),
            "df_model": int(df_model),
            "df_resid": int(df_resid),
            "r_squared": float(r_squared),
            "ss_model": float(ss_model),
            "ss_resid": float(ss_resid),
            "ss_total": float(ss_total),
        },
        "coefficients": {
            "values": linear_model.coef_.tolist(),
            "std_errors": se_coef.tolist(),
            "t_statistics": t_stats.tolist(),
            "p_values": p_values.tolist(),
        }
    }


def classify_term(name: str) -> str:
    """Classify a term as linear, interaction, or quadratic."""
    if "^2" in name:
        return "quadratic"
    elif " " in name:  # space indicates interaction
        return "interaction"
    else:
        return "linear"


def main():
    print("=" * 70)
    print("  RSM ANOVA Significance Analysis")
    print("=" * 70)
    
    # Load model
    rsm_artifact = joblib.load(RSM_MODEL_IN)
    pipeline = rsm_artifact["pipeline"]
    factors = rsm_artifact["factors"]
    
    print(f"\nLoaded: {RSM_MODEL_IN.relative_to(ROOT)}")
    print(f"Factors: {', '.join(factors)}")
    
    # Load data
    df = load_and_clean()
    X = df[factors].copy()
    y = df[TARGET]
    
    print(f"Dataset: {len(df)} rows\n")
    
    # Compute ANOVA
    print("Computing ANOVA...")
    anova = compute_anova(X, y, pipeline)
    
    # Get term names
    from rsm_model import get_response_surface_terms
    term_names = get_response_surface_terms(pipeline)
    
    # Overall model test
    overall = anova["overall"]
    print(f"\n{'─' * 70}")
    print("Overall Model F-Test")
    print(f"{'─' * 70}")
    print(f"  F-statistic = {overall['f_statistic']:.4f}")
    print(f"  p-value     = {overall['p_value']:.2e}")
    print(f"  df          = ({overall['df_model']}, {overall['df_resid']})")
    print(f"  R²          = {overall['r_squared']:.4f}")
    
    sig = "SIGNIFICANT" if overall['p_value'] < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  Overall model: {sig} (p < 0.05)")
    
    # Individual terms
    coefs = anova["coefficients"]
    p_values = coefs["p_values"]
    
    # Group by type
    terms_by_type = {"linear": [], "interaction": [], "quadratic": []}
    
    for i, name in enumerate(term_names):
        term_type = classify_term(name)
        terms_by_type[term_type].append({
            "name": name,
            "coefficient": coefs["values"][i],
            "std_error": coefs["std_errors"][i],
            "t_statistic": coefs["t_statistics"][i],
            "p_value": p_values[i],
            "significant": p_values[i] < 0.05,
        })
    
    # Print by type
    for term_type in ["linear", "interaction", "quadratic"]:
        terms = terms_by_type[term_type]
        if not terms:
            continue
        
        print(f"\n{'─' * 70}")
        print(f"{term_type.capitalize()} Terms")
        print(f"{'─' * 70}")
        print(f"{'Term':<50} {'Coef':>10} {'p-value':>10} {'Sig':>5}")
        print(f"{'─' * 70}")
        
        for term in terms:
            sig = "✓" if term["significant"] else ""
            p_str = f"{term['p_value']:.2e}" if term['p_value'] >= 0.0001 else "<1e-4"
            print(f"{term['name']:<50} {term['coefficient']:>10.4f} {p_str:>10} {sig:>5}")
    
    # Summary statistics
    total_terms = len(term_names)
    significant_terms = sum(1 for p in p_values if p < 0.05)
    
    print(f"\n{'─' * 70}")
    print("Summary")
    print(f"{'─' * 70}")
    print(f"  Total terms: {total_terms}")
    print(f"  Significant (p < 0.05): {significant_terms} ({significant_terms/total_terms*100:.1f}%)")
    print(f"  Not significant: {total_terms - significant_terms} ({(total_terms-significant_terms)/total_terms*100:.1f}%)")
    
    # Count by type
    for term_type in ["linear", "interaction", "quadratic"]:
        terms = terms_by_type[term_type]
        sig_count = sum(1 for t in terms if t["significant"])
        print(f"    {term_type.capitalize()}: {sig_count}/{len(terms)} significant")
    
    # Save ANOVA results
    import json
    
    output = {
        "overall_model": overall,
        "terms": {
            "linear": terms_by_type["linear"],
            "interaction": terms_by_type["interaction"],
            "quadratic": terms_by_type["quadratic"],
        },
        "summary": {
            "total_terms": total_terms,
            "significant_terms": significant_terms,
            "nonsignificant_terms": total_terms - significant_terms,
            "alpha": 0.05,
        }
    }
    
    with open(ANOVA_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved: {ANOVA_OUT.relative_to(ROOT)}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()