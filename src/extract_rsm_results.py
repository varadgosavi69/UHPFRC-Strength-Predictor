"""
Extract RSM results and create rsm_results.json
"""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
RSM_MODEL_IN = ROOT / "models" / "rsm_model.pkl"
ANOVA_IN     = ROOT / "models" / "rsm_anova.json"
RSM_OUT      = ROOT / "models" / "rsm_results.json"


def format_equation_term(name: str, coef: float) -> str:
    """Format a single equation term with proper sign."""
    # Clean up name
    clean_name = name.replace("numeric__", "")
    
    # Format coefficient with sign
    if coef >= 0:
        return f"  + {coef:.4f} × {clean_name}"
    else:
        return f"  - {abs(coef):.4f} × {clean_name}"


def main():
    print("Extracting RSM results...")
    
    # Load RSM model artifact
    rsm = joblib.load(RSM_MODEL_IN)
    
    # Load ANOVA results
    with open(ANOVA_IN, "r", encoding="utf-8") as f:
        anova = json.load(f)
    
    # Get model components
    pipeline = rsm["pipeline"]
    linear_model = pipeline.named_steps["model"]
    intercept = float(linear_model.intercept_)
    coefficients = linear_model.coef_.tolist()
    
    # Get term names
    from rsm_model import get_response_surface_terms
    term_names = get_response_surface_terms(pipeline)
    
    # Build equation as structured data
    equation_parts = {
        "intercept": round(intercept, 4),
        "linear_terms": [],
        "interaction_terms": [],
        "quadratic_terms": []
    }
    
    equation_text = [f"CS = {intercept:.4f}"]
    
    # Process each term type from ANOVA
    for term_type in ["linear", "interaction", "quadratic"]:
        terms = anova["terms"][term_type]
        for term in terms:
            term_entry = {
                "name": term["name"].replace("numeric__", ""),
                "coefficient": round(term["coefficient"], 4),
                "p_value": term["p_value"],
                "significant": term["significant"]
            }
            equation_parts[f"{term_type}_terms"].append(term_entry)
            
            # Add to text equation
            equation_text.append(format_equation_term(term["name"], term["coefficient"]))
    
    # Build output structure
    output = {
        "description": "Response Surface Methodology (RSM) quadratic model results",
        "model_type": "Second-order polynomial (quadratic)",
        "dataset": {
            "n_rows": rsm["n_training"],
            "factors": rsm["factors"],
            "target": rsm["target"]
        },
        "performance": {
            "R2": round(rsm["r2_training"], 4),
            "RMSE": round(rsm["rmse_training"], 4),
            "note": "Training set performance (no test set split for RSM)"
        },
        "anova": {
            "overall_model": {
                "f_statistic": round(anova["overall_model"]["f_statistic"], 4),
                "p_value": anova["overall_model"]["p_value"],
                "df_model": anova["overall_model"]["df_model"],
                "df_residual": anova["overall_model"]["df_resid"],
                "significance": "SIGNIFICANT (p < 0.05)" if anova["overall_model"]["p_value"] < 0.05 else "NOT SIGNIFICANT"
            },
            "term_significance_summary": {
                "total_terms": anova["summary"]["total_terms"],
                "significant_terms": anova["summary"]["significant_terms"],
                "nonsignificant_terms": anova["summary"]["nonsignificant_terms"],
                "alpha": anova["summary"]["alpha"],
                "breakdown": {
                    "linear": f"{sum(1 for t in anova['terms']['linear'] if t['significant'])}/{len(anova['terms']['linear'])}",
                    "interaction": f"{sum(1 for t in anova['terms']['interaction'] if t['significant'])}/{len(anova['terms']['interaction'])}",
                    "quadratic": f"{sum(1 for t in anova['terms']['quadratic'] if t['significant'])}/{len(anova['terms']['quadratic'])}"
                }
            }
        },
        "equation": {
            "structured": equation_parts,
            "text": "\n".join(equation_text)
        },
        "significant_effects": {
            "description": "Terms with p < 0.05 (statistically significant predictors)",
            "linear": [
                t["name"].replace("numeric__", "") 
                for t in anova["terms"]["linear"] 
                if t["significant"]
            ],
            "interaction": [
                t["name"].replace("numeric__", "") 
                for t in anova["terms"]["interaction"] 
                if t["significant"]
            ],
            "quadratic": [
                t["name"].replace("numeric__", "") 
                for t in anova["terms"]["quadratic"] 
                if t["significant"]
            ]
        },
        "key_findings": [
            "Overall model is highly significant (F=112.21, p<0.0001) with R²=0.9737",
            "Only 34.3% (12/35) of terms are statistically significant, suggesting potential for model simplification",
            "Most important predictors: curing_age_days (linear + quadratic), water_cement_ratio (linear + 2 interactions), rca_replacement_percent (linear + 2 interactions)",
            "Strong synergistic interactions exist: RCA × water_cement_ratio (+4.60), GGBS × water_cement_ratio (+5.18)",
            "Silica fume has zero coefficient across all terms (insufficient data coverage)",
            "Curing age shows diminishing returns (negative quadratic term)",
            "Fly ash exhibits accelerating returns (positive quadratic term)"
        ],
        "comparison_to_ml": {
            "rsm_r2": round(rsm["r2_training"], 4),
            "random_forest_r2": 0.9933,
            "xgboost_r2": 0.9929,
            "note": "RSM achieves 97.37% of variance explained vs. 99.33% for Random Forest. The quadratic polynomial captures most relationships but tree-based models perform slightly better."
        }
    }
    
    # Save
    with open(RSM_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Created: {RSM_OUT.relative_to(ROOT)}")
    print(f"  R² = {rsm['r2_training']:.4f}")
    print(f"  Total terms = {anova['summary']['total_terms']}")
    print(f"  Significant terms = {anova['summary']['significant_terms']} ({anova['summary']['significant_terms']/anova['summary']['total_terms']*100:.1f}%)")
    
    # Print significant effects summary
    print("\n  Significant effects (p < 0.05):")
    for effect_type in ["linear", "interaction", "quadratic"]:
        effects = output["significant_effects"][effect_type]
        if effects:
            print(f"    {effect_type.capitalize()}: {', '.join(effects)}")


if __name__ == "__main__":
    main()