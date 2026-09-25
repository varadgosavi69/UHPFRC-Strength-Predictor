# Model Training and Evaluation Results

**Dataset:** `master_rac_fiber_dataset.csv`  
**Training rows:** 142 (after excluding 3 RSM-predicted rows)  
**Source papers:** 3 (Jaglan_2025: 100 rows, Afroughsabet_2015: 36 rows, Kakooei_2011: 6 rows)  
**Evaluation date:** 2026-08-25

---

## Standard 80/20 Random Split Results

All models were trained on 113 rows and tested on 29 rows using a standard random split.

| Model | R² | RMSE (MPa) | MAE (MPa) | Saved As |
|---|---:|---:|---:|---|
| **Random Forest** | **0.9933** | **1.74** | **1.36** | `best_model.pkl` ✓ |
| XGBoost | 0.9929 | 1.80 | 1.37 | `advanced_model.pkl` |
| ANN (MLP) | 0.9515 | 4.69 | 4.35 | — |
| Linear Regression | 0.9497 | 4.78 | 4.37 | `baseline_model.pkl` |

### Model Selection

**Random Forest** was selected as the best model and saved to `models/best_model.pkl` based on:
- Highest R² (0.9933) among all models
- Lowest RMSE (1.74 MPa) and MAE (1.36 MPa)
- Strong predictive performance on the held-out test set

XGBoost performed nearly identically (R²=0.9929, RMSE=1.80 MPa) and was saved as `advanced_model.pkl` for comparison. Linear Regression achieved R²=0.9497 and was saved as `baseline_model.pkl`.

---

## Leave-One-Paper-Out (LOPO) Cross-Validation

LOPO evaluates generalization to completely unseen source papers by holding out one paper at a time for testing.

### Results Summary

| Held-out Paper | Test Rows | Best Model | R² | RMSE (MPa) |
|---|---:|---|---:|---:|
| Afroughsabet_2015 | 36 | XGBoost | −20.55 | 48.2 |
| Jaglan_2025 | 100 | ANN | −0.93 | 12.6 |
| Kakooei_2011 | 6 | XGBoost ⚠ | −3.52 | 15.5 |

⚠ **Kakooei_2011 results are unreliable** due to only 6 test rows (see `models/generalization_metrics.json`).

### Domain Shift Finding

**All models exhibit poor generalization across papers (negative R²).** This occurs because the three source papers represent fundamentally different concrete systems: Jaglan_2025 covers normal-strength concrete with recycled coarse aggregate (RCA) at 0–100% replacement; Afroughsabet_2015 focuses on high-strength concrete with fly ash, silica fume, and hybrid steel+polypropylene fibers; Kakooei_2011 examines low-strength concrete with coral aggregate. When a model is trained on two of these systems and tested on the third, it cannot reliably extrapolate due to the non-overlapping mix design spaces and strength regimes.

**Interpretation:** The standard 80/20 split R² (~0.99) measures **interpolation** within the combined dataset. LOPO measures **extrapolation** to unseen concrete systems, where all models fail. This is a data coverage limitation, not a model architecture problem.

---

## Implications for Deployment

- The deployed model (`best_model.pkl` — Random Forest) performs excellently **within the training distribution** (R²=0.99).
- Predictions for mix designs **outside the range of the three source papers** should be treated as unreliable extrapolations.
- Future work should expand the dataset to include:
  - Additional source papers covering intermediate mix designs
  - RCA replacement percentages between 0–100%
  - Broader ranges of cement-to-binder ratios, curing ages, and fiber types

---

## Artifacts

| File | Description |
|---|---|
| `models/best_model.pkl` | Random Forest (R²=0.9933) — used by `/predict` endpoint |
| `models/advanced_model.pkl` | XGBoost (R²=0.9929) |
| `models/baseline_model.pkl` | Linear Regression (R²=0.9497) |
| `models/metrics_real_dataset.json` | Full metrics for standard split + LOPO |
| `models/generalization_metrics.json` | Structured LOPO results with domain-shift analysis |
| `data/raw/master_rac_fiber_dataset.csv` | Training dataset (142 rows, 3 papers) |

---

## Methodology Notes

- **Blank-fill rule:** Numeric blanks → 0, categorical blanks → "unspecified"
- **Preprocessing:** Median imputation + StandardScaler for numerics; most-frequent imputation + OneHotEncoder for categoricals
- **Hyperparameters:** 
  - Random Forest: 200 estimators
  - XGBoost: 300 estimators, learning_rate=0.05, max_depth=6
  - ANN: (64, 32) hidden layers, 2000 max iterations
- **Random state:** 42 (reproducible splits)