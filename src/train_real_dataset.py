"""
train_real_dataset.py
=====================
Trains 4 models on master_rac_fiber_dataset.csv using two evaluation strategies:

  (a) Standard 80/20 random split
  (b) Leave-One-Paper-Out (LOPO) cross-validation

Results are saved to models/metrics_real_dataset.json and printed to stdout.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_utils import build_preprocessor  # noqa: E402

DATA_FILE      = ROOT / "data" / "raw" / "master_rac_fiber_dataset.csv"
MODELS_DIR     = ROOT / "models"
METRICS_OUT    = MODELS_DIR / "metrics_real_dataset.json"
BEST_MODEL_OUT = MODELS_DIR / "best_model.pkl"

TARGET     = "compressive_strength_MPa"
SOURCE_COL = "source_paper"
# Columns that carry no physical information for prediction
DROP_META  = ["mix_id", "data_notes", SOURCE_COL]

# Rows whose data_notes flag them as non-experimental
EXCLUDE_PATTERNS = ["RSM-PREDICTED", "NOT an experimental"]

RANDOM_STATE          = 42
SMALL_PAPER_THRESHOLD = 10  # flag folds with fewer test rows than this


# ── data loading & cleaning ───────────────────────────────────────────────────

def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    # 1. Exclude RSM-predicted / non-experimental rows
    if "data_notes" in df.columns:
        excl = df["data_notes"].fillna("").str.contains(
            "|".join(EXCLUDE_PATTERNS), case=False, na=False
        )
        print(f"[DATA] Excluded {int(excl.sum())} RSM-predicted rows.")
        df = df[~excl].copy()

    # 2. Must have a target value
    df = df.dropna(subset=[TARGET])

    # 3. Fix rca_replacement_percent: "~100" → 100.0
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

    # 4. Fill blanks: numeric → 0, string/categorical → "unspecified"
    #    Do this AFTER type coercion above so rca_replacement_percent is float.
    for col in df.columns:
        if col in [TARGET, SOURCE_COL] + DROP_META:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)
        else:
            # cast to plain Python str first to avoid mixed ArrowString / NaN
            df[col] = df[col].astype(object).fillna("unspecified")

    # 5. Ensure all string columns contain only strings (no ints slipping through)
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        if col in [TARGET, SOURCE_COL] + DROP_META:
            continue
        df[col] = df[col].astype(str)

    print(f"[DATA] Clean rows: {len(df)}   Source papers: {df[SOURCE_COL].nunique()}")
    for paper, cnt in df[SOURCE_COL].value_counts().items():
        pct = cnt / len(df) * 100
        note = "  ⚠ dominates dataset" if pct > 60 else ""
        note = "  ⚠ very small paper"  if cnt < SMALL_PAPER_THRESHOLD else note
        print(f"       {paper}: {cnt} rows ({pct:.0f}%){note}")

    return df


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    drop = [TARGET] + [c for c in DROP_META if c in df.columns]
    return df.drop(columns=drop)


# ── model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_STATE, verbosity=0
        ),
        "ANN": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=2000,
            random_state=RANDOM_STATE
        ),
    }


def calc_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 6),
        "RMSE": round(rmse, 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "N":    int(len(y_true)),
    }


def build_pipeline(x_train: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(x_train)),
        ("model", estimator),
    ])


# ── (a) Standard 80/20 random split ──────────────────────────────────────────

def run_standard_split(df: pd.DataFrame):
    X = get_features(df)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\n[STANDARD SPLIT] Train={len(X_train)}  Test={len(X_test)}\n")

    results = []
    best_pipeline = None
    best_r2 = -999.0

    for name, estimator in get_models().items():
        pipe = build_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        m = calc_metrics(y_test, pipe.predict(X_test))
        record = {"split": "standard_80_20", "model": name, **m}
        results.append(record)
        print(f"  {name:22s}  R2={m['R2']:+.4f}  RMSE={m['RMSE']:7.4f}  MAE={m['MAE']:7.4f}")
        if m["R2"] > best_r2:
            best_r2 = m["R2"]
            best_pipeline = (name, pipe, X_train.columns.tolist())

    return results, best_pipeline


# ── (b) Leave-One-Paper-Out ───────────────────────────────────────────────────

def run_lopo(df: pd.DataFrame) -> list:
    papers = sorted(df[SOURCE_COL].unique())
    results = []

    print(f"\n[LOPO] Folds: {len(papers)}\n")

    for held_out in papers:
        train_df = df[df[SOURCE_COL] != held_out]
        test_df  = df[df[SOURCE_COL] == held_out]

        X_train = get_features(train_df)
        y_train = train_df[TARGET]
        X_test  = get_features(test_df)
        y_test  = test_df[TARGET]

        n_test = len(test_df)
        flag   = "  ⚠ TOO FEW TEST ROWS — UNRELIABLE" if n_test < SMALL_PAPER_THRESHOLD else ""
        print(f"  Held-out: {held_out}  (test n={n_test}){flag}")

        for name, estimator in get_models().items():
            if n_test < 2:
                m    = {"R2": None, "RMSE": None, "MAE": None, "N": n_test}
                note = "SKIPPED — only 1 test row, metrics undefined"
            else:
                try:
                    pipe = build_pipeline(X_train, estimator)
                    pipe.fit(X_train, y_train)
                    m    = calc_metrics(y_test, pipe.predict(X_test))
                    note = "unreliable (few test rows)" if n_test < SMALL_PAPER_THRESHOLD else ""
                except Exception as exc:
                    m    = {"R2": None, "RMSE": None, "MAE": None, "N": n_test}
                    note = f"ERROR: {exc}"

            record = {
                "split": "LOPO",
                "held_out_paper": held_out,
                "model": name,
                **m,
            }
            if note:
                record["note"] = note
            results.append(record)

            r2s   = f"{m['R2']:+.4f}"  if m["R2"]   is not None else "    N/A "
            rmses = f"{m['RMSE']:.4f}" if m["RMSE"] is not None else "    N/A "
            suffix = f"  [{note}]" if note else ""
            print(f"    {name:22s}  R2={r2s}  RMSE={rmses}{suffix}")
        print()

    return results


def lopo_summary(lopo_results: list) -> None:
    df = pd.DataFrame(lopo_results)
    valid = df[df["R2"].notna()].copy()
    if valid.empty:
        print("[LOPO SUMMARY] No valid folds with ≥2 test rows.")
        return
    print("[LOPO SUMMARY] Mean R²/RMSE/MAE across valid folds:")
    summary = (
        valid.groupby("model")[["R2", "RMSE", "MAE"]]
             .mean()
             .sort_values("R2", ascending=False)
    )
    print(summary.to_string(float_format=lambda v: f"{v:+.4f}"))
    print()
    print("  Note: Kakooei_2011 (6 rows) and any 1-row paper results are flagged")
    print("  unreliable. Metrics for those folds should not be used for model selection.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Model Training — master_rac_fiber_dataset.csv")
    print("=" * 65)

    df = load_and_clean()

    # (a) Standard split
    print("\n" + "─" * 65)
    print("(a) STANDARD 80/20 RANDOM SPLIT")
    print("─" * 65)
    std_results, best_pipeline = run_standard_split(df)

    # (b) LOPO
    print("\n" + "─" * 65)
    print("(b) LEAVE-ONE-PAPER-OUT (LOPO)")
    print("─" * 65)
    lopo_results = run_lopo(df)
    lopo_summary(lopo_results)

    # Save all metrics to JSON
    all_results = std_results + lopo_results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n[SAVED] Metrics → {METRICS_OUT.relative_to(ROOT)}")

    # Save best model from standard split
    import joblib
    model_name, pipe, feat_cols = best_pipeline
    joblib.dump(
        {
            "model_name": model_name,
            "pipeline":   pipe,
            "target_column": TARGET,
            "feature_columns": feat_cols,
        },
        BEST_MODEL_OUT,
    )
    print(f"[SAVED] Best model ({model_name}) → {BEST_MODEL_OUT.relative_to(ROOT)}")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()