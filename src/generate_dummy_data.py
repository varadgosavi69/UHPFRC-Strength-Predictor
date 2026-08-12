from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data/raw/dummy_test_data.csv")
RANDOM_SEED = 42
ROW_COUNT = 60
SCHEMA = [
    "cement",
    "silica_fume",
    "fly_ash",
    "sand",
    "coarse_aggregate",
    "water",
    "superplasticizer",
    "fiber_type",
    "fiber_content_percent",
    "water_binder_ratio",
    "curing_age_days",
    "curing_temp_celsius",
    "specimen_type",
    "compressive_strength_MPa",
    "source_paper",
]


def clipped_normal(rng, mean: float, std: float, low: float, high: float, size: int):
    return np.round(np.clip(rng.normal(mean, std, size), low, high), 2)


def generate_dummy_data(
    row_count: int = ROW_COUNT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cement = rng.uniform(550, 1100, row_count)
    silica_fume = rng.uniform(100, 200, row_count)
    fly_ash = rng.uniform(0, 180, row_count)
    sand = rng.uniform(700, 1200, row_count)
    coarse_aggregate = rng.uniform(0, 650, row_count)
    water_binder_ratio = rng.uniform(0.16, 0.28, row_count)
    binder = cement + silica_fume + fly_ash
    water = binder * water_binder_ratio
    fiber_content = rng.uniform(0, 2, row_count)
    curing_age = rng.choice(
        [7, 14, 28, 56, 90],
        size=row_count,
        p=[0.1, 0.15, 0.45, 0.2, 0.1],
    )
    curing_temp = rng.choice(
        [20, 23, 40, 60, 90],
        size=row_count,
        p=[0.35, 0.25, 0.15, 0.15, 0.1],
    )

    strength = (
        55
        + 0.055 * cement
        + 0.16 * silica_fume
        + 12 * fiber_content
        - 175 * (water_binder_ratio - 0.16)
        + 0.18 * np.sqrt(curing_age) * 10
        + 0.04 * np.maximum(curing_temp - 20, 0)
        + rng.normal(0, 8, row_count)
    )

    data = pd.DataFrame(
        {
            "cement": np.round(cement, 2),
            "silica_fume": np.round(silica_fume, 2),
            "fly_ash": np.round(fly_ash, 2),
            "sand": np.round(sand, 2),
            "coarse_aggregate": np.round(coarse_aggregate, 2),
            "water": np.round(water, 2),
            "superplasticizer": clipped_normal(rng, 22, 7, 5, 45, row_count),
            "fiber_type": rng.choice(["steel", "polypropylene", "hybrid"], row_count),
            "fiber_content_percent": np.round(fiber_content, 2),
            "water_binder_ratio": np.round(water_binder_ratio, 3),
            "curing_age_days": curing_age,
            "curing_temp_celsius": curing_temp,
            "specimen_type": rng.choice(["cube", "cylinder", "prism"], row_count),
            "compressive_strength_MPa": np.round(np.clip(strength, 60, 200), 2),
            "source_paper": [f"dummy_source_{index % 6 + 1}" for index in range(row_count)],
        }
    )
    return data[SCHEMA]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dummy_data()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
