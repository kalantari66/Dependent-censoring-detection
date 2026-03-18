from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from data import load_real_data

REAL_DATASETS = (
    "SUPPORT",
    "METABRIC",
    "NACD",
    "FLCHAIN",
    "GBSG2",
    "NWTCO",
    "PBC",
    "GBM",
    "NPC",
    "AIDS",
    "HFCR",
    "WPBC",
    "BMT",
    "leukemia",
    "Rossi",
    "COVID",
)
TARGET_COLUMNS = ("time", "event")


def encode_feature_block(series: pd.Series) -> pd.DataFrame:
    """Encode a single feature into the design matrix used by the Cox model."""
    feature_name = str(series.name)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce").astype(float)
        return pd.DataFrame({feature_name: numeric_series}, index=series.index)

    categories = pd.Index(series.dropna().astype(str).unique())
    if categories.empty:
        return pd.DataFrame(index=series.index)

    categorical = pd.Categorical(series.astype("string"), categories=categories)
    encoded = pd.get_dummies(categorical, prefix=feature_name, drop_first=True, dtype=float)
    encoded.index = series.index
    return encoded


def make_survival_target(df: pd.DataFrame) -> np.ndarray:
    """Create the survival target expected by scikit-survival."""
    clean_df = df.assign(
        event=df["event"].astype(bool),
        time=pd.to_numeric(df["time"], errors="coerce").astype(float),
    )
    return Surv.from_dataframe(event="event", time="time", data=clean_df)


def evaluate_feature(
    df: pd.DataFrame,
    feature_name: str,
    repeats: int,
    test_size: float,
    seed: int,
) -> tuple[str, float]:
    """Return the mean held-out concordance for one feature."""
    feature_block = encode_feature_block(df[feature_name])
    if feature_block.empty:
        return feature_name, float("-inf")

    model_df = pd.concat([df[list(TARGET_COLUMNS)], feature_block], axis=1).dropna().reset_index(drop=True)
    n_rows = len(model_df)
    n_events = int(model_df["event"].astype(bool).sum())
    if n_rows < 10 or n_events < 2:
        return feature_name, float("-inf")

    X = model_df.drop(columns=list(TARGET_COLUMNS))
    y = make_survival_target(model_df)
    scores: list[float] = []

    for repeat_idx in range(repeats):
        split_seed = seed + repeat_idx
        train_idx, test_idx = train_test_split(
            np.arange(n_rows),
            test_size=test_size,
            random_state=split_seed,
            shuffle=True,
        )

        y_train = y[train_idx]
        y_test = y[test_idx]
        if y_train["event"].sum() == 0 or y_test["event"].sum() == 0:
            continue

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        active_columns = [column for column in X_train.columns if X_train[column].nunique(dropna=True) > 1]
        if not active_columns:
            continue

        X_train = X_train[active_columns]
        X_test = X_test[active_columns]

        try:
            model = CoxPHSurvivalAnalysis(alpha=1e-6)
            model.fit(X_train, y_train)
            scores.append(float(model.score(X_test, y_test)))
        except Exception:  # pragma: no cover
            continue

    mean_c_index = float(np.mean(scores)) if scores else float("-inf")
    return feature_name, mean_c_index


def rank_dataset_features(
    dataset_name: str,
    repeats: int,
    test_size: float,
    seed: int,
) -> dict[str, int | list[str]]:
    """Rank all features for one dataset."""
    df = load_real_data(dataset_name, onehot_encode=False)
    feature_names = [column for column in df.columns if column not in TARGET_COLUMNS]
    feature_scores = [
        evaluate_feature(
            df=df,
            feature_name=feature_name,
            repeats=repeats,
            test_size=test_size,
            seed=seed,
        )
        for feature_name in feature_names
    ]
    ranked_features = [feature for feature, _ in sorted(feature_scores, key=lambda item: (-item[1], item[0]))]

    return {
        "n_samples": len(df),
        "n_features": len(feature_names),
        "feature_ranking": ranked_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank real datasets with repeated univariate Cox models.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(REAL_DATASETS),
        help="Datasets to rank. Defaults to all real datasets.",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated train/test splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction for evaluation.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for repeated splits.")
    args = parser.parse_args()

    invalid_datasets = sorted(set(args.datasets) - set(REAL_DATASETS))
    if invalid_datasets:
        raise ValueError(f"Unsupported datasets: {', '.join(invalid_datasets)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "feature_rankings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}.json"

    results = {
        "created_at": timestamp,
        "config": {
            "datasets": args.datasets,
            "repeats": args.repeats,
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "datasets": {
            dataset_name: rank_dataset_features(
                dataset_name=dataset_name,
                repeats=args.repeats,
                test_size=args.test_size,
                seed=args.seed,
            )
            for dataset_name in tqdm(args.datasets)
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Wrote feature rankings to {output_path}")


if __name__ == "__main__":
    main()
