import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cmi import detect_dependent_censoring, preprocess_dataset
from data import dgp, load_real_data, semiDGP

REAL_DATASETS = {"METABRIC", "NACD", "GBSG2", "NWTCO", "NPC", "AIDS", "HFCR", "leukemia", "Rossi", "COVID"}
SEMI_SYNTH_DATASETS = {f"SEMI_{data}" for data in REAL_DATASETS}


def select_feature_by_strata_size(
        df: pd.DataFrame, 
        x_cols: List[str], 
        min_size: int
) -> Tuple[List[str], int]:
    """
    Select features based on the number of strata that meet the minimum size requirement.
    Iteratively drop the most complex feature (based on cardinality) until at least one stratum
    satisfies the minimum size requirement.
    """
    for k in range(len(x_cols), 0, -1):
        cand = x_cols[:k]
        group_sizes = df.groupby(cand).size()
        valid = (group_sizes >= min_size).sum()
        if valid > 0:
            return cand, valid
    group_sizes = df.groupby([x_cols[0]]).size()
    return [x_cols[0]], (group_sizes >= min_size).sum()


def sample_hyperparameters(
        config: Dict[str, Any], 
        n_trials: int, 
        seed: int
) -> List[Dict[str, Any]]:
    """Sample hyperparameters for the dependent-censoring detection method based on the provided configuration."""
    rng = np.random.default_rng(seed)

    n_quantiles_choices = [v for v in config["n_quantiles"]]
    bootstrap_choices = [v for v in config["bootstrap_samples"]]
    min_strata_choices = [v for v in config["min_stratum_size"]]
    seed_range = config["seed_range"]
    seed_low, seed_high = seed_range[0], seed_range[1]
    q_low, q_high = config["quantile_range"][0], config["quantile_range"][1]
    if not (0 < q_low < q_high < 1):
        raise ValueError("quantile_range must satisfy 0 < low < high < 1.")
    if seed_low > seed_high:
        raise ValueError("seed_range must be [low, high] with low <= high.")

    sampled: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        n_q = rng.choice(n_quantiles_choices)
        quantiles = np.linspace(q_low, q_high, n_q).tolist()
        sampled.append(
            {
                "n_quantiles": n_q,
                "quantiles": quantiles,
                "B": rng.choice(bootstrap_choices),
                "seed": rng.integers(seed_low, seed_high + 1),
                "min_stratum_size": rng.choice(min_strata_choices),
            }
        )
    return sampled


def format_strength_for_label(value: float) -> str:
    """Return a file-name-friendly representation of a numeric dependence strength."""
    return f"{value:g}".replace("-", "neg").replace("+", "").replace(".", "p")


def resolve_dataset(
    dataset: str,
    dependency_kind: str,
    copula_type: str,
    feature_kind: str,
    seed: int,
    theta: float,
    alpha: float,
) -> tuple[str, Path, pd.DataFrame]:
    """Resolve the experiment dataset and return its label, config, and frame."""
    # TODO: right now all the experiments use the same config file.
    config_path = Path("config/real_exp.json")

    if dataset == "SYNTH":
        kind = f"{dependency_kind}_{feature_kind}"
        raw_df = dgp(
            kind=kind,
            n_subjects=1000,
            n_features=4,
            copula=copula_type,
            seed=seed,
            theta=theta,
            alpha_E=alpha,
            alpha_C=alpha,
        )
        dataset_label = f"SYNTH_{kind}"
        if dependency_kind == "copula":
            dataset_label += f"_{copula_type}_theta{format_strength_for_label(theta)}"
        else:
            dataset_label += f"_alpha{format_strength_for_label(alpha)}"
    elif dataset in SEMI_SYNTH_DATASETS:
        dataset_label = f"{dataset}_{dependency_kind}"
        if dependency_kind == "copula":
            dataset_label += f"_{copula_type}_theta{theta}"
        raw_df = semiDGP(
            dataset=dataset.split("_", 1)[1], # extract the real dataset name from the SEMI_ prefix
            kind=dependency_kind,
            model="coxph",
            seed=seed,
            copula=copula_type,
            theta=theta,
        )
    elif dataset in REAL_DATASETS:
        raw_df = load_real_data(data_name=dataset, onehot_encode=False)
        dataset_label = f"REAL_{dataset}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return dataset_label, config_path, raw_df


def prepare_experiment_dataset(
    raw_df: pd.DataFrame,
    config: Dict[str, Any],
    dataset: str,
) -> tuple[pd.DataFrame, List[str], str, str]:
    """Prepare the experiment dataset and return the feature columns used downstream."""
    if dataset != "SYNTH":
        df, features_all = preprocess_dataset(
            raw_df=raw_df,
            bins=config["preprocessing"]["discretization_bins"],
            max_features=config["preprocessing"]["max_selected_features"],
            event_col="event",
            time_col="time",
            feature_exclude=None
        )
        return df, features_all, "time", "event"

    feature_cols = [
        col for col in raw_df.columns
        if col.startswith("x") and not col.endswith("_continuous")
    ]
    if not feature_cols:
        raise ValueError("Synthetic dataset does not include usable strata columns.")
    df = raw_df[["time", "event"] + feature_cols].copy()
    return df, feature_cols, "time", "event"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dependent-censoring detection.")
    parser.add_argument(
        "--dataset",
        type=str, 
        default="SEMI_GBSG2",
        help="Dataset name, including real, semi-synthetic, and synthetic options.",
    )
    parser.add_argument(
        "--dependency-kind",
        choices=("copula", "frailty"),
        default="frailty",
        help="Dependence construction used when the selected --dataset is synthetic (SYNTH) or semi-synthetic (SEMI_*).",
    )
    parser.add_argument(
        "--copula-type",
        choices=("gaussian", "clayton", "gumbel", "frank"),
        default="clayton",
        help="Type of copula to use for synthetic or semi-synthetic copula dependence construction.",
    )
    parser.add_argument(
        "--feature-kind",
        choices=("discrete", "continuous"),
        default="discrete",
        help=(
            "Kind of features to use for fully synthetic data generation; "
            "this affects --dataset SYNTH for both copula and frailty dependence."
        ),
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=3.0,
        help=(
            "Theta parameter controlling the strength of dependence for copula-based data generation; "
            "ignored for frailty-based generation."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help=(
            "Frailty dependence strength for --dataset SYNTH when --dependency-kind frailty; "
            "the same value is used for alpha_E and alpha_C."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=10, help="Number of hyperparameter combinations to sample.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for hyperparameter sampling and DGP.")
    args = parser.parse_args()

    if args.dataset == "SYNTH" and args.dependency_kind == "copula" and not np.isfinite(args.theta):
        parser.error("--theta must be finite for synthetic copula data.")
    if args.dataset == "SYNTH" and args.dependency_kind == "frailty" and not np.isfinite(args.alpha):
        parser.error("--alpha must be finite for synthetic frailty data.")

    dataset_label, config_path, raw_df = resolve_dataset(
        dataset=args.dataset,
        dependency_kind=args.dependency_kind,
        copula_type=args.copula_type,
        feature_kind=args.feature_kind,
        seed=args.seed,
        theta=args.theta,
        alpha=args.alpha,
    )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    sampled_hyperparameters = sample_hyperparameters(config=config, n_trials=args.n_trials, seed=args.seed)

    df, features_all, time_col, event_col = prepare_experiment_dataset(
        raw_df=raw_df,
        config=config,
        dataset=args.dataset,
    )

    msg = f"Running {args.n_trials} trials on {dataset_label} with {len(df)} samples and {len(features_all)} covariates."
    # TODO: add parallel processing
    records: List[Dict[str, Any]] = []
    for run_id, hyperparameters in enumerate(tqdm(sampled_hyperparameters, desc=msg), start=1):
        row: Dict[str, Any] = {
            "run_id": run_id,
            "dataset": dataset_label,
            "n_samples": len(df),
            "n_quantiles": hyperparameters["n_quantiles"],
            "quantiles": json.dumps(hyperparameters["quantiles"]),
            "B": hyperparameters["B"],
            "seed": hyperparameters["seed"],
            "min_stratum_size": hyperparameters["min_stratum_size"],
            "p_value": np.nan,
            "observed_fisher_stat": np.nan,
            "ncov_used": 0,
            "cov_used": "",
            "n_strata": 0,
            "n_excluded_strata": 0,
            "status": "success",
            "error": "",
        }

        features_select, n_valid_strata = select_feature_by_strata_size(
            df=df,
            x_cols=features_all,
            min_size=hyperparameters["min_stratum_size"],
        )
        row["ncov_used"] = len(features_select)
        row["cov_used"] = json.dumps(features_select)

        if n_valid_strata == 0:
            row["status"] = "error"
            row["error"] = "No strata meet min_stratum_size after dataset preparation."
            records.append(row)
            continue

        run_df = df[[time_col, event_col] + features_select].copy()
        try:
            test_details = detect_dependent_censoring(
                run_df,
                quantiles=hyperparameters["quantiles"],
                t_col=time_col,
                e_col=event_col,
                x_cols=features_select,
                B=hyperparameters["B"],
                seed=hyperparameters["seed"],
                min_stratum_size=hyperparameters["min_stratum_size"],
                return_details=True,
            )
            row["p_value"] = test_details["final_p_value"]
            row["observed_fisher_stat"] = test_details["observed_fisher_stat"]

            # Sanity check: the number of per-stratum p-values should match the number of valid strata (after excluding those that don't meet min size)
            excluded = test_details["excluded_strata"]
            per_stratum = test_details["per_stratum_p_values"]
            assert len(per_stratum) + len(excluded) == n_valid_strata, "Number of per-stratum p-values plus excluded strata should equal number of valid strata."
            row["n_strata"] = len(per_stratum)
            row["n_excluded_strata"] = len(excluded)
            
            row["per_stratum_p_values"] = json.dumps(per_stratum) if isinstance(per_stratum, dict) else ""
            
            if pd.isna(row["p_value"]):
                row["status"] = "error"
                row["error"] = "Invalid p-value."
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        records.append(row)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = Path("results") / f"{dataset_label}_{args.n_trials}_{now}.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df.to_csv(file, index=False)

    success_count = (results_df["status"] == "success").sum()
    error_count = (results_df["status"] == "error").sum()
    print(f"Saved {len(results_df)} runs to {file} (success={success_count}, error={error_count}).")


if __name__ == "__main__":
    main()
