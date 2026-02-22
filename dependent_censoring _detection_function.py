"""
Dependent censoring detection via a stratified permutation test.

This module provides a reusable API to test the conditional independence
assumption between event and censoring processes by returning a global p-value.

Main public function:
- get_final_p_value_for_dataset(...)

A command-line interface is also provided to run on CSV/Excel files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler


DEFAULT_TIME_COL = "observed_time"
DEFAULT_EVENT_COL = "event_indicator"


def sample_time_from_survival_curve(
    times: np.ndarray,
    survival_probs: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Sample one time from a discrete survival curve S(t)."""
    if len(times) == 0 or len(survival_probs) == 0:
        return np.inf

    survival_probs = np.minimum.accumulate(survival_probs)
    extended_survival_probs = np.concatenate(([1.0], survival_probs))
    interval_probs = extended_survival_probs[:-1] - extended_survival_probs[1:]
    interval_probs[interval_probs < 0] = 0

    prob_survive_past_end = survival_probs[-1]
    full_pmf = np.concatenate((interval_probs, [prob_survive_past_end]))

    pmf_sum = full_pmf.sum()
    if pmf_sum <= 1e-12:
        return float(times[-1]) + 1e-6
    full_pmf /= pmf_sum

    time_bins = np.concatenate((times, [times[-1] + 1e-6]))
    return float(rng.choice(time_bins, p=full_pmf))


def sample_time_conditionally(
    times: np.ndarray,
    survival_probs: np.ndarray,
    conditioning_time: float,
    rng: np.random.Generator,
) -> float:
    """Sample latent time from S(t), conditioned on time > conditioning_time."""
    start_index = int(np.searchsorted(times, conditioning_time, side="right"))

    if start_index >= len(times):
        return float(times[-1]) + 1e-6

    tail_times = times[start_index:]
    tail_probs = survival_probs[start_index:]

    if start_index > 0:
        prob_survive_past_conditioning_time = survival_probs[start_index - 1]
    else:
        prob_survive_past_conditioning_time = 1.0

    if prob_survive_past_conditioning_time <= 1e-12:
        return float(tail_times[-1]) + 1e-6

    conditional_survival_probs = tail_probs / prob_survive_past_conditioning_time
    return sample_time_from_survival_curve(tail_times, conditional_survival_probs, rng)


def _to_structured_y(time_col: pd.Series, event_col: pd.Series) -> np.ndarray:
    return np.array(
        list(zip(event_col.astype(bool), time_col.astype(float))),
        dtype=[("status", bool), ("time", float)],
    )


def _normalize_stratum_key(key: Any, x_cols: List[str]) -> Tuple[Any, ...]:
    if len(x_cols) == 1:
        return (key,)
    if isinstance(key, tuple):
        return key
    return (key,)


def _validate_input_dataframe(
    df: pd.DataFrame,
    x_cols: List[str],
    time_col: str,
    event_col: str,
) -> None:
    if df.empty:
        raise ValueError("Input dataset is empty.")

    if not x_cols:
        raise ValueError("x_cols must include at least one covariate column.")

    required_cols = {time_col, event_col, *x_cols}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df[[time_col, event_col] + x_cols].isnull().any().any():
        raise ValueError("Input contains missing values in required columns.")

    event_values = set(pd.Series(df[event_col]).unique().tolist())
    if not event_values.issubset({0, 1, False, True}):
        raise ValueError(f"{event_col} must be binary (0/1). Found values: {sorted(event_values)}")

    if (df[time_col] <= 0).any():
        raise ValueError(f"{time_col} must be strictly positive.")


def infer_x_cols(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> List[str]:
    """Infer covariate columns by excluding time/event columns."""
    x_cols = [c for c in df.columns if c not in {time_col, event_col}]
    if not x_cols:
        raise ValueError(
            f"No covariate columns found. Dataset must include columns besides "
            f"'{time_col}' and '{event_col}'."
        )
    return x_cols


def generate_null_nonparametric(
    df: pd.DataFrame,
    t_col: str,
    e_col: str,
    x_cols: List[str],
    rng: np.random.Generator,
    rsf_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generate one null dataset where event and censoring are independent given X.

    Procedure:
    1. Fit RSF for event process and censoring process.
    2. Impute latent times conditionally for each subject.
    3. Permute imputed censoring times within strata of X.
    4. Reconstruct observed (T, D).
    """
    if rsf_params is None:
        rsf_params = {"n_estimators": 100, "min_samples_leaf": 15, "n_jobs": -1}

    df_for_fit = pd.get_dummies(df[[t_col, e_col] + x_cols], columns=x_cols, drop_first=True)
    fit_cols = [c for c in df_for_fit.columns if c not in {t_col, e_col}]
    if not fit_cols:
        raise ValueError("No model features available after encoding x_cols.")

    scaler = StandardScaler()
    x_features = scaler.fit_transform(df_for_fit[fit_cols])

    y_event = _to_structured_y(df[t_col], df[e_col])
    y_censor = _to_structured_y(df[t_col], 1 - df[e_col].astype(int))

    model_event = RandomSurvivalForest(**rsf_params, random_state=int(rng.integers(1_000_000_000)))
    model_censor = RandomSurvivalForest(**rsf_params, random_state=int(rng.integers(1_000_000_000)))
    model_event.fit(x_features, y_event)
    model_censor.fit(x_features, y_censor)

    surv_funcs_event = model_event.predict_survival_function(x_features)
    surv_funcs_censor = model_censor.predict_survival_function(x_features)

    n_subjects = len(df)
    event_full = np.zeros(n_subjects, dtype=float)
    censor_full = np.zeros(n_subjects, dtype=float)

    for i in range(n_subjects):
        t_obs = float(df.iloc[i][t_col])
        e_obs = int(df.iloc[i][e_col])

        if e_obs == 1:
            event_full[i] = t_obs
            sf_c = surv_funcs_censor[i]
            censor_full[i] = sample_time_conditionally(sf_c.x, sf_c.y, t_obs, rng)
        else:
            censor_full[i] = t_obs
            sf_e = surv_funcs_event[i]
            event_full[i] = sample_time_conditionally(sf_e.x, sf_e.y, t_obs, rng)

    censor_permuted = censor_full.copy()
    grouped = df.groupby(x_cols, dropna=False)
    for _, indices in grouped.groups.items():
        idx = np.array(list(indices), dtype=int)
        censor_permuted[idx] = rng.permutation(censor_full[idx])

    df_null = df[x_cols].copy()
    df_null[t_col] = np.minimum(event_full, censor_permuted)
    df_null[e_col] = (event_full <= censor_permuted).astype(int)

    return df_null.reset_index(drop=True)


def compute_counts_for_time(
    df: pd.DataFrame,
    t: float,
    x_cols: List[str],
    time_col: str,
    event_col: str,
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    out = defaultdict(lambda: {"Nobs": {(1, 1): 0}, "ell": 0, "m": 0, "N": 0})

    for _, row in df.iterrows():
        key = tuple(row[c] for c in x_cols)
        observed_time = float(row[time_col])
        event_indicator = int(row[event_col])

        out[key]["N"] += 1
        if observed_time > t:
            out[key]["Nobs"][(1, 1)] += 1
        elif event_indicator == 0:
            out[key]["ell"] += 1
        else:
            out[key]["m"] += 1

    return out


def _compute_stratum_cmi_bounds(
    stratum_counts: Dict[str, Any],
    n_total_overall: int,
) -> Tuple[float, float]:
    n_total_stratum = int(stratum_counts["N"])
    if n_total_stratum == 0:
        return 0.0, 0.0

    n11_obs = int(stratum_counts["Nobs"][(1, 1)])
    ell_obs = int(stratum_counts["ell"])
    m_obs = int(stratum_counts["m"])

    best_min_mi = np.inf
    best_max_mi = -np.inf

    for u10 in range(m_obs + 1):
        for u01 in range(ell_obs + 1):
            n_e_gt_c_gt = n11_obs
            n_e_le_c_gt = u10
            n_e_gt_c_le = u01
            n_e_le_c_le = (m_obs - u10) + (ell_obs - u01)

            if n_e_le_c_le < 0:
                continue

            n_e_gt_row = n_e_gt_c_gt + n_e_gt_c_le
            n_e_le_row = n_e_le_c_gt + n_e_le_c_le
            n_c_gt_col = n_e_gt_c_gt + n_e_le_c_gt
            n_c_le_col = n_e_gt_c_le + n_e_le_c_le

            current_mi = 0.0
            for nij, n_row, n_col in [
                (n_e_gt_c_gt, n_e_gt_row, n_c_gt_col),
                (n_e_gt_c_le, n_e_gt_row, n_c_le_col),
                (n_e_le_c_gt, n_e_le_row, n_c_gt_col),
                (n_e_le_c_le, n_e_le_row, n_c_le_col),
            ]:
                if nij > 0 and n_row > 0 and n_col > 0:
                    current_mi += (nij / n_total_overall) * np.log((n_total_stratum * nij) / (n_row * n_col))

            best_min_mi = min(best_min_mi, current_mi)
            best_max_mi = max(best_max_mi, current_mi)

    if best_min_mi == np.inf:
        return 0.0, 0.0

    return float(best_min_mi), float(best_max_mi - best_min_mi)


def _get_delta_i_for_stratum(
    df: pd.DataFrame,
    t: float,
    stratum_key: Tuple[Any, ...],
    x_cols: List[str],
    n_total: int,
    time_col: str,
    event_col: str,
) -> float:
    counts = compute_counts_for_time(df, t, x_cols, time_col, event_col).get(stratum_key)
    if counts is None or counts["N"] <= 0:
        return 0.0

    _, delta_i = _compute_stratum_cmi_bounds(counts, n_total)
    return float(delta_i)


def stratified_fisher_test_standardized_strata(
    df: pd.DataFrame,
    times: Iterable[float],
    x_cols: List[str],
    B: int = 500,
    seed: int = 123,
    min_stratum_size: int = 30,
    variance_threshold: float = 1e-9,
    time_col: str = DEFAULT_TIME_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> Dict[str, Any]:
    """Compute global p-value from stratified standardized Fisher combination."""
    rng = np.random.default_rng(seed)
    times = [float(t) for t in times]
    if not times:
        raise ValueError("times must contain at least one time point.")
    if B < 10:
        raise ValueError("B should be at least 10 for a stable permutation p-value.")

    n_total = len(df)
    epsilon = 1e-18

    df_encoded = pd.get_dummies(df[[time_col, event_col] + x_cols], columns=x_cols, drop_first=True)
    fit_cols = [c for c in df_encoded.columns if c not in {time_col, event_col}]

    cph = CoxPHFitter()
    cph.fit(
        df_encoded,
        duration_col=time_col,
        event_col=event_col,
        formula=" + ".join(fit_cols) if fit_cols else None,
    )

    strata_groups: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    for key, grp in df.groupby(x_cols, dropna=False):
        normalized_key = _normalize_stratum_key(key, x_cols)
        if len(grp) >= min_stratum_size:
            strata_groups[normalized_key] = grp

    if not strata_groups:
        raise ValueError(f"No strata satisfy min_stratum_size={min_stratum_size}.")

    unique_strata = list(strata_groups.keys())

    permuted_dfs = [
        generate_null_nonparametric(df, t_col=time_col, e_col=event_col, x_cols=x_cols, rng=rng)
        for _ in range(B)
    ]

    per_stratum_results: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    n_times = len(times)

    for s in unique_strata:
        null_delta_i_matrix = np.zeros((B, n_times), dtype=float)
        obs_delta_i_vector = np.zeros(n_times, dtype=float)

        for k, t in enumerate(times):
            obs_delta_i_vector[k] = _get_delta_i_for_stratum(df, t, s, x_cols, n_total, time_col, event_col)
            for b in range(B):
                null_delta_i_matrix[b, k] = _get_delta_i_for_stratum(
                    permuted_dfs[b], t, s, x_cols, n_total, time_col, event_col
                )

        sigma_all = null_delta_i_matrix.std(axis=0)
        stable_idx = np.where(sigma_all > variance_threshold)[0]

        if len(stable_idx) == 0:
            continue

        null_stable = null_delta_i_matrix[:, stable_idx]
        obs_stable = obs_delta_i_vector[stable_idx]

        mu_stable = null_stable.mean(axis=0)
        sigma_stable = null_stable.std(axis=0)

        d_obs = np.abs((obs_stable - mu_stable) / (sigma_stable + epsilon))
        lambda_obs = float(np.max(d_obs)) if d_obs.size > 0 else 0.0

        d_perm = np.abs((null_stable - mu_stable) / (sigma_stable + epsilon))
        lambda_perm_dist = np.max(d_perm, axis=1)

        p_s = float((1 + np.sum(lambda_perm_dist >= lambda_obs)) / (B + 1))

        per_stratum_results[s] = {
            "p_value": p_s,
            "lambda_obs": lambda_obs,
            "lambda_perm_dist": lambda_perm_dist,
        }

    if not per_stratum_results:
        return {
            "final_p_value": np.nan,
            "observed_fisher_stat": np.nan,
            "per_stratum_p_values": {},
            "excluded_strata": [str(s) for s in unique_strata],
            "notes": "No stable strata found after variance filtering.",
        }

    stable_p_values = np.array([v["p_value"] for v in per_stratum_results.values()], dtype=float)
    observed_fisher_stat = float(-2.0 * np.sum(np.log(stable_p_values + 1e-12)))

    fisher_null = np.zeros(B, dtype=float)
    for b in range(B):
        fisher_stat_b = 0.0
        for s_data in per_stratum_results.values():
            lam_b = s_data["lambda_perm_dist"][b]
            p_b = (1 + np.sum(s_data["lambda_perm_dist"] >= lam_b)) / (B + 1)
            fisher_stat_b += -2.0 * np.log(p_b + 1e-12)
        fisher_null[b] = fisher_stat_b

    final_p_value = float((1 + np.sum(fisher_null >= observed_fisher_stat)) / (B + 1))

    return {
        "final_p_value": final_p_value,
        "observed_fisher_stat": observed_fisher_stat,
        "per_stratum_p_values": {str(k): v["p_value"] for k, v in per_stratum_results.items()},
        "excluded_strata": [str(s) for s in unique_strata if s not in per_stratum_results],
    }


def get_final_p_value_for_dataset(
    dataset: pd.DataFrame,
    quantiles: Iterable[float],
    B: int = 100,
    seed: int = 123,
    min_stratum_size: int = 30,
    variance_threshold: float = 0.001,
) -> float:
    """
    Public API for users: return only the final p-value.

    Interpretation:
    - Small p-value (e.g. < 0.05): evidence against conditional independence.
    - Large p-value: no strong evidence against the independence assumption.
    """
    time_col = DEFAULT_TIME_COL
    event_col = DEFAULT_EVENT_COL
    x_cols = infer_x_cols(dataset, time_col=time_col, event_col=event_col)
    _validate_input_dataframe(dataset, x_cols, time_col, event_col)

    times_list = build_times_from_quantiles(dataset, quantiles, time_col=time_col)

    fisher_result = stratified_fisher_test_standardized_strata(
        dataset,
        times=times_list,
        x_cols=x_cols,
        B=B,
        seed=seed,
        min_stratum_size=min_stratum_size,
        variance_threshold=variance_threshold,
        time_col=DEFAULT_TIME_COL,
        event_col=DEFAULT_EVENT_COL,
    )
    return float(fisher_result["final_p_value"])


def build_times_from_quantiles(
    dataset: pd.DataFrame,
    quantiles: Iterable[float],
    time_col: str = DEFAULT_TIME_COL,
) -> List[float]:
    """Helper to derive valid time grid from quantiles of observed time."""
    q = list(float(v) for v in quantiles)
    if not q:
        raise ValueError("quantiles is empty.")

    times = list(np.quantile(dataset[time_col].to_numpy(dtype=float), q))
    max_time = float(dataset[time_col].max())
    times = sorted(set(float(t) for t in times if 0 < t < max_time))
    if not times:
        raise ValueError("No valid times generated from quantiles.")
    return times


def _load_dataset(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv, .xls, or .xlsx")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect dependent censoring by computing a global p-value. "
            "Input dataset must include 'observed_time' and 'event_indicator'."
        )
    )
    parser.add_argument("--input", required=True, help="Path to CSV or Excel dataset")
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Quantile list used to generate time points from observed_time",
    )

    parser.add_argument("--B", type=int, default=100, help="Number of null simulations/permutations")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--min-stratum-size", type=int, default=30, help="Minimum stratum size")
    parser.add_argument("--variance-threshold", type=float, default=0.001, help="Stability threshold")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    df = _load_dataset(args.input)
    p_value = get_final_p_value_for_dataset(
        dataset=df,
        quantiles=args.quantiles,
        B=args.B,
        seed=args.seed,
        min_stratum_size=args.min_stratum_size,
        variance_threshold=args.variance_threshold,
    )

    print(f"Final p-value: {p_value:.6f}")


if __name__ == "__main__":
    main()
