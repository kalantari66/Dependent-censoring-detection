"""Core detection logic for conditional independence in censored data."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd

from cmi.null_sampling import prepare_null_nonparametric, generate_null_nonparametric_fast


def detect_dependent_censoring(
    df: pd.DataFrame,
    quantiles: Iterable[float],
    t_col: str,
    e_col: str,
    x_cols: Optional[List[str]] = None,
    B: int = 500,
    seed: int = 123,
    min_stratum_size: int = 30,
    variance_threshold: float = 1e-9,   # TODO: how to choose this variance threshold? Hamid uses 0.001
    return_details: bool = False,
    verbose: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Public function for dependent censoring detection.

    Inputs:
        df: pandas DataFrame containing the data
        quantiles: iterable of quantiles to use for time points
        t_col: column name for observed times
        e_col: column name for event indicators (1=event, 0=censoring)
        x_cols: list of column names to use as covariates for stratification (if None, uses all columns except t_col and e_col)
        B: number of bootstrap samples
        seed: random seed for reproducibility
        min_stratum_size: minimum size of each stratum
        variance_threshold: threshold for variance in Fisher's exact test
        return_details: whether to return detailed results or just the final p-value
        verbose: whether to print detailed information during computation

    Output:
      global p-value (or dict with details if return_details=True)

    Data requirements:
      - df[t_col] numeric, df[e_col] in {0,1}
      - covariates: by default all columns starting with 'x' are used as strata
    """
    df = df.copy()

    if x_cols is None:
        x_cols = [c for c in df.columns if c not in {t_col, e_col}]
        if not x_cols:
            raise ValueError("No covariate columns found.")
        if verbose:
            print(f"Using all columns except '{t_col}' and '{e_col}' as features: {x_cols}")


    validate_data(df, t_col, e_col, x_cols)

    # TODO: pass times as an optional argument
    times = list(np.quantile(df[t_col].to_numpy(), list(quantiles)))
    times = [t for t in times if np.isfinite(t) and t > 0 and t < df[t_col].max()]
    if not times:
        raise ValueError("No valid time points produced from quantiles. Check quantiles and data range.")

    res = stratified_fisher_test_standardized_strata(
        df=df,
        times=times,
        x_cols=x_cols,
        B=B,
        seed=seed,
        min_stratum_size=min_stratum_size,
        variance_threshold=variance_threshold,
        t_col=t_col,
        e_col=e_col,
        verbose=verbose,
    )

    return res if return_details else float(res["final_p_value"])


def validate_data(
        df: pd.DataFrame, 
        t_col: str, 
        e_col: str, 
        x_cols: List[str]
) -> None:
    """
    Validate that the input DataFrame has the required structure and types for the dependent censoring detection.
    """
    # Check for required columns
    missing = [c for c in [t_col, e_col] + x_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    
    # Check if t_col is numeric and positive
    if not np.issubdtype(df[t_col].dtype, np.number):
        raise ValueError(f"{t_col} must be numeric.")
    if (df[t_col] <= 0).any():
        raise ValueError(f"{t_col} must be positive.")

    # Check if e_col is binary 0/1
    vals = set(pd.unique(df[e_col].dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(f"{e_col} must be 0/1. Found values: {sorted(vals)}")


def _stratum_key(row: pd.Series, x_cols: List[str]) -> Tuple:
    return tuple(row[c] for c in x_cols)


def _compute_counts_for_time(
    df: pd.DataFrame, t: float, x_cols: List[str], t_col: str, e_col: str
) -> Dict[Tuple, Dict[str, Any]]:
    out = defaultdict(lambda: {"Nobs": {(1, 1): 0}, "ell": 0, "m": 0, "N": 0})
    for _, r in df.iterrows():
        key = _stratum_key(r, x_cols)
        T, D = r[t_col], r[e_col]
        out[key]["N"] += 1
        if T > t:
            out[key]["Nobs"][(1, 1)] += 1
        else:
            if D == 0:
                out[key]["ell"] += 1
            else:
                out[key]["m"] += 1
    return out


def _compute_stratum_cmi_bounds(stratum_counts: Dict[str, Any], n_total_overall: int) -> Tuple[float, float]:
    N_total_stratum = stratum_counts["N"]
    if N_total_stratum == 0:
        return 0.0, 0.0

    N11_obs = stratum_counts["Nobs"][(1, 1)]
    ell_obs = stratum_counts["ell"]
    m_obs = stratum_counts["m"]

    best_min_MI, best_max_MI = np.inf, -np.inf

    for u10 in range(m_obs + 1):
        for u01 in range(ell_obs + 1):
            N_E_gt_C_gt = N11_obs
            N_E_le_C_gt = u10
            N_E_gt_C_le = u01
            N_E_le_C_le = (m_obs - u10) + (ell_obs - u01)
            if N_E_le_C_le < 0:
                continue

            N_E_gt_row = N_E_gt_C_gt + N_E_gt_C_le
            N_E_le_row = N_E_le_C_gt + N_E_le_C_le
            N_C_gt_col = N_E_gt_C_gt + N_E_le_C_gt
            N_C_le_col = N_E_gt_C_le + N_E_le_C_le

            current_MI = 0.0
            for Nij, Ns_row, Nd_col in [
                (N_E_gt_C_gt, N_E_gt_row, N_C_gt_col),
                (N_E_gt_C_le, N_E_gt_row, N_C_le_col),
                (N_E_le_C_gt, N_E_le_row, N_C_gt_col),
                (N_E_le_C_le, N_E_le_row, N_C_le_col),
            ]:
                if Nij > 0 and Ns_row > 0 and Nd_col > 0:
                    current_MI += (1.0 / n_total_overall) * Nij * np.log(
                        (N_total_stratum * Nij) / (Ns_row * Nd_col)
                    )

            best_min_MI = min(best_min_MI, current_MI)
            best_max_MI = max(best_max_MI, current_MI)

    if best_min_MI == np.inf:
        return 0.0, 0.0
    return best_min_MI, best_max_MI - best_min_MI


def _get_delta_I(
    df: pd.DataFrame,
    t: float,
    stratum_key: Tuple,
    x_cols: List[str],
    n_total: int,
    t_col: str,
    e_col: str,
) -> float:
    counts = _compute_counts_for_time(df, t, x_cols, t_col, e_col).get(stratum_key)
    if counts and counts["N"] > 0:
        _, delta_I = _compute_stratum_cmi_bounds(counts, n_total)
        return delta_I
    return 0.0


def stratified_fisher_test_standardized_strata(
    df: pd.DataFrame,
    times: Iterable[float],
    x_cols: List[str],
    B: int,
    seed: int,
    min_stratum_size: int,
    variance_threshold: float,
    t_col: str,
    e_col: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Performs a stratified permutation test using Fisher's method with a robust,
    standardized statistic for each stratum.

    For each stratum, this function:
    1. Standardizes the ΔI statistic at each time point by its null mean and variance.
    2. Uses the maximum of these standardized scores as the stratum's test statistic.
    3. Calculates a p-value for this robust statistic.
    4. Combines these p-values using Fisher's method.

    Input:
    - df: input DataFrame
    - times: list of time points to evaluate
    - x_cols: columns to define strata
    - B: number of bootstrap samples
    - seed: random seed for reproducibility
    - min_stratum_size: minimum size of each stratum to be included in the test
    - variance_threshold: minimum variance threshold for including a stratum in the Fisher combination
    - t_col: column name for observed times
    - e_col: column name for event indicators
    - verbose: whether to print progress and warnings

    Output:
    - dict with final p-value, observed Fisher statistic, per-stratum p-values, and excluded strata
    """
    rng = np.random.default_rng(seed)
    times = list(times)
    n_times = len(times)
    n_total = len(df)
    eps = 1e-18 # TODO: change to a global constant

    # --- Standard setup: find strata, generate permutations (DVFM-based null) ---
    strata_groups = {k: grp for k, grp in df.groupby(x_cols) if len(grp) >= min_stratum_size}
    if not strata_groups:
        raise ValueError(f"No strata with size >= {min_stratum_size}. Reduce min_stratum_size or change strata.")
    unique_strata = list(strata_groups.keys())

    null_pre = prepare_null_nonparametric(
        df,
        t_col=t_col,
        e_col=e_col,
        x_cols=x_cols,
        rng=rng,
    )
    permuted_dfs = [
        generate_null_nonparametric_fast(null_pre, t_col=t_col, e_col=e_col, rng=rng)
        for _ in range(B)
    ]

    # Dictionary to store results for each stratum
    per_s: Dict[Tuple, Dict[str, Any]] = {}

    # Calculate p-value for each stratum
    for s in unique_strata:
        # For this stratum, compute the null matrix (B x K) and observed vector (K) of ΔI
        null_mat = np.zeros((B, n_times))
        obs_vec = np.zeros(n_times)

        for k, t in enumerate(times):
            obs_vec[k] = _get_delta_I(df, t, s, x_cols, n_total, t_col, e_col)
            for b in range(B):
                null_mat[b, k] = _get_delta_I(permuted_dfs[b], t, s, x_cols, n_total, t_col, e_col)

        # Filter out unstable time points WITHIN this stratum
        sigma_all = null_mat.std(axis=0)
        valid_idx = np.where(sigma_all > variance_threshold)[0]
        if len(valid_idx) == 0:
            warn(f"Stratum {s} has no stable time points after filtering. Excluding from Fisher combination.")
            continue

        null_stable = null_mat[:, valid_idx]
        obs_stable = obs_vec[valid_idx]

        # Calculate mean and std dev for the stable null distributions
        mu = null_stable.mean(axis=0)
        sigma = null_stable.std(axis=0)

        # Calculate the OBSERVED standardized statistic (Λ_std^s) for the stratum
        d_obs = np.abs((obs_stable - mu) / (sigma + eps))
        lambda_obs = float(np.max(d_obs)) if d_obs.size else 0.0

        # Calculate the NULL DISTRIBUTION of the standardized statistic (Λ_std^s)
        d_perm = np.abs((null_stable - mu) / (sigma + eps))
        lambda_perm = np.max(d_perm, axis=1)

        # Calculate the p-value
        p_s = (1 + np.sum(lambda_perm >= lambda_obs)) / (B + 1)
        per_s[s] = {"p_value": float(p_s), "lambda_obs": lambda_obs, "lambda_perm_dist": lambda_perm}

    # Fisher combination logic (now correct)
    if not per_s:
        warn("Warning: No strata remained after stability checks. Cannot compute a valid p-value.")
        return {"final_p_value": np.nan, "notes": "No stable strata found."}

    if verbose:
        print(f"Using {len(per_s)} out of {len(unique_strata)} strata for final test.")

    stable_p = np.array([v["p_value"] for v in per_s.values()])
    F_obs = float(-2 * np.sum(np.log(stable_p + 1e-12)))

    # null Fisher distribution
    F_null = np.zeros(B)
    for b in range(B):
        Fb = 0.0
        for v in per_s.values():
            lam_b = v["lambda_perm_dist"][b]
            p_b = (1 + np.sum(v["lambda_perm_dist"] >= lam_b)) / (B + 1)
            Fb += -2 * np.log(p_b + 1e-12)
        F_null[b] = Fb

    final_p = float((1 + np.sum(F_null >= F_obs)) / (B + 1))

    return {
        "final_p_value": final_p,
        "observed_fisher_stat": F_obs,
        "per_stratum_p_values": {str(k): v["p_value"] for k, v in per_s.items()},
        "excluded_strata": [str(s) for s in unique_strata if s not in per_s],
    }
