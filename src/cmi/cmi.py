"""Core detection logic for conditional independence in censored data."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest


def detect_dependent_censoring(
    df: pd.DataFrame,
    quantiles: Iterable[float],
    B: int = 500,
    seed: int = 123,
    min_stratum_size: int = 30,
    variance_threshold: float = 1e-9,
    t_col: str = "observed_time",
    e_col: str = "event_indicator",
    x_cols: Optional[List[str]] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Public function for dependent censoring detection.

    Inputs:
      df: pandas DataFrame containing the data
      quantiles: iterable of quantiles to use for time points
      B: number of bootstrap samples
      seed: random seed for reproducibility
      min_stratum_size: minimum size of each stratum
      variance_threshold: threshold for variance in Fisher's exact test

    Output:
      global p-value (or dict with details if return_details=True)

    Data requirements:
      - df[t_col] numeric, df[e_col] in {0,1}
      - covariates: by default all columns starting with 'x' are used as strata
    """
    df = df.copy()

    if x_cols is None:
        x_cols = [c for c in df.columns if c.startswith("x")]
        if not x_cols:
            raise ValueError("Could not infer x_cols. Provide x_cols or name covariates like x0, x1, ...")

    _validate_input_df(df, t_col, e_col, x_cols)

    times = list(np.quantile(df[t_col].to_numpy(), list(quantiles)))
    times = [t for t in times if np.isfinite(t) and t > 0 and t < df[t_col].max()]
    if not times:
        raise ValueError("No valid time points produced from quantiles. Check quantiles and data range.")

    res = _stratified_fisher_test_standardized_strata(
        df=df,
        times=times,
        x_cols=x_cols,
        B=B,
        seed=seed,
        min_stratum_size=min_stratum_size,
        variance_threshold=variance_threshold,
        t_col=t_col,
        e_col=e_col,
    )

    return res if return_details else float(res["final_p_value"])


def _validate_input_df(df: pd.DataFrame, t_col: str, e_col: str, x_cols: List[str]) -> None:
    missing = [c for c in [t_col, e_col] + x_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    if not np.issubdtype(df[t_col].dtype, np.number):
        raise ValueError(f"{t_col} must be numeric.")
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


def _sample_time_from_survival_curve(
    times: np.ndarray, 
    survival_probs: np.ndarray, 
    rng: np.random.Generator
) -> float:
    if len(times) == 0 or len(survival_probs) == 0:
        return np.inf
    survival_probs = np.minimum.accumulate(survival_probs)
    extended = np.concatenate(([1.0], survival_probs))
    interval_probs = extended[:-1] - extended[1:]
    interval_probs[interval_probs < 0] = 0
    prob_survive_past_end = survival_probs[-1]
    pmf = np.concatenate((interval_probs, [prob_survive_past_end]))

    s = pmf.sum()
    if s > 1e-9:
        pmf /= s
    else:
        return times[-1] + 1e-6

    time_bins = np.concatenate((times, [times[-1] + 1e-6]))
    return float(rng.choice(time_bins, p=pmf))


def _sample_time_conditionally(
    times: np.ndarray, survival_probs: np.ndarray, conditioning_time: float, rng: np.random.Generator
) -> float:
    start = np.searchsorted(times, conditioning_time, side="right")
    if start >= len(times):
        return float(times[-1] + 1e-6)

    tail_times = times[start:]
    tail_probs = survival_probs[start:]
    prob_survive = survival_probs[start - 1] if start > 0 else 1.0
    if prob_survive <= 1e-9:
        return float(tail_times[-1] + 1e-6)

    cond_probs = tail_probs / prob_survive
    return _sample_time_from_survival_curve(tail_times, cond_probs, rng)


def _generate_null_nonparametric(
    df: pd.DataFrame,
    t_col: str,
    e_col: str,
    x_cols: List[str],
    rng: np.random.Generator,
    rsf_params: Optional[dict] = None,
) -> pd.DataFrame:
    df_for_fit = pd.get_dummies(df, columns=[c for c in x_cols if df[c].dtype == "object"], drop_first=True)
    fit_cols = [
        c
        for c in df_for_fit.columns
        if c not in [t_col, e_col] and (c in df.columns or c.startswith(tuple(x_cols)))
    ]

    scaler = StandardScaler()
    x_features = scaler.fit_transform(df_for_fit[fit_cols])

    if rsf_params is None:
        rsf_params = {"n_estimators": 100, "min_samples_leaf": 15, "n_jobs": -1}

    def structured_y(time_col: pd.Series, event_col: pd.Series) -> np.ndarray:
        return np.array(list(zip(event_col.astype(bool), time_col.astype(float))), dtype=[("status", bool), ("time", float)])

    y_E = structured_y(df[t_col], df[e_col])
    y_C = structured_y(df[t_col], 1 - df[e_col])

    model_E = RandomSurvivalForest(**rsf_params, random_state=int(rng.integers(1_000_000)))
    model_C = RandomSurvivalForest(**rsf_params, random_state=int(rng.integers(1_000_000)))
    model_E.fit(x_features, y_E)
    model_C.fit(x_features, y_C)

    surv_E = model_E.predict_survival_function(x_features)
    surv_C = model_C.predict_survival_function(x_features)

    n = len(df)
    E_full = np.zeros(n)
    C_full = np.zeros(n)

    for i in range(n):
        t_obs = df.iloc[i][t_col]
        e_obs = df.iloc[i][e_col]
        if e_obs == 1:
            E_full[i] = t_obs
            sf_C = surv_C[i]
            C_full[i] = _sample_time_conditionally(sf_C.x, sf_C.y, t_obs, rng)
        else:
            C_full[i] = t_obs
            sf_E = surv_E[i]
            E_full[i] = _sample_time_conditionally(sf_E.x, sf_E.y, t_obs, rng)

    C_perm = C_full.copy()
    for _, idx in df.groupby(x_cols).groups.items():
        idx = idx.to_numpy()
        C_perm[idx] = rng.permutation(C_full[idx])

    df_null = df[x_cols].copy()
    df_null[t_col] = np.minimum(E_full, C_perm)
    df_null[e_col] = (E_full <= C_perm).astype(int)
    return df_null


def _stratified_fisher_test_standardized_strata(
    df: pd.DataFrame,
    times: Iterable[float],
    x_cols: List[str],
    B: int,
    seed: int,
    min_stratum_size: int,
    variance_threshold: float,
    t_col: str,
    e_col: str,
) -> Dict[str, Any]:
    """
    Stratified test using Fisher's method to combine p-values across strata, with standardized strata across permutations.

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

    Output:
    - dict with final p-value, observed Fisher statistic, per-stratum p-values, and excluded strata
    """
    rng = np.random.default_rng(seed)
    times = list(times)
    n_times = len(times)
    n_total = len(df)
    eps = 1e-18

    strata_groups = {k: grp for k, grp in df.groupby(x_cols) if len(grp) >= min_stratum_size}
    if not strata_groups:
        raise ValueError(f"No strata with size >= {min_stratum_size}. Reduce min_stratum_size or change strata.")

    unique_strata = list(strata_groups.keys())

    permuted_dfs = [
        _generate_null_nonparametric(df, t_col=t_col, e_col=e_col, x_cols=x_cols, rng=rng)
        for _ in range(B)
    ]

    per_s: Dict[Tuple, Dict[str, Any]] = {}

    for s in unique_strata:
        null_mat = np.zeros((B, n_times))
        obs_vec = np.zeros(n_times)

        for k, t in enumerate(times):
            obs_vec[k] = _get_delta_I(df, t, s, x_cols, n_total, t_col, e_col)
            for b in range(B):
                null_mat[b, k] = _get_delta_I(permuted_dfs[b], t, s, x_cols, n_total, t_col, e_col)

        sigma_all = null_mat.std(axis=0)
        valid_idx = np.where(sigma_all > variance_threshold)[0]
        if len(valid_idx) == 0:
            continue

        null_stable = null_mat[:, valid_idx]
        obs_stable = obs_vec[valid_idx]
        mu = null_stable.mean(axis=0)
        sigma = null_stable.std(axis=0)

        d_obs = np.abs((obs_stable - mu) / (sigma + eps))
        lambda_obs = float(np.max(d_obs)) if d_obs.size else 0.0

        d_perm = np.abs((null_stable - mu) / (sigma + eps))
        lambda_perm = np.max(d_perm, axis=1)

        p_s = (1 + np.sum(lambda_perm >= lambda_obs)) / (B + 1)
        per_s[s] = {"p_value": float(p_s), "lambda_obs": lambda_obs, "lambda_perm_dist": lambda_perm}

    if not per_s:
        return {"final_p_value": np.nan, "notes": "No stable strata found."}

    stable_p = np.array([v["p_value"] for v in per_s.values()])
    F_obs = float(-2 * np.sum(np.log(stable_p + 1e-12)))

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
