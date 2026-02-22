# dependent_censoring.py
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from collections import defaultdict
from scipy.stats import norm, levy_stable

from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1) DATA GENERATOR (PUBLIC)
# =============================================================================

def generate_survival_data(
    kind: Literal["copula_direct", "frailty_discrete", "frailty_continuous"] = "copula_direct",
    n_subjects: int = 1000,
    n_features: int = 3,
    seed: int = 42,
    # copula_direct params
    copula: Literal["gaussian", "clayton", "gumbel", "frank"] = "clayton",
    theta: float = 2.0,
    gamma: float = 0.0,
    event_params: Optional[Dict[str, Any]] = None,
    censoring_params: Optional[Dict[str, Any]] = None,
    # frailty params
    alpha_E: float = 1.0,
    alpha_C: float = 0.0,
    censoring_rate: float = 0.3,
    # continuous discretization
    n_bins: int = 2,
) -> pd.DataFrame:
    """
    Public generator that returns a DataFrame with required columns:
      - observed_time
      - event_indicator
    plus covariates x0..x{p-1} (or binned x0..), and optional latent times.
    """
    if kind == "copula_direct":
        return _generate_direct_dependence_data(
            n_subjects=n_subjects,
            n_features=n_features,
            copula=copula,
            theta=theta,
            gamma=gamma,
            event_params=event_params,
            censoring_params=censoring_params,
            seed=seed,
        )
    elif kind == "frailty_discrete":
        return _generate_dependent_via_frailty_mod(
            n_subjects=n_subjects,
            n_features=n_features,
            alpha_E=alpha_E,
            alpha_C=alpha_C,
            censoring_rate=censoring_rate,
            seed=seed,
        )
    elif kind == "frailty_continuous":
        return _generate_dependent_continuous_features(
            n_subjects=n_subjects,
            n_features=n_features,
            alpha_E=alpha_E,
            alpha_C=alpha_C,
            censoring_rate=censoring_rate,
            n_bins=n_bins,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown kind={kind}")


# --- Copula + direct dependence (from your code; cleaned) ---

def _generate_direct_dependence_data(
    n_subjects: int = 1000,
    n_features: int = 3,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"] = "gaussian",
    theta: float = 2.0,
    gamma: float = 0.0,
    event_params: Optional[Dict[str, Any]] = None,
    censoring_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # covariates (binary)
    X = rng.integers(0, 2, size=(n_subjects, n_features))

    if event_params is None:
        event_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 100, "rho": 1.5}
    if censoring_params is None:
        censoring_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 150, "rho": 1.0}

    eps = 1e-12

    # sample copula uniforms u, v
    if copula == "gaussian":
        if not (-1 < theta < 1):
            raise ValueError("For gaussian copula, theta must be in (-1,1).")
        cov = np.array([[1.0, theta], [theta, 1.0]])
        Z = rng.standard_normal(size=(n_subjects, 2))
        corr_normals = Z @ np.linalg.cholesky(cov).T
        u, v = norm.cdf(corr_normals[:, 0]), norm.cdf(corr_normals[:, 1])

    elif copula == "clayton":
        if theta < 0:
            raise ValueError("For clayton copula, theta must be >= 0.")
        if theta == 0:
            u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        else:
            w = rng.gamma(1.0 / theta, 1.0, size=n_subjects)
            e = rng.exponential(scale=1.0, size=(n_subjects, 2))
            u = (1.0 + e[:, 0] / w) ** (-1.0 / theta)
            v = (1.0 + e[:, 1] / w) ** (-1.0 / theta)

    elif copula == "gumbel":
        if theta < 1:
            raise ValueError("For gumbel copula, theta must be >= 1.")
        if theta == 1:
            u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        else:
            alpha = 1.0 / theta
            scale = (np.cos(np.pi * alpha / 2.0)) ** (1.0 / alpha)
            S = levy_stable.rvs(alpha=alpha, beta=1.0, scale=scale, loc=0, size=n_subjects, random_state=rng)
            E1 = rng.exponential(scale=1.0, size=n_subjects)
            E2 = rng.exponential(scale=1.0, size=n_subjects)
            u = np.exp(-E1 / S)
            v = np.exp(-E2 / S)

    elif copula == "frank":
        if theta == 0:
            u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        elif theta > 0:
            w = rng.logseries(1.0 - np.exp(-theta), size=n_subjects)
            e1 = rng.exponential(scale=1.0, size=n_subjects)
            e2 = rng.exponential(scale=1.0, size=n_subjects)
            u = 1.0 - np.exp(-e1 / w)
            v = 1.0 - np.exp(-e2 / w)
        else:
            alpha = -theta
            w = rng.logseries(1.0 - np.exp(-alpha), size=n_subjects)
            e1 = rng.exponential(scale=1.0, size=n_subjects)
            e2 = rng.exponential(scale=1.0, size=n_subjects)
            u = 1.0 - np.exp(-e1 / w)
            v_pos = 1.0 - np.exp(-e2 / w)
            v = 1.0 - v_pos
    else:
        raise NotImplementedError(copula)

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    # latent event time
    g_E = np.exp(X @ event_params["beta"])
    T_E = event_params["scale"] * (-np.log(u) / g_E) ** (1.0 / event_params["rho"])

    # latent censoring time with *direct* dependence on T_E
    median_E = np.median(T_E)
    direct_dependence_factor = np.exp(-gamma * (T_E - median_E) / max(median_E, 1e-12))
    scale_C_individual = censoring_params["scale"] * direct_dependence_factor

    g_C = np.exp(X @ censoring_params["beta"])
    T_C = scale_C_individual * (-np.log(v) / g_C) ** (1.0 / censoring_params["rho"])

    observed_time = np.minimum(T_E, T_C)
    event_indicator = (T_E <= T_C).astype(int)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["subject_id"] = np.arange(n_subjects)
    df["observed_time"] = observed_time
    df["event_indicator"] = event_indicator
    df["true_event_time"] = T_E
    df["censoring_time"] = T_C
    return df


# --- Frailty generators (from your code; cleaned) ---

def _generate_dependent_via_frailty_mod(
    n_subjects: int,
    n_features: int,
    alpha_E: float = 1.0,
    alpha_C: float = 0.0,
    censoring_rate: float = 0.3,
    seed: int = 1,
) -> pd.DataFrame:
    main_rng = np.random.default_rng(seed)
    X = main_rng.integers(0, 2, size=(n_subjects, n_features))
    beta = main_rng.uniform(-0.5, 0.5, size=n_features)
    gamma = main_rng.uniform(-0.5, 0.5, size=n_features)
    Z = main_rng.normal(0, 1, size=n_subjects)
    lamE = np.exp(X @ beta + alpha_E * Z)

    def censoring_rate_for(baseline_C: float) -> float:
        temp_rng = np.random.default_rng(seed)
        lamC = baseline_C * np.exp(X @ gamma + alpha_C * Z)
        E_temp = temp_rng.exponential(scale=1 / lamE)
        C_temp = temp_rng.exponential(scale=1 / lamC)
        return (C_temp < E_temp).mean()

    lo, hi = 1e-3, 50.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        r = censoring_rate_for(mid)
        if r < censoring_rate:
            lo = mid
        else:
            hi = mid
    baseline_C = 0.5 * (lo + hi)

    lamC = baseline_C * np.exp(X @ gamma + alpha_C * Z)
    E = main_rng.exponential(scale=1 / lamE)
    C = main_rng.exponential(scale=1 / lamC)
    T = np.minimum(E, C)
    Delta = (E <= C).astype(int)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["subject_id"] = np.arange(n_subjects)
    df["true_event_time"] = E
    df["censoring_time"] = C
    df["observed_time"] = T
    df["event_indicator"] = Delta
    return df


def _generate_dependent_continuous_features(
    n_subjects: int,
    n_features: int,
    alpha_E: float = 1.0,
    alpha_C: float = 0.0,
    censoring_rate: float = 0.3,
    n_bins: int = 2,
    seed: int = 1,
) -> pd.DataFrame:
    main_rng = np.random.default_rng(seed)
    X_cont = main_rng.normal(0, 1, size=(n_subjects, n_features))
    beta = main_rng.uniform(-0.5, 0.5, size=n_features)
    gamma = main_rng.uniform(-0.5, 0.5, size=n_features)
    Z = main_rng.normal(0, 1, size=n_subjects)
    lamE = np.exp(X_cont @ beta + alpha_E * Z)

    def censoring_rate_for(baseline_C: float) -> float:
        temp_rng = np.random.default_rng(seed)
        lamC_temp = baseline_C * np.exp(X_cont @ gamma + alpha_C * Z)
        E_temp = temp_rng.exponential(scale=1 / lamE)
        C_temp = temp_rng.exponential(scale=1 / lamC_temp)
        return (C_temp < E_temp).mean()

    lo, hi = 1e-4, 100.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        r = censoring_rate_for(mid)
        if r < censoring_rate:
            lo = mid
        else:
            hi = mid
    baseline_C = 0.5 * (lo + hi)

    lamC = baseline_C * np.exp(X_cont @ gamma + alpha_C * Z)
    E = main_rng.exponential(scale=1 / lamE)
    C = main_rng.exponential(scale=1 / lamC)
    T = np.minimum(E, C)
    Delta = (E <= C).astype(int)

    X_binned = np.zeros_like(X_cont, dtype=int)
    for i in range(n_features):
        X_binned[:, i] = pd.qcut(X_cont[:, i], q=n_bins, labels=False, duplicates="drop")

    df = pd.DataFrame({**{f"x{i}": X_binned[:, i] for i in range(n_features)},
                       **{f"x{i}_continuous": X_cont[:, i] for i in range(n_features)}})
    df["subject_id"] = np.arange(n_subjects)
    df["true_event_time"] = E
    df["censoring_time"] = C
    df["observed_time"] = T
    df["event_indicator"] = Delta
    return df


# =============================================================================
# 2) DEPENDENT CENSORING DETECTION (PUBLIC)
# =============================================================================

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
    Public function required by your GitHub users.

    Inputs (as you requested):
      df, quantiles, B, seed, min_stratum_size, variance_threshold

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


def load_csv(path: str) -> pd.DataFrame:
    """Convenience: user loads CSV then passes df into detect_dependent_censoring."""
    return pd.read_csv(path)


# =============================================================================
# INTERNALS (test + null world)  -- based on your code, lightly cleaned
# =============================================================================

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


def _compute_counts_for_time(df: pd.DataFrame, t: float, x_cols: List[str],
                             t_col: str, e_col: str) -> Dict[Tuple, Dict[str, Any]]:
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


def _get_delta_I(df: pd.DataFrame, t: float, stratum_key: Tuple, x_cols: List[str],
                 n_total: int, t_col: str, e_col: str) -> float:
    counts = _compute_counts_for_time(df, t, x_cols, t_col, e_col).get(stratum_key)
    if counts and counts["N"] > 0:
        _, delta_I = _compute_stratum_cmi_bounds(counts, n_total)
        return delta_I
    return 0.0


def _sample_time_from_survival_curve(times: np.ndarray, survival_probs: np.ndarray, rng: np.random.Generator) -> float:
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


def _sample_time_conditionally(times: np.ndarray, survival_probs: np.ndarray, conditioning_time: float,
                               rng: np.random.Generator) -> float:
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
    fit_cols = [c for c in df_for_fit.columns if c not in [t_col, e_col] and (c in df.columns or c.startswith(tuple(x_cols)))]

    scaler = StandardScaler()
    x_features = scaler.fit_transform(df_for_fit[fit_cols])

    if rsf_params is None:
        rsf_params = {"n_estimators": 100, "min_samples_leaf": 15, "n_jobs": -1}

    def structured_y(time_col: pd.Series, event_col: pd.Series) -> np.ndarray:
        return np.array(list(zip(event_col.astype(bool), time_col.astype(float))),
                        dtype=[("status", bool), ("time", float)])

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
        t_obs, e_obs = df.loc[i, t_col], df.loc[i, e_col]
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
    rng = np.random.default_rng(seed)
    times = list(times)
    n_times = len(times)
    n_total = len(df)
    eps = 1e-18

    strata_groups = {k: grp for k, grp in df.groupby(x_cols) if len(grp) >= min_stratum_size}
    if not strata_groups:
        raise ValueError(f"No strata with size >= {min_stratum_size}. Reduce min_stratum_size or change strata.")

    unique_strata = list(strata_groups.keys())

    # null datasets
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