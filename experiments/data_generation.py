"""Synthetic data generators for experiments and method checks."""

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import levy_stable, norm


def generate_survival_data(
    kind: Literal["copula_direct", "frailty_discrete", "frailty_continuous"] = "copula_direct",
    n_subjects: int = 1000,
    n_features: int = 3,
    seed: int = 42,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"] = "clayton",
    theta: float = 2.0,
    gamma: float = 0.0,
    event_params: Optional[Dict[str, Any]] = None,
    censoring_params: Optional[Dict[str, Any]] = None,
    alpha_E: float = 1.0,
    alpha_C: float = 0.0,
    censoring_rate: float = 0.3,
    n_bins: int = 2,
) -> pd.DataFrame:
    """
    Generate synthetic right-censored survival data with controllable dependence.

    Returns a DataFrame containing:
      - observed_time
      - event_indicator
      - x0..x{p-1} strata covariates
    and latent timing columns used for simulation diagnostics.
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
    if kind == "frailty_discrete":
        return _generate_dependent_via_frailty_mod(
            n_subjects=n_subjects,
            n_features=n_features,
            alpha_E=alpha_E,
            alpha_C=alpha_C,
            censoring_rate=censoring_rate,
            seed=seed,
        )
    if kind == "frailty_continuous":
        return _generate_dependent_continuous_features(
            n_subjects=n_subjects,
            n_features=n_features,
            alpha_E=alpha_E,
            alpha_C=alpha_C,
            censoring_rate=censoring_rate,
            n_bins=n_bins,
            seed=seed,
        )
    raise ValueError(f"Unknown kind={kind}")


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
    X = rng.integers(0, 2, size=(n_subjects, n_features))

    if event_params is None:
        event_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 100, "rho": 1.5}
    if censoring_params is None:
        censoring_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 150, "rho": 1.0}

    eps = 1e-12

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

    g_E = np.exp(X @ event_params["beta"])
    T_E = event_params["scale"] * (-np.log(u) / g_E) ** (1.0 / event_params["rho"])

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

    df = pd.DataFrame(
        {
            **{f"x{i}": X_binned[:, i] for i in range(n_features)},
            **{f"x{i}_continuous": X_cont[:, i] for i in range(n_features)},
        }
    )
    df["subject_id"] = np.arange(n_subjects)
    df["true_event_time"] = E
    df["censoring_time"] = C
    df["observed_time"] = T
    df["event_indicator"] = Delta
    return df
