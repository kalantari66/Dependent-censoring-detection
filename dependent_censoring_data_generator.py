"""
Synthetic data generator for dependent-censoring experiments.

This module creates survival datasets compatible with:
`dependent_censoring _detection_function.py`

Default output columns are detector-friendly:
- observed_time
- event_indicator
- x0, x1, ... (discrete covariates)
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import levy_stable, norm


def _bin_continuous_features(x_continuous: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin each column into integer categories."""
    x_binned = np.zeros_like(x_continuous, dtype=int)
    for i in range(x_continuous.shape[1]):
        x_binned[:, i] = pd.qcut(
            x_continuous[:, i],
            q=n_bins,
            labels=False,
            duplicates="drop",
        ).astype(int)
    return x_binned


def generate_copula_continuous_features(
    n_subjects: int = 1000,
    n_features: int = 3,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"] = "gaussian",
    theta: float = 0.4,
    gamma: float = 0.0,
    n_bins: int = 4,
    event_params: Optional[Dict[str, Any]] = None,
    censoring_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    include_latent_columns: bool = False,
    include_continuous_features: bool = False,
) -> pd.DataFrame:
    """
    Generate survival data with copula-based dependence and optional direct dependence.

    Notes:
    - `gamma > 0` introduces direct dependence by making censoring scale depend on event time.
    - Default output is detector-friendly (discrete x columns only).
    """
    rng = np.random.default_rng(seed)
    x_continuous = rng.normal(0.0, 1.0, size=(n_subjects, n_features))

    if event_params is None:
        event_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 100.0, "rho": 1.5}
    if censoring_params is None:
        censoring_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 150.0, "rho": 1.0}

    eps = 1e-12
    if copula == "gaussian":
        if not (-1.0 < theta < 1.0):
            raise ValueError("For gaussian copula, theta must be in (-1, 1).")
        cov = np.array([[1.0, theta], [theta, 1.0]])
        z = rng.standard_normal(size=(n_subjects, 2))
        corr_normals = z @ np.linalg.cholesky(cov).T
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
            s = levy_stable.rvs(
                alpha=alpha,
                beta=1.0,
                scale=scale,
                loc=0.0,
                size=n_subjects,
                random_state=rng,
            )
            e1 = rng.exponential(scale=1.0, size=n_subjects)
            e2 = rng.exponential(scale=1.0, size=n_subjects)
            u, v = np.exp(-e1 / s), np.exp(-e2 / s)
    elif copula == "frank":
        if theta == 0:
            u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        elif theta > 0:
            w = rng.logseries(1.0 - np.exp(-theta), size=n_subjects)
            e1 = rng.exponential(scale=1.0, size=n_subjects)
            e2 = rng.exponential(scale=1.0, size=n_subjects)
            u, v = 1.0 - np.exp(-e1 / w), 1.0 - np.exp(-e2 / w)
        else:
            alpha = -theta
            w = rng.logseries(1.0 - np.exp(-alpha), size=n_subjects)
            e1 = rng.exponential(scale=1.0, size=n_subjects)
            e2 = rng.exponential(scale=1.0, size=n_subjects)
            u = 1.0 - np.exp(-e1 / w)
            v_pos = 1.0 - np.exp(-e2 / w)
            v = 1.0 - v_pos
    else:
        raise ValueError(f"Unsupported copula: {copula}")

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    g_e = np.exp(x_continuous @ np.asarray(event_params["beta"]))
    t_e = float(event_params["scale"]) * (-np.log(u) / g_e) ** (1.0 / float(event_params["rho"]))

    median_e = np.median(t_e)
    direct_factor = np.exp(gamma * (t_e - median_e) / (median_e + eps))
    scale_c_individual = float(censoring_params["scale"]) * direct_factor
    g_c = np.exp(x_continuous @ np.asarray(censoring_params["beta"]))
    t_c = scale_c_individual * (-np.log(v) / g_c) ** (1.0 / float(censoring_params["rho"]))

    observed_time = np.minimum(t_e, t_c)
    event_indicator = (t_e <= t_c).astype(int)

    x_binned = _bin_continuous_features(x_continuous, n_bins=n_bins)

    data = {f"x{i}": x_binned[:, i] for i in range(n_features)}
    df = pd.DataFrame(data)
    df["observed_time"] = observed_time
    df["event_indicator"] = event_indicator

    if include_continuous_features:
        for i in range(n_features):
            df[f"x{i}_continuous"] = x_continuous[:, i]

    if include_latent_columns:
        df["true_event_time"] = t_e
        df["censoring_time"] = t_c

    cols = ["observed_time", "event_indicator"] + [f"x{i}" for i in range(n_features)]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def generate_dependent_via_frailty_mod(
    n_subjects: int = 1000,
    n_features: int = 3,
    alpha_e: float = 1.0,
    alpha_c: float = 0.0,
    censoring_rate: float = 0.3,
    seed: int = 1,
    include_latent_columns: bool = False,
) -> pd.DataFrame:
    """
    Generate binary-feature survival data with shared frailty dependence.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_subjects, n_features))
    beta = rng.uniform(-0.5, 0.5, size=n_features)
    gamma = rng.uniform(-0.5, 0.5, size=n_features)
    z = rng.normal(0.0, 1.0, size=n_subjects)

    lam_e = np.exp(x @ beta + alpha_e * z)

    def censoring_rate_for(baseline_c: float) -> float:
        temp_rng = np.random.default_rng(seed)
        lam_c_tmp = baseline_c * np.exp(x @ gamma + alpha_c * z)
        e_tmp = temp_rng.exponential(scale=1.0 / lam_e)
        c_tmp = temp_rng.exponential(scale=1.0 / lam_c_tmp)
        return float((c_tmp < e_tmp).mean())

    lo, hi = 1e-3, 50.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        r = censoring_rate_for(mid)
        if r < censoring_rate:
            lo = mid
        else:
            hi = mid
    baseline_c = 0.5 * (lo + hi)

    lam_c = baseline_c * np.exp(x @ gamma + alpha_c * z)
    e = rng.exponential(scale=1.0 / lam_e)
    c = rng.exponential(scale=1.0 / lam_c)
    t = np.minimum(e, c)
    d = (e <= c).astype(int)

    df = pd.DataFrame({f"x{i}": x[:, i] for i in range(n_features)})
    df["observed_time"] = t
    df["event_indicator"] = d

    if include_latent_columns:
        df["true_event_time"] = e
        df["censoring_time"] = c

    cols = ["observed_time", "event_indicator"] + [f"x{i}" for i in range(n_features)]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def generate_dependent_continuous_features(
    n_subjects: int = 1000,
    n_features: int = 3,
    alpha_e: float = 1.0,
    alpha_c: float = 0.0,
    censoring_rate: float = 0.3,
    n_bins: int = 4,
    seed: int = 1,
    include_latent_columns: bool = False,
    include_continuous_features: bool = False,
) -> pd.DataFrame:
    """
    Generate continuous-feature frailty data, then discretize to x0..xK.
    """
    rng = np.random.default_rng(seed)
    x_continuous = rng.normal(0.0, 1.0, size=(n_subjects, n_features))
    beta = rng.uniform(-0.5, 0.5, size=n_features)
    gamma = rng.uniform(-0.5, 0.5, size=n_features)
    z = rng.normal(0.0, 1.0, size=n_subjects)

    lam_e = np.exp(x_continuous @ beta + alpha_e * z)

    def censoring_rate_for(baseline_c: float) -> float:
        temp_rng = np.random.default_rng(seed)
        lam_c_tmp = baseline_c * np.exp(x_continuous @ gamma + alpha_c * z)
        e_tmp = temp_rng.exponential(scale=1.0 / lam_e)
        c_tmp = temp_rng.exponential(scale=1.0 / lam_c_tmp)
        return float((c_tmp < e_tmp).mean())

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
    baseline_c = 0.5 * (lo + hi)

    lam_c = baseline_c * np.exp(x_continuous @ gamma + alpha_c * z)
    e = rng.exponential(scale=1.0 / lam_e)
    c = rng.exponential(scale=1.0 / lam_c)
    t = np.minimum(e, c)
    d = (e <= c).astype(int)

    x_binned = _bin_continuous_features(x_continuous, n_bins=n_bins)
    df = pd.DataFrame({f"x{i}": x_binned[:, i] for i in range(n_features)})
    df["observed_time"] = t
    df["event_indicator"] = d

    if include_continuous_features:
        for i in range(n_features):
            df[f"x{i}_continuous"] = x_continuous[:, i]

    if include_latent_columns:
        df["true_event_time"] = e
        df["censoring_time"] = c

    cols = ["observed_time", "event_indicator"] + [f"x{i}" for i in range(n_features)]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for dependent-censoring tests.")
    parser.add_argument("--generator", choices=["copula", "frailty_binary", "frailty_continuous"], default="copula")
    parser.add_argument("--out", required=True, help="Output CSV path")

    parser.add_argument("--n-subjects", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--copula", choices=["gaussian", "clayton", "gumbel", "frank"], default="gaussian")
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--n-bins", type=int, default=4)

    parser.add_argument("--alpha-e", type=float, default=1.0)
    parser.add_argument("--alpha-c", type=float, default=0.0)
    parser.add_argument("--censoring-rate", type=float, default=0.3)

    parser.add_argument("--include-latent-columns", action="store_true")
    parser.add_argument("--include-continuous-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.generator == "copula":
        df = generate_copula_continuous_features(
            n_subjects=args.n_subjects,
            n_features=args.n_features,
            copula=args.copula,
            theta=args.theta,
            gamma=args.gamma,
            n_bins=args.n_bins,
            seed=args.seed,
            include_latent_columns=args.include_latent_columns,
            include_continuous_features=args.include_continuous_features,
        )
    elif args.generator == "frailty_binary":
        df = generate_dependent_via_frailty_mod(
            n_subjects=args.n_subjects,
            n_features=args.n_features,
            alpha_e=args.alpha_e,
            alpha_c=args.alpha_c,
            censoring_rate=args.censoring_rate,
            seed=args.seed,
            include_latent_columns=args.include_latent_columns,
        )
    else:
        df = generate_dependent_continuous_features(
            n_subjects=args.n_subjects,
            n_features=args.n_features,
            alpha_e=args.alpha_e,
            alpha_c=args.alpha_c,
            censoring_rate=args.censoring_rate,
            n_bins=args.n_bins,
            seed=args.seed,
            include_latent_columns=args.include_latent_columns,
            include_continuous_features=args.include_continuous_features,
        )

    df.to_csv(args.out, index=False)
    print(f"Wrote dataset: {args.out}")
    print(f"Shape: {df.shape}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()

