from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import levy_stable, norm


def dgp(
    kind: Literal["copula", "frailty_discrete", "frailty_continuous"] = "copula",
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
      - time
      - event
      - x0..x{p-1} strata covariates
    """
    if kind == "copula_discrete":
        return generate_direct_dependence_data(
            n_subjects=n_subjects,
            n_features=n_features,
            copula=copula,
            theta=theta,
            gamma=gamma,
            event_params=event_params,
            censoring_params=censoring_params,
            seed=seed,
        )
    elif kind == "copula_continuous":
        return generate_copula_continuous_features(
            n_subjects=n_subjects,
            n_features=n_features,
            copula=copula,
            theta=theta,
            gamma=gamma,
            n_bins=n_bins,
            event_params=event_params,
            censoring_params=censoring_params,
            seed=seed,
        )
    elif kind == "frailty_discrete":
        return generate_dependent_via_frailty_mod(
            n_subjects=n_subjects,
            n_features=n_features,
            alpha_E=alpha_E,
            alpha_C=alpha_C,
            censoring_rate=censoring_rate,
            seed=seed,
        )
    elif kind == "frailty_continuous":
        return generate_dependent_continuous_features(
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


def sample_copula_uniform_pairs(
    rng: np.random.Generator,
    n_subjects: int,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"],
    theta: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample dependent uniform pairs from the requested copula."""
    if copula == "gaussian":
        if not (-1 < theta < 1):
            raise ValueError("For gaussian copula, theta must be in (-1,1).")
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
            s = levy_stable.rvs(alpha=alpha, beta=1.0, scale=scale, loc=0, size=n_subjects, random_state=rng)
            e1 = rng.exponential(scale=1.0, size=n_subjects)
            e2 = rng.exponential(scale=1.0, size=n_subjects)
            u = np.exp(-e1 / s)
            v = np.exp(-e2 / s)

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
    return u, v


def generate_direct_dependence_data(
    n_subjects: int = 1000,
    n_features: int = 3,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"] = "gaussian",
    theta: float = 2.0,
    gamma: float = 0.0,
    event_params: Optional[Dict[str, Any]] = None,
    censoring_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    eps: float = 1e-12
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_subjects, n_features))

    if event_params is None:
        event_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 100, "rho": 1.5}
    if censoring_params is None:
        censoring_params = {"beta": rng.uniform(-0.5, 0.5, n_features), "scale": 150, "rho": 1.0}

    u, v = sample_copula_uniform_pairs(rng=rng, n_subjects=n_subjects, copula=copula, theta=theta, eps=eps)

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
    df["time"] = observed_time
    df["event"] = event_indicator
    return df


def generate_copula_continuous_features(
    n_subjects: int = 1000,
    n_features: int = 3,
    copula: Literal['gaussian', 'clayton', 'gumbel', 'frank'] = 'gaussian',
    theta: float = 2.0,
    gamma: float = 0.0,
    n_bins: int = 4,
    event_params: Optional[Dict[str, Any]] = None,
    censoring_params: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates survival data with copula-based dependence and continuous features,
    which are then discretized into bins for the final output.

    The underlying data generation process uses the continuous features. The final
    DataFrame includes both the original continuous features and the binned versions.

    Args:
        n_subjects: Number of subjects to generate.
        n_features: Number of continuous features.
        copula: The copula family to use ('gaussian', 'clayton', 'gumbel', 'frank').
        theta: Copula dependence parameter.
        gamma: Direct dependence strength. If gamma > 0, shorter event times lead to
               shorter censoring times.
        n_bins: The number of quantile-based bins to categorize continuous features into.
        event_params: Parameters for the event time Weibull distribution.
        censoring_params: Parameters for the censoring time Weibull distribution.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with generated data, containing both 'x_continuous'
        and 'x_binned' columns for each feature.
    """
    rng = np.random.default_rng(seed)

    # === STEP 1: Generate CONTINUOUS features from a standard normal distribution ===
    X_continuous = rng.normal(0, 1, size=(n_subjects, n_features))

    # --- Defaults ---
    if event_params is None:
        event_params = {'beta': rng.uniform(-0.5, 0.5, n_features), 'scale': 100, 'rho': 1.5}
    if censoring_params is None:
        censoring_params = {'beta': rng.uniform(-0.5, 0.5, n_features), 'scale': 150, 'rho': 1.0}

    # === STEP 2: Generate correlated uniforms (u, v) using the chosen copula ===
    eps = 1e-12
    if copula == 'gaussian':
        if not -1 < theta < 1: raise ValueError("For Gaussian, theta must be in (-1, 1).")
        cov = np.array([[1.0, theta], [theta, 1.0]])
        Z = rng.standard_normal(size=(n_subjects, 2))
        corr_normals = Z @ np.linalg.cholesky(cov).T
        u, v = norm.cdf(corr_normals[:, 0]), norm.cdf(corr_normals[:, 1])

    elif copula == 'clayton':
        if theta < 0: raise ValueError("For Clayton, theta must be >= 0.")
        if theta == 0: u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        else:
            w = rng.gamma(1.0 / theta, 1.0, size=n_subjects)
            e = rng.exponential(scale=1.0, size=(n_subjects, 2))
            u, v = (1.0 + e[:, 0] / w) ** (-1.0 / theta), (1.0 + e[:, 1] / w) ** (-1.0 / theta)
            
    elif copula == 'gumbel':
        if theta < 1: raise ValueError("For Gumbel, theta must be >= 1.")
        if theta == 1: u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        else:
            alpha = 1.0 / theta
            scale = (np.cos(np.pi * alpha / 2.0))**(1.0 / alpha)
            S = levy_stable.rvs(alpha=alpha, beta=1.0, scale=scale, loc=0, size=n_subjects, random_state=rng)
            E1, E2 = rng.exponential(scale=1.0, size=n_subjects), rng.exponential(scale=1.0, size=n_subjects)
            u, v = np.exp(-E1 / S), np.exp(-E2 / S)

    elif copula == 'frank':
        if theta == 0: u, v = rng.uniform(size=n_subjects), rng.uniform(size=n_subjects)
        elif theta > 0:
            w = rng.logseries(1.0 - np.exp(-theta), size=n_subjects)
            e1, e2 = rng.exponential(scale=1.0, size=n_subjects), rng.exponential(scale=1.0, size=n_subjects)
            u, v = 1.0 - np.exp(-e1 / w), 1.0 - np.exp(-e2 / w)
        else: # theta < 0
            alpha = -theta
            w = rng.logseries(1.0 - np.exp(-alpha), size=n_subjects)
            e1, e2 = rng.exponential(scale=1.0, size=n_subjects), rng.exponential(scale=1.0, size=n_subjects)
            u = 1.0 - np.exp(-e1 / w)
            v_pos = 1.0 - np.exp(-e2 / w)
            v = 1.0 - v_pos
    else:
        raise NotImplementedError(f"Copula '{copula}' is not implemented.")
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    # --- Generate latent times using the CONTINUOUS covariates ---
    g_E = np.exp(X_continuous @ event_params['beta'])
    T_E = event_params['scale'] * (-np.log(u) / g_E) ** (1.0 / event_params['rho'])
    
    median_E = np.median(T_E)
    direct_dependence_factor = np.exp(gamma * (T_E - median_E) / median_E)
    scale_C_individual = censoring_params['scale'] * direct_dependence_factor
    g_C = np.exp(X_continuous @ censoring_params['beta'])
    T_C = scale_C_individual * (-np.log(v) / g_C) ** (1.0 / censoring_params['rho'])

    # === STEP 3: Construct the observed data ===
    observed_time = np.minimum(T_E, T_C)
    event_indicator = (T_E <= T_C).astype(int)

    # === STEP 4: Discretize the continuous features into bins ===
    X_binned = np.zeros_like(X_continuous, dtype=int)
    for i in range(n_features):
        X_binned[:, i] = pd.qcut(X_continuous[:, i], q=n_bins, labels=False, duplicates='drop')

    # === STEP 5: Create the final DataFrame with both feature sets ===
    binned_cols = {f'x{i}_binned': X_binned[:, i] for i in range(n_features)}
    continuous_cols = {f'x{i}_continuous': X_continuous[:, i] for i in range(n_features)}

    df = pd.DataFrame({**binned_cols, **continuous_cols})
    df['time']   = observed_time
    df['event'] = event_indicator

    return df


def generate_dependent_via_frailty_mod(
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
    df["time"] = T
    df["event"] = Delta
    return df


def generate_dependent_continuous_features(
    n_subjects: int,
    n_features: int,
    alpha_E: float = 1.0,
    alpha_C: float = 0.0,
    censoring_rate: float = 0.3,
    n_bins: int = 2,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Generates a survival dataset with dependent censoring via a shared frailty term.

    This modified version first generates continuous features from a standard normal
    distribution. These continuous features are used in the underlying data generating
    process for event and censoring times. Finally, it discretizes these continuous
    features into a specified number of bins for inclusion in the final DataFrame,
    simulating a scenario where a continuous process is modeled using categorical data.

    Args:
        n_subjects: The number of subjects in the dataset.
        n_features: The number of continuous features to generate.
        alpha_E: The coefficient for the frailty term in the event model.
        alpha_C: The coefficient for the frailty term in the censoring model.
                 If alpha_C > 0 and alpha_E > 0, dependence is induced.
                 If alpha_C = 0, event and censoring are conditionally independent.
        censoring_rate: The target proportion of subjects who are censored.
        n_bins: The number of quantile-based bins to categorize continuous features into.
        seed: The random seed for reproducibility.

    Returns:
        A pandas DataFrame containing the generated survival data. The DataFrame includes
        both the original continuous features ('x_continuous') and the binned
        categorical features ('x_binned').
    """

    main_rng = np.random.default_rng(seed)

    # 1. Generate CONTINUOUS features from a standard normal distribution
    X_cont = main_rng.normal(0, 1, size=(n_subjects, n_features))
    
    # Coefficients for the continuous features
    beta = main_rng.uniform(-0.5, 0.5, size=n_features)
    gamma = main_rng.uniform(-0.5, 0.5, size=n_features)

    # Shared frailty term to induce dependence
    Z = main_rng.normal(0, 1, size=n_subjects)

    # 2. Use CONTINUOUS features to define the hazard rates
    lamE = np.exp(X_cont @ beta + alpha_E * Z)

    # Helper function to find the baseline censoring rate that achieves the target
    def censoring_rate_for(baseline_C: float) -> float:
        # Note: This uses a temporary RNG to not affect the main sequence
        temp_rng = np.random.default_rng(seed)
        lamC_temp = baseline_C * np.exp(X_cont @ gamma + alpha_C * Z)
        E_temp = temp_rng.exponential(scale=1 / lamE)
        C_temp = temp_rng.exponential(scale=1 / lamC_temp)
        return (C_temp < E_temp).mean()

    # Binary search to find the correct baseline_C for the target censoring rate
    lo, hi = 1e-4, 100.0
    for _ in range(30): # Increased iterations for better precision
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        r = censoring_rate_for(mid)
        if r < censoring_rate:
            lo = mid
        else:
            hi = mid
    baseline_C = 0.5 * (lo + hi)

    # Generate final event and censoring times
    lamC = baseline_C * np.exp(X_cont @ gamma + alpha_C * Z)
    E = main_rng.exponential(scale=1 / lamE)
    C = main_rng.exponential(scale=1 / lamC)
    T = np.minimum(E, C)
    Delta = (E <= C).astype(int)

    # 3. Discretize the continuous features into bins
    X_binned = np.zeros_like(X_cont, dtype=int)
    for i in range(n_features):
        # Use qcut for quantile-based bins. `labels=False` gives integer codes.
        # `duplicates='drop'` handles cases where quantiles are not unique.
        X_binned[:, i] = pd.qcut(X_cont[:, i], q=n_bins, labels=False, duplicates="drop")

    # 4. Create the final DataFrame
    # Create column names for both binned and continuous features
    df = pd.DataFrame(
        {
            **{f"x{i}": X_binned[:, i] for i in range(n_features)},
            **{f"x{i}_continuous": X_cont[:, i] for i in range(n_features)},
        }
    )
    df["time"] = T
    df["event"] = Delta
    return df
