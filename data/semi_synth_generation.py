from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .data_generation import sample_copula_uniform_pairs
from .real_data import load_real_data


def semiDGP(
    dataset: str,
    kind: Literal["copula", "frailty"],
    model: Literal["coxph", "rsf"],
    seed: int = 42,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"] = "clayton",
    theta: float = 2.0,
    drop_cov: int = 1,
    tail_strategy: Literal["last_segment_hazard", "global_exponential", "linear"] = "last_segment_hazard",
    eps: float = 1e-12,
    **model_params,
) -> pd.DataFrame:
    """Generate a semi-synthetic survival dataset from a real covariate table.

    Args:
        dataset: Real dataset name to use.
        kind: Dependence construction, either ``"copula"`` or ``"frailty"``.
        model: Survival model family, either ``"coxph"`` or ``"rsf"``.
        seed: Random seed for model fitting and latent time sampling.
        copula: Copula family used when ``kind="copula"``.
        theta: Copula parameter used when ``kind="copula"``.
        drop_cov: Number of top-ranked raw covariates to drop per model when
            ``kind="frailty"``.
        tail_strategy: Extrapolation rule used beyond the support of predicted
            survival curves.
        model_params: Optional estimator-specific overrides.

    Returns:
        A DataFrame with original covariate names plus ``time``, ``event``.
    """
    raw_df = load_real_data(dataset, onehot_encode=True)
    raw_covariates = raw_df.drop(columns=["time", "event"], errors="ignore")
    covariate_cols = raw_covariates.columns.tolist()

    event_model, censoring_model = _fit_event_and_censoring_models(
        df=raw_df,
        model=model,
        seed=seed,
        **model_params
    )

    event_survival_functions = event_model.predict_survival_function(raw_covariates)
    censoring_survival_functions = censoring_model.predict_survival_function(raw_covariates)

    dropped_features: set[str] = set()
    if kind == "frailty":
        dropped_features = _select_features_to_drop(
            event_model=event_model,
            censoring_model=censoring_model,
            X=raw_covariates,
            T=raw_df["time"],
            E=raw_df["event"],
            drop_cov=drop_cov,
            seed=seed,
        )
        kept_covariates = [col for col in covariate_cols if col not in dropped_features]
        if not kept_covariates:
            raise ValueError("Dropping the selected covariates would remove all covariate columns. "
                             "Consider reducing drop_cov or using kind='copula' instead.")
        true_event_time, censoring_time = _sample_times_from_independent_survivals(
            event_survival_functions=event_survival_functions,
            censoring_survival_functions=censoring_survival_functions,
            rng=np.random.default_rng(seed),
            tail_strategy=tail_strategy,
            eps=eps,
        )
    elif kind == "copula":
        kept_covariates = covariate_cols
        true_event_time, censoring_time = _sample_times_from_copula_survivals(
            event_survival_functions=event_survival_functions,
            censoring_survival_functions=censoring_survival_functions,
            rng=np.random.default_rng(seed),
            copula=copula,
            theta=theta,
            tail_strategy=tail_strategy,
        )
    else:
        raise ValueError(f"Unknown kind={kind}")

    observed_time = np.minimum(true_event_time, censoring_time)
    event_indicator = (true_event_time <= censoring_time).astype(int)

    result = raw_df[kept_covariates].copy()
    result["time"] = observed_time
    result["event"] = event_indicator
    result = result[result["time"] > 0]
    return result.reset_index(drop=True)


def _build_model(model: Literal["coxph", "rsf"], seed: int, **model_params) -> Any:
    """Instantiate one survival model with stable defaults."""
    if model == "coxph":
        return CoxPHSurvivalAnalysis(**model_params)
    elif model == "rsf":
        return RandomSurvivalForest(**model_params, random_state=seed)
    else:
        raise ValueError(f"Unknown model={model}")


def _fit_event_and_censoring_models(
    df: pd.DataFrame,
    model: Literal["coxph", "rsf"],
    seed: int,
    **model_params
) -> tuple[Any, Any]:
    """Fit paired survival models for the event and censoring processes."""
    X = df.drop(columns=["time", "event"], errors="ignore")
    y_event = Surv.from_arrays(time=df["time"], event=df["event"])
    y_censor = Surv.from_arrays(time=df["time"], event=1 - df["event"])

    event_model = _build_model(model=model, seed=seed, **model_params).fit(X, y_event)
    censoring_model = _build_model(model=model, seed=seed, **model_params).fit(X, y_censor)
    return event_model, censoring_model


def _estimate_global_exponential_hazard(times: np.ndarray, survival_probs: np.ndarray, eps: float = 1e-12) -> float:
    """Estimate a constant tail hazard from the endpoint survival probability."""
    t_last = times[-1]
    s_last = np.clip(survival_probs[-1], eps, 1.0 - eps)
    hazard = -np.log(s_last) / max(t_last, eps)
    return max(hazard, eps)


def _estimate_last_segment_hazard(times: np.ndarray, survival_probs: np.ndarray, eps: float = 1e-12) -> float:
    """Estimate the final-interval hazard for exponential tail continuation."""
    if len(times) < 2:
        return _estimate_global_exponential_hazard(times, survival_probs, eps)

    delta_t = times[-1] - times[-2]
    if delta_t <= 0:
        return _estimate_global_exponential_hazard(times, survival_probs, eps)

    s_prev = np.clip(survival_probs[-2], eps, 1.0 - eps)
    s_last = np.clip(survival_probs[-1], eps, 1.0 - eps)
    hazard = -np.log(s_last / s_prev) / delta_t
    if hazard <= 0:
        return _estimate_global_exponential_hazard(times, survival_probs, eps)
    return hazard


def _extrapolate_survival_tail(
    times: np.ndarray,
    survival_probs: np.ndarray,
    u: float,
    tail_strategy: Literal["last_segment_hazard", "global_exponential", "linear"],
    eps: float = 1e-12,
) -> float:
    """Extrapolate the inverse survival time beyond the final observed knot."""
    t_last = times[-1]
    s_last = survival_probs[-1]

    if tail_strategy == "last_segment_hazard":
        hazard = _estimate_last_segment_hazard(times, survival_probs, eps)
        return t_last + np.log(s_last / u) / hazard
    elif tail_strategy == "global_exponential":
        hazard = _estimate_global_exponential_hazard(times, survival_probs, eps)
        return t_last + np.log(s_last / u) / hazard
    elif tail_strategy == "linear":
        slope = min((s_last - 1.0) / max(t_last, eps), -eps)
        zero_crossing = -1.0 / slope
        target_time = (u - 1.0) / slope
        return min(max(target_time, t_last), zero_crossing)
    else:
        raise ValueError(f"Unknown tail_strategy={tail_strategy}")


def _invert_survival_curve(
    times: np.ndarray,
    survival_probs: np.ndarray,
    u: float,
    tail_strategy: Literal["last_segment_hazard", "global_exponential", "linear"],
    eps: float = 1e-12,
) -> float:
    """Map a latent uniform draw into a survival time using the fitted curve."""
    u = float(np.clip(u, eps, 1.0 - eps))
    crossing_idx = np.flatnonzero(survival_probs <= u)
    if crossing_idx.size > 0:
        return float(times[int(crossing_idx[0])])
    else:
        return _extrapolate_survival_tail(times, survival_probs, u=u, tail_strategy=tail_strategy, eps=eps)


def _sample_times_from_uniforms(
    event_survival_functions: np.ndarray,
    censoring_survival_functions: np.ndarray,
    u_event: np.ndarray,
    u_censoring: np.ndarray,
    tail_strategy: Literal["last_segment_hazard", "global_exponential", "linear"],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent event and censoring times given survival-function draws."""
    n_samples = len(event_survival_functions)
    true_event_times = np.empty(n_samples, dtype=float)
    censoring_times = np.empty(n_samples, dtype=float)

    for idx in range(n_samples):
        event_times = event_survival_functions[idx].x
        event_probs = event_survival_functions[idx].y
        censor_times = censoring_survival_functions[idx].x
        censor_probs = censoring_survival_functions[idx].y
        true_event_times[idx] = _invert_survival_curve(
            event_times,
            event_probs,
            u=u_event[idx],
            tail_strategy=tail_strategy,
        )
        censoring_times[idx] = _invert_survival_curve(
            censor_times,
            censor_probs,
            u=u_censoring[idx],
            tail_strategy=tail_strategy,
        )

    return true_event_times, censoring_times


def _sample_times_from_independent_survivals(
    event_survival_functions: np.ndarray,
    censoring_survival_functions: np.ndarray,
    rng: np.random.Generator,
    tail_strategy: Literal["last_segment_hazard", "global_exponential", "linear"],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample event and censoring times independently from fitted survival curves."""
    n_samples = len(event_survival_functions)
    u_event = np.clip(rng.uniform(size=n_samples), eps, 1.0 - eps)
    u_censoring = np.clip(rng.uniform(size=n_samples), eps, 1.0 - eps)
    return _sample_times_from_uniforms(
        event_survival_functions=event_survival_functions,
        censoring_survival_functions=censoring_survival_functions,
        u_event=u_event,
        u_censoring=u_censoring,
        tail_strategy=tail_strategy,
    )


def _sample_times_from_copula_survivals(
    event_survival_functions: np.ndarray,
    censoring_survival_functions: np.ndarray,
    rng: np.random.Generator,
    copula: Literal["gaussian", "clayton", "gumbel", "frank"],
    theta: float,
    tail_strategy: Literal["last_segment_hazard", "global_exponential", "linear"],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample event and censoring times using paired copula uniforms."""
    u, v = sample_copula_uniform_pairs(
        rng=rng,
        n_subjects=len(event_survival_functions),
        copula=copula,
        theta=theta,
    )
    return _sample_times_from_uniforms(
        event_survival_functions=event_survival_functions,
        censoring_survival_functions=censoring_survival_functions,
        u_event=u,
        u_censoring=v,
        tail_strategy=tail_strategy,
    )


def _rank_raw_features(
    fitted_model: Any,
    X: pd.DataFrame,
    T: pd.Series,
    E: pd.Series,
    seed: int,
) -> list[str]:
    """Rank raw features by aggregated model importance."""
    if isinstance(fitted_model, CoxPHSurvivalAnalysis):
        importances = pd.Series(np.abs(fitted_model.coef_), index=X.columns, dtype=float)
    elif isinstance(fitted_model, RandomSurvivalForest):
        y = Surv.from_arrays(time=T, event=E)
        permutation = permutation_importance(
            estimator=fitted_model,
            X=X,
            y=y,
            n_repeats=5,
            random_state=seed,
            n_jobs=1,
        )
        importances = pd.Series(np.abs(permutation.importances_mean), index=X.columns, dtype=float)
    else:
        raise ValueError(f"Unsupported model type for feature importance: {type(fitted_model)}")

    ranked = importances.sort_values(ascending=False, kind="stable")
    return ranked.index.tolist()


def _select_features_to_drop(
    event_model: Any,
    censoring_model: Any,
    X: pd.DataFrame,
    T: pd.Series,
    E: pd.Series,
    drop_cov: int,
    seed: int,
) -> set[str]:
    """Select the union of top-ranked event and censoring features to hide."""
    ranked_event = _rank_raw_features(
        fitted_model=event_model,
        X=X,
        T=T,
        E=E,
        seed=seed,
    )
    ranked_censor = _rank_raw_features(
        fitted_model=censoring_model,
        X=X,
        T=T,
        E=1 - E,
        seed=seed,
    )
    return set(ranked_event[:drop_cov]) | set(ranked_censor[:drop_cov])
