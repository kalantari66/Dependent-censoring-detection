from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def _ensure_binary_event(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
    else:
        normalized = s.astype("string").str.strip().str.lower()
        mapped = normalized.map(
            {
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "dead": 1,
                "alive": 0,
                "event": 1,
                "censored": 0,
            }
        )
        out = pd.to_numeric(mapped, errors="coerce")
    out = out.dropna()
    out = out[out.isin([0, 1])]
    return out.astype(int)


def _discretize_numeric(series: pd.Series, bins: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0, index=series.index, dtype=int)
    s = s.fillna(s.median())
    uniq = s.nunique(dropna=True)
    if uniq <= 1:
        return pd.Series(0, index=series.index, dtype=int)
    q = max(2, min(bins, uniq))
    try:
        out = pd.qcut(s, q=q, labels=False, duplicates="drop")
    except ValueError:
        out = pd.cut(s, bins=q, labels=False, duplicates="drop")
    if out.isna().all():
        return pd.Series(0, index=series.index, dtype=int)
    return out.fillna(0).astype(int)


def _encode_categorical(series: pd.Series, top_k: int = 20) -> pd.Series:
    s = series.astype("string").fillna("missing")
    counts = s.value_counts(dropna=False)
    if len(counts) > top_k:
        keep = set(counts.index[:top_k])
        s = s.where(s.isin(keep), "other")
    return s.astype("category").cat.codes.astype(int)


def preprocess_dataset(
    raw_df: pd.DataFrame,
    bins: int,
    max_features: int | None = None,
    event_col: str = "event_indicator",
    time_col: str = "observed_time",
    feature_exclude: List[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    df = raw_df.copy()
    if time_col not in df.columns or event_col not in df.columns:
        raise ValueError("Dataset must include time/event columns or observed_time/event_indicator columns.")

    # drop rows with missing or non-positive time, and ensure event indicator is binary
    df = df.replace([np.inf, -np.inf], np.nan)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col])
    df = df[df[time_col] > 0].copy()

    event_binary = _ensure_binary_event(df[event_col])
    df = df.loc[event_binary.index].copy()
    df[event_col] = event_binary.values

    if feature_exclude is None:
        feature_exclude = {time_col, event_col}
    else:
        feature_exclude = set(feature_exclude) + {time_col, event_col}
    feature_cols = [c for c in df.columns if c not in feature_exclude]

    # encode features and select top ones by cardinality
    encoded = pd.DataFrame(index=df.index)
    card: Dict[str, int] = {}
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(df[col]):
            series = df[col].astype(int)
        elif pd.api.types.is_numeric_dtype(df[col]):
            series = _discretize_numeric(df[col], bins=bins)
        else:
            series = _encode_categorical(df[col])
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            continue
        encoded[col] = series.astype(int)
        card[col] = n_unique

    if encoded.empty:
        raise ValueError("No usable covariates found after preprocessing.")

    # rank features by cardinality (ascending) and select top ones
    ranked = sorted(card.items(), key=lambda x: (x[1], x[0]))
    if max_features is not None and max_features < len(ranked):
        selected_raw = [name for name, _ in ranked[:max_features]]
    else:
        selected_raw = [name for name, _ in ranked]

    preprocessed_df = pd.DataFrame(index=df.index)
    preprocessed_df[time_col] = df[time_col].astype(float)
    preprocessed_df[event_col] = df[event_col].astype(int)
    preprocessed_df = pd.concat([preprocessed_df, encoded[selected_raw]], axis=1)

    return preprocessed_df.reset_index(drop=True), selected_raw
