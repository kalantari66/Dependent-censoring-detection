# Dependent-Censoring-Detection

A Python implementation for detecting **dependent censoring** in right-censored survival data using a stratified permutation-based global test.

This repository provides:

- ✅ Synthetic survival data generator  
- ✅ Dependent censoring detection function  
- ✅ Global p-value output  
- ✅ Automatic covariate stratification  
- ✅ CSV-ready workflow  

---

## Installation

Python >= 3.10 is recommended.

For local development in this repository:

```bash
pip install -e .
```

For a direct dependency-only install:

```bash
pip install -r requirements.txt
```

Future PyPI target package name: `cmi`.

---

## Project Structure

```text
.
├── src/
│   └── cmi/
│       ├── __init__.py
│       └── cmi.py               # package API and internal detection logic
├── data/
│   ├── __init__.py
│   ├── data_generation.py       # synthetic data generation utilities
│   ├── semi_synth_generation.py # semi-synthetic dataset generation utilities
│   └── real_data.py             # real-world dataset loaders
├── pyproject.toml
├── requirements.txt
└── env.yaml
```

---

# Default Dataset Format

By default, `detect_dependent_censoring()` expects:

- `observed_time` → positive survival/censoring time  
- `event_indicator` → 0 (censored) or 1 (event)  
- All other columns are automatically treated as covariates (strata variables)

If your dataset uses different column names such as `time` and `event`, pass them explicitly with
`t_col="time"` and `e_col="event"`.

Example structure:

| x0 | x1 | x2 | observed_time | event_indicator |
|----|----|----|---------------|-----------------|
| 0  | 1  | 0  | 12.5          | 1               |
| 1  | 0  | 1  | 7.3           | 0               |

---

# Usage

## 1️⃣ Real Data (CSV Input)

```python
import pandas as pd
from cmi import detect_dependent_censoring

df = pd.read_csv("mydata.csv")

p_global = detect_dependent_censoring(
    df,
    quantiles=[0.25, 0.5, 0.75],
    B=500,
    seed=123,
    min_stratum_size=20,
    variance_threshold=1e-4
)

print("Global p-value:", p_global)
```

---

## 2️⃣ Synthetic Data Example

This generator module is kept in-repo for experiments and is not part of the `cmi` package API.

```python
from cmi import detect_dependent_censoring
from data import dgp

df = dgp(
    kind="copula_discrete",
    n_subjects=500,
    n_features=3,
    copula="clayton",
    theta=4.0,
    gamma=0.5,
    seed=1
)

p_global = detect_dependent_censoring(
    df,
    quantiles=[0.3, 0.5, 0.7, 0.9],
    B=200,
    seed=123,
    min_stratum_size=30,
    variance_threshold=1e-3,
    t_col="time",
    e_col="event",
)

print("Global p-value:", p_global)
```

---

# Interpretation

Common rule of thumb:

- **p < 0.05** → Evidence against conditional independence (dependent censoring detected)  
- **p ≥ 0.05** → No strong evidence against the independence assumption  

---

# Main Function

```python
detect_dependent_censoring(
    df,
    quantiles,
    B=500,
    seed=123,
    min_stratum_size=30,
    variance_threshold=1e-9,
    t_col="observed_time",
    e_col="event_indicator",
    x_cols=None,
    return_details=False,
    verbose=False,
)
```

### Arguments

| Argument | Description |
|----------|------------|
| `df` | pandas DataFrame |
| `quantiles` | List of time quantiles to evaluate |
| `B` | Number of permutation samples |
| `seed` | Random seed |
| `min_stratum_size` | Minimum size per covariate stratum |
| `variance_threshold` | Minimum null variance for stability |
| `t_col` | Time column name; defaults to `observed_time` |
| `e_col` | Event indicator column name; defaults to `event_indicator` |
| `x_cols` | Optional covariate columns to stratify on |
| `return_details` | Return the full results dictionary instead of just the global p-value |
| `verbose` | Print progress and diagnostic messages |

### Output

Returns:

```
float  → Global p-value
```

---

# Synthetic Data Generator

```python
from data import dgp

dgp(
    kind="copula_discrete",
    n_subjects=1000,
    n_features=3,
    seed=42,
    ...
)
```

Available generator types:

- `"copula_discrete"`  
- `"copula_continuous"`  
- `"frailty_discrete"`  
- `"frailty_continuous"`  

---

# Computational Notes

- The method is permutation-based.
- Runtime increases with:
  - Larger `B`
  - Larger sample size
  - More strata
- For exploratory analysis, start with `B=200`.
- For publication-level results, consider `B ≥ 1000`.

---

# Citation

If you use this software in academic work, please cite:

```
()
```

---

# Author

Hamid Kalantari
University of Alberta  
2026
