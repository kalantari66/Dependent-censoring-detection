import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cmi.DVFM import DVFM, train_dvfm


class SurvivalDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.time = torch.tensor(time, dtype=torch.float32)
        self.event = torch.tensor(event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]


def sample_weibull_truncated(shape, scale, t_obs, rng):
    """Exact Inverse Transform Sampling for T > t_obs"""
    # Clip t_obs to avoid numerical issues if scale is very small
    t_safe = max(t_obs, 1e-6)
    survival_at_t = np.exp(-(t_safe / scale) ** shape)
    u = rng.random()
    # Sample from the tail: p is a random probability between 0 and S(t)
    p = u * survival_at_t
    p = np.clip(p, 1e-10, 1.0)
    return scale * (-np.log(p)) ** (1.0 / shape)


def prepare_null_nonparametric(df, t_col, e_col, x_cols, rng, rsf_params=None):
    """Pre-compute E_full and C_full ONCE using individual-specific imputation."""
    df = df.reset_index(drop=True)
    scaler = StandardScaler()
    # Dummy encoding and scaling
    df_fit = pd.get_dummies(df[x_cols], drop_first=True)
    x_features = scaler.fit_transform(df_fit)
    latent_dim = len(x_features)
    # Model parameters
    p = rsf_params or {}
    device = p.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    time_scale = float(df[t_col].max())
    
    dataset = SurvivalDataset(x_features, df[t_col].values / time_scale, df[e_col].values)
    loader = DataLoader(dataset, batch_size=p.get("batch_size", 64), shuffle=True)
    
    model = DVFM(input_dim=x_features.shape[1], latent_dim=p.get("latent_dim", latent_dim)).to(device)
    train_dvfm(model, loader, loader, n_epochs=p.get("n_epochs", 200), device=device)
    
    model.eval()
    n = len(df)
    E_full, C_full = np.zeros(n), np.zeros(n)
    
    with torch.no_grad():
        x_t = torch.tensor(x_features, dtype=torch.float32).to(device)
        t_t = torch.tensor(df[t_col].values / time_scale, dtype=torch.float32).to(device)
        e_t = torch.tensor(df[e_col].values, dtype=torch.float32).to(device)
        mu, logvar = model.encoder(x_t, t_t, e_t)
        
        for i in range(n):
            t_obs_raw = float(df.iloc[i][t_col])
            # Use patient's OWN mu/logvar for consistent trait imputation
            z = mu[i] + torch.exp(0.5 * logvar[i]) * torch.randn_like(mu[i])
            s_T, sc_T, s_C, sc_C = model.decoder(x_t[i:i+1], z.unsqueeze(0))
            
            # Convert to scalars and rescale time
            st, sct = s_T.item(), sc_T.item() * time_scale
            sc, scc = s_C.item(), sc_C.item() * time_scale
            
            if df.iloc[i][e_col] == 1:
                E_full[i] = t_obs_raw
                C_full[i] = sample_weibull_truncated(sc, scc, t_obs_raw, rng)
            else:
                C_full[i] = t_obs_raw
                E_full[i] = sample_weibull_truncated(st, sct, t_obs_raw, rng)

    return {
        "x_df": df[x_cols],
        "E_full": E_full,
        "C_full": C_full,
        "group_indices": [idx.to_numpy() for idx in df.groupby(x_cols).groups.values()]
    }


def generate_null_nonparametric_fast(pre, t_col, e_col, rng):
    """Fast permutation step (repeated 200x)"""
    C_perm = pre["C_full"].copy()
    for idx in pre["group_indices"]:
        C_perm[idx] = rng.permutation(pre["C_full"][idx])

    df_null = pre["x_df"].copy()
    df_null[t_col] = np.minimum(pre["E_full"], C_perm)
    df_null[e_col] = (pre["E_full"] <= C_perm).astype(int)
    return df_null


def generate_null_nonparametric(df, t_col, e_col, x_cols, rng, rsf_params=None):
    pre = prepare_null_nonparametric(df, t_col, e_col, x_cols, rng, rsf_params)
    return generate_null_nonparametric_fast(pre, t_col, e_col, rng)
