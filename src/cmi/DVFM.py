import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim + 2  # X + time + event
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x, time, event):
        inputs = torch.cat([x, time.unsqueeze(1), event.unsqueeze(1)], dim=1)
        h = self.network(inputs)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[32, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim + latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.fc_params = nn.Linear(prev_dim, 4)

    def forward(self, x, z):
        inputs = torch.cat([x, z], dim=1)
        h = self.network(inputs)
        params = self.fc_params(h)
        # Softplus ensures positive parameters for Weibull
        shape_T = nn.functional.softplus(params[:, 0]) + 1e-2
        scale_T = nn.functional.softplus(params[:, 1]) + 1e-2
        shape_C = nn.functional.softplus(params[:, 2]) + 1e-2
        scale_C = nn.functional.softplus(params[:, 3]) + 1e-2
        return shape_T, scale_T, shape_C, scale_C


class DVFM(nn.Module):
    def __init__(self, input_dim, latent_dim=8, encoder_hidden=[64, 32], decoder_hidden=[32, 64]):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden)
        self.decoder = Decoder(input_dim, latent_dim, decoder_hidden)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def weibull_log_pdf(self, t, shape, scale):
        return torch.log(shape) - torch.log(scale) + (shape - 1) * (torch.log(t) - torch.log(scale)) - (t / scale) ** shape

    def weibull_log_survival(self, t, shape, scale):
        return -(t / scale) ** shape

    def forward(self, x, time, event):
        mu, logvar = self.encoder(x, time, event)
        z = self.reparameterize(mu, logvar)
        return self.decoder(x, z), mu, logvar

    def loss_function(self, params, mu, logvar, time, event, beta=1.0, free_bits=0.0):
        shape_T, scale_T, shape_C, scale_C = params
        log_f_T = self.weibull_log_pdf(time, shape_T, scale_T)
        log_S_T = self.weibull_log_survival(time, shape_T, scale_T)
        log_f_C = self.weibull_log_pdf(time, shape_C, scale_C)
        log_S_C = self.weibull_log_survival(time, shape_C, scale_C)

        recon_loss = (event * (log_f_T + log_S_C) + (1 - event) * (log_S_T + log_f_C)).mean()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return -recon_loss + beta * kl_div


def train_dvfm(model, train_loader, val_loader, n_epochs=200, lr=1e-3, beta_max=1.0, warmup_epochs=50, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(n_epochs):
        beta = min(beta_max, epoch / warmup_epochs) if warmup_epochs > 0 else beta_max
        model.train()
        for x, t, e in train_loader:
            x, t, e = x.to(device), t.to(device), e.to(device)
            optimizer.zero_grad()
            params, mu, logvar = model(x, t, e)
            loss = model.loss_function(params, mu, logvar, t, e, beta)
            loss.backward()
            optimizer.step()
