"""Latent flow matching baseline (Dao et al., 2023).

An autoencoder maps x to a latent z, and an unconditional CNF trained with flow
matching learns the latent distribution p(z). Training has two stages:

    1. autoencoder only                       (`baseline.ae_epochs` epochs)
    2. autoencoder and latent flow together   (remaining epochs)

The flow matching loss sees detached latents: letting it update the encoder
would reward shrinking z towards a point instead of learning p(z).

Two variants:
    LatentFM        deterministic MLP encoder / decoder, MSE reconstruction
    LatentFM w/VAE  Gaussian encoder / decoder (the VAE of benchmarks/vae.py)

Optional config entries (section `baseline:`):
    ae_epochs    length of stage 1 (default: half of training.n_epochs)
    flow_depth   hidden layers of the latent vector field (default 3)
    beta, min_std   as in benchmarks/vae.py, for the w/VAE variant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchdiffeq import odeint

from models.nn import MLP
from benchmarks.common import (
    BaselineModule, BaselineRunner, baseline_opt, decoder_net, encoder_net)
from benchmarks.vae import VAE


class LatentFlow(nn.Module):
    """Unconditional vector field on z. As in SFA, t=0 is data and t=1 is N(0, I)."""

    def __init__(self, z_dim, hidden_features, fct, freqs=2, sig_min=1e-4):
        super().__init__()
        self.z_dim = z_dim
        self.sig_min = sig_min
        self.net = MLP(z_dim + 2 * freqs, z_dim, hidden_features=hidden_features, fct=fct)
        self.register_buffer("freqs", torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t, z):
        t = self.freqs.to(z.dtype) * t[..., None]
        temb = torch.cat((t.cos(), t.sin()), dim=-1).expand(*z.shape[:-1], -1)
        return self.net(torch.cat((z, temb), dim=-1))

    def fm_loss(self, z):
        t = torch.rand(len(z), device=z.device, dtype=z.dtype)
        eps = torch.randn_like(z)
        zt = (1 - t[:, None]) * z + (self.sig_min + (1 - self.sig_min) * t[:, None]) * eps
        ut = (1 - self.sig_min) * eps - z
        return (self(t, zt) - ut).square().mean()

    def _t_span(self, t0, t1, z):
        return torch.tensor([t0, t1], device=z.device, dtype=z.dtype)

    @torch.no_grad()
    def sample(self, n):
        z1 = torch.randn(n, self.z_dim, device=self.freqs.device, dtype=self.net[0].weight.dtype)
        return odeint(self, z1, self._t_span(1.0, 0.0, z1), atol=1e-5, rtol=1e-5)[-1]

    def log_prob(self, z):
        """log p(z) by the instantaneous change of variables, with the exact divergence."""
        def augmented(t, state):
            with torch.enable_grad():
                zt = state[0].detach().requires_grad_(True)
                dz = self(t, zt)
                div = sum(
                    torch.autograd.grad(dz[:, i].sum(), zt, retain_graph=True)[0][:, i]
                    for i in range(self.z_dim))
            return dz.detach(), div.detach()

        z1, ladj = odeint(
            augmented, (z, z.new_zeros(len(z))), self._t_span(0.0, 1.0, z),
            atol=1e-5, rtol=1e-5)
        return Normal(0.0, 1.0).log_prob(z1[-1]).sum(-1) + ladj[-1]


class LatentFMModule(BaselineModule):
    probabilistic = False  # True: LatentFM w/VAE

    def __init__(self, config, args, spec):
        super().__init__(config, args, spec)
        n_epochs = config.training.n_epochs
        self.ae_epochs = baseline_opt(config, "ae_epochs", n_epochs // 2)

        if self.probabilistic:
            self.beta = baseline_opt(config, "beta", 1.0)
            self.vae = VAE(spec, min_std=baseline_opt(config, "min_std", 1e-3))
        else:
            self.encoder = encoder_net(spec, spec.z_dim)
            self.decoder = decoder_net(spec, spec.x_dim)

        self.flow = LatentFlow(
            spec.z_dim,
            hidden_features=[config.flow.ngf] * baseline_opt(config, "flow_depth", 3),
            fct=spec.fct, freqs=max(getattr(config.flow, "freqs", 2), 2))

    def stage_epochs(self):
        return [self.ae_epochs, self.config.training.n_epochs - self.ae_epochs]

    def loss(self, x):
        if self.probabilistic:
            log_lik, kl, z = self.vae.elbo_terms(x)
            ae_loss = (self.beta * kl - log_lik).mean() / self.spec.x_dim
        else:
            z = self.encoder(x)
            ae_loss = F.mse_loss(self.decoder(z), x)
        fm_loss = self.flow.fm_loss(z.detach())

        train_loss = ae_loss + fm_loss if self.current_epoch >= self.ae_epochs else ae_loss
        # Validation always includes the flow term, so checkpoint selection on
        # val_loss cannot prefer a stage-1 model whose flow is untrained.
        return train_loss, ae_loss + fm_loss, {"ae_loss": ae_loss, "fm_loss": fm_loss}

    def posterior(self, x):
        if self.probabilistic:
            q = self.vae.encoder(x)
            z = q.sample()
            return z, q.log_prob(z).sum(-1)
        return self.encoder(x), None

    def regenerate(self, z):
        if self.probabilistic:
            return self.vae.decoder(z).sample()
        return self.decoder(z)

    def generate(self, n):
        return self.regenerate(self.flow.sample(n))

    def extra_test_metrics(self, x, z):
        return {"log_prior_z": self.flow.log_prob(z).mean().item()}


class LatentFMVAEModule(LatentFMModule):
    probabilistic = True


class HVGRunner(BaselineRunner):
    dataset, module_cls = "hvg", LatentFMModule


class PinWheelRunner(BaselineRunner):
    dataset, module_cls = "pinwheel", LatentFMModule


class MNISTRunner(BaselineRunner):
    dataset, module_cls = "mnist", LatentFMModule


class HVGVAERunner(BaselineRunner):
    dataset, module_cls = "hvg", LatentFMVAEModule


class PinWheelVAERunner(BaselineRunner):
    dataset, module_cls = "pinwheel", LatentFMVAEModule


class MNISTVAERunner(BaselineRunner):
    dataset, module_cls = "mnist", LatentFMVAEModule
