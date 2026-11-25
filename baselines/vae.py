"""Gaussian VAE baseline for MNIST (paper Table 2 "VAE", Fig. 6c/6g, Fig. 7a, Sec. 4 run time).

This is NOT the authors' code (their baselines are not in the repository). What the paper states
about the VAE, and how it is implemented here:
  - posterior and likelihood are Gaussians with diagonal covariance whose means and covariances are
    MLP outputs (App. B.1, B.3)                    -> GaussianEncoder / GaussianDecoder
  - the prior is fixed (Sec. 4: "we restrict the prior to be fixed")   -> N(0, I)
  - softplus activations, 64-dim latent (App. B.3)  -> model.fct, flow.z_dim in the config
  - comparable size/cost to SFA (Sec. 4 compares their s/epoch; SFA has 2.4M parameters)
                                                    -> vae.hidden / vae.depth give ~2.4M
Everything the paper does not state (data split, logit-space preprocessing, optimizer, LR schedule,
batch size, epochs, gradient clipping, checkpointing, evaluation) is shared with the SFA runs: the
VAE is trained and evaluated by the authors' own LightningModule (runner_continuous.py), with only
the loss swapped for the negative ELBO. To make that possible the encoder/decoder expose the same
methods as SFA's posterior (GaussianPrior: sample, log_prob(x, z)) and likelihood (LLK:
decode(noise, z), log_prob(x, z, t, source)).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from models.nn import MLP
from runner_continuous import LightningModule, MNISTRunner, get_mnist, DataLoader
from utils import get_activation, find_checkpoint


class GaussianEncoder(nn.Module):
    """q(z|x) = N(mu(x), diag sigma(x)^2); same interface as sfa.GaussianPrior."""

    def __init__(self, x_features, z_features, hidden_features, fct, min_std):
        super().__init__()
        self.net = MLP(x_features, 2 * z_features, hidden_features=hidden_features, fct=fct)
        self.min_std = min_std

    def forward(self, x, t=None):
        mu, s = self.net(x.flatten(start_dim=1)).chunk(2, dim=-1)
        return Normal(mu, F.softplus(s) + self.min_std)

    def rsample(self, x, t=None):
        return self(x).rsample()

    def sample(self, x, t=None):
        return self(x).sample()

    def log_prob(self, x, z, t=None):
        return self(x).log_prob(z).sum(-1)


class GaussianDecoder(nn.Module):
    """p(x|z) = N(mu(z), diag sigma(z)^2) over the logit-transformed pixels; same interface as LLK."""

    def __init__(self, x_shape, z_features, hidden_features, fct, min_std):
        super().__init__()
        self.x_shape = tuple(x_shape)
        self.net = MLP(z_features, 2 * math.prod(self.x_shape), hidden_features=hidden_features, fct=fct)
        self.min_std = min_std

    def forward(self, z):
        mu, s = self.net(z).chunk(2, dim=-1)
        return Normal(mu.view(-1, *self.x_shape), (F.softplus(s) + self.min_std).view(-1, *self.x_shape))

    def decode(self, noise, z, t=None, s=None):
        """x ~ p(x|z), reparameterised with the given N(0, I) noise. For SFA the same argument is
        the flow's starting point x1 ~ N(0, I), so the authors' sampling code works unchanged."""
        d = self(z)
        return d.loc + d.scale * noise.reshape(d.loc.shape)

    def log_prob(self, x, z, t=0, source=None, s=None):
        """log p(x|z), summed over pixels (t / source / s only mirror LLK.log_prob's signature)."""
        lp = self(z).log_prob(x.reshape(-1, *self.x_shape))
        return lp.flatten(start_dim=1).sum(-1)


class NegativeELBO(nn.Module):
    """-E_q[log p(x|z)] + beta * KL(q(z|x) || N(0, I)), averaged over the batch (nats per image)."""

    def __init__(self, vt, rt, beta=1.0):
        super().__init__()
        self.vt, self.rt, self.beta = vt, rt, beta

    def forward(self, x):
        q = self.rt(x)
        z = q.rsample()
        rec = self.vt.log_prob(x, z)
        kl = kl_divergence(q, Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))).sum(-1)
        return (-rec + self.beta * kl).mean()


class VAELightningModule(LightningModule):
    """The authors' continuous-latent LightningModule with the loss swapped for the negative ELBO.
    Training step, optimizer, LR schedule, clipping, validation and test code are inherited."""

    def setup(self, stage=None):
        super().setup(stage)  # builds priory (and an SFA loss that is replaced below)
        self.flow_matching_loss = NegativeELBO(self.vt, self.rt, beta=vae_cfg(self.config, "beta", 1.0))


def vae_cfg(config, key, default):
    return getattr(getattr(config, "vae", None), key, default)


class VAERunner(MNISTRunner):
    """MNISTRunner (same data, trainer, callbacks, checkpointing) with a VAE instead of vt / rt."""

    def __init__(self, args, config):
        super().__init__(args, config)  # builds SFA's vt / rt, replaced below
        c, p, d = self.c, self.p, self.d
        hidden = [vae_cfg(config, "hidden", 630)] * vae_cfg(config, "depth", 2)
        fct = getattr(config.model, "fct", "softplus")
        min_std = vae_cfg(config, "min_std", 1e-5)
        self.rt = GaussianEncoder(c * p * p, d, hidden, get_activation(fct), min_std).to(config.device)
        self.vt = GaussianDecoder((c, p, p), d, hidden, get_activation(fct), min_std).to(config.device)

    def train(self):
        dataset, val_dataset, sampler, val_sampler = get_mnist(
            self.config.data.n_classes, "data", self.config.data.samplesize, self.config.data.test_samplesize)
        train_dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size,
                                      num_workers=self.config.data.num_workers, sampler=sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.training.batch_size,
                                    num_workers=self.config.data.num_workers, sampler=val_sampler, drop_last=True)
        model = VAELightningModule(self.vt, self.rt, self.priorz, self.config, self.args)
        ckpt_path = find_checkpoint(self.args.log_path, self.checkpoint_callback) if self.args.resume_training else None
        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
