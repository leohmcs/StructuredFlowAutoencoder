"""Mixture-SVAE (GMVAE) baseline for MNIST (paper Table 3, Fig. 4a, 6h, 7c, 8).

This is NOT the authors' code (their baselines are not in the repository). What the paper states
about Mixture-SVAE, and how it is implemented here:
  - graphical model of Sec. 3.2 / Fig. 2b: xi ~ Cat(pi), z | xi ~ p(z | xi), x | z ~ p(x | z)
  - "we restrict the prior to be fixed" (Sec. 4)  -> pi uniform, p(z | xi=k) = N(mu_k, I) with
    means mu_k drawn once (seeded) and never trained   (FixedGaussianMixturePrior). Their spread
    (mixture_svae.prior_scale) is not stated; widely separated components (scale 0.3 or 1.0)
    make q(xi|x) collapse onto one component, so the config uses 0.1.
  - "Gaussian distribution with diagonal covariance for posterior and likelihood family. MLPs are
    used to map context vectors to the means and covariances" (App. B.3)
        q(z | x, xi): the authors' GaussianMixtureComponent (linear embedding of xi, concatenated
                      with x, MLP to mean / std), conditioned on the xi sample and not sorted
        p(x | z):     GaussianDecoder from baselines/vae.py
  - "The latent class probability is via a Gumbel-Softmax network, which uses MLP to map from
    context vector to logits, then apply Gumbel-Softmax trick for sampling" (App. B.3)
        q(xi | x):    the authors' CatNF_fixed (the same Gumbel-Softmax network Mixture-SFA uses)
  - softplus activations, 64-dim latent, K = 10 (App. B.3, Sec. 4.2)
The authors' two networks take a time input (SFA posteriors are indexed by t); a VAE has no time, so
they are always evaluated at t = 0, which only adds a constant input.

Objective: single-sample relaxed ELBO (Jang et al., 2017)
    log p(x|z) - KL(q(z|x,xi) || p(z|xi)) - KL(q(xi|x) || Cat(1/K)),
with xi a Gumbel-Softmax sample, p(z|xi) = N(sum_k xi_k mu_k, I), the categorical KL in closed form.

Training (data, optimizer, LR schedule, clipping, checkpointing) and evaluation run through the
authors' Mixture-SFA LightningModule (runner_mixture.py) with the loss replaced, exactly like the
VAE baseline does with the continuous module.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from sfa_discrete import CatNF_fixed, GaussianMixtureComponent
from runner_mixture import FlowMatchingLightningModule, MNISTMixtureRunner, get_mnist, DataLoader
from utils import get_activation, find_checkpoint
from baselines.vae import GaussianDecoder


class FixedGaussianMixturePrior(nn.Module):
    """p(z | xi) = N(xi @ mu, I) with fixed component means mu (K x d); for a one-hot xi this is the
    k-th component. Same interface as the authors' GaussianMixturePrior."""

    def __init__(self, K, z_features, scale=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("means", scale * torch.randn(K, z_features, generator=g, dtype=torch.float64))

    def forward(self, pi):
        loc = pi @ self.means.to(pi.dtype)
        return Normal(loc, torch.ones_like(loc))

    def sample(self, pi, size):
        return self(pi).sample(size)[0]

    def rsample(self, pi, size):
        return self(pi).rsample(size)[0]

    def log_prob(self, pi, z):
        return self(pi).log_prob(z).sum(-1)


class MixtureNegativeELBO(nn.Module):
    """-[log p(x|z) - KL(q(z|x,xi) || p(z|xi)) - KL(q(xi|x) || uniform)], averaged over the batch."""

    def __init__(self, vt, Rt, rt, priorz, beta=1.0):
        super().__init__()
        self.vt, self.Rt, self.rt, self.priorz, self.beta = vt, Rt, rt, priorz, beta

    def forward(self, x):
        t0 = torch.zeros((), device=x.device, dtype=x.dtype)
        logits, xi = self.Rt.rsample(x, t0)           # q(xi|x) logits and a Gumbel-Softmax sample
        q_z = self.rt(xi, x, t0)                       # q(z | x, xi)
        z = q_z.rsample()
        rec = self.vt.log_prob(x, z)
        kl_z = kl_divergence(q_z, self.priorz(xi)).sum(-1)
        log_q_xi = F.log_softmax(logits, dim=-1)
        kl_xi = (log_q_xi.exp() * (log_q_xi + math.log(logits.shape[-1]))).sum(-1)
        return (-rec + self.beta * (kl_z + kl_xi)).mean()


def svae_cfg(config, key, default):
    return getattr(getattr(config, "mixture_svae", None), key, default)


class MixtureSVAELightningModule(FlowMatchingLightningModule):
    """The authors' Mixture-SFA LightningModule with the loss swapped for the mixture ELBO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_on_xi = True  # q(z | x, xi) is part of the model definition here

    def setup(self, stage=None):
        super().setup(stage)  # builds priorpi / priory (and an SFA loss that is replaced below)
        self.flow_matching_loss = MixtureNegativeELBO(
            self.vt, self.Rt, self.rt, self.priorz, beta=svae_cfg(self.config, "beta", 1.0))


class MixtureSVAERunner(MNISTMixtureRunner):
    """MNISTMixtureRunner (same data, trainer, callbacks, checkpointing) with the SVAE networks."""

    def __init__(self, args, config):
        super().__init__(args, config)  # builds Mixture-SFA's networks, replaced below
        k, d, c, p = self.k, self.d, self.c, self.p
        hidden = [svae_cfg(config, "hidden", 454)] * svae_cfg(config, "depth", 2)
        fct = getattr(config.model, "fct", "softplus")
        self.Rt = CatNF_fixed(c * p * p, k, hidden_features=hidden, temp=svae_cfg(config, "temp", config.flow.beta),
                              freqs=config.flow.freqs, fct=get_activation(fct), in_ch=c).to(config.device)
        self.rt = GaussianMixtureComponent(k, d, c * p * p, hidden_features=hidden, hidden_dim=c * p * p,
                                           fct=get_activation(fct), in_ch=c, freqs=config.flow.freqs,
                                           sort_pi=False).to(config.device)
        self.vt = GaussianDecoder((c, p, p), d, hidden, get_activation(fct),
                                  svae_cfg(config, "min_std", 1e-5)).to(config.device)
        self.priorz = FixedGaussianMixturePrior(k, d, scale=svae_cfg(config, "prior_scale", 0.1))

    def train(self):
        dataset, val_dataset, sampler, val_sampler = get_mnist(
            self.config.data.n_classes, "data", self.config.data.samplesize, self.config.data.test_samplesize)
        train_dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size,
                                      num_workers=self.config.data.num_workers, sampler=sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.training.batch_size,
                                    num_workers=self.config.data.num_workers, sampler=val_sampler, drop_last=True)
        model = MixtureSVAELightningModule(self.vt, self.Rt, self.rt, self.priorz, self.embx, self.config, self.args)
        ckpt_path = find_checkpoint(self.args.log_path, self.checkpoint_callback) if self.args.resume_training else None
        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
