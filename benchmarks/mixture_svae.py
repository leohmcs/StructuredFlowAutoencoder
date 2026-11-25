"""Mixture-SVAE (GMVAE) baseline (paper Table 1 / Fig. 1 on pinwheel, Table 3 / Fig. 4a, 6h, 7c, 8 on MNIST).

Not the authors' code (their baselines are not in the repository). What the paper states, and how it
is implemented here:
  - graphical model of Sec. 3.2 / Fig. 2b: xi ~ Cat(pi), z | xi ~ p(z | xi), x | z ~ p(x | z)
  - "we restrict the prior to be fixed" (Sec. 4) -> pi uniform, p(z | xi=k) = N(mu_k, prior_std^2 I)
    with means mu_k fixed at construction and never trained (FixedGaussianMixturePrior)
  - "Gaussian distribution with diagonal covariance for posterior and likelihood family. MLPs are
    used to map context vectors to the means and covariances" (App. B.3)
        q(z | x, xi): the authors' GaussianMixtureComponent (linear embedding of xi, concatenated
                      with x, MLP to mean / std), conditioned on the xi sample and not sorted
        p(x | z):     GaussianDecoder (benchmarks/common.py)
  - "The latent class probability is via a Gumbel-Softmax network, which uses MLP to map from
    context vector to logits, then apply Gumbel-Softmax trick for sampling" (App. B.3)
        q(xi | x):    the authors' CatNF_fixed (the same Gumbel-Softmax network Mixture-SFA uses)
The authors' two networks take a time input (SFA posteriors are indexed by t); a VAE has no time, so
they are always evaluated at t = 0, which only adds a constant input. Reusing them (and the SFA
interfaces of GaussianDecoder) lets reproduce_mnist.py evaluate this model with the authors'
Mixture-SFA test code.

Objective: single-sample relaxed ELBO (Jang et al., 2017)
    log p(x|z) - KL(q(z|x,xi) || p(z|xi)) - KL(q(xi|x) || Cat(1/K)),
with xi a Gumbel-Softmax sample, p(z|xi) = N(sum_k xi_k mu_k, prior_std^2 I), the categorical KL in
closed form; scaled per observed dimension like the VAE.

Optional config entries (section `baseline:` or `baseline.mixture_svae:`), besides the network
options listed in benchmarks/common.py (enc_* sizes both q(xi|x) and q(z|x,xi)):
    k            number of components (default flow.k_dim / flow.pi_dim, else data.n_classes)
    beta         weight of both KL terms (default 1.0)
    prior_means  'random': mu_k ~ N(0, prior_scale^2 I), drawn once with seed 0 (default)
                 'linspace': mu_k = (a_k, ..., a_k), a = linspace(-prior_scale, prior_scale, K)
    prior_scale  spread of the component means (default 0.1)
    prior_std    standard deviation of each component (default 1.0)
    temp         Gumbel-Softmax temperature (default flow.beta, as Mixture-SFA)
    temp_min     anneal the temperature geometrically from temp to temp_min over training
                 (default: temp, no annealing); evaluation uses temp_min
    xi_embed     width of the linear xi embedding in q(z|x,xi) (default: x dimension)
    min_std      lower bound on the decoder standard deviation (default 1e-5)
The paper does not state the prior's spread; on MNIST widely separated components (prior_scale 0.3
or 1.0) make q(xi|x) collapse onto one component, so configs/mnist_paper.yml uses 0.1.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sfa_discrete import CatNF_fixed, GaussianMixtureComponent
from utils import soft_nmi
from benchmarks.common import BaselineModule, BaselineRunner, GaussianDecoder


class FixedGaussianMixturePrior(nn.Module):
    """p(z | xi) = N(xi @ mu, std^2 I) with fixed component means mu (K x d); for a one-hot xi this
    is the k-th component. Same interface as the authors' GaussianMixturePrior."""

    def __init__(self, K, z_dim, means="random", scale=0.1, std=1.0, seed=0):
        super().__init__()
        if means == "random":
            g = torch.Generator().manual_seed(seed)
            mu = scale * torch.randn(K, z_dim, generator=g, dtype=torch.float64)
        elif means == "linspace":
            mu = torch.linspace(-scale, scale, K, dtype=torch.float64)[:, None].repeat(1, z_dim)
        else:
            raise ValueError(f"prior_means must be 'random' or 'linspace', not {means!r}")
        self.register_buffer("means", mu)
        self.std = std

    def forward(self, pi):
        loc = pi @ self.means.to(pi.dtype)
        return Normal(loc, torch.full_like(loc, self.std))

    def sample(self, pi, size):
        return self(pi).sample(size)[0]

    def rsample(self, pi, size):
        return self(pi).rsample(size)[0]

    def log_prob(self, pi, z):
        return self(pi).log_prob(z).sum(-1)


class MixtureSVAEModule(BaselineModule):
    name = "mixture_svae"

    def __init__(self, config, args, spec):
        super().__init__(config, args, spec)
        flow = config.flow
        self.k = self.opt("k", getattr(flow, "k_dim", getattr(flow, "pi_dim", spec.n_classes)))
        self.beta = self.opt("beta", 1.0)
        self.temp = self.opt("temp", getattr(flow, "beta", 1.0))
        self.temp_min = self.opt("temp_min", self.temp)
        x_dim, z_dim = spec.x_dim, spec.z_dim

        # attribute names follow the authors' Mixture-SFA module (Rt, rt, vt, priorz)
        self.Rt = CatNF_fixed(x_dim, self.k, temp=self.temp, freqs=2,
                              hidden_features=self.enc_hidden, fct=self.fct)
        self.rt = GaussianMixtureComponent(self.k, z_dim, x_dim, freqs=2,
                                           hidden_dim=self.opt("xi_embed", x_dim), sort_pi=False,
                                           hidden_features=self.enc_hidden, fct=self.fct)
        self.vt = GaussianDecoder(x_dim, z_dim, self.dec_hidden, self.fct, self.opt("min_std", 1e-5))
        self.priorz = FixedGaussianMixturePrior(
            self.k, z_dim, means=self.opt("prior_means", "random"),
            scale=self.opt("prior_scale", 0.1), std=self.opt("prior_std", 1.0))

    # ── temperature schedule ──
    def on_train_epoch_start(self):
        frac = self.current_epoch / max(self.config.training.n_epochs - 1, 1)
        self.Rt.temp = self.temp * (self.temp_min / self.temp) ** frac

    def on_test_start(self):
        self.Rt.temp = self.temp_min

    def _t0(self, x):
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def loss(self, x):
        t0 = self._t0(x)
        logits, xi = self.Rt.rsample(x, t0)        # q(xi|x) logits and a Gumbel-Softmax sample
        q_z = self.rt(xi, x, t0)                    # q(z | x, xi)
        z = q_z.rsample()
        log_lik = self.vt.log_prob(x, z)
        kl_z = kl_divergence(q_z, self.priorz(xi)).sum(-1)
        log_q_xi = F.log_softmax(logits, dim=-1)
        kl_xi = (log_q_xi.exp() * (log_q_xi + math.log(self.k))).sum(-1)
        loss = (self.beta * (kl_z + kl_xi) - log_lik).mean() / self.spec.x_dim
        logs = {"elbo": (log_lik - kl_z - kl_xi).mean(), "log_lik": log_lik.mean(),
                "kl_z": kl_z.mean(), "kl_xi": kl_xi.mean(), "temp": self.Rt.temp}
        return loss, loss, logs

    def posterior(self, x):
        t0 = self._t0(x)
        _, xi = self.Rt.rsample(x, t0)
        q = self.rt(xi, x, t0)
        z = q.sample()
        return z, q.log_prob(z).sum(-1)

    def regenerate(self, z):
        return self.vt(z).sample()

    def generate(self, n):
        dtype = next(self.parameters()).dtype
        xi = F.one_hot(torch.randint(self.k, (n,), device=self.device), self.k).to(dtype)
        return self.regenerate(self.priorz(xi).sample())

    def extra_test_metrics(self, x, z, y):
        """Clustering by the mixture component, argmax q(xi|x) (as Mixture-SFA's test_plot_latent)."""
        probs = self.Rt(x, self._t0(x)).detach().double().cpu().numpy()
        y_np = y.cpu().numpy()
        k_hat = probs.argmax(-1)
        m = {"nmi_xi": normalized_mutual_info_score(y_np, k_hat),
             "ari_xi": adjusted_rand_score(y_np, k_hat),
             "log_lik": self.vt.log_prob(x, z).mean().item()}
        if probs.shape[1] > y_np.max():  # soft_nmi one-hot encodes the labels with K columns
            m["soft_nmi"] = soft_nmi(probs, y_np)
        return m


class PinWheelRunner(BaselineRunner):
    dataset, module_cls = "pinwheel", MixtureSVAEModule


class MNISTRunner(BaselineRunner):
    dataset, module_cls = "mnist", MixtureSVAEModule
