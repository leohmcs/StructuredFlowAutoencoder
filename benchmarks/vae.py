"""VAE baseline: diagonal Gaussian encoder q(z|x) and decoder p(x|z), N(0, I) prior.

Both networks are time-independent MLPs sized like the SFA posterior and vector
field of the same dataset (see benchmarks/common.py); configs/mnist_paper.yml
instead matches SFA's parameter count (2 x 630 hidden units, ~2.4M parameters).

Optional config entries (section `baseline:` or `baseline.vae:`), besides the
network options listed in benchmarks/common.py:
    beta      weight of the KL term (default 1.0)
    min_std   lower bound on the decoder standard deviation (default 1e-3)
    enc_min_std  lower bound on the encoder standard deviation (default 1e-5)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from benchmarks.common import BaselineModule, BaselineRunner, GaussianDecoder, GaussianEncoder


class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, enc_hidden, dec_hidden, fct, layer_norm=False,
                 min_std=1e-3, enc_min_std=1e-5):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = GaussianEncoder(x_dim, z_dim, enc_hidden, fct, layer_norm, enc_min_std)
        self.decoder = GaussianDecoder(x_dim, z_dim, dec_hidden, fct, min_std)

    @classmethod
    def from_module(cls, m):
        """Build from a BaselineModule's resolved options."""
        return cls(m.spec.x_dim, m.spec.z_dim, m.enc_hidden, m.dec_hidden, m.fct, m.layer_norm,
                   m.opt("min_std", 1e-3), m.opt("enc_min_std", 1e-5))

    def elbo_terms(self, x):
        """Per-sample log p(x|z) and KL(q(z|x) || p(z)) for one z ~ q(z|x), plus that z."""
        q = self.encoder(x)
        z = q.rsample()
        log_lik = self.decoder(z).log_prob(x).sum(-1)
        kl = kl_divergence(q, Normal(torch.zeros_like(z), torch.ones_like(z))).sum(-1)
        return log_lik, kl, z


class VAEModule(BaselineModule):
    name = "vae"

    def __init__(self, config, args, spec):
        super().__init__(config, args, spec)
        self.beta = self.opt("beta", 1.0)
        self.vae = VAE.from_module(self)

    def loss(self, x):
        log_lik, kl, _ = self.vae.elbo_terms(x)
        # negative (beta-)ELBO, scaled per observed dimension to keep gradients O(1)
        loss = (self.beta * kl - log_lik).mean() / self.spec.x_dim
        logs = {"elbo": (log_lik - kl).mean(), "log_lik": log_lik.mean(), "kl": kl.mean()}
        return loss, loss, logs

    def posterior(self, x):
        q = self.vae.encoder(x)
        z = q.sample()
        return z, q.log_prob(z).sum(-1)

    def regenerate(self, z):
        return self.vae.decoder(z).sample()

    def generate(self, n):
        z = torch.randn(n, self.spec.z_dim, device=self.device,
                        dtype=next(self.parameters()).dtype)
        return self.regenerate(z)

    def extra_test_metrics(self, x, z, y):
        return {"log_lik": self.vae.decoder.log_prob(x, z).mean().item()}


class HVGRunner(BaselineRunner):
    dataset, module_cls = "hvg", VAEModule


class PinWheelRunner(BaselineRunner):
    dataset, module_cls = "pinwheel", VAEModule


class MNISTRunner(BaselineRunner):
    dataset, module_cls = "mnist", VAEModule
