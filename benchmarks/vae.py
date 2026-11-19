"""VAE baseline: diagonal Gaussian encoder q(z|x) and decoder p(x|z), N(0, I) prior.

Both networks are time-independent MLPs sized like the SFA posterior and vector
field of the same dataset (see benchmarks/common.py).

Optional config entries (section `baseline:`):
    beta      weight of the KL term (default 1.0)
    min_std   lower bound on the decoder standard deviation (default 1e-3)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from benchmarks.common import (
    BaselineModule, BaselineRunner, GaussianHead, baseline_opt, decoder_net, encoder_net)


class VAE(nn.Module):
    def __init__(self, spec, min_std=1e-3):
        super().__init__()
        self.z_dim = spec.z_dim
        self.encoder = GaussianHead(encoder_net(spec, 2 * spec.z_dim))
        self.decoder = GaussianHead(decoder_net(spec, 2 * spec.x_dim), min_std=min_std)

    def elbo_terms(self, x):
        """Per-sample log p(x|z) and KL(q(z|x) || p(z)) for one z ~ q(z|x), plus that z."""
        q = self.encoder(x)
        z = q.rsample()
        log_lik = self.decoder(z).log_prob(x).sum(-1)
        kl = kl_divergence(q, Normal(torch.zeros_like(z), torch.ones_like(z))).sum(-1)
        return log_lik, kl, z


class VAEModule(BaselineModule):
    def __init__(self, config, args, spec):
        super().__init__(config, args, spec)
        self.beta = baseline_opt(config, "beta", 1.0)
        self.vae = VAE(spec, min_std=baseline_opt(config, "min_std", 1e-3))

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

    def extra_test_metrics(self, x, z):
        return {"log_lik": self.vae.decoder(z).log_prob(x).sum(-1).mean().item()}


class HVGRunner(BaselineRunner):
    dataset, module_cls = "hvg", VAEModule


class PinWheelRunner(BaselineRunner):
    dataset, module_cls = "pinwheel", VAEModule


class MNISTRunner(BaselineRunner):
    dataset, module_cls = "mnist", VAEModule
