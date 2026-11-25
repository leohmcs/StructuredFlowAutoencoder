"""Shared pieces for the baseline benchmarks (VAE, LatentFM, Mixture-SVAE, FM).

The baselines are dataset-agnostic: every model works on flattened observations
and gets its layer sizes from a `DataSpec`, so the same architectures run on
HVG, pinwheel and MNIST. Only the data loading differs per dataset.

Run through the main entry point from the project root, e.g.

    python src_sfa/main.py --config hvg.yml --runner hvg_vae --ni            # train
    python src_sfa/main.py --config hvg.yml --runner hvg_vae --sample --ni   # evaluate

Baseline-specific hyperparameters are read from an optional `baseline:` section
of the config (see `baseline_opt`); everything else (z_dim, widths, lr, epochs,
batch size) is shared with the SFA runner of the same config. Options can be set
for all baselines or for one model only:

    baseline:
      beta: 1.            # every baseline
      vae:                # only benchmarks/vae.py (overrides the shared value)
        enc_width: 630

Network options understood by every model: enc_width, enc_depth, dec_width,
dec_depth (hidden layer sizes; defaults mirror the SFA runner of the dataset)
and layer_norm (LayerNorm on x before the encoder).

The Gaussian encoder / decoder expose the same methods as SFA's posterior
(sfa.GaussianPrior) and likelihood (sfa_llk.LLK), so a trained baseline can also
be evaluated with the authors' own test code (see reproduce_mnist.py).
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from models.nn import MLP
from callbacks import PlotLossCallback
from utils import find_checkpoint, find_resume_checkpoint


def baseline_opt(config, name, default, model=None):
    """Read `baseline.<model>.<name>`, then `baseline.<name>`, falling back to `default`."""
    section = getattr(config, "baseline", None)
    if model is not None and hasattr(getattr(section, model, None), name):
        return getattr(getattr(section, model), name)
    return getattr(section, name, default)


# ─── Datasets ──────────────────────────────────────────────────────────────

class DataSpec:
    """What the models need to know about a dataset.

    The default depths and activations mirror the SFA runner of the same
    dataset (posterior network -> encoder, vector field MLP -> decoder).
    `reference(n)` optionally draws n fresh samples of the true distribution,
    for a data-vs-data noise floor of the EMD.
    """

    def __init__(self, config, x_shape, fct, enc_depth, dec_depth,
                 layer_norm=True, image=False, perplexity=30, reference=None):
        self.config = config
        self.x_shape = tuple(x_shape)
        self.x_dim = int(np.prod(x_shape))
        self.z_dim = config.flow.z_dim
        self.n_classes = config.data.n_classes
        self.fct = fct
        self.enc_width, self.enc_depth = config.flow.ngf, enc_depth
        self.dec_width, self.dec_depth = config.model.ngf, dec_depth
        self.layer_norm = layer_norm  # normalise x before the encoder (as SFA's GaussianPrior)
        self.image = image
        self.perplexity = perplexity
        self.reference = reference


def _hvg(config):
    return DataSpec(config, (config.data.size,), nn.Softplus(), enc_depth=0, dec_depth=3)


def _hvg_train(config):
    from dataloader.dataloader_hvg import get_hvg_dataloaders
    train, val, _ = get_hvg_dataloaders(
        dat_dir="data", batch_size=config.training.batch_size,
        num_workers=config.data.num_workers)
    return train, val


def _hvg_eval(config):
    # Same evaluation cells as HVGRunner.sample(): the 'valid' split in one batch
    from dataloader.dataloader_hvg import get_hvg_eval_loader
    return get_hvg_eval_loader(dat_dir="data", num_workers=config.data.num_workers)


def _pinwheel(config):
    from dataloader.dataloader_pinwheel import true_pinwheel
    # LayerNorm over 2 features would discard the data, so it is off here
    return DataSpec(config, (config.data.size,), nn.Tanh(), enc_depth=1, dec_depth=5,
                    layer_norm=False,
                    reference=lambda n: true_pinwheel(n, config.data.n_classes))


def _pinwheel_train(config):
    from dataloader.dataloader_pinwheel import get_dataset
    dataset, test_dataset = get_dataset(
        config.data.n_classes, "data", config.data.samplesize, config.data.test_samplesize)
    train = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                       num_workers=config.data.num_workers)
    val = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=True,
                     num_workers=config.data.num_workers, drop_last=True)
    return train, val


def _pinwheel_eval(config):
    # Same protocol as PinWheelRunner.sample(): 2000 points per arm, shuffled batches of 500
    from dataloader.dataloader_pinwheel import get_dataset
    _, test_dataset = get_dataset(config.data.n_classes, "data", 2000, 2000)
    return DataLoader(test_dataset, batch_size=500, shuffle=True,
                      num_workers=config.data.num_workers, drop_last=True)


def _mnist(config):
    shape = (config.data.channel, config.data.size, config.data.size)
    return DataSpec(config, shape, nn.SiLU(), enc_depth=0, dec_depth=3,
                    image=True, perplexity=50)


def _mnist_train(config):
    from dataloader.dataloader_mnist import get_mnist
    dataset, val_dataset, sampler, val_sampler = get_mnist(
        config.data.n_classes, "data", config.data.samplesize, config.data.test_samplesize)
    train = DataLoader(dataset, batch_size=config.training.batch_size,
                       num_workers=config.data.num_workers, sampler=sampler)
    val = DataLoader(val_dataset, batch_size=config.training.batch_size,
                     num_workers=config.data.num_workers, sampler=val_sampler, drop_last=True)
    return train, val


def _mnist_eval(config):
    from dataloader.dataloader_mnist import get_mnist
    dataset, _, sampler, _ = get_mnist(config.data.n_classes, "data", 300, 50)
    return DataLoader(dataset, batch_size=3000,
                      num_workers=config.data.num_workers, sampler=sampler)


# name -> (spec, train/val loaders, evaluation loader)
DATASETS = {
    "hvg": (_hvg, _hvg_train, _hvg_eval),
    "pinwheel": (_pinwheel, _pinwheel_train, _pinwheel_eval),
    "mnist": (_mnist, _mnist_train, _mnist_eval),
}


# ─── Networks ──────────────────────────────────────────────────────────────

ACTIVATIONS = {"silu": nn.SiLU, "softplus": nn.Softplus, "tanh": nn.Tanh, "relu": nn.ReLU}


class GaussianEncoder(nn.Module):
    """q(z|x) = N(mu(x), diag sigma(x)^2); same interface as sfa.GaussianPrior."""

    def __init__(self, x_dim, z_dim, hidden, fct, layer_norm=False, min_std=1e-5):
        super().__init__()
        mlp = MLP(x_dim, 2 * z_dim, hidden_features=hidden, fct=fct)
        self.net = nn.Sequential(nn.LayerNorm(x_dim), mlp) if layer_norm else mlp
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
    """p(x|z) = N(mu(z), diag sigma(z)^2) over flattened x; same interface as sfa_llk.LLK."""

    def __init__(self, x_dim, z_dim, hidden, fct, min_std=1e-3):
        super().__init__()
        self.net = MLP(z_dim, 2 * x_dim, hidden_features=hidden, fct=fct)
        self.min_std = min_std

    def forward(self, z):
        mu, s = self.net(z).chunk(2, dim=-1)
        return Normal(mu, F.softplus(s) + self.min_std)

    def decode(self, noise, z, t=None, s=None):
        """x ~ p(x|z), reparameterised with the given N(0, I) noise (shaped like x). For SFA the
        same argument is the flow's starting point x1 ~ N(0, I), so the authors' sampling code
        works unchanged."""
        d = self(z)
        return (d.loc + d.scale * noise.flatten(start_dim=1)).view_as(noise)

    def log_prob(self, x, z, t=0, source=None, s=None):
        """log p(x|z), summed over dimensions (t / source / s only mirror LLK.log_prob)."""
        return self(z).log_prob(x.flatten(start_dim=1)).sum(-1)


def staged_cosine(stage_epochs, warmup=5):
    """LR multiplier per epoch: linear warmup then cosine decay, restarted for each stage."""
    def factor(epoch):
        start = 0
        for n in stage_epochs:
            if epoch < start + n or start + n >= sum(stage_epochs):
                break
            start += n
        e = min(epoch - start, n)
        return min(1.0, (e + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * e / max(n, 1)))
    return factor


# ─── Metrics ───────────────────────────────────────────────────────────────

def _np(t):
    return t.detach().double().cpu().numpy()


def vendi_score(x_np):
    from vendi_score import vendi
    return float(vendi.score_dual(x_np, normalize=True))


def emd(x_np, x_new_np, normalise=True):
    """Earth mover's distance between two equally weighted samples.

    normalise=True is the SFA runners' metric (cost matrix rescaled so its max is 10);
    normalise=False is the plain empirical W1 in data units.
    """
    import ot
    M = ot.dist(x_np, x_new_np, "euclidean")
    if normalise:
        M /= M.max() * 0.1
    return float(ot.emd2(np.full(len(x_np), 1 / len(x_np)),
                         np.full(len(x_new_np), 1 / len(x_new_np)), M))


def clustering_scores(z_np, y_np, k):
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(z_np).labels_
    return normalized_mutual_info_score(y_np, labels), adjusted_rand_score(y_np, labels)


def plot_latent(z_np, y_np, x_np, path, perplexity):
    fig, ax = plt.subplots(figsize=(8, 6))
    if z_np.shape[1] == 1 and x_np.shape[1] == 2:
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=z_np[:, 0], cmap="hsv", s=10)
    else:
        if z_np.shape[1] > 2:
            z_np = TSNE(n_components=2, perplexity=min(perplexity, len(z_np) - 1),
                        random_state=0).fit_transform(z_np)
        elif z_np.shape[1] == 1:
            z_np = np.concatenate([z_np, np.zeros_like(z_np)], axis=1)
        sc = ax.scatter(z_np[:, 0], z_np[:, 1], c=y_np, cmap="tab10", s=10)
    fig.colorbar(sc, ax=ax, shrink=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─── Lightning module and runner shared by all baselines ───────────────────

class BaselineModule(pl.LightningModule):
    """Training / evaluation loop. Subclasses set `name` (their `baseline.<name>` config
    section) and provide the model through:

        loss(x)        -> (training loss, validation loss, dict of extra logs)
        posterior(x)   -> (z, log q(z|x) or None for a deterministic encoder),
                          or None for a model without latents
        regenerate(z)  -> x~ given latents of observed data (posterior predictive)
        generate(n)    -> n unconditional samples of x
    """

    name = None

    def __init__(self, config, args, spec):
        super().__init__()
        self.config = config
        self.args = args
        self.spec = spec
        self.enc_hidden = [self.opt("enc_width", spec.enc_width)] * self.opt("enc_depth", spec.enc_depth)
        self.dec_hidden = [self.opt("dec_width", spec.dec_width)] * self.opt("dec_depth", spec.dec_depth)
        self.layer_norm = self.opt("layer_norm", spec.layer_norm)
        fct = self.opt("fct", None)
        self.fct = ACTIVATIONS[fct.lower()]() if fct else spec.fct

    def opt(self, key, default):
        return baseline_opt(self.config, key, default, self.name)

    def stage_epochs(self):
        return [self.config.training.n_epochs]

    def extra_test_metrics(self, x, z, y):
        return {}

    def prep(self, X):
        return X.flatten(start_dim=1).to(next(self.parameters()).dtype)

    def training_step(self, batch, batch_idx):
        loss, _, logs = self.loss(self.prep(batch[0]))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, val_loss, logs = self.loss(self.prep(batch[0]))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, staged_cosine(self.stage_epochs()))
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def test_step(self, batch, batch_idx):
        X, y = batch
        x = self.prep(X)
        x_np, y_np = _np(x), y.cpu().numpy()
        x_gen = self.generate(len(x))
        x_gen_np = _np(x_gen)

        metrics = {"vendi_x_gen": vendi_score(x_gen_np), "emd_gen": emd(x_np, x_gen_np)}
        if self.spec.reference is not None:
            # low-dim data with a known distribution: plain W1 and the data-vs-data noise floor
            ref = self.spec.reference(len(x))
            metrics["w1_raw_gen"] = emd(x_np, x_gen_np, normalise=False)
            metrics["emd_ref"] = emd(x_np, ref)
            metrics["w1_raw_ref"] = emd(x_np, ref, normalise=False)

        post = self.posterior(x)
        if post is not None:  # models with latents: posterior and posterior predictive
            z, log_q = post
            x_new = self.regenerate(z)
            z_np, x_new_np = _np(z), _np(x_new)
            if log_q is not None:
                metrics["log_post_z"] = log_q.mean().item()
            metrics["nmi"], metrics["ari"] = clustering_scores(z_np, y_np, self.spec.n_classes)
            metrics["vendi"] = vendi_score(z_np)
            metrics["vendi_x"] = vendi_score(x_new_np)
            metrics["emd"] = emd(x_np, x_new_np)
            metrics.update(self.extra_test_metrics(x, z, y))

            if self.spec.image:
                from dataloader.dataloader_mnist import inv_transform
                from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
                as_img = lambda t: inv_transform(t.detach().float().cpu()).view(-1, *self.spec.x_shape)
                metrics["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0)(
                    as_img(x_new), as_img(x)).item()

            if batch_idx == 0:
                plot_latent(z_np, y_np, x_np,
                            os.path.join(self.args.log_sample_path, "test_latent.png"),
                            self.spec.perplexity)

        self.log_dict({f"test_{k}": float(v) for k, v in metrics.items()},
                      on_step=False, on_epoch=True, batch_size=len(x))


class BaselineRunner:
    """Runner with the interface main.py expects (train / sample).

    Subclasses set `dataset` (a key of DATASETS) and `module_cls`.
    """

    dataset = None
    module_cls = None

    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, "samples")
        os.makedirs(args.log_sample_path, exist_ok=True)

        make_spec, self._train_loaders, self._eval_loader = DATASETS[self.dataset]
        self.model = self.module_cls(config, args, make_spec(config))

        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True,
            dirpath=args.log_path,
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}")
        plot_loss = PlotLossCallback(
            save_path=os.path.join(args.log_sample_path, "loss.png"), update_interval=1)

        clipval = getattr(config.training, "clipval", getattr(config.optim, "clip_value", None))
        self.trainer = pl.Trainer(
            max_epochs=config.training.n_epochs,
            default_root_dir=args.log_path,
            accelerator="auto", devices=1,
            gradient_clip_val=clipval,
            # no_grad instead of inference mode: LatentFM's log-density needs autograd at test time
            inference_mode=False,
            callbacks=[plot_loss, self.checkpoint_callback])

    def train(self):
        train_loader, val_loader = self._train_loaders(self.config)
        ckpt_path = None
        if self.args.resume_training:
            ckpt_path = find_resume_checkpoint(self.args.log_path, self.checkpoint_callback)
            print(f"Resuming from {ckpt_path}")
        self.trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)

    def sample(self):
        ckpt_path = find_checkpoint(self.args.log_path, self.checkpoint_callback)
        print(f"Evaluating checkpoint: {ckpt_path}")
        results = self.trainer.test(
            self.model, dataloaders=self._eval_loader(self.config), ckpt_path=ckpt_path)
        metrics = dict(results[0], checkpoint=os.path.basename(ckpt_path))
        with open(os.path.join(self.args.log_path, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
