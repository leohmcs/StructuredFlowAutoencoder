"""Shared pieces for the baseline benchmarks (VAE, LatentFM).

The baselines are dataset-agnostic: every model works on flattened observations
and gets its layer sizes from a `DataSpec`, so the same architectures run on
HVG, pinwheel and MNIST. Only the data loading differs per dataset.

Run through the main entry point from the project root, e.g.

    python src_sfa/main.py --config hvg.yml --runner hvg_vae --ni            # train
    python src_sfa/main.py --config hvg.yml --runner hvg_vae --sample --ni   # evaluate

Baseline-specific hyperparameters are read from an optional `baseline:` section
of the config (see `baseline_opt`); everything else (z_dim, widths, lr, epochs,
batch size) is shared with the SFA runner of the same config.
"""

import json
import math
import os
import re

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


def baseline_opt(config, name, default):
    """Read `baseline.<name>` from the config, falling back to `default`."""
    return getattr(getattr(config, "baseline", None), name, default)


# ─── Datasets ──────────────────────────────────────────────────────────────

class DataSpec:
    """What the models need to know about a dataset.

    The default depths and activations mirror the SFA runner of the same
    dataset (posterior network -> encoder, vector field MLP -> decoder).
    """

    def __init__(self, config, x_shape, fct, enc_depth, dec_depth,
                 layer_norm=True, image=False, perplexity=30):
        self.x_shape = tuple(x_shape)
        self.x_dim = int(np.prod(x_shape))
        self.z_dim = config.flow.z_dim
        self.n_classes = config.data.n_classes
        self.fct = fct
        self.enc_hidden = [config.flow.ngf] * baseline_opt(config, "enc_depth", enc_depth)
        self.dec_hidden = [config.model.ngf] * baseline_opt(config, "dec_depth", dec_depth)
        self.layer_norm = layer_norm  # normalise x before the encoder (as SFA's GaussianPrior)
        self.image = image
        self.perplexity = perplexity


def _hvg(config):
    return DataSpec(config, (config.data.size,), nn.Softplus(), enc_depth=0, dec_depth=3)


def _hvg_train(config):
    from dataloader.dataloader_hvg import get_hvg_dataloaders
    train, val, _ = get_hvg_dataloaders(
        dat_dir="data", batch_size=config.training.batch_size,
        num_workers=config.data.num_workers)
    return train, val


def _hvg_eval(config):
    # Same evaluation cells as HVGRunner.sample(): 500 random 'valid' cells in one batch
    from dataloader.dataloader_hvg import get_hvg_eval_loader
    return get_hvg_eval_loader(dat_dir="data", num_workers=config.data.num_workers)


def _pinwheel(config):
    # LayerNorm over 2 features would discard the data, so it is off here
    return DataSpec(config, (config.data.size,), nn.Tanh(), enc_depth=1, dec_depth=5,
                    layer_norm=False)


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

def encoder_net(spec, out_dim):
    """x -> out_dim: optional LayerNorm followed by an MLP."""
    mlp = MLP(spec.x_dim, out_dim, hidden_features=spec.enc_hidden, fct=spec.fct)
    return nn.Sequential(nn.LayerNorm(spec.x_dim), mlp) if spec.layer_norm else mlp


def decoder_net(spec, out_dim):
    """z -> out_dim MLP."""
    return MLP(spec.z_dim, out_dim, hidden_features=spec.dec_hidden, fct=spec.fct)


class GaussianHead(nn.Module):
    """Wraps a network with 2*d outputs into a diagonal Gaussian over d dims."""

    def __init__(self, net, min_std=1e-5):
        super().__init__()
        self.net = net
        self.min_std = min_std

    def forward(self, h):
        mu, raw_std = self.net(h).chunk(2, dim=-1)
        return Normal(mu, F.softplus(raw_std) + self.min_std)


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


def emd(x_np, x_new_np):
    """Earth mover's distance between two equally weighted samples (as in the SFA runner)."""
    import ot
    M = ot.dist(x_np, x_new_np, "euclidean")
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
    """Training / evaluation loop. Subclasses provide the model through:

        loss(x)        -> (training loss, validation loss, dict of extra logs)
        posterior(x)   -> (z, log q(z|x) or None for a deterministic encoder)
        regenerate(z)  -> x~ given latents of observed data (posterior predictive)
        generate(n)    -> n unconditional samples of x
    """

    def __init__(self, config, args, spec):
        super().__init__()
        self.config = config
        self.args = args
        self.spec = spec

    def stage_epochs(self):
        return [self.config.training.n_epochs]

    def extra_test_metrics(self, x, z):
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
        z, log_q = self.posterior(x)
        x_new = self.regenerate(z)
        x_gen = self.generate(len(x))

        metrics = {}
        if log_q is not None:
            metrics["log_post_z"] = log_q.mean().item()
        metrics.update(self.extra_test_metrics(x, z))

        x_np, z_np, y_np = _np(x), _np(z), y.cpu().numpy()
        x_new_np, x_gen_np = _np(x_new), _np(x_gen)
        metrics["nmi"], metrics["ari"] = clustering_scores(z_np, y_np, self.spec.n_classes)
        metrics["vendi"] = vendi_score(z_np)
        metrics["vendi_x"] = vendi_score(x_new_np)
        metrics["vendi_x_gen"] = vendi_score(x_gen_np)
        metrics["emd"] = emd(x_np, x_new_np)
        metrics["emd_gen"] = emd(x_np, x_gen_np)

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


def best_ckpt(log_path):
    """Return the 'best-checkpoint-...-val_loss=X.ckpt' in log_path with the lowest X."""
    best, best_loss = None, float("inf")
    for f in os.listdir(log_path):
        m = re.match(r"best-checkpoint-.*val_loss=(-?\d+(?:\.\d+)?)\.ckpt$", f)
        if m and float(m.group(1)) < best_loss:
            best, best_loss = os.path.join(log_path, f), float(m.group(1))
    if best is None:
        raise FileNotFoundError(f"No best-checkpoint-*.ckpt found in {log_path}")
    return best


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

        self.trainer = pl.Trainer(
            max_epochs=config.training.n_epochs,
            default_root_dir=args.log_path,
            accelerator="auto", devices=1,
            gradient_clip_val=config.training.clipval,
            # no_grad instead of inference mode: LatentFM's log-density needs autograd at test time
            inference_mode=False,
            callbacks=[plot_loss, self.checkpoint_callback])

    def train(self):
        train_loader, val_loader = self._train_loaders(self.config)
        ckpt_path = None
        if self.args.resume_training:
            ckpt_path = os.path.join(self.args.log_path, "last.ckpt")
            print(f"Resuming from {ckpt_path}")
        self.trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)

    def sample(self):
        ckpt_path = self.checkpoint_callback.best_model_path or best_ckpt(self.args.log_path)
        print(f"Evaluating checkpoint: {ckpt_path}")
        results = self.trainer.test(
            self.model, dataloaders=self._eval_loader(self.config), ckpt_path=ckpt_path)
        with open(os.path.join(self.args.log_path, "test_metrics.json"), "w") as f:
            json.dump(results[0], f, indent=2)
