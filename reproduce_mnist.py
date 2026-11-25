#!/usr/bin/env python
"""Reproduce the MNIST / EMNIST results of "Structured Flow Autoencoders" (ICLR 2026)
with the authors' implementation in src_sfa/, used as-is.

Models are built by the authors' runner classes, weights come from the checkpoints
their training loop writes, and every metric/figure that has evaluation code in the
repo is produced by calling that code. Where the paper reports something the repo has
no evaluation code for, it is computed here from the model's own methods
(vt.decode, vt.log_prob, rt.sample, ...) and labelled as such in the outputs.

Variants (the only MNIST models the repo implements):
    sfa      runner `mnist`,          flow.fix_z=true  -> Gaussian posterior   (Table 2 "SFA")
    sfa_cnf  runner `mnist`,          flow.fix_z=false -> CNF posterior        (Table 2 "w/CNF Posterior")
    mixture  runner `mnist_mixture`                                            (Table 3 "Mixture-SFA")
The baselines (VAE, VampVAE, Latent FM, Mixture-SVAE) and "SFA w/Deterministic" have
no code in this repo, so their rows only show the paper's numbers.

Outputs (default exp/reproduction/):
    table2_mnist_continuous.{md,csv}   Table 2, paper vs reproduced
    table3_mnist_mixture.{md,csv}      Table 3, paper vs reproduced
    fig3b_trajectories_<variant>.png   Figure 3b (no plotting code in repo; see fig3b())
    fig4b_mixture_display.png          Figure 4b
    fig6{a,e,d}_*.png                  Figure 6 t-SNE panels for SFA / OOD / Mixture
    fig7{b,d}_*.png                    Figure 7 posterior predictive grids
    fig9{a,b}_*.png                    Figure 9 Mixture-SFA posterior samples
    raw/<variant>/                     files as written by the authors' functions + metrics.json
    README.md                          protocol, provenance and file index

Configs:
    configs/mnist_paper.yml (default)  hyperparameters the paper states (z_dim 64, softplus, 2.4M
                                       parameters, no regularizer, z ~ q(z|x, xi)); runs in
                                       exp/logs/mnist_paper{,_cnf_posterior,_mixture}
    configs/mnist.yml                  the repository's original settings; runs in
                                       exp/logs/mnist{,_cnf_posterior,_mixture}

Usage:
    python reproduce_mnist.py train --variant sfa          # SFA, Gaussian posterior
    python reproduce_mnist.py train --variant sfa_cnf      # slow: ODE solve per training step
    python reproduce_mnist.py train --variant mixture
    python reproduce_mnist.py eval                         # all variants with a checkpoint
    python reproduce_mnist.py eval --config mnist.yml      # the original-settings runs

Time convention: in the code t=0 is data and t=1 is noise (the paper uses the reverse).
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "src_sfa")]
os.chdir(REPO)  # main.py and the dataloaders use paths relative to the repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torchdiffeq import odeint
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from vendi_score import vendi

import main as sfa_main
from utils import dict2namespace, find_checkpoint
from dataloader.dataloader_mnist import get_mnist, get_emnist, inv_transform

DEFAULT_CONFIG = "mnist_paper.yml"
IMAGE_DATASETS = {"MNIST", "EMNIST", "CIFAR10", "CIFAR100"}

# run folder = <config stem><suffix>, e.g. mnist_paper_mixture
VARIANTS = {
    "sfa": dict(runner="mnist", suffix="", overrides={}, label="SFA"),
    "sfa_cnf": dict(runner="mnist", suffix="_cnf_posterior", overrides={"flow.fix_z": False},
                    label="w/CNF Posterior"),
    "mixture": dict(runner="mnist_mixture", suffix="_mixture", overrides={}, label="Mixture-SFA"),
    # Table 2 "w/Deterministic": z = posterior mean in training and evaluation (flow.deterministic);
    # the paper names this variant but the repository has no code for it
    "sfa_det": dict(runner="mnist", suffix="_deterministic", overrides={"flow.deterministic": True},
                    label="w/Deterministic"),
    # baseline, not the authors' code: baselines/vae.py
    "vae": dict(runner="mnist_vae", suffix="_vae", overrides={}, label="VAE"),
    # baseline, not the authors' code: baselines/mixture_svae.py
    "mixture_svae": dict(runner="mnist_mixture_svae", suffix="_mixture_svae", overrides={}, label="Mixture-SVAE"),
    # ── one-factor ablations of the paper config: each flips one setting the paper does not state
    #    explicitly back to the repository's value (see "INFERRED:" / "PAPER:" in configs/mnist_paper.yml)
    "sfa_floor1e5": dict(runner="mnist", suffix="_floor1e-5", overrides={"flow.min_std": 1e-5}, ablation=True,
                         label="SFA", change="posterior std floor 1e-5 (published code) instead of 1e-6"),
    "sfa_alpha": dict(runner="mnist", suffix="_alpha0.01", overrides={"training.alpha": 0.01}, ablation=True,
                      label="SFA", change="alpha 0.01 (repository's z^2 penalty) instead of 0"),
    "sfa_w800": dict(runner="mnist", suffix="_w800", overrides={"flow.ngf": 800, "model.ngf": 800}, ablation=True,
                     label="SFA", change="width 800 (repository; 3.35M parameters) instead of 618 (2.40M)"),
    "sfa_legacy": dict(runner="mnist", suffix="_legacyloop", overrides={"training.legacy_loop": True}, ablation=True,
                       label="SFA", change="training loop as published (flat lr 5e-4, no effective clipping)"),
    "sfa_cnf_alpha": dict(runner="mnist", suffix="_cnf_posterior_alpha0.01",
                          overrides={"flow.fix_z": False, "training.alpha": 0.01}, ablation=True, base="sfa_cnf",
                          label="w/CNF Posterior", change="alpha 0.01 (repository's z^2 penalty) instead of 0"),
    "mixture_alpha0": dict(runner="mnist_mixture", suffix="_mixture_alpha0", overrides={"training.alpha_mixture": 0.0},
                           ablation=True, label="Mixture-SFA", change="no batch-entropy term (alpha_mixture 0) instead of 0.01"),
    "mixture_logits": dict(runner="mnist_mixture", suffix="_mixture_logits", overrides={"flow.condition_on_xi": False},
                           ablation=True, label="Mixture-SFA",
                           change="z conditioned on the sorted logits (repository) instead of the xi sample"),
}
ABLATION_BASE = {"mnist": "sfa", "mnist_mixture": "mixture"}  # the main run each ablation is compared with
BASELINE_RUNNERS = {"mnist_vae", "mnist_mixture_svae"}
MIXTURE_RUNNERS = {"mnist_mixture", "mnist_mixture_svae"}


def config_stem(config_name):
    return os.path.splitext(os.path.basename(config_name))[0]

# Paper values. Table 2: NMI/ARI are "in-distribution (OOD)".
PAPER_T2 = [
    # model,            log p(x|z), log p(z|x), Vendi,  SSIM,  NMI,  NMI_ood, ARI,  ARI_ood, variant
    ("VAE",             -453.648,   -85.448,    63.286, 0.419, 0.039, 0.033, 0.017, 0.012, "vae"),
    ("VampVAE",         -584.845,   155.820,    1.140,  0.866, 0.006, 0.006, 0.000, 0.000, None),
    ("Latent FM",       None,       None,       8.380,  0.980, 0.488, 0.392, 0.381, 0.194, None),
    ("Latent FM (VAE)", -910.925,   -11.192,    19.631, 0.697, 0.309, 0.152, 0.205, 0.073, None),
    ("SFA",             -916.901,   793.262,    25.589, 0.716, 0.490, 0.394, 0.356, 0.208, "sfa"),
    ("w/Deterministic", -858.385,   None,       10.189, 0.732, 0.501, 0.333, 0.379, 0.155, "sfa_det"),
    ("w/CNF Posterior", -907.998,   356.141,    23.166, 0.654, 0.485, 0.325, 0.355, 0.118, "sfa_cnf"),
]
PAPER_T3 = [
    # model,          log p(x|z), log p(z|x), SSIM,  softNMI, NMI,  ARI,  variant
    ("Mixture-SVAE",  -1133,      0.667,      0.634, 0.698,   0.161, 0.072, "mixture_svae"),
    ("Mixture-SFA",   -906,       725,        0.779, 0.728,   0.489, 0.332, "mixture"),
]


# ─── config / model loading (mirrors main.parse_args_and_config) ──────────────

def apply_overrides(config, overrides):
    for key, value in overrides.items():
        *path, leaf = key.split(".")
        node = config
        for p in path:
            node = getattr(node, p)
        setattr(node, leaf, value)


def load_config(config_name, overrides, device):
    with open(os.path.join("configs", config_name)) as f:
        config = dict2namespace(yaml.load(f, Loader=yaml.SafeLoader))
    if not hasattr(config.model, "cnn"):
        config.model.cnn = config.data.dataset in IMAGE_DATASETS
    config.device = device
    apply_overrides(config, overrides)
    return config


def pick_checkpoint(log_path, which):
    """'best' = lowest val_loss, as the authors' ModelCheckpoint and evaluation code use (default;
    the VAE baselines overfit, so the final weights are not a fair choice for them).
    'last' = final weights; falls back to 'best' for runs saved before save_last was enabled."""
    last = os.path.join(log_path, "last.ckpt")
    if which == "last" and os.path.exists(last):
        return last, "last"
    return find_checkpoint(log_path), "best"


def load_variant(name, doc, exp, raw_dir, config_name, which):
    """Build the variant with the authors' runner, load its checkpoint."""
    import runner_continuous
    import runner_mixture

    v = VARIANTS[name]
    log_path = os.path.join(exp, "logs", doc)
    ckpt_path, ckpt_kind = pick_checkpoint(log_path, which)
    config = load_config(config_name, v["overrides"], torch.device("cpu"))
    args = argparse.Namespace(
        config=config_name, runner=v["runner"], doc=doc, exp=exp, log_path=log_path, seed=0,
        comment="", verbose="info", test=False, sample=True, inference=False, ood=False,
        predict=False, figure=False, fast_fid=False, resume_training=False,
        image_folder="data", ni=True)
    runner = sfa_main._get_runner_cls(v["runner"])(args, config)
    # the authors' plotting helpers save into args.log_sample_path
    os.makedirs(raw_dir, exist_ok=True)
    args.log_sample_path = raw_dir

    if v["runner"] == "mnist_mixture":
        module = runner_mixture.FlowMatchingLightningModule(
            runner.vt, runner.Rt, runner.rt, runner.priorz, runner.embx, config, args)
    elif v["runner"] == "mnist_vae":
        from baselines.vae import VAELightningModule
        module = VAELightningModule(runner.vt, runner.rt, runner.priorz, config, args)
    elif v["runner"] == "mnist_mixture_svae":
        from baselines.mixture_svae import MixtureSVAELightningModule
        module = MixtureSVAELightningModule(runner.vt, runner.Rt, runner.rt, runner.priorz, runner.embx, config, args)
    else:
        module = runner_continuous.LightningModule(runner.vt, runner.rt, runner.priorz, config, args)
    module.setup("test")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"])  # strict: architecture must match the run
    module.eval()
    return module, config, dict(checkpoint=ckpt_path, checkpoint_kind=ckpt_kind, epoch=ckpt.get("epoch"))


# ─── data ─────────────────────────────────────────────────────────────────────

def mnist_heldout(n_per_class, seed):
    """Stratified sample of the MNIST *test* split (paper: held-out set of size 1000)."""
    np.random.seed(seed)
    _, test_ds, _, test_sampler = get_mnist(10, "data", 1, n_per_class)
    idx = list(test_sampler)
    X = torch.stack([test_ds[i][0] for i in idx]).to(torch.float64)
    y = torch.tensor([int(test_ds[i][1]) for i in idx])
    return X, y


def mnist_train_sample(n_per_class, seed):
    """Stratified sample of the MNIST *training* split, as MNISTRunner.sample() draws it
    (300 per class). The paper's Table 2 Vendi values match this set size (see README)."""
    np.random.seed(seed)
    train_ds, _, train_sampler, _ = get_mnist(10, "data", n_per_class, 1)
    idx = list(train_sampler)
    X = torch.stack([train_ds[i][0] for i in idx]).to(torch.float64)
    y = torch.tensor([int(train_ds[i][1]) for i in idx])
    return X, y


def emnist_ood(n_per_class, seed):
    """Same OOD set as MNISTRunner.sample(in_sample=False): EMNIST 'balanced',
    first 20 classes = digits 0-9 + letters A-J."""
    np.random.seed(seed)
    ds, sampler = get_emnist(20, "data", n_per_class, 50, split="balanced")
    idx = list(sampler)
    X = torch.stack([ds[i][0] for i in idx]).to(torch.float64)
    y = torch.tensor([int(ds[i][1]) for i in idx])
    return X, y


def one_per_class(y, k=10):
    return torch.tensor([int(torch.nonzero(y == c)[0]) for c in range(k)])


# ─── model calls (mirroring the authors' sampling code) ───────────────────────

def is_baseline(name):
    return VARIANTS[name]["runner"] in BASELINE_RUNNERS


def is_mixture(name):
    return VARIANTS[name]["runner"] in MIXTURE_RUNNERS


def posterior_sample(module, config, name, X):
    """z ~ q(z|x) exactly as the runners' evaluation code draws it.
    Returns (z, logits, pi, ctx); ctx is what the mixture posterior rt was conditioned on."""
    if is_mixture(name):
        return module.sample_with_context(X)  # FlowMatchingLightningModule
    xf = X.flatten(start_dim=1)
    if config.flow.fix_z:
        return module.rt.sample(xf), None, None, None
    z1 = module.priorz.sample((len(X),)).to(X.device)
    return module.rt.decode(z1, xf), None, None, None


def posterior_predictive(module, z0):
    """x ~ p1(x|z): integrate vt from noise (t=1) to data (t=0), as in generate_img."""
    x1 = module.priory.sample((len(z0),))
    return module.vt.decode(x1, z0)


def in_chunks(fn, n, size):
    return torch.cat([fn(slice(i, min(i + size, n))) for i in range(0, n, size)])


def ssim_score(real, gen):
    # same preprocessing as LightningModule.generate_img: pixel space, 3 channels, float32
    ssim = StructuralSimilarityIndexMeasure()
    return float(ssim(inv_transform(real).repeat(1, 3, 1, 1).float(),
                      inv_transform(gen).repeat(1, 3, 1, 1).float()))


# ─── Figure 3b ────────────────────────────────────────────────────────────────

def fig3b(module, config, name, X, y, path, per_class=5, n_t=60):
    """1D-PCA projected trajectories (Figure 3b). The repo has no code for this figure;
    the paths are the model's own ODE / posterior path recorded on a time grid:
      top    - x_t from shared noise (t=1) to data (t=0) under vt(t, x | z), z ~ q(z|x)
      bottom - z_t for the same images, from shared noise:
               CNF posterior: the posterior ODE rt(t, z | x) (same as rt.decode)
               Gaussian / mixture posterior: z_t = mu_t + sigma_t * eps on the training
               interpolation x_t = (1-t) x + (sig_min + (1-sig_min) t) noise
    """
    idx = torch.cat([torch.nonzero(y == c)[:per_class, 0] for c in range(10)])
    Xs, ys = X[idx], y[idx].numpy()
    B = len(Xs)
    ts = torch.linspace(1.0, 0.0, n_t)
    z0, _, _, _ = posterior_sample(module, config, name, Xs)

    noise = module.priory.sample((1,)).expand(B, *Xs.shape[1:]).clone()
    x_path = odeint(lambda t, x: module.vt(t, x, z0), noise, ts, atol=1e-5, rtol=1e-5)

    xf = Xs.flatten(start_dim=1)
    if not is_mixture(name) and not config.flow.fix_z:
        z1 = module.priorz.sample((1,)).expand(B, -1).clone()
        z_path = odeint(lambda t, z: module.rt(t, z, xf), z1, ts, atol=1e-5, rtol=1e-5)
    else:
        sig = module.flow_matching_loss.sig_min
        eps = torch.randn(1, z0.shape[-1]).expand(B, -1)
        z_path = []
        for t in ts:
            tv = t.expand(B)
            xt = (1 - t) * Xs + (sig + (1 - sig) * t) * noise
            if is_mixture(name):
                logits_t, _ = module.Rt.rsample(xt, tv)
                # noise-free relaxed xi (softmax(logits / temp)) so the plotted path is smooth
                xi_t = torch.softmax(logits_t / module.Rt.temp, dim=-1)
                dist = module.rt(module._rt_context(logits_t, xi_t), xt, tv)
            else:
                dist = module.rt(xt.flatten(start_dim=1), tv)
            noise_scale = 0. if getattr(module.rt, "deterministic", False) else 1.
            z_path.append(dist.mean + noise_scale * dist.stddev * eps)
        z_path = torch.stack(z_path)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    cmap = plt.colormaps["tab10"]
    for ax, traj, title in [(axes[0], x_path, "observation paths $x_t$ given $z \\sim q(z|x)$"),
                            (axes[1], z_path, "latent paths $z_t$ given $x$")]:
        flat = traj.reshape(n_t, B, -1).numpy()
        pc = PCA(n_components=1).fit(flat.reshape(n_t * B, -1))
        proj = pc.transform(flat.reshape(n_t * B, -1)).reshape(n_t, B)
        for b in range(B):
            ax.plot(ts.numpy(), proj[:, b], color=cmap(ys[b]), lw=0.8, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("PC 1")
    axes[1].set_xlabel("t   (code convention: 1 = noise, 0 = data)")
    axes[1].invert_xaxis()
    handles = [plt.Line2D([], [], color=cmap(c), label=str(c)) for c in range(10)]
    fig.legend(handles=handles, loc="center right", title="digit")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─── evaluation per variant ───────────────────────────────────────────────────

def evaluate(name, module, config, data, out, raw_dir, opts):
    X, y = data["mnist"]
    m, timings = {}, {}

    def timed(key, fn):
        t0 = time.time()
        res = fn()
        timings[key] = round(time.time() - t0, 1)
        print(f"  [{name}] {key}: {timings[key]}s", flush=True)
        return res

    with torch.no_grad():
        torch.manual_seed(opts.seed)
        # posterior sample and posterior predictive for the whole held-out set
        z0, logits0, pi0, ctx0 = timed("posterior", lambda: posterior_sample(module, config, name, X))
        xpred = timed("posterior_predictive", lambda: in_chunks(
            lambda s: posterior_predictive(module, z0[s]), len(X), opts.chunk))
        m["ssim"] = ssim_score(X, xpred)
        m["vendi_x"] = float(vendi.score_dual(inv_transform(xpred).flatten(start_dim=1).numpy(),
                                              normalize=True))

        if not opts.skip_loglik:
            # log p(x|z): vt.log_prob(x, z, 0, priory) as in the runners' validation code
            m["log_p_x_given_z"] = float(timed("log_p_x_given_z", lambda: in_chunks(
                lambda s: module.vt.log_prob(X[s], z0[s], 0., module.priory),
                len(X), opts.chunk)).mean())

        if is_mixture(name):
            # their validation-time posterior density, with the context z was drawn with
            m["log_p_z_given_x"] = float(module.rt.log_prob(ctx0, X.flatten(start_dim=1), z0).mean())
            # their test-time clustering metrics (argmax of q(xi|x), softNMI) + Fig 6d
            nmi, ari, soft = timed("test_plot_latent", lambda: module.test_plot_latent(X, y))
            m.update(nmi=float(nmi), ari=float(ari), soft_nmi=float(soft))
            timed("generate", lambda: module.generate(X, y))                # Fig 7d, 9a
            timed("plot_latent", lambda: module.plot_latent(X, y))          # Fig 9b
            i10 = one_per_class(y)
            timed("test_generation_figure",
                  lambda: module.test_generation_figure(X[i10], y[i10]))   # Fig 4b
        else:
            # their posterior density (LightningModule.sample_and_log); undefined for a
            # deterministic encoder (the paper reports "-")
            m["log_p_z_given_x"] = (None if getattr(config.flow, "deterministic", False)
                                    else float(module.sample_and_log(X, y)))
            # their figure + SSIM protocol (first image per class, 7 samples) - Fig 7b
            m["ssim_code_protocol"] = float(timed("generate_img",
                                                  lambda: module.generate_img(X, y, latent=True)))
            # their clustering: KMeans with k given on z, Vendi on z - Fig 6a / 6e
            nmi, ari, vz = timed("test_plot_latent", lambda: module.test_plot_latent(X, y, in_sample=True))
            m.update(nmi=float(nmi), ari=float(ari), vendi_z=float(vz))
            if data.get("emnist") is not None:
                Xo, yo = data["emnist"]
                nmi, ari, vz = timed("test_plot_latent_ood",
                                     lambda: module.test_plot_latent(Xo, yo, in_sample=False))
                m.update(nmi_ood=float(nmi), ari_ood=float(ari), vendi_z_ood=float(vz))

        # Figure 3b (flow trajectories; the VAE baselines have none)
        if not is_baseline(name):
            torch.manual_seed(opts.seed)
            timed("fig3b", lambda: fig3b(module, config, name, X, y,
                                         os.path.join(out, f"fig3b_trajectories_{name}.png")))

        # the authors' clustering / latent metrics again, on the set their `--sample` command uses
        # (training images). Runs last so the numbers above do not depend on it.
        if data.get("mnist_train") is not None:
            Xt, yt = data["mnist_train"]
            sample_dir = module.args.log_sample_path
            module.args.log_sample_path = os.path.join(raw_dir, "train_protocol")
            os.makedirs(module.args.log_sample_path, exist_ok=True)
            torch.manual_seed(opts.seed)
            try:
                if is_mixture(name):
                    nmi, ari, soft = timed("train_protocol", lambda: module.test_plot_latent(Xt, yt))
                    m.update(nmi_train=float(nmi), ari_train=float(ari), soft_nmi_train=float(soft))
                else:
                    nmi, ari, vz = timed("train_protocol",
                                         lambda: module.test_plot_latent(Xt, yt, in_sample=True))
                    m.update(nmi_train=float(nmi), ari_train=float(ari), vendi_z_train=float(vz))
                    m["log_p_z_given_x_train"] = (None if getattr(config.flow, "deterministic", False)
                                                  else float(module.sample_and_log(Xt, yt)))
            finally:
                module.args.log_sample_path = sample_dir

    # copy the authors' files to paper-named files
    mapping = ({"image_grid_epoch_0.png": f"fig7c_posterior_predictive_{name}.png",
                "postpi_grid_epoch_0.png": f"fig8a_xi_posterior_{name}.png",
                "postz_grid_epoch_0.png": f"fig8b_z_posterior_tsne3d_{name}.png",
                "eval_grid_epoch_0.png": f"fig6h_latent_tsne_{name}.png",
                "eval_sfa_mixture_eval_display.png": "fig4a_mixture_svae_display.png"}
               if is_mixture(name) and is_baseline(name) else
               {"image_grid_epoch_0.png": f"fig7d_posterior_predictive_{name}.png",
                "postpi_grid_epoch_0.png": f"fig9a_xi_posterior_{name}.png",
                "postz_grid_epoch_0.png": f"fig9b_z_posterior_tsne3d_{name}.png",
                "eval_grid_epoch_0.png": f"fig6d_latent_tsne_{name}.png",
                "eval_sfa_mixture_eval_display.png": "fig4b_mixture_display.png"}
               if is_mixture(name) else
               {"image_grid_epoch_0.png": f"fig7a_posterior_predictive_{name}.png",
                "test_latent_mnist.png": f"fig6c_latent_tsne_{name}.png",
                "test_latent_emnist.png": f"fig6g_latent_tsne_ood_{name}.png",
                "postz_grid_epoch_0.png": f"extra_latent_tsne3d_{name}.png"}
               if is_baseline(name) else
               {"image_grid_epoch_0.png": f"fig7b_posterior_predictive_{name}.png",
                "test_latent_mnist.png": f"fig6a_latent_tsne_{name}.png",
                "test_latent_emnist.png": f"fig6e_latent_tsne_ood_{name}.png",
                "postz_grid_epoch_0.png": f"extra_latent_tsne3d_{name}.png"})
    for src, dst in mapping.items():
        if os.path.exists(os.path.join(raw_dir, src)):
            shutil.copy(os.path.join(raw_dir, src), os.path.join(out, dst))
    m["timings_s"] = timings
    return m


# ─── tables / report ──────────────────────────────────────────────────────────

def fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def pair(a, b):
    return "–" if a is None and b is None else f"{fmt(a)} ({fmt(b)})"


# How to read each metric. All MNIST/EMNIST metrics in the paper are marked "higher is better" (↑);
# none is lower-is-better (those, W1 and RMSE, belong to the Pinwheel / Pendulum experiments).
METRIC_NOTES = {
    "log p(x|z)": "higher is better. Log-likelihood (nats) of the test image under the decoder, given "
                  "z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one "
                  "model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances "
                  "on the constant background pixels.",
    "log p(z|x)": "higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at "
                  "its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a "
                  "1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores "
                  "highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a "
                  "64-dim posterior sitting on a 1e-6 floor.",
    "Vendi": "higher = more diverse latent samples z ~ q(z|x) (effective number of distinct directions; at most "
             "the latent dimension). This is what the repository's MNIST test code reports as Vendi "
             "(`vendi.score_dual(z, normalize=True)`), and what the paper's Table 2 reports: its VAE value, 63.29, "
             "is what 3000 draws from a 64-dim standard normal give (63.33). The value depends on the number of "
             "images.",
    "Vendi x": "diversity of the posterior-predictive images (higher = more diverse). Not a quantity the "
               "paper's tables report. Noisy or unrelated samples also score high, so read it together with SSIM.",
    "SSIM": "higher is better (0 to 1): structural similarity between each test image and its "
            "posterior-predictive sample.",
    "NMI (OOD)": "higher is better (0 = no relation to the labels, 1 = perfect). Agreement of the latent "
                 "clustering with the labels: KMeans on z with k given (k = 10; 20 for OOD). First value MNIST, "
                 "in parentheses EMNIST (out of distribution).",
    "NMI": "higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with "
           "the digit labels.",
    "ARI (OOD)": "higher is better (0 = chance level, can be negative; 1 = perfect). First value MNIST, in "
                 "parentheses EMNIST (out of distribution).",
    "ARI": "higher is better (0 = chance level, can be negative; 1 = perfect).",
    "softNMI": "the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect "
               "clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component "
               "0.800. Use NMI / ARI to judge clustering.",
    "SSIM (code protocol)": "higher is better (0 to 1). SSIM as returned by the authors' generate_img: first "
                            "test image of each digit, 7 posterior-predictive samples each.",
    "checkpoint epoch": "epoch of the evaluated checkpoint; best = lowest validation loss.",
}


def metric_key(column):
    return column.replace(" ↑", "").strip()


def metric_notes(header):
    return [f"- **{h}**: {METRIC_NOTES[metric_key(h)]}" for h in header if metric_key(h) in METRIC_NOTES]


def write_table(path_stem, header, rows):
    # CSV: spell the direction out in the column name
    csv_header = [h.replace(" ↑", " (higher is better)") for h in header]
    with open(path_stem + ".csv", "w", newline="") as f:
        csv.writer(f).writerows([csv_header] + rows)
    with open(path_stem + ".md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        f.write("\n↑ = higher is better (as marked in the paper). No metric here is lower-is-better.\n\n")
        f.write("\n".join(metric_notes(header)) + "\n")


def ckpt_label(r):
    return f"{r['epoch']} ({r.get('checkpoint_kind', 'best')})"


def status(results, var):
    if var not in results:
        return "not evaluated"
    return "not trained" if results[var] is None else "failed (see raw/metrics.json)"


def tables(results, out):
    na = "not in repo"
    h2 = ["Model", "Source", "log p(x|z) ↑", "log p(z|x) ↑", "Vendi ↑", "SSIM ↑",
          "NMI (OOD) ↑", "ARI (OOD) ↑", "Vendi x ↑", "SSIM (code protocol) ↑", "checkpoint epoch"]
    rows = []
    for model, lx, lz, ve, ss, nmi, nmio, ari, ario, var in PAPER_T2:
        rows.append([model, "paper", fmt(lx), fmt(lz), fmt(ve), fmt(ss), pair(nmi, nmio),
                     pair(ari, ario), "", "", ""])
        r = results.get(var) if var else None
        if var is None:
            rows.append([model, "reproduced", na, na, na, na, na, na, "", "", ""])
        elif r is None or "error" in r:
            rows.append([model, "reproduced", status(results, var), "", "", "", "", "", "", "", ""])
        else:
            rows.append([model, "reproduced", fmt(r.get("log_p_x_given_z")), fmt(r["log_p_z_given_x"]),
                         fmt(r.get("vendi_z")), fmt(r["ssim"]), pair(r["nmi"], r.get("nmi_ood")),
                         pair(r["ari"], r.get("ari_ood")), fmt(r["vendi_x"]),
                         fmt(r.get("ssim_code_protocol")), ckpt_label(r)])
    write_table(os.path.join(out, "table2_mnist_continuous"), h2, rows)

    h3 = ["Model", "Source", "log p(x|z) ↑", "log p(z|x) ↑", "SSIM ↑", "softNMI ↑", "NMI ↑", "ARI ↑",
          "Vendi x ↑", "checkpoint epoch"]
    rows = []
    for model, lx, lz, ss, snmi, nmi, ari, var in PAPER_T3:
        rows.append([model, "paper", fmt(lx), fmt(lz), fmt(ss), fmt(snmi), fmt(nmi), fmt(ari), "", ""])
        r = results.get(var) if var else None
        if var is None:
            rows.append([model, "reproduced", na, na, na, na, na, na, "", ""])
        elif r is None or "error" in r:
            rows.append([model, "reproduced", status(results, var), "", "", "", "", "", "", ""])
        else:
            rows.append([model, "reproduced", fmt(r.get("log_p_x_given_z")), fmt(r["log_p_z_given_x"]),
                         fmt(r["ssim"]), fmt(r["soft_nmi"]), fmt(r["nmi"]), fmt(r["ari"]),
                         fmt(r["vendi_x"]), ckpt_label(r)])
    write_table(os.path.join(out, "table3_mnist_mixture"), h3, rows)

    def ok(var):
        return var and results.get(var) is not None and "error" not in results[var]

    # clustering / latent metrics on the repository's `--sample` set (training images)
    ht = ["Model", "Source", "NMI ↑", "ARI ↑", "Vendi ↑", "softNMI ↑", "log p(z|x) ↑"]
    rows = []
    for model, lx, lz, ve, ss, nmi, nmio, ari, ario, var in PAPER_T2:
        if ok(var) and "nmi_train" in results[var]:
            r = results[var]
            rows.append([model, "paper", fmt(nmi), fmt(ari), fmt(ve), "", fmt(lz)])
            rows.append([model, "reproduced", fmt(r["nmi_train"]), fmt(r["ari_train"]), fmt(r.get("vendi_z_train")),
                         "", fmt(r.get("log_p_z_given_x_train"))])
    for model, lx, lz, ss, snmi, nmi, ari, var in PAPER_T3:
        if ok(var) and "nmi_train" in results[var]:
            r = results[var]
            rows.append([model, "paper", fmt(nmi), fmt(ari), "", fmt(snmi), ""])
            rows.append([model, "reproduced", fmt(r["nmi_train"]), fmt(r["ari_train"]), "",
                         fmt(r.get("soft_nmi_train")), ""])
    if rows:
        write_table(os.path.join(out, "table_train_protocol"), ht, rows)

    # one-factor ablations next to the main run they differ from
    ha = ["Model", "Run", "Setting changed", "SSIM ↑", "NMI ↑", "ARI ↑", "softNMI ↑", "Vendi ↑", "log p(z|x) ↑",
          "log p(x|z) ↑", "checkpoint epoch"]

    def abl_row(var, change):
        r = results[var]
        return [VARIANTS[var]["label"], var, change, fmt(r["ssim"]), fmt(r["nmi"]), fmt(r["ari"]),
                fmt(r.get("soft_nmi")), fmt(r.get("vendi_z")), fmt(r["log_p_z_given_x"]),
                fmt(r.get("log_p_x_given_z")), ckpt_label(r)]
    rows = []
    bases = list(dict.fromkeys(d.get("base", ABLATION_BASE[d["runner"]]) for d in VARIANTS.values() if d.get("ablation")))
    for base in bases:
        abl = [v for v, d in VARIANTS.items()
               if d.get("ablation") and d.get("base", ABLATION_BASE[d["runner"]]) == base and ok(v)]
        if abl and ok(base):
            rows.append(abl_row(base, "none (paper config)"))
        rows += [abl_row(v, VARIANTS[v]["change"]) for v in abl]
    if rows:
        write_table(os.path.join(out, "table_ablations"), ha, rows)


def git_state():
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout.rstrip()
        except OSError:
            return "?"
    return run("git", "rev-parse", "--short", "HEAD"), run("git", "status", "--short")


def write_readme(results, opts, out):
    commit, status = git_state()
    lines = [
        "# SFA reproduction: MNIST / EMNIST",
        "",
        f"Generated by `reproduce_mnist.py eval` on {time.strftime('%Y-%m-%d %H:%M')}.",
        f"Config: `configs/{opts.config}`; checkpoint: `{opts.checkpoint}` "
        "(best = lowest val_loss, the authors' ModelCheckpoint convention; last = final weights, "
        "falling back to best for runs without last.ckpt).",
        f"Code: commit `{commit}`; working-tree changes at evaluation time:",
        "```", status or "(clean)", "```",
        "",
        "## Protocol",
        f"- In-distribution: {10 * opts.n_per_class} images from the MNIST **test** split "
        f"({opts.n_per_class} per class, seed {opts.seed}). The paper evaluates on a held-out set of "
        "1000; note the repo's own `--sample` evaluates on 3000 *training* images instead.",
        f"- OOD: EMNIST `balanced`, classes 0-19 (digits 0-9 + letters A-J), {opts.ood_per_class} per "
        "class, as in `MNISTRunner.sample()` with `in_sample: false`.",
        "- One posterior sample z ~ q(z|x) per image; one posterior predictive sample x ~ p1(x|z) per z.",
        "- log p(x|z): `vt.log_prob(x, z, 0, priory)` (Hutchinson trace, adjoint ODE), computed in the "
        "logit-transformed space the model is trained in (`LogitTrans(0.01, 0.99)`), as in the code.",
        "- log p(z|x): continuous — `LightningModule.sample_and_log`; mixture — "
        "`rt.log_prob(logits0, x, z0)` as in the validation loop.",
        "- SSIM: posterior predictive vs. input, pixel space, torchmetrics defaults "
        "(same preprocessing as `generate_img`). 'SSIM (code protocol)' is the value returned by "
        "`generate_img` itself (first image of each class, 7 samples each).",
        "- Vendi: `vendi.score_dual(z, normalize=True)` on the posterior samples z, as the repo's MNIST test "
        "code computes it (this is the quantity in the paper's Table 2). 'Vendi x' is the same score on the "
        "posterior-predictive images (pixel space), which the paper does not report.",
        f"- `table_train_protocol`: NMI / ARI / Vendi / softNMI / log p(z|x) recomputed on {10 * opts.train_per_class} "
        f"MNIST **training** images ({opts.train_per_class} per class), the set the repo's `--sample` command uses. "
        "The paper's Vendi values fit this set size rather than the 1000 held-out images of its captions.",
        "- `table_ablations`: runs that differ from the paper config in one setting the paper does not state "
        "explicitly (marked INFERRED / PAPER in the config).",
        "- NMI/ARI: continuous — `test_plot_latent` (KMeans on z, k=10 / k=20 for OOD); mixture — "
        "`test_plot_latent` (argmax of q(xi|x)) and `soft_nmi`.",
        "- VAE and Mixture-SVAE: our implementations (`baselines/vae.py`, `baselines/mixture_svae.py`), "
        "not the authors' code; trained and evaluated by the authors' continuous / mixture LightningModules "
        "with the loss replaced by the (mixture) negative ELBO. Choices the paper does not state are in "
        "the `vae:` / `mixture_svae:` sections of the config.",
        "- SFA w/Deterministic: named in the paper but not implemented in the repository; here it is the "
        "authors' SFA with z = posterior mean in training and evaluation (`flow.deterministic`).",
        "- Not reproducible from this repo: VampVAE, Latent FM, Latent FM (VAE) (no code), and the "
        "corresponding figure panels.",
        "",
        "## Metrics",
        "",
        "↑ = higher is better, as marked in the paper; no MNIST/EMNIST metric is lower-is-better. "
        "How to read each one:",
        "",
        *[f"- **{k} ↑**: {v}" if k != "checkpoint epoch" else f"- **{k}**: {v}" for k, v in METRIC_NOTES.items()],
        "",
        "## Runs",
    ]
    for name, r in results.items():
        if r is None:
            lines.append(f"- `{name}`: no checkpoint found — train with "
                         f"`python reproduce_mnist.py train --variant {name} --config {opts.config}`")
        elif "error" in r:
            lines.append(f"- `{name}`: evaluation failed: `{r['error']}`")
        else:
            lines.append(f"- `{name}` ({VARIANTS[name]['label']}): `{r['checkpoint']}` "
                         f"(epoch {r['epoch']}, {r.get('checkpoint_kind')}); timings {r['timings_s']}")
    lines += ["", "## Files", ""]
    for f in sorted(os.listdir(out)):
        if f not in ("README.md", "raw"):
            lines.append(f"- `{f}`")
    lines += ["- `raw/<variant>/` — files exactly as written by the authors' functions, plus `metrics.json`",
              "", "Figure 3b has no plotting code in the repo; `fig3b()` in `reproduce_mnist.py` documents "
              "how its paths are generated."]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── commands ─────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_SEED = 123  # main.py's default; runs with other seeds get an _s<seed> suffix


def seed_suffix(seed):
    return "" if seed == DEFAULT_TRAIN_SEED else f"_s{seed}"


def parse_docs(items, config_name, train_seed=DEFAULT_TRAIN_SEED):
    stem = config_stem(config_name)
    docs = {k: stem + v["suffix"] + seed_suffix(train_seed) for k, v in VARIANTS.items()}
    for item in items or []:
        k, _, v = item.partition("=")
        if k not in VARIANTS or not v:
            sys.exit(f"--doc expects VARIANT=DOC with VARIANT in {list(VARIANTS)}")
        docs[k] = v
    return docs


def save_code_state(log_path):
    """Write the commit and the full uncommitted diff (tracked + untracked text files) next to the run,
    so results can be tied to the code that trained them."""
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout
    parts = ["# commit " + run("git", "rev-parse", "HEAD"), run("git", "diff", "HEAD")]
    for f in run("git", "ls-files", "--others", "--exclude-standard").splitlines():
        if f.endswith((".py", ".yml", ".yaml", ".md", ".txt")):
            parts.append(run("git", "diff", "--no-index", "--", os.devnull, f))
    with open(os.path.join(log_path, "code_state.patch"), "w") as fh:
        fh.write("\n".join(parts))


def cmd_train(opts):
    v = VARIANTS[opts.variant]
    doc = parse_docs(opts.doc, opts.config, opts.seed)[opts.variant]
    log_path = os.path.join(opts.exp, "logs", doc)
    if not opts.resume and glob.glob(os.path.join(log_path, "*.ckpt")) and not opts.overwrite:
        sys.exit(f"{log_path} already has a checkpoint; training would delete it. "
                 "Use --overwrite, --resume, or another --doc.")
    sys.argv = ["main.py", "--config", opts.config, "--runner", v["runner"], "--doc", doc,
                "--exp", opts.exp, "--seed", str(opts.seed), "--ni"] + (["--resume_training"] if opts.resume else [])
    args, config = sfa_main.parse_args_and_config()
    if opts.threads:
        torch.set_num_threads(opts.threads)
    if opts.num_workers is not None:
        config.data.num_workers = opts.num_workers
    if v["overrides"] or opts.num_workers is not None:
        apply_overrides(config, v["overrides"])
        if not opts.resume:
            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False)
    if not opts.resume:
        save_code_state(args.log_path)
    runner = sfa_main._get_runner_cls(args.runner)(args, config)
    runner.train()


def cmd_eval(opts):
    out = opts.out or os.path.join("exp", "reproduction", config_stem(opts.config) + seed_suffix(opts.train_seed))
    os.makedirs(out, exist_ok=True)
    docs = parse_docs(opts.doc, opts.config, opts.train_seed)
    torch.manual_seed(opts.seed)
    data = {"mnist": mnist_heldout(opts.n_per_class, opts.seed)}
    needs_ood = any(not is_mixture(v) for v in opts.variants)
    data["emnist"] = emnist_ood(opts.ood_per_class, opts.seed) if needs_ood and opts.ood_per_class > 0 else None
    data["mnist_train"] = mnist_train_sample(opts.train_per_class, opts.seed) if opts.train_per_class > 0 else None

    results = {}
    for name in opts.variants:
        raw_dir = os.path.join(out, "raw", name)
        try:
            module, config, info = load_variant(name, docs[name], opts.exp, raw_dir, opts.config, opts.checkpoint)
        except FileNotFoundError as e:
            print(f"[{name}] skipped: {e}")
            results[name] = None
            continue
        print(f"[{name}] {info['checkpoint']} (epoch {info['epoch']})", flush=True)
        try:
            r = dict(info, **evaluate(name, module, config, data, out, raw_dir, opts))
        except Exception as e:  # keep going with the other variants, record the failure
            import traceback
            traceback.print_exc()
            r = dict(info, error=f"{type(e).__name__}: {e}")
        with open(os.path.join(raw_dir, "metrics.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        results[name] = r

    # variants not evaluated in this call keep their earlier results (same output folder) in the tables
    for name in VARIANTS:
        saved = os.path.join(out, "raw", name, "metrics.json")
        if name not in opts.variants and os.path.exists(saved):
            results[name] = json.load(open(saved))
    results = {name: results[name] for name in VARIANTS if name in results}

    tables(results, out)
    write_readme(results, opts, out)
    print(f"\nWrote {out}/")
    for stem in ("table2_mnist_continuous", "table3_mnist_mixture", "table_train_protocol", "table_ablations"):
        if os.path.exists(os.path.join(out, stem + ".md")):
            print(open(os.path.join(out, stem + ".md")).read())


AGG_COLUMNS = {
    "table2_mnist_continuous": [("log p(x|z) ↑", "log_p_x_given_z"), ("log p(z|x) ↑", "log_p_z_given_x"),
                                ("Vendi ↑", "vendi_z"), ("SSIM ↑", "ssim"), ("NMI ↑", "nmi"),
                                ("NMI (OOD) ↑", "nmi_ood"), ("ARI ↑", "ari"), ("ARI (OOD) ↑", "ari_ood"),
                                ("Vendi x ↑", "vendi_x"), ("NMI (train set) ↑", "nmi_train"),
                                ("ARI (train set) ↑", "ari_train"), ("Vendi (train set) ↑", "vendi_z_train")],
    "table3_mnist_mixture": [("log p(x|z) ↑", "log_p_x_given_z"), ("log p(z|x) ↑", "log_p_z_given_x"),
                             ("SSIM ↑", "ssim"), ("softNMI ↑", "soft_nmi"), ("NMI ↑", "nmi"), ("ARI ↑", "ari"),
                             ("Vendi x ↑", "vendi_x"), ("NMI (train set) ↑", "nmi_train"),
                             ("ARI (train set) ↑", "ari_train")],
}


def mean_std(vals):
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"


def cmd_aggregate(opts):
    """Mean ± sample std over training seeds, from each seed's eval output (raw/<variant>/metrics.json)."""
    stem = config_stem(opts.config)
    out = opts.out or os.path.join("exp", "reproduction", stem + "_seeds")
    os.makedirs(out, exist_ok=True)
    for table, cols in AGG_COLUMNS.items():
        variants = [v for v in VARIANTS if is_mixture(v) == (table == "table3_mnist_mixture")]
        agg_rows, seed_rows = [], []
        for v in variants:
            runs = {}
            for seed in opts.train_seeds:
                f = os.path.join("exp", "reproduction", stem + seed_suffix(seed), "raw", v, "metrics.json")
                if os.path.exists(f):
                    m = json.load(open(f))
                    if "error" not in m:
                        runs[seed] = m
            if not runs:
                continue
            label = VARIANTS[v]["label"] + (f" [{v}: {VARIANTS[v]['change']}]" if VARIANTS[v].get("ablation") else "")
            agg_rows.append([label, f"{len(runs)} ({', '.join(map(str, runs))})"] +
                            [mean_std([r[k] for r in runs.values() if r.get(k) is not None]) for _, k in cols])
            for seed, r in runs.items():
                seed_rows.append([label, seed, f"{r['epoch']} ({r.get('checkpoint_kind', 'best')})"] +
                                 [fmt(r.get(k)) for _, k in cols])
        write_table(os.path.join(out, table), ["Model", "seeds (n)"] + [c for c, _ in cols], agg_rows)
        write_table(os.path.join(out, table + "_per_seed"), ["Model", "seed", "checkpoint epoch"] + [c for c, _ in cols],
                    seed_rows)
        print(open(os.path.join(out, table + ".md")).read().split("\n\n")[0], "\n")
    print(f"Wrote {out}/ (mean ± sample std over training seeds; evaluation seed fixed)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help="file in configs/ (default: %(default)s)")
    common.add_argument("--exp", default="exp", help="experiment root (logs under <exp>/logs/<doc>)")
    common.add_argument("--doc", action="append", metavar="VARIANT=DOC",
                        help="use a different run folder for a variant (repeatable)")

    t = sub.add_parser("train", parents=[common], help="train one variant via the authors' runner")
    t.add_argument("--variant", choices=list(VARIANTS), required=True)
    t.add_argument("--seed", type=int, default=123, help="main.py's default seed")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--overwrite", action="store_true", help="allow deleting an existing run folder")
    t.add_argument("--num-workers", type=int, default=None,
                   help="override data.num_workers (data-loader processes; does not change what is trained)")
    t.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads, for running several trainings side by side")

    e = sub.add_parser("eval", parents=[common], help="evaluate trained variants, write tables/figures")
    e.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    e.add_argument("--out", default=None, help="default: exp/reproduction/<config stem>")
    e.add_argument("--checkpoint", choices=["best", "last"], default="best",
                   help="lowest val_loss (the authors' ModelCheckpoint convention; default) or final weights")
    e.add_argument("--n-per-class", type=int, default=100, help="MNIST test images per class (paper: 1000 total)")
    e.add_argument("--ood-per-class", type=int, default=300, help="EMNIST images per class (code: 300); 0 = skip")
    e.add_argument("--train-per-class", type=int, default=300,
                   help="MNIST training images per class for table_train_protocol (code's --sample: 300); 0 = skip")
    e.add_argument("--seed", type=int, default=0, help="evaluation seed (test-image sample, posterior samples)")
    e.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED,
                   help="which training seed's runs to evaluate (run folders <config><variant>_s<seed>)")

    a = sub.add_parser("aggregate", help="combine eval outputs of several training seeds into mean ± std tables")
    a.add_argument("--config", default=DEFAULT_CONFIG)
    a.add_argument("--train-seeds", type=int, nargs="+", required=True)
    a.add_argument("--out", default=None, help="default: exp/reproduction/<config stem>_seeds")
    e.add_argument("--chunk", type=int, default=250, help="batch size for ODE solves")
    e.add_argument("--skip-loglik", action="store_true", help="skip log p(x|z) (slowest metric)")

    opts = p.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "aggregate": cmd_aggregate}[opts.cmd](opts)


if __name__ == "__main__":
    main()
