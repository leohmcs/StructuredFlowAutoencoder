# Mixture-SVAE baseline for pinwheel (Table 1 / Figure 1).
# Not in the repo, so this is written from the paper (sec 4, appendix B.1 and B.3):
#   GMVAE with a fixed prior: p(c) uniform, p(z|c) = N(mu_c, sd^2)
#   q(c|x): MLP -> logits -> gumbel softmax
#   q(z|x,c) and p(x|z): diagonal gaussians from MLPs, tanh everywhere
#   samples: c ~ p(c), z ~ p(z|c), x ~ p(x|z), then z_post ~ q(z|x)
#
# Things the paper doesn't say, so I picked them:
#   prior means -4,-2,0,2,4 and sd 0.5
#   200 hidden units, 2 layers for the encoders, 4 for the decoder
#   gumbel temperature goes from 1.0 to 0.5 during training
#   everything else from configs/pinwheel_discrete.yml (same as mixture-sfa)
#
# Run from the repo root:
#   .venv\Scripts\python scripts\pinwheel_mixture_svae.py --seed 1
#   .venv\Scripts\python scripts\pinwheel_mixture_svae.py --seed 1 --eval-only
# Output goes to exp/logs/pinwheel_msvae_seed<N>/ (best and last model, metrics, figure samples)

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributions import Normal, kl_divergence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT, os.path.join(ROOT, "src_sfa")] + sys.path

from models.nn import MLP
from dataloader.dataloader_pinwheel import make_pinwheel
from pinwheel_eval import w1_hat, w1_raw, true_pinwheel, write_json

torch.set_default_dtype(torch.float64)

PRIOR_SD = 0.5


def gauss(h):
    mean, s = h.chunk(2, dim=-1)
    return Normal(mean, F.softplus(s) + 1e-4)


class MixtureSVAE(nn.Module):
    def __init__(self, x_dim, k, z_dim, hidden):
        super().__init__()
        self.k = k
        self.enc_c = MLP(x_dim, k, [hidden] * 2, fct=nn.Tanh())
        self.enc_z = MLP(x_dim + k, 2 * z_dim, [hidden] * 2, fct=nn.Tanh())
        self.dec = MLP(z_dim, 2 * x_dim, [hidden] * 4, fct=nn.Tanh())
        # fixed prior means, not trained
        self.register_buffer("prior_mu", torch.linspace(-4, 4, k)[:, None].repeat(1, z_dim))

    def loss(self, x, tau):
        logits = self.enc_c(x)
        q_c = F.softmax(logits, dim=-1)
        c = F.gumbel_softmax(logits, tau=tau)
        qz = gauss(self.enc_z(torch.cat([x, c], dim=-1)))
        z = qz.rsample()
        rec = gauss(self.dec(z)).log_prob(x).sum(-1)
        kl_z = kl_divergence(qz, Normal(c @ self.prior_mu, PRIOR_SD)).sum(-1)
        kl_c = (q_c * (torch.log(q_c + 1e-12) + math.log(self.k))).sum(-1)  # KL to uniform
        loss = -(rec - kl_z - kl_c).mean()
        return loss, rec.mean().item(), kl_z.mean().item(), kl_c.mean().item(), q_c.mean(0)

    @torch.no_grad()
    def generate(self, n):
        c = F.one_hot(torch.randint(self.k, (n,)), self.k).double()
        z = Normal(c @ self.prior_mu, PRIOR_SD).sample()
        px = gauss(self.dec(z))
        x = px.sample()
        pi = F.softmax(self.enc_c(x), dim=-1)
        c_post = F.one_hot(torch.multinomial(pi, 1)[:, 0], self.k).double()
        z_post = gauss(self.enc_z(torch.cat([x, c_post], dim=-1))).sample()
        return {"x": x.numpy(), "x_mean": px.mean.numpy(), "z_prior": z.numpy(),
                "c_prior": c.argmax(-1).numpy(), "z_post": z_post.numpy(), "pi_post": pi.numpy()}


def train(model, cfg, epochs, out_dir, log):
    k = cfg["data"]["n_classes"]
    X = torch.tensor(make_pinwheel(k, cfg["data"]["samplesize"])[:, :2])
    Xval = torch.tensor(make_pinwheel(k, cfg["data"]["test_samplesize"])[:, :2])
    bs = cfg["training"]["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]),
                            weight_decay=float(cfg["optim"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = float("inf")
    start = time.time()

    for epoch in range(epochs):
        tau = 0.5 ** (epoch / max(epochs - 1, 1))  # 1.0 -> 0.5
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), bs):
            x = X[perm[i:i + bs]]
            loss = model.loss(x, tau)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()

        model.eval()
        with torch.no_grad():
            val, rec, kl_z, kl_c, usage = model.loss(Xval, tau)
        val = val.item()
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val},
                       os.path.join(out_dir, "msvae_best.pt"))
        if epoch % 25 == 0 or epoch == epochs - 1:
            usage = sorted(usage.tolist(), reverse=True)
            log(f"epoch {epoch}/{epochs} train {total / len(X):.4f} val {val:.4f} rec {rec:.3f} "
                f"kl_z {kl_z:.3f} kl_c {kl_c:.3f} tau {tau:.2f} usage {np.round(usage, 2)} "
                f"({time.time() - start:.0f}s)")

    torch.save({"state_dict": model.state_dict(), "epoch": epochs - 1, "val_loss": val},
               os.path.join(out_dir, "msvae_last.pt"))
    log(f"best val_loss {best:.4f}, took {(time.time() - start) / 60:.1f} min")


def evaluate(model, cfg, seed, ckpt, tag, out_dir, log):
    k = cfg["data"]["n_classes"]

    # Table 1, same as the mixture-sfa runner: one batch of 2500 (500 per arm)
    test = make_pinwheel(k, 500)[:, :2]
    s = model.generate(len(test))
    ref = true_pinwheel(len(test), k)
    metrics = {"runner": "mixture_svae", "seed": seed, "checkpoint": ckpt, "batch_size": len(test),
               "n_batches": 1,
               "test_emd": w1_hat(test, s["x"]), "test_w1_raw": w1_raw(test, s["x"]),
               "test_emd_ref": w1_hat(test, ref), "test_w1_raw_ref": w1_raw(test, ref),
               "test_emd_decoder_mean": w1_hat(test, s["x_mean"])}  # extra: decoder mean instead of samples
    write_json(os.path.join(out_dir, f"test_metrics{tag}.json"), metrics)
    log(f"[{tag or 'best'}] test_emd {metrics['test_emd']:.4f} (ref {metrics['test_emd_ref']:.4f}, "
        f"decoder mean {metrics['test_emd_decoder_mean']:.4f})")

    # Figure 1 samples
    n = cfg["sample"]["n_gen"]
    s = model.generate(n)
    truth = make_pinwheel(k, int(np.ceil(n / k)))
    np.savez(os.path.join(out_dir, f"figure_samples{tag}.npz"), **s, truth_x=truth[:, :2],
             truth_label=truth[:, 2].astype(int), checkpoint=ckpt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax[0].scatter(truth[:, 0], truth[:, 1], c=truth[:, 2], cmap="gist_rainbow", s=2)
    ax[0].set_title("Truth")
    ax[1].scatter(s["x"][:, 0], s["x"][:, 1], c=s["z_post"][:, 0], cmap="hsv", s=2)
    ax[1].set_title("Mixture-SVAE, colour = z")
    ax[2].scatter(s["x"][:, 0], s["x"][:, 1], c=s["pi_post"].argmax(-1), cmap="tab10", s=2)
    ax[2].set_title("Mixture-SVAE, colour = cluster")
    for a in ax:
        a.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"figure_samples{tag}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "pinwheel_discrete.yml"))
    parser.add_argument("--epochs", type=int, default=None)  # default: from the config
    parser.add_argument("--doc", default=None)  # run folder name
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = args.epochs or cfg["training"]["n_epochs"]
    out_dir = os.path.join(ROOT, "exp", "logs", args.doc or f"pinwheel_msvae_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "msvae_log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] seed {args.seed}: {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MixtureSVAE(cfg["data"]["size"], cfg["flow"]["k_dim"], cfg["flow"]["z_dim"], cfg["model"]["ngf"])
    if not args.eval_only:
        log(f"training {epochs} epochs")
        train(model, cfg, epochs, out_dir, log)

    for tag, name in [("", "msvae_best.pt"), ("_last", "msvae_last.pt")]:
        ckpt = os.path.join(out_dir, name)
        state = torch.load(ckpt, weights_only=True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        # same random draws for the evaluation whether we trained now or not
        np.random.seed(args.seed + 10000)
        torch.manual_seed(args.seed + 10000)
        evaluate(model, cfg, args.seed, f"{os.path.relpath(ckpt, ROOT)} (epoch {state['epoch']})", tag,
                 out_dir, log)
    log("done")


if __name__ == "__main__":
    main()
