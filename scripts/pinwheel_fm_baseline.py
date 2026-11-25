"""Flow-matching (FM) baseline for Pinwheel Table 1 / Figure 1.

The repo has no baselines (upstream's .gitignore excludes src_fm/), so this one is ours. It differs from
SFA in one thing only: it is SFA's vector field `vt` (the same LLK "mlp" network: tanh, 5 x 200 hidden
units), trained with the same flow-matching loss, optimiser (AdamW, lr 1e-3, weight decay 1e-4), cosine
schedule, epochs, batch size, float64 and training data, and sampled with the same ODE solver. The
difference is that the latent z is fixed at 0, so the model is plain (unconditional) flow matching.
Hyperparameters are read from configs/pinwheel.yml.

Run from the repo root:
    .venv\\Scripts\\python scripts\\pinwheel_fm_baseline.py --seed 1
    .venv\\Scripts\\python scripts\\pinwheel_fm_baseline.py --seed 1 --eval-only     # reuse fm.pt

Writes to exp/logs/pinwheel_fm_seed<N>/:
    fm.pt                    trained weights (final epoch; FM has no posterior, so no checkpoint selection)
    test_metrics.json        W1 with the same protocol as SFA's --sample (20 batches of 500)
    figure_samples.npz/.png  5000 samples for Figure 1 and scripts/pinwheel_eval_n.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src_sfa")]

from sfa_llk import LLK  # noqa: E402
from dataloader.dataloader_pinwheel import make_pinwheel  # noqa: E402
from pinwheel_eval import w1_hat, w1_raw, true_pinwheel, write_json  # noqa: E402

torch.set_default_dtype(torch.float64)  # after the imports: models/unet.py resets it to float32


def build_vt(cfg):
    # identical to PinWheelRunner.vt
    return LLK(cfg["data"]["size"], 1, model="mlp", hidden_features=[cfg["model"]["ngf"]] * 5,
               hidden_dim=cfg["data"]["size"])


def generate(vt, n, dim):
    with torch.no_grad():
        return vt.decode(torch.randn(n, dim), torch.zeros(n, 1)).numpy()


def train(vt, cfg, epochs, log):
    k, per_class = cfg["data"]["n_classes"], cfg["data"]["samplesize"]
    X = torch.tensor(make_pinwheel(k, per_class)[:, :2])
    bs, sig_min = cfg["training"]["batch_size"], float(cfg["flow"]["sig_min"])
    opt = torch.optim.AdamW(vt.parameters(), lr=float(cfg["optim"]["lr"]),
                            weight_decay=float(cfg["optim"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
    t0 = time.time()
    for epoch in range(epochs):
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), bs):
            x = X[perm[i:i + bs]]
            t = torch.rand(len(x))
            x0 = torch.randn_like(x)
            tt = t[:, None]
            # same probability path and target as sfa.FlowMatchingLoss (t=0 data, t=1 noise)
            xt = (1 - tt) * x + (sig_min + (1 - sig_min) * tt) * x0
            ut = (1 - sig_min) * x0 - x
            loss = (vt(t, xt, torch.zeros(len(x), 1)) - ut).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            log(f"epoch {epoch:4d}/{epochs}  loss {total / len(X):.4f}  ({time.time() - t0:.0f}s)")


def evaluate(vt, cfg, seed, ckpt, out_dir, log):
    k, dim = cfg["data"]["n_classes"], cfg["data"]["size"]

    # Table 1, same protocol as PinWheelRunner.sample(): eval set of 2000 per arm, shuffled batches of 500
    test = make_pinwheel(k, 2000)[:, :2]
    test = test[np.random.permutation(len(test))]
    m = {"test_emd": [], "test_w1_raw": [], "test_emd_ref": [], "test_w1_raw_ref": []}
    for i in range(0, len(test), 500):
        xb = test[i:i + 500]
        gen = generate(vt, len(xb), dim)
        ref = true_pinwheel(len(xb), k)
        m["test_emd"].append(w1_hat(xb, gen))
        m["test_w1_raw"].append(w1_raw(xb, gen))
        m["test_emd_ref"].append(w1_hat(xb, ref))
        m["test_w1_raw_ref"].append(w1_raw(xb, ref))
    metrics = {"runner": "fm_baseline", "seed": seed, "checkpoint": ckpt, "batch_size": 500,
               "n_batches": len(m["test_emd"]), **{key: float(np.mean(v)) for key, v in m.items()}}
    write_json(os.path.join(out_dir, "test_metrics.json"), metrics)
    log(f"test_emd {metrics['test_emd']:.4f} (ref {metrics['test_emd_ref']:.4f})")

    # Figure 1 samples
    n = cfg["sample"]["n_gen"]
    x = generate(vt, n, dim)
    truth = make_pinwheel(k, int(np.ceil(n / k)))
    np.savez(os.path.join(out_dir, "figure_samples.npz"), x=x, truth_x=truth[:, :2],
             truth_label=truth[:, 2].astype(int), checkpoint=ckpt)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    axes[0].scatter(truth[:, 0], truth[:, 1], c=truth[:, 2], cmap="gist_rainbow", s=2)
    axes[0].set_title("Truth")
    axes[1].scatter(x[:, 0], x[:, 1], c="k", s=2)
    axes[1].set_title("FM baseline")
    for ax in axes:
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_samples.png"), dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pinwheel.yml"))
    p.add_argument("--epochs", type=int, default=None, help="default: training.n_epochs from the config")
    p.add_argument("--doc", default=None, help="run folder name (default pinwheel_fm_seed<N>)")
    p.add_argument("--eval-only", action="store_true", help="load fm.pt instead of training")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = args.epochs or cfg["training"]["n_epochs"]
    out_dir = os.path.join(ROOT, "exp", "logs", args.doc or f"pinwheel_fm_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "fm_log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] seed {args.seed}: {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vt = build_vt(cfg)
    ckpt = os.path.join(out_dir, "fm.pt")
    if args.eval_only:
        vt.load_state_dict(torch.load(ckpt))
    else:
        log(f"training {epochs} epochs")
        train(vt, cfg, epochs, log)
        torch.save(vt.state_dict(), ckpt)
    vt.eval()
    np.random.seed(args.seed + 10_000)  # evaluation draws do not depend on whether we trained in this call
    torch.manual_seed(args.seed + 10_000)
    evaluate(vt, cfg, args.seed, os.path.relpath(ckpt, ROOT), out_dir, log)
    log("done")


if __name__ == "__main__":
    main()
