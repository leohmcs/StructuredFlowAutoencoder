"""How much the posterior mean z_t moves with t, and what the decoder's flow-matching loss is when
it gets its own z_t, the clean-image z_0, or another image's z_t (REPORT.md, Section 7, point 3).

usage: python diagnostics/posterior_drift.py     # sfa and mixture, seeds 123 124 125, best checkpoints
"""
import sys, os, tempfile, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src_sfa")]; os.chdir(REPO)
sys.argv = sys.argv[:1]
TMP = tempfile.mkdtemp()
import torch, numpy as np
import reproduce_mnist as R
torch.set_num_threads(2)
X, y = R.mnist_heldout(30, 0)
sig = 1e-4
out = {}
for name in ("sfa", "mixture"):
    for seed in (123, 124, 125):
        doc = "mnist_paper" + R.VARIANTS[name]["suffix"] + ("" if seed == 123 else f"_s{seed}")
        m, cfg, info = R.load_variant(name, doc, "exp", TMP, "mnist_paper.yml", "best")
        def posterior_mean(xt, t):
            tv = torch.full((len(xt),), float(t))
            if name == "mixture":
                logits, _ = m.Rt.rsample(xt, tv)
                xi = torch.softmax(logits / m.Rt.temp, -1)
                return m.rt(m._rt_context(logits, xi), xt, tv).mean
            return m.rt(xt.flatten(1), tv).mean
        with torch.no_grad():
            torch.manual_seed(0)
            noise = torch.randn_like(X)
            z0 = posterior_mean(X, 0.)
            drift = {}
            for t in (0.25, 0.5, 0.75, 0.95):
                zt = posterior_mean((1 - t) * X + (sig + (1 - sig) * t) * noise, t)
                drift[t] = ((zt - z0).norm(dim=1) / z0.norm(dim=1)).mean().item()
            loss = {}
            for t in (0.1, 0.5, 0.9):
                tv = torch.full((len(X),), t); tt = tv.view(-1, 1, 1, 1)
                xt = (1 - tt) * X + (sig + (1 - sig) * tt) * noise; ut = (1 - sig) * noise - X
                zt = posterior_mean(xt, t)
                L = lambda z: (m.vt(tv, xt, z) - ut).square().mean().item()
                loss[t] = (L(zt), L(z0), L(zt[torch.randperm(len(zt))]))
        out[(name, seed)] = (drift, loss)
        print(f"[{name} s{seed}, epoch {info['epoch']}] drift: " + ", ".join(f"t={t}: {v:.2f}" for t, v in drift.items()))
        print(f"[{name} s{seed}] loss (own z_t | clean z_0 | other image's z_t): " + " || ".join(f"t={t}: {a:.3f} | {b:.3f} | {c:.3f}" for t, (a, b, c) in loss.items()), flush=True)
for name in ("sfa", "mixture"):
    d = np.array([out[(name, s)][0][0.5] for s in (123, 124, 125)])
    l = np.array([out[(name, s)][1][0.5] for s in (123, 124, 125)])
    print(f"== {name}: drift at t=0.5 {d.mean():.2f} ± {d.std(ddof=1):.2f} {np.round(d,2).tolist()}; loss at t=0.5 own {l[:,0].mean():.2f} ± {l[:,0].std(ddof=1):.2f} | clean z_0 {l[:,1].mean():.2f} ± {l[:,1].std(ddof=1):.2f} {np.round(l[:,1],2).tolist()} | other image {l[:,2].mean():.2f} ± {l[:,2].std(ddof=1):.2f} {np.round(l[:,2],2).tolist()}")
