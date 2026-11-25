"""Structure of a trained CNF posterior (REPORT.md, Section 2a).

The MNIST CNF vector field is linear, so the posterior is z0 = M z1 + c(x). This prints how large the
noise (M), the signal (spread of c(x) over images) and the common offset (mean of c(x)) are, and how
much class information c(x) carries.

usage: python diagnostics/cnf_posterior.py [variant[:seed[:best|last]] ...]
       default: sfa_cnf and sfa_cnf_alpha, seeds 123 124 125, best checkpoints
"""
import os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src_sfa")]; os.chdir(REPO)
jobs = sys.argv[1:] or [f"{v}:{s}" for v in ("sfa_cnf", "sfa_cnf_alpha") for s in (123, 124, 125)]
sys.argv = sys.argv[:1]
import torch, numpy as np, reproduce_mnist as R
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from vendi_score import vendi
torch.set_num_threads(4)
TMP = tempfile.mkdtemp()
X, y = R.mnist_heldout(100, 0)
xf = X.flatten(1)


def km(z):
    return nmi(y.numpy(), KMeans(10, n_init=10, random_state=0).fit_predict(z.numpy()))


for job in jobs:
    parts = job.split(":")
    variant, seed, which = parts[0], (parts[1:2] or ["123"])[0], (parts[2:3] or ["best"])[0]
    doc = "mnist_paper" + R.VARIANTS[variant]["suffix"] + ("" if seed == "123" else f"_s{seed}")
    m, config, info = R.load_variant(variant, doc, "exp", TMP, "mnist_paper.yml", which)
    rt, d = m.rt, config.flow.z_dim
    with torch.no_grad():
        torch.manual_seed(0)
        c = rt.decode(torch.zeros(len(xf), d, dtype=xf.dtype), xf)                      # posterior mean c(x)
        one = xf[:1].expand(d, -1)
        M = (rt.decode(torch.eye(d, dtype=xf.dtype), one) - c[:1]).T                    # noise factor
        z = c + m.priorz.sample((len(xf),)) @ M.T                                       # posterior samples
        # linear in x?  c(a x1 + (1-a) x2) == a c(x1) + (1-a) c(x2)
        mix = rt.decode(torch.zeros(1, d, dtype=xf.dtype), 0.3 * xf[:1] + 0.7 * xf[1:2])
        lin = (mix - (0.3 * c[:1] + 0.7 * c[1:2])).abs().max().item()
    sv = torch.linalg.svdvals(M)
    off = c.mean(0)
    sig = (c - off).std(0)
    print(f"== {variant} seed {seed}: {info['checkpoint']} (epoch {info['epoch']}), z_dim {d}, alpha {config.training.alpha}")
    print(f"   noise: singular values of M  min {sv.min():.2e} median {sv.median():.2e} max {sv.max():.2e};"
          f"  mean log sv {sv.log().mean():.3f}  ->  log q(z|x) at a sample = {-(d/2)*np.log(2*np.pi*np.e) - sv.log().sum():.1f}")
    print(f"   signal: per-dim std of c(x) over images  min {sig.min():.2e} median {sig.median():.2e} max {sig.max():.2e};"
          f"  total signal var {(sig**2).sum():.3f} vs total noise var {(sv**2).sum():.3f}")
    print(f"   common offset |mean c(x)| = {off.norm():.3f};  mean |c(x) - offset| = {(c - off).norm(dim=1).mean():.3f};"
          f"  c linear in x: max dev {lin:.1e}")
    print(f"   k-means NMI: samples z {km(z):.3f} | posterior means c(x) {km(c):.3f} | raw pixels x {km(xf):.3f}")
    print(f"   Vendi (normalize=True): samples z {vendi.score_dual(z.numpy(), normalize=True):.2f} | centred z"
          f" {vendi.score_dual((z - z.mean(0)).numpy(), normalize=True):.2f} | means c(x) {vendi.score_dual(c.numpy(), normalize=True):.2f}", flush=True)
