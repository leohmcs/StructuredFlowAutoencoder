"""Probabilistic PCA baseline for the HVG metrics in Table 4a.

Plain PCA is deterministic, so it has no p(z|x) and no generative model for Vendi(x).
Probabilistic PCA (Tipping & Bishop, 1999) does, in closed form:

    x = W z + mu + eps,   z ~ N(0, I),   eps ~ N(0, s2 I)
    posterior:  p(z|x) = N(M^-1 W^T (x - mu), s2 M^-1),   M = W^T W + s2 I

At the maximum-likelihood fit, W = U (L - s2 I)^(1/2), so M = L (the top PCA eigenvalues)
is diagonal. The metrics follow the SFA test step (runner_continuous.py, hvg.yml branch):
    log p(z|x)  log-density of a posterior sample z ~ p(z|x), averaged over cells
    Vendi(x)    vendi.score_dual on cells regenerated as W z + mu + eps, z ~ p(z|x)
    Vendi(z)    vendi.score_dual on the sampled latents
    NMI / ARI   k-means (k = data.n_classes) on the sampled latents vs. cell type
    EMD         optimal-transport distance between real and regenerated cells

Fitted on the 'train' split, evaluated on the same 500 random held-out cells as the
SFA, VAE and LatentFM evaluations (dataloader_hvg.hvg_eval_indices).

    python benchmarks/pca.py                  # writes exp/logs/hvg_pca/test_metrics.json
    python benchmarks/pca.py --n-components 50 --doc hvg_pca50
"""
import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import ot
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from vendi_score import vendi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataloader_hvg import hvg_eval_indices


def dense(X):
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Probabilistic PCA baseline on HVG")
    parser.add_argument("--config", default="hvg.yml")
    parser.add_argument("--data", default="data/kang_normalized_hvg.h5ad")
    parser.add_argument("--n-components", type=int, default=None,
                        help="Latent size (default: flow.z_dim from the config, 32)")
    parser.add_argument("--k", type=int, default=None,
                        help="k-means clusters (default: data.n_classes from the config, 7)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--exp", default="exp")
    parser.add_argument("--doc", default="hvg_pca")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config)) as f:
        cfg = yaml.safe_load(f)
    d = args.n_components or cfg["flow"]["z_dim"]
    k = args.k or cfg["data"]["n_classes"]
    rng = np.random.default_rng(args.seed)

    adata = ad.read_h5ad(args.data)
    split = adata.obs["split_B"].astype(str).values
    X_train = dense(adata[split == "train"].X)
    held_out = adata[hvg_eval_indices(split)]   # same 500 cells as the other HVG evaluations
    X = dense(held_out.X)
    y = held_out.obs["cell_type"].astype(str).values
    print(f"train {X_train.shape}, eval {X.shape}, {len(np.unique(y))} cell types, k={k}, d={d}")

    # ML probabilistic PCA from the sample eigendecomposition
    pca = PCA(n_components=d, random_state=args.seed).fit(X_train)
    lam = pca.explained_variance_             # top-d eigenvalues = diagonal of M
    s2 = pca.noise_variance_                  # mean of the discarded eigenvalues
    W = pca.components_.T * np.sqrt(np.maximum(lam - s2, 0.0))   # (p, d)
    mu = pca.mean_

    # posterior p(z|x): diagonal Gaussian
    post_mean = (X - mu) @ W / lam
    post_std = np.sqrt(s2 / lam)
    z = post_mean + post_std * rng.standard_normal(post_mean.shape)
    log_post_z = (-0.5 * ((z - post_mean) / post_std) ** 2
                  - np.log(post_std) - 0.5 * np.log(2 * np.pi)).sum(-1).mean()

    # regenerate x from the posterior sample, with observation noise
    x_gen = z @ W.T + mu + np.sqrt(s2) * rng.standard_normal(X.shape)

    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(z).labels_

    n = len(X)
    w = np.full(n, 1.0 / n)
    M = ot.dist(X, x_gen, "euclidean")
    M /= M.max() * 0.1

    metrics = {
        "test_nmi": normalized_mutual_info_score(y, labels),
        "test_ari": adjusted_rand_score(y, labels),
        "test_vendi": vendi.score_dual(z, normalize=True),
        "test_vendi_x": vendi.score_dual(x_gen, normalize=True),
        "test_log_post_z": log_post_z,
        "test_emd": ot.emd2(w, w, M),
        # references for reading the numbers above
        "ref_vendi_x_real": vendi.score_dual(X, normalize=True),
        "ref_vendi_x_recon_noisefree": vendi.score_dual(z @ W.T + mu, normalize=True),
        "ppca_noise_variance": s2,
        "ppca_posterior_std_median": float(np.median(post_std)),
        "ppca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "n_components": d,
        "k": k,
    }
    metrics = {key: float(v) if not isinstance(v, int) else v for key, v in metrics.items()}

    out_dir = os.path.join(args.exp, "logs", args.doc)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "test_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out}")

    cond = held_out.obs["condition"].astype(str).values if "condition" in held_out.obs else None
    png = os.path.join(out_dir, "samples", "test_latent.png")
    plot_latent(z, y, cond, labels, metrics, png, seed=args.seed)
    print(f"Saved {png}")


# Categorical slots in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"


def plot_latent(z, y, cond, kmeans_labels, metrics, path, seed=0):
    """t-SNE of the sampled latents, coloured by cell type, condition and k-means cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    emb = TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(z)  # as paper Fig. 5

    def panel(ax, groups, title):
        # largest group first, so small groups are drawn on top and stay visible
        names, counts = np.unique(groups, return_counts=True)
        order = names[np.argsort(-counts)]
        for i, g in enumerate(order):
            m = groups == g
            ax.scatter(emb[m, 0], emb[m, 1], s=16, c=SERIES[i % len(SERIES)],
                       edgecolors="white", linewidths=0.5, label=f"{g} ({m.sum()})")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, frameon=False,
                  fontsize=8, labelcolor=INK_2, handletextpad=0.2, columnspacing=0.8,
                  markerscale=1.2)

    panels = [(y, "Cell type")]
    if cond is not None:
        panels.append((cond, "Condition"))
    panels.append((np.array([f"cluster {c}" for c in kmeans_labels]),
                   f"k-means clusters (NMI {metrics['test_nmi']:.2f}, ARI {metrics['test_ari']:.2f})"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6.2), facecolor="white")
    for ax, (groups, title) in zip(axes, panels):
        panel(ax, groups, title)
    fig.suptitle(f"Probabilistic PCA latent ({z.shape[1]} dims, {len(z)} held-out cells), t-SNE",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
