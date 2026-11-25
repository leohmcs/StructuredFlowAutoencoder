# Baselines (`benchmarks/`)

This folder holds all of the repo's baseline models. Before November 2026 they were spread over three branches:
- `sinne`: `benchmarks/`
- `leo`: `baselines/`
- `dante/pinwheel`: `scripts/pinwheel_*.py`

Those branches had overlapping implementations, which are now merged into one set here. None of this is the authors' code: the paper's baselines aren't in the original repository.

## Models and runners

All runners are registered in [`src_sfa/main.py`](../src_sfa/main.py) and use the same config as the SFA runner for the same dataset.

| Model | File | Runners | Paper |
|---|---|---|---|
| VAE | [`vae.py`](vae.py) | `hvg_vae`, `pinwheel_vae`, `mnist_vae` | Table 2, Table 4a |
| LatentFM / LatentFM w/VAE | [`latentfm.py`](latentfm.py) | `{hvg,pinwheel,mnist}_latentfm`, `{hvg,pinwheel,mnist}_latentfm_vae` | Table 2, Table 4a |
| Mixture-SVAE (GMVAE) | [`mixture_svae.py`](mixture_svae.py) | `mnist_mixture_svae`, `pinwheel_mixture_svae` | Table 1 / Fig. 1, Table 3 |
| Plain flow matching (FM) | [`fm.py`](fm.py) | `pinwheel_fm` | Table 1 / Fig. 1 |
| Probabilistic PCA | [`pca.py`](pca.py) | standalone script | Table 4a |
| GLD-SVAE | `gld-svae.py` | not implemented yet (empty stub) | Table 4b |

Train and evaluate from the repo root:

```bash
python src_sfa/main.py --config pinwheel_discrete.yml --runner pinwheel_mixture_svae --ni           # train
python src_sfa/main.py --config pinwheel_discrete.yml --runner pinwheel_mixture_svae --sample --ni  # evaluate
```

The evaluation step writes `exp/logs/<runner>/test_metrics.json`.

For the MNIST paper tables, `reproduce_mnist.py train --variant vae` (or `--variant mixture_svae`) trains through these runners. `reproduce_mnist.py eval` then evaluates them; see the evaluation section below.

## One harness for every baseline

[`common.py`](common.py) is the only training and evaluation loop:
- `BaselineModule` provides the Lightning training and test steps.
- `BaselineRunner` provides the trainer, checkpointing, `train()` and `sample()`.
- `DataSpec` and `DATASETS` define each dataset: its shapes, default network sizes, and train and eval loaders.

The settings that are shared:
- **Optimizer:** AdamW with linear warmup then cosine decay, restarted for each training stage.
- **Gradient clipping:** `training.clipval`, falling back to `optim.clip_value`.
- **Checkpoints:** the best `val_loss` checkpoint plus `last.ckpt`. They're found with `utils.find_checkpoint` and `utils.find_resume_checkpoint`, the same helpers the SFA runners use.
- **Loss scale:** the losses are divided by the observed dimension (per-dimension negative ELBO).

The test step computes the same metrics for every model:

| Metric | What it is |
|---|---|
| `log_post_z` | log q(z\|x) of one posterior sample |
| `nmi`, `ari` | k-means on z, compared with the class labels |
| `vendi` | Vendi score of z |
| `vendi_x` | Vendi score of the reconstructions |
| `vendi_x_gen` | Vendi score of unconditional samples |
| `emd` | EMD between real data and reconstructions |
| `emd_gen` | EMD between real data and unconditional samples |
| `ssim` | SSIM of the reconstructions (images only) |
| `w1_raw_gen` | plain W1 without cost normalisation (pinwheel only) |
| `emd_ref`, `w1_raw_ref` | data-vs-data noise floor of the two distances (pinwheel only) |

Notes on the metrics:
- `emd` uses the SFA runners' normalisation: the cost matrix is rescaled so its largest entry is 10.
- Models without latents (FM) only report the generative metrics.
- Each model can add its own metrics. Mixture-SVAE adds `nmi_xi`, `ari_xi` and `soft_nmi`, computed from argmax q(ξ|x); LatentFM adds `log_prior_z`.

### Evaluating with the authors' test code

The Gaussian encoder and decoder in `common.py` have the same methods as SFA's posterior (`sfa.GaussianPrior`) and likelihood (`sfa_llk.LLK`). Mixture-SVAE reuses the authors' `CatNF_fixed` and `GaussianMixtureComponent` networks.

Because of this, `reproduce_mnist.py` can load a trained baseline and pass its networks to the authors' `LightningModule` or `FlowMatchingLightningModule`. The baselines are then scored by exactly the same test code as SFA, while still being trained by the shared loop above.

## Config: the `baseline:` section

Model-specific options go under `baseline.<model>:`, where `<model>` is one of `vae`, `latentfm`, `mixture_svae` or `fm`. Options set directly under `baseline:` apply to all baselines, and a value under `baseline.<model>:` overrides the shared one.

```yaml
baseline:
  fct: softplus      # every baseline
  layer_norm: false
  vae:
    enc_width: 630
    enc_depth: 2
```

**Network options (every model):**
- `enc_width`, `enc_depth`, `dec_width`, `dec_depth`. The defaults mirror the SFA runner for the dataset:
  - encoder width: `flow.ngf`
  - decoder width: `model.ngf`
- `layer_norm`: LayerNorm on x before the encoder. It's on by default except for pinwheel.
- `fct`: activation, one of silu, softplus, tanh or relu. The default comes from the dataset.

**Per-model options:** listed in each file's docstring.
- VAE: `beta`, `min_std`, `enc_min_std`.
- LatentFM: `ae_epochs`, `flow_depth`.
- Mixture-SVAE: `k`, `beta`, `prior_means` (`random` or `linspace`), `prior_scale`, `prior_std`, `temp`, `temp_min`, `xi_embed`, `min_std`.

The settings carried over from the old implementations:
- [`configs/mnist_paper.yml`](../configs/mnist_paper.yml) has `leo`'s MNIST settings:
  - VAE: 2 × 630 units, about 2.4M parameters, matching SFA.
  - Mixture-SVAE: 2 × 454 units, about 2.5M parameters; random prior means with scale 0.1.
  - Softplus activations.
- [`configs/pinwheel_discrete.yml`](../configs/pinwheel_discrete.yml) has `dante`'s pinwheel Mixture-SVAE:
  - Encoders 2 × 200 units, decoder 4 × 200.
  - Prior means at −4, −2, 0, 2, 4 with sd 0.5.
  - Gumbel temperature annealed from 1.0 to 0.5.

## What was merged, and what changed

| Before | Now |
|---|---|
| `sinne` `benchmarks/vae.py` and `leo` `baselines/vae.py` | `vae.py` |
| `leo` `baselines/mixture_svae.py` (MNIST) and `dante` `scripts/pinwheel_mixture_svae.py` | `mixture_svae.py` |
| `dante` `scripts/pinwheel_fm_baseline.py` | `fm.py` |
| `dante` `src_sfa/pinwheel_eval.py` | its W1 metrics are in `common.emd`; `true_pinwheel` is in `dataloader/dataloader_pinwheel.py`; checkpoint lookup uses `utils.find_checkpoint` |

Differences from the old implementations:

- **MNIST VAE and Mixture-SVAE** (from `leo`)
  - They're trained by the shared loop rather than the authors' loop.
  - The authors' loop used a warmup scheduler, clipping after the optimizer step, and an unscaled loss.
  - Evaluation through `reproduce_mnist.py` is unchanged.
- **Pinwheel Mixture-SVAE** (from `dante`)
  - Evaluation is now 20 batches of 500, like the other pinwheel runners, instead of one batch of 2,500.
  - q(z|x,ξ) is the authors' `GaussianMixtureComponent`, so its std floor is 1e-8 instead of 1e-4.
  - The model is trained by the shared loop, with warmup.
- **Pinwheel FM** (from `dante`)
  - It's trained by the shared loop: warmup plus cosine decay, and the best validation checkpoint is kept.
  - The old script used cosine decay without warmup and kept the final epoch's weights.

Because of these differences, results committed from the old implementations should be re-run before they're compared with new runs:
- `exp/logs/pinwheel_msvae_seed*/`
- `exp/logs/pinwheel_fm_seed*/`
- the VAE and Mixture-SVAE rows under `exp/reproduction/`

## Known issues

- `runner_mixture.FlowMatchingLightningModule.test_plot_latent` (authors' code) calls `TSNE(n_iter=...)`, which scikit-learn 1.9 rejects. It breaks the Mixture-SFA and Mixture-SVAE evaluation in `reproduce_mnist.py`. Renaming the argument to `max_iter` should fix it.
- GLD-SVAE (pendulum, Table 4b) isn't implemented yet.
