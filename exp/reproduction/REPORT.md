# Reproducing the MNIST / EMNIST results of *Structured Flow Autoencoders* (ICLR 2026)

Status as of 2026-11-19. The main models (SFA, SFA w/Deterministic, SFA w/CNF Posterior, Mixture-SFA, VAE, Mixture-SVAE) are trained with **three seeds** (123, 124, 125) and reported as mean ± sample standard deviation. The ablations use the same three seeds (Section 4). The original-config runs are one seed (123).

Scope: the paper's MNIST and EMNIST results only: Table 2, Table 3, the §4 run-time claim, and Figures 3b, 4, 6, 7, 8 and 9.

## What changed since the 2026-11-16 version

An audit of the code against the paper found that some of our own choices were unfair to the paper, and that three statements in the report were wrong. This version fixes both.

- **Mixture-SFA was trained without the repository's component-usage term.** We had removed it because Eq. 8 doesn't show it. It is the only term that keeps more than one mixture component in use. With it restored, clustering reproduces: NMI 0.529 ± 0.057 against the paper's 0.489 (it was 0.143).
- **The paper's log p(z|x) of 793 is not impossible.** It is the value of a 64-dimensional posterior sitting on a standard-deviation floor of 1e-6. The published code uses 1e-5. The paper config now uses 1e-6.
- **The paper's Table 2 "Vendi" is computed on the latent samples z, not on generated images.** Table 2 now compares that quantity.
- **"The implemented objective differs from the paper's" was wrong.** The single-sample estimate is what the paper prescribes in §3.1.
- **Each inferred setting now has a one-factor ablation** (Section 4), and every run stores the exact code it was trained from (`code_state.patch` in the run folder).
- **Added on 2026-11-18: SFA w/CNF Posterior with the paper config, three seeds, and its `alpha` ablation** (Table 2, Sections 2a and 4). Its implementation was reviewed against §3.1 first and needed no change.

## Summary

- **SFA (continuous latent): clustering reproduces, reconstruction doesn't.**
  - NMI 0.533 ± 0.015 against the paper's 0.490; ARI 0.381 ± 0.024 against 0.356.
  - SSIM 0.538 ± 0.003 against 0.716, and latent Vendi 16.7 ± 0.1 against 25.6.
  - This needs the paper's stated hyperparameters and no penalty on z². The fix of two training bugs adds a little SSIM on average (Section 4). With the repository's own settings SFA is far off: SSIM 0.225, NMI 0.173.
- **Mixture-SFA: clustering reproduces, generation fails.**
  - NMI 0.529 ± 0.057 against 0.489; ARI 0.391 ± 0.056 against 0.332.
  - Posterior-predictive samples are noise unrelated to the input: SSIM 0.054 ± 0.004 against 0.779. No setting we tried changes this (Section 7, point 3).
- **Our stochastic and deterministic SFA are indistinguishable** (SSIM 0.538 and 0.531, NMI 0.533 and 0.539). The posterior standard deviation shrinks to about 2.6e-6, so the "stochastic" model is deterministic in effect. The paper's own log p(z|x) says the same about its model, but the paper reports a Vendi of 25.6 against 10.2 for the two.
- **SFA w/CNF Posterior does not reproduce.**
  - SSIM 0.167 ± 0.007 against the paper's 0.654; NMI 0.158 ± 0.021 against 0.485; latent Vendi 1.5 against 23.2. The three seeds agree, and the repository's own settings give the same picture (SSIM 0.221, NMI 0.122).
  - The code implements the paper's posterior ODE correctly. But the vector field the repository builds for MNIST has no hidden layer, which makes the posterior a *linear* map of the image plus Gaussian noise (Section 2a).
  - Our inferred `alpha` = 0 is not what holds it back. With the repository's penalty on z² (`alpha` = 0.01) SSIM is 0.214 ± 0.009 and NMI 0.075 ± 0.009: one metric slightly better, the other worse, both far from the paper (Section 4).
- **The settings we inferred do not handicap the method in any way that matters.** Over three seeds, one ablation is slightly better than the paper config: the repository's width of 800 gives SSIM 0.549 ± 0.007 against 0.538 ± 0.003, with the same clustering. Every other alternative is equal or worse (Section 4).
- **The baselines are our own implementations,** because the repository contains none.
  - Our **VAE** is much stronger than the paper's: NMI 0.425 ± 0.024 against 0.039. The paper's VAE had collapsed onto its prior; ours hasn't. SFA still clusters better, by a much smaller margin than the paper reports.
  - Our **Mixture-SVAE** is unstable across seeds: NMI 0.000, 0.180 and 0.150 (paper: 0.161).
- **What still can't be explained from the paper and the published code:** the SSIM and Vendi gaps for SFA, the CNF-posterior results, the Mixture-SFA generation failure, and the paper's baselines. The published code is not the code that produced the paper's numbers (Section 7, point 1).

## 1. What was run

### Code and configurations

All SFA models use the authors' code (commit `05e8907`) plus the changes listed below. Two configurations are compared:

| Setting | `configs/mnist.yml` (repository) | `configs/mnist_paper.yml` (paper) | Basis | Status |
|---|---|---|---|---|
| Latent dimension | 32 | 64 | App. B.3; Fig. 8 and 9 captions; Table 2's VAE Vendi and SFA log p(z\|x) are only possible with 64 | stated |
| Posterior std floor (SFA) | 1e-5 | 1e-6 | Table 2: log p(z\|x) = 793.26; a 64-dim posterior on a 1e-6 floor gives 793.38 | pinned by a reported number |
| Likelihood-network activation | SiLU | softplus | App. B.3 (written about the mixture models) | stated |
| Likelihood-network width (3 hidden layers) | 800 (SFA: 3.27M parameters) | 618 (SFA: 2.40M parameters) | §4: "SFA (2.4M parameters)"; other shapes also give 2.4M | inferred; ablated |
| Regularizer on z, `alpha` (SFA, SFA w/CNF) | 0.01 | 0 | Eq. 6 has no such term | inferred; ablated |
| Component-usage term, `alpha_mixture` (Mixture-SFA) | 0.01 | 0.01 | not in Eq. 8; kept from the repository | repository value; ablated |
| Mixture model: z is conditioned on | the logits, sorted | a Gumbel-Softmax sample ξ | Eq. 8; App. B.3 | inferred; ablated |
| Learning-rate schedule / gradient clipping | as published (buggy, see below) | fixed | not in the paper; the authors' pendulum runner has the fixed version | inferred; ablated |

The paper doesn't state anything else, so both configurations keep the repository's values for everything else:
- AdamW, learning rate 5e-4, weight decay 1e-4, batch size 256, 201 epochs;
- 5000 training images per digit;
- logit-transformed pixels, float64.

**The rule behind the paper config:** follow the paper where it states a value, or where a number it reports pins the value down; otherwise keep the repository's value. Where we had to read between the lines, the setting is marked "inferred" and has an ablation.

**Bug fixes** (no change to the method's equations):

| Fix | Applies to |
|---|---|
| Shape error in the mixture posterior (a linear layer for ξ was commented out); the published Mixture-SFA crashed on its first batch. Re-enabling it agrees with App. B.3 ("a linear map … to vectors of the same size") | all mixture runs |
| CNF posterior crashed on image input (missing flatten) | CNF-posterior runs |
| Resuming an interrupted run restarted the cosine decay from its first step; it now continues where it stopped, from the last completed epoch | the two resumed CNF runs (below) |
| Evaluation commands couldn't load checkpoints; `--figure` called a missing method; OOD code had shape bugs; dependencies missing | evaluation only |
| Cosine learning-rate decay never took effect (rate stayed at 5e-4 for all 201 epochs) | paper-config runs only |
| Gradient clipping ran after the optimizer step, so it did nothing | paper-config runs only |

The **original-config runs were trained before the last two fixes.** They therefore show the authors' code as published, plus only the crash fixes.

**SFA w/CNF Posterior** is the repository's own model (`flow.fix_z: false`), with the paper config otherwise unchanged. Two of its three runs (seeds 123 and 125) were stopped by a crash inside PyTorch's thread pool at epochs 190 and 184 of 201, and were resumed from the last completed epoch. The resumed learning rates match an uninterrupted schedule exactly; only the random stream of the remaining epochs differs. Seed 124 ran without interruption.

**SFA w/Deterministic** is named in the paper but has no code in the repository. Ours is the authors' SFA with z set to the posterior mean in training and evaluation.

**Baselines (our implementations):** `baselines/vae.py` and `baselines/mixture_svae.py`.
- They follow App. B.1/B.3 and §4: diagonal-Gaussian MLP encoder and decoder, a Gumbel-Softmax network for q(ξ|x), a fixed prior, softplus activations, z_dim 64. They are sized to match SFA (2.40M and 2.51M parameters).
- They train and evaluate through the authors' own training and evaluation code, with only the loss replaced by the ELBO. Data, optimizer, schedule, checkpointing and metrics are therefore identical to SFA's.
- Choices the paper doesn't state (β = 1, network widths, variance floor, spacing of the fixed prior's components) are in the `vae:` and `mixture_svae:` sections of `configs/mnist_paper.yml`.

### Evaluation protocol

- **In-distribution:** 1000 MNIST test images, 100 per digit (evaluation seed 0), matching the paper's "held-out set of size 1000".
- **The repository's own protocol:** its `--sample` command evaluates on 3000 *training* images (300 per digit). The paper's VAE Vendi fits 3000 images, not 1000 (Section 7, point 7), so the clustering and latent metrics are also reported on that set (end of Sections 2 and 3). The conclusions are the same on both.
- **Out-of-distribution:** EMNIST `balanced`, classes 0–19 (digits 0–9 and letters A–J), 300 images per class. This is the set the authors' code uses.
- **Sampling:** one posterior sample z ~ q(z|x) per image, and one posterior-predictive sample x ~ p(x|z) per z. The posterior predictive is *sampled*, not taken at the decoder mean.
- **Checkpoint:** the one with the lowest validation loss, the authors' `ModelCheckpoint` convention. The validation images are drawn from the MNIST test split, the same split the evaluation set comes from.
- **Metrics:**
  - NMI/ARI (k-means on z, or argmax of q(ξ|x)), Vendi (on z), softNMI, log p(z|x), and "SSIM (code protocol)" use the authors' own functions.
  - SSIM on the posterior-predictive images of all 1000 test images, and log p(x|z), are computed by us from the models' own methods, with the same preprocessing. The repository has no code that reports SSIM, so the paper's SSIM protocol is unknown.
  - log-likelihoods are in nats, in the logit-transformed space the models are trained in. The paper doesn't say which space it uses.

## 2. Table 2: continuous latent (MNIST; EMNIST OOD in parentheses)

All metrics are higher-is-better (↑), as marked in the paper. "—" means not reported; "n/a" means the model isn't implemented in the repository. Our paper-config rows are mean ± std over three seeds; the original-config rows are one seed.

| Model | Source | log p(x\|z) ↑ | log p(z\|x) ↑ | Vendi (z) ↑ | SSIM ↑ | NMI (OOD) ↑ | ARI (OOD) ↑ |
|---|---|---|---|---|---|---|---|
| VAE | paper | −453.648 | −85.448 | 63.286 | 0.419 | 0.039 (0.033) | 0.017 (0.012) |
| VAE | **ours** (paper config) | 262.6 ± 77.4 | −11.3 ± 12.5 | 13.9 ± 3.4 | 0.503 ± 0.005 | 0.425 ± 0.024 (0.214 ± 0.011) | 0.288 ± 0.039 (0.084 ± 0.007) |
| VampVAE | paper | −584.845 | 155.820 | 1.140 | 0.866 | 0.006 (0.006) | 0.000 (0.000) |
| VampVAE | ours | n/a | n/a | n/a | n/a | n/a | n/a |
| Latent FM | paper | — | — | 8.380 | 0.980 | 0.488 (0.392) | 0.381 (0.194) |
| Latent FM | ours | n/a | n/a | n/a | n/a | n/a | n/a |
| Latent FM (VAE) | paper | −910.925 | −11.192 | 19.631 | 0.697 | 0.309 (0.152) | 0.205 (0.073) |
| Latent FM (VAE) | ours | n/a | n/a | n/a | n/a | n/a | n/a |
| SFA | paper | −916.901 | 793.262 | 25.589 | 0.716 | 0.490 (0.394) | 0.356 (0.208) |
| SFA | **ours, paper config** | −861.8 ± 0.5 | 731.7 ± 2.6 | 16.7 ± 0.1 | 0.538 ± 0.003 | 0.533 ± 0.015 (0.370 ± 0.007) | 0.381 ± 0.024 (0.185 ± 0.004) |
| SFA | ours, original config | −727.015 | 164.438 | 4.481 | 0.225 | 0.173 (0.147) | 0.091 (0.052) |
| SFA w/Deterministic | paper | −858.385 | — | 10.189 | 0.732 | 0.501 (0.333) | 0.379 (0.155) |
| SFA w/Deterministic | **ours, paper config** | −867.7 ± 1.6 | — | 16.5 ± 0.2 | 0.531 ± 0.003 | 0.539 ± 0.016 (0.366 ± 0.006) | 0.390 ± 0.018 (0.182 ± 0.006) |
| SFA w/CNF Posterior | paper | −907.998 | 356.141 | 23.166 | 0.654 | 0.485 (0.325) | 0.355 (0.118) |
| SFA w/CNF Posterior | **ours, paper config** | −1186.3 ± 11.5 | −41.2 ± 1.4 | 1.5 ± 0.0 | 0.167 ± 0.007 | 0.158 ± 0.021 (0.270 ± 0.013) | 0.088 ± 0.019 (0.121 ± 0.010) |
| SFA w/CNF Posterior | ours, original config | −866.063 | 149.049 | 1.277 | 0.221 | 0.122 (0.170) | 0.056 (0.066) |

Additional values. "Vendi x" is the Vendi score of the generated images, which the paper doesn't report. "SSIM (code protocol)" is what the authors' `generate_img` returns: the first test image of each digit with 7 samples each (seed 123).

| Run | Vendi x ↑ | SSIM (code protocol) ↑ | Checkpoint epochs |
|---|---|---|---|
| VAE, paper config | 34.6 ± 1.4 | 0.485 | 105, 77, 69 |
| SFA, paper config | 15.7 ± 0.1 | 0.536 | 197, 180, 187 |
| SFA w/Deterministic, paper config | 15.8 ± 0.1 | 0.520 | 172, 194, 186 |
| SFA w/CNF, paper config | 20.1 ± 2.7 | 0.155 | 187, 116, 190 |
| SFA, original config | 29.648 | 0.214 | 185 |
| SFA w/CNF, original config | 21.987 | 0.223 | 191 |

**The same checkpoints on the repository's `--sample` set** (3000 training images; mean ± std over three seeds):

| Model | Source | NMI ↑ | ARI ↑ | Vendi (z) ↑ |
|---|---|---|---|---|
| VAE | paper | 0.039 | 0.017 | 63.286 |
| VAE | ours | 0.405 ± 0.013 | 0.285 ± 0.024 | 14.1 ± 3.5 |
| SFA | paper | 0.490 | 0.356 | 25.589 |
| SFA | ours | 0.502 ± 0.012 | 0.368 ± 0.013 | 17.1 ± 0.1 |
| SFA w/Deterministic | paper | 0.501 | 0.379 | 10.189 |
| SFA w/Deterministic | ours | 0.511 ± 0.014 | 0.378 ± 0.014 | 16.9 ± 0.2 |
| SFA w/CNF Posterior | paper | 0.485 | 0.355 | 23.166 |
| SFA w/CNF Posterior | ours | 0.152 ± 0.007 | 0.091 ± 0.003 | 1.5 ± 0.0 |

**How to read Table 2:**
- **Vendi is the diversity of the latent samples z**, as the repository's MNIST test code computes it. It is at most the latent dimension (64).
  - The paper's VAE value, 63.29, is what 3000 draws from a 64-dimensional standard normal give (63.33 ± 0.01). Its VAE posterior had collapsed onto the prior, which is also why its NMI is 0.04.
  - The value depends on the number of images: 1000 such draws give 62.0.
- **log p(x|z) can't be compared across model families.** The VAE's positive values come from very tight per-pixel variances on the background pixels, which are exactly constant after the logit transform. They don't indicate better samples.
- **log p(z|x) is capped by each posterior's std floor.** A collapsed, deterministic posterior scores highest, so higher isn't automatically better.
  - At 64 dimensions the cap is 646 for a 1e-5 floor and 793.4 for a 1e-6 floor.
  - The paper's SFA, at 793.26, sits on a 1e-6 floor. Ours, at 731.7, is close to it: an average std of about 2.6e-6.
  - With the published code's 1e-5 floor our SFA gives 626.9 ± 1.1; nothing else changes (Section 4).

## 2a. SFA w/CNF Posterior: what the trained models look like

The CNF posterior reproduces neither the paper's numbers nor the quality of our Gaussian-posterior SFA. This section records what we checked.

**The implementation matches §3.1.** The posterior sample is the solution of the ODE dz/ds = r_s(z | x_t), started from prior noise, and gradients go through the adjoint method. Checked on the code:
- the adjoint gradient equals direct backpropagation through the solver (relative difference about 1e-9);
- the flow inverts to 1.6e-7;
- `log_prob` equals the analytic density.

**The vector field the repository builds for MNIST is linear.** `MNISTRunner` constructs it with `hidden_features=[ngf]*0`, an empty list, so r is one linear layer of (z, time features, a linear embedding of x). The posterior is therefore exactly

  z = G·x + c + M·ε, ε ~ N(0, I),

a linear map of the logit-transformed image plus Gaussian noise whose covariance is the same for every image. On the trained models, the mean is linear in x to 1e-7 and M doesn't depend on x. The paper calls the CNF family "maximally expressive"; this instance is less expressive than the Gaussian posterior, whose mean and variance come from a 3-hidden-layer network. We kept it because it is what the repository builds.

**What the three trained models have in common** (best checkpoints, 1000 held-out images):

| Quantity | SFA w/CNF, paper config (seeds 123, 124, 125) | SFA, paper config (seed 123) |
|---|---|---|
| Distance of the mean latent from the origin | 370, 247, 366 | 6.3 |
| Spread of the latents over images, per dimension (median std) | 12.0, 7.4, 11.7 | 3.6 |
| Posterior noise, per dimension (median) | 0.46, 0.46, 0.47 | about 2.6e-6 |
| k-means NMI of the noise-free posterior means | 0.176, 0.148, 0.175 | n/a |
| Vendi (z) after subtracting the mean latent | 13.9, 14.0, 13.9 | 16.3 |

For reference, k-means on the raw pixels of the same images gives NMI 0.512.

- **The latents sit far from the origin and all point the same way.** That is why Vendi, which the repository computes on length-normalised z, is 1.5. After centring it is 14, close to SFA's.
- **Noise isn't what limits clustering.** The noise-free posterior means cluster no better (NMI 0.15 to 0.18) than the samples. The learned linear map itself separates the digits far worse than the raw pixels do.
- **The posterior stays wide,** which is why log p(z|x) is −41.

**The paper's log p(z|x) points to a different training setup.** For a posterior of this form, log p(z|x) depends only on M. This reading assumes the paper's CNF model has 64 latent dimensions and the repository's linear form.
- The paper's 356.1 corresponds to a noise scale of about 9e-4 per dimension.
- Ours is 0.46.
- The original-config run, which has the repository's penalty on z² (`alpha` = 0.01) and 32 dimensions, reaches 2.3e-3.

The penalty on z² is the only term in the loss that pushes this flow to contract, so the paper's CNF run probably had one, although Eq. 6 doesn't show it. We tested that: the paper config with `alpha` = 0.01, on the same three seeds (123, 124, 125; best checkpoints from epochs 187, 116 and 137).

| Quantity | `alpha` 0 | `alpha` 0.01 | Paper |
|---|---|---|---|
| Posterior noise, per dimension (median) | 0.46, 0.46, 0.47 | 0.011, 0.019, 0.014 | about 9e-4 (from log p(z\|x)) |
| log p(z\|x) | −41.2 ± 1.4 | 188.3 ± 17.2 | 356.1 |
| log p(x\|z) | −1186.3 ± 11.5 | −885.5 ± 7.8 | −908.0 |
| Distance of the mean latent from the origin | 370, 247, 366 | 0.59, 0.58, 0.55 | — |
| Spread of the latents over images, per dimension (median std) | 12.0, 7.4, 11.7 | 0.026, 0.021, 0.021 | — |
| k-means NMI of the noise-free posterior means | 0.176, 0.148, 0.175 | 0.081, 0.093, 0.067 | — |
| SSIM | 0.167 ± 0.007 | 0.214 ± 0.009 | 0.654 |
| NMI | 0.158 ± 0.021 | 0.075 ± 0.009 | 0.485 |
| Vendi (z) | 1.5 ± 0.0 | 1.9 ± 0.3 | 23.2 |

- **The penalty moves the two likelihoods toward the paper's values,** which supports the guess that the paper's CNF run had one.
- **It does not make the model work.** The latents now shrink toward a single point instead of growing: they differ between images by 0.02 to 0.03 per dimension, around a common offset of about 0.6. SSIM rises a little and clustering gets worse, in each of the three seeds.
- **So neither value of `alpha` gives a usable CNF posterior with this vector field.** The Table 2 row keeps `alpha` = 0, the same setting as the other paper-config rows.

**Checkpoint choice doesn't matter here.** The CNF loss draws one time t per batch, so its validation loss is noisy and the "best" epoch is partly luck (187, 116, 190). The final weights (epoch 200) give the same results:

| Checkpoint | SSIM ↑ | NMI (OOD) ↑ | ARI (OOD) ↑ | Vendi (z) ↑ | log p(z\|x) ↑ | log p(x\|z) ↑ |
|---|---|---|---|---|---|---|
| lowest validation loss | 0.167 ± 0.007 | 0.158 ± 0.021 (0.270 ± 0.013) | 0.088 ± 0.019 (0.121 ± 0.010) | 1.5 ± 0.0 | −41.2 ± 1.4 | −1186.3 ± 11.5 |
| final (epoch 200) | 0.162 ± 0.004 | 0.168 ± 0.003 (0.279 ± 0.014) | 0.101 ± 0.007 (0.128 ± 0.007) | 1.5 ± 0.0 | −41.0 ± 1.5 | −1188.4 ± 8.0 |

## 3. Table 3: mixture latent (MNIST)

Our paper-config rows are mean ± std over three seeds; the original-config row is one seed.

| Model | Source | log p(x\|z) ↑ | log p(z\|x) ↑ | SSIM ↑ | softNMI ↑* | NMI ↑ | ARI ↑ |
|---|---|---|---|---|---|---|---|
| Mixture-SVAE | paper | −1133 | 0.667 | 0.634 | 0.698 | 0.161 | 0.072 |
| Mixture-SVAE | **ours** (paper config) | 317.2 ± 134.1 | −24.8 ± 3.4 | 0.512 ± 0.010 | 0.672 ± 0.111 | 0.110 ± 0.096 | 0.034 ± 0.030 |
| Mixture-SFA | paper | −906 | 725 | 0.779 | 0.728 | 0.489 | 0.332 |
| Mixture-SFA | **ours, paper config** | −1281.2 ± 19.3 | 1087.9 ± 0.0 | 0.054 ± 0.004 | 0.635 ± 0.014 | 0.529 ± 0.057 | 0.391 ± 0.056 |
| Mixture-SFA | ours, original config | −783.326 | 543.906 | 0.192 | 0.598 | 0.373 | 0.207 |

Per seed (123, 124, 125): Mixture-SFA NMI 0.522, 0.589, 0.475; Mixture-SVAE NMI 0.000, 0.180, 0.150. Checkpoint epochs: Mixture-SFA 199, 168, 173; Mixture-SVAE 113, 102, 127; original config 182.

On the repository's `--sample` set (3000 training images): Mixture-SFA NMI 0.509 ± 0.058, ARI 0.386 ± 0.059; Mixture-SVAE NMI 0.118 ± 0.102, ARI 0.043 ± 0.037.

Vendi of posterior-predictive images (not reported in the paper's Table 3): Mixture-SVAE 35.2 ± 0.6; Mixture-SFA 17.6 ± 1.3 (paper config) and 13.4 (original config).

\* **softNMI isn't reliably higher-is-better** with the authors' implementation (`utils.soft_nmi`). On 1000 images:
- a perfect clustering scores 0.750;
- an uninformative, uniform q(ξ|x) scores 0.600;
- putting every image in one component scores 0.800.

The score also depends on the number of images. Use NMI and ARI to judge clustering.

**What the mixture runs show:**
- **Mixture-SFA clusters as well as the paper reports,** with a distinct q(ξ|x) peak for most digits (Fig. 4).
- **Mixture-SFA can't generate.** Its posterior over z has collapsed onto the code's 1e-8 std floor: log p(z|x) = 1087.9 ≈ 64 × 17.0, the maximum. Its samples are unrelated to the input (Section 7, point 3).
- **Mixture-SVAE collapses in one seed of three** (seed 123: every test image in the same component).

## 4. Ablations of the paper config

Each run changes one setting of the paper config back to the repository's value. Mean ± std over the same three seeds as the main runs (123, 124, 125), on the 1000 held-out images. The reference rows are the rows of Tables 2 and 3.

**SFA**

| Setting changed | SSIM ↑ | NMI ↑ | ARI ↑ | Vendi (z) ↑ | log p(z\|x) ↑ | log p(x\|z) ↑ |
|---|---|---|---|---|---|---|
| none (paper config) | 0.538 ± 0.003 | 0.533 ± 0.015 | 0.381 ± 0.024 | 16.7 ± 0.1 | 731.7 ± 2.6 | −861.8 ± 0.5 |
| posterior std floor 1e-5 instead of 1e-6 | 0.539 ± 0.003 | 0.536 ± 0.019 | 0.379 ± 0.024 | 16.7 ± 0.1 | 626.9 ± 1.1 | −861.2 ± 2.0 |
| width 800 (3.35M parameters) instead of 618 | 0.549 ± 0.007 | 0.531 ± 0.023 | 0.380 ± 0.009 | 16.4 ± 0.2 | 720.6 ± 2.5 | −840.4 ± 3.2 |
| training loop as published (flat learning rate, no effective clipping) | 0.512 ± 0.026 | 0.536 ± 0.025 | 0.396 ± 0.035 | 18.8 ± 0.1 | 617.3 ± 3.7 | −828.2 ± 3.8 |
| `alpha` 0.01 (the repository's penalty on z²) instead of 0 | 0.263 ± 0.011 | 0.441 ± 0.029 | 0.283 ± 0.015 | 34.6 ± 0.9 | 562.8 ± 2.8 | −941.2 ± 5.0 |

**Mixture-SFA**

| Setting changed | SSIM ↑ | NMI ↑ | ARI ↑ | softNMI ↑ | log p(z\|x) ↑ | log p(x\|z) ↑ |
|---|---|---|---|---|---|---|
| none (paper config) | 0.054 ± 0.004 | 0.529 ± 0.057 | 0.391 ± 0.056 | 0.635 ± 0.014 | 1087.9 ± 0.0 | −1281.2 ± 19.3 |
| z conditioned on the sorted logits instead of the ξ sample | 0.059 ± 0.002 | 0.341 ± 0.058 | 0.199 ± 0.040 | 0.597 ± 0.012 | 1087.9 ± 0.0 | −1248.3 ± 5.8 |
| no component-usage term (`alpha_mixture` 0) | 0.054 ± 0.003 | 0.205 ± 0.096 | 0.044 ± 0.041 | 0.768 ± 0.014 | 1087.8 ± 0.1 | −1292.1 ± 11.1 |

Per seed (123, 124, 125), SSIM / NMI:
- SFA, paper config: 0.536 / 0.549, 0.541 / 0.519, 0.536 / 0.530
- floor 1e-5: 0.536 / 0.552, 0.541 / 0.542, 0.539 / 0.515
- width 800: 0.541 / 0.556, 0.552 / 0.512, 0.554 / 0.523
- training loop as published: 0.487 / 0.537, 0.539 / 0.511, 0.510 / 0.560
- `alpha` 0.01: 0.272 / 0.472, 0.267 / 0.414, 0.251 / 0.436
- Mixture-SFA, paper config: 0.050 / 0.522, 0.059 / 0.589, 0.054 / 0.475
- sorted logits: 0.058 / 0.327, 0.061 / 0.292, 0.058 / 0.405
- no component-usage term: 0.050 / 0.143, 0.057 / 0.316, 0.055 / 0.156
- SFA w/CNF, paper config: 0.160 / 0.175, 0.173 / 0.134, 0.167 / 0.164
- SFA w/CNF, `alpha` 0.01: 0.217 / 0.079, 0.203 / 0.081, 0.221 / 0.065

**SFA w/CNF Posterior**

| Setting changed | SSIM ↑ | NMI ↑ | ARI ↑ | Vendi (z) ↑ | log p(z\|x) ↑ | log p(x\|z) ↑ |
|---|---|---|---|---|---|---|
| none (paper config) | 0.167 ± 0.007 | 0.158 ± 0.021 | 0.088 ± 0.019 | 1.5 ± 0.0 | −41.2 ± 1.4 | −1186.3 ± 11.5 |
| `alpha` 0.01 (the repository's penalty on z²) instead of 0 | 0.214 ± 0.009 | 0.075 ± 0.009 | 0.030 ± 0.009 | 1.9 ± 0.3 | 188.3 ± 17.2 | −885.5 ± 7.8 |

**What the ablations show:**
- **One inferred setting costs a little SSIM: the width.** The repository's 800 gives SSIM 0.549 ± 0.007 against 0.538 ± 0.003 for 618, higher in each of the three seeds, with the same NMI and ARI. The difference, 0.011, is small next to the gap to the paper's 0.716. We keep 618 because it is the width that gives the paper's 2.4M parameters.
- **No other alternative beats the paper config** on SSIM, NMI or ARI by more than the spread between seeds.
- **The penalty on z² is the setting that matters most for SFA.** With the repository's value SSIM halves (0.263 against 0.538) and NMI drops from 0.533 to 0.441, in every seed. The paper's Eq. 6 has no such term, and dropping it is the choice that favours the method.
- **The floor doesn't matter** for anything except log p(z|x).
- **The training fixes help SSIM on average but not in every seed:** 0.538 ± 0.003 with them, 0.512 ± 0.026 without (0.487, 0.539, 0.510). Clustering is the same. Their clearest effect is that results vary less between seeds.
- **Mixture-SFA needs the component-usage term to cluster** (NMI 0.205 ± 0.096 without it, against 0.529 ± 0.057), and clusters better when z is conditioned on the ξ sample, as Eq. 8 says (0.341 ± 0.058 with the sorted logits).
- **Nothing here closes the SSIM gap for SFA or repairs Mixture-SFA's generation** (SSIM 0.050 to 0.061 in all nine Mixture-SFA runs).
- **For the CNF posterior the penalty on z² trades one failure for another:** SSIM 0.214 ± 0.009 against 0.167 ± 0.007, NMI 0.075 ± 0.009 against 0.158 ± 0.021, with the same direction in each seed. Both are far from the paper's 0.654 and 0.485 (Section 2a). The final-epoch weights give the same picture with less spread between seeds (SSIM 0.217 ± 0.001, NMI 0.084 ± 0.007, log p(z|x) 204.1 ± 1.3); the spread in log p(z|x) at the best checkpoints comes from their different epochs (187, 116, 137).
- Not ablated: latent dimension 64 and softplus, which the paper states.

## 5. Other reported numbers

| Quantity | Paper | Ours |
|---|---|---|
| SFA parameters | 2.4M | 2.40M (paper config); 3.27M (original config) |
| SFA s/epoch | 13.220 ± 1.848 | 10.5 ± 0.3 (paper config); 6.7 ± 0.1 (original config) |
| VAE s/epoch | 12.789 ± 2.011 | median 10.2 (mean 31.5 ± 57.3, inflated by an external slowdown) |
| CNF-posterior s/epoch | 167.460 ± 176.817 | 222.6 ± 169.2 (original config; median 159.1); median 154 (paper config, three runs at a time) |
| Ratio CNF / SFA | 12.7× | 33× (original config); about 15× (paper config) |

The paper's numbers come from an Apple M2 Pro; ours from an Apple Silicon Mac, CPU only. Our times are from the 2026-11-16 runs, which trained one at a time. The runs of this version trained four at a time, and the machine slept in between, so their times aren't usable. The architecture is unchanged, so the earlier times still apply. The paper-config CNF time is from seed 124, which trained alongside two other CNF runs on 4 threads each; it rose from about 110 to about 165 seconds per epoch over training. Our times are measured from the snapshot files written every 10 epochs, so they include validation. They also depend on the number of data-loader processes, which differs between runs: `num_workers` was 0 for the original-config SFA and 13 for the paper-config runs.

## 6. Figures: paper vs. ours

All of our figures come from the paper-config runs with seed 123 unless noted. The original-config figures are in `mnist/`; the other seeds' in `mnist_paper_s124/` and `mnist_paper_s125/`.

### Figure 7: posterior predictive (first column: test image; other columns: samples)

**Paper**

<img src="paper_figures/paper_fig7_posterior_predictive.png" width="760">

**Ours**

<table>
<tr><td align="center"><img src="mnist_paper/fig7a_posterior_predictive_vae.png" width="370"><br>(a) VAE (ours)</td>
<td align="center"><img src="mnist_paper/fig7b_posterior_predictive_sfa.png" width="370"><br>(b) SFA</td></tr>
<tr><td align="center"><img src="mnist_paper/fig7c_posterior_predictive_mixture_svae.png" width="370"><br>(c) Mixture-SVAE (ours)</td>
<td align="center"><img src="mnist_paper/fig7d_posterior_predictive_mixture.png" width="370"><br>(d) Mixture-SFA</td></tr>
</table>

- **(b) SFA** reproduces the right digit in the input's style, as in the paper, but with speckle noise.
- **(d) Mixture-SFA** produces noise unrelated to the input (see Section 7, point 3).
- **(a) VAE and (c) Mixture-SVAE:** our samples keep the digit's shape and are grainy, because pixels are sampled independently.
  - The paper's VAE samples are mostly the wrong digit.
  - The paper's Mixture-SVAE samples are clean. That suggests the paper plotted decoder means for that model but samples for the VAE.

### Figure 4: mixture models, real / generated / q(ξ|x)

**Paper**

<img src="paper_figures/paper_fig4_mixture_display.png" width="760">

**Ours**

<table>
<tr><td align="center"><img src="mnist_paper/fig4a_mixture_svae_display.png" width="370"><br>(a) Mixture-SVAE (ours)</td>
<td align="center"><img src="mnist_paper/fig4b_mixture_display.png" width="370"><br>(b) Mixture-SFA</td></tr>
</table>

- **Mixture-SVAE:** in this seed, q(ξ|x) puts all its mass on the same component for every digit. Seeds 124 and 125 don't collapse.
- **Mixture-SFA:** q(ξ|x) has a clear peak for most digits, on different components, as in the paper. Some digits share a component, and two (0 and 1 here) are spread over several.
- **Mixture-SFA's generated row is noise,** unlike the paper's.
- **The paper's** Mixture-SVAE has spread-out distributions.

### Figure 6: t-SNE of posterior samples z (perplexity 50)

**Paper**

<img src="paper_figures/paper_fig6_latent_tsne.png" width="760">

**Ours**

<table>
<tr><td align="center"><img src="mnist_paper/fig6a_latent_tsne_sfa.png" width="250"><br>(a) SFA</td>
<td align="center"><img src="mnist_paper/fig6c_latent_tsne_vae.png" width="250"><br>(c) VAE (ours)</td>
<td align="center"><img src="mnist_paper/fig6d_latent_tsne_mixture.png" width="250"><br>(d) SFA-Mixture</td></tr>
<tr><td align="center"><img src="mnist_paper/fig6e_latent_tsne_ood_sfa.png" width="250"><br>(e) SFA, EMNIST OOD</td>
<td align="center"><img src="mnist_paper/fig6g_latent_tsne_ood_vae.png" width="250"><br>(g) VAE, EMNIST OOD (ours)</td>
<td align="center"><img src="mnist_paper/fig6h_latent_tsne_mixture_svae.png" width="250"><br>(h) Mixture-SVAE (ours)</td></tr>
</table>

- **SFA, (a) and (e):** clearly clustered, as in the paper.
- **VAE, (c) and (g):** partly clustered. The paper's VAE panels look structureless.
- **SFA-Mixture (d):** partly clustered. The 1s are separate, the 0s and 2s form loose groups, and the remaining digits overlap.
- Panels (b) and (f), Latent FM, aren't reproduced.

### Figures 8 and 9: posterior samples of ξ (histograms) and z (3-D t-SNE)

**Paper:** Mixture-SVAE (Fig. 8), then Mixture-SFA (Fig. 9)

<img src="paper_figures/paper_fig8_mixture_svae_posterior.png" width="600">
<img src="paper_figures/paper_fig9_mixture_sfa_posterior.png" width="600">

**Ours**

<table>
<tr><td align="center"><img src="mnist_paper/fig8a_xi_posterior_mixture_svae.png" width="250"><br>8a Mixture-SVAE ξ</td>
<td align="center"><img src="mnist_paper/fig8b_z_posterior_tsne3d_mixture_svae.png" width="250"><br>8b Mixture-SVAE z</td>
<td align="center"><img src="mnist_paper/fig9a_xi_posterior_mixture.png" width="250"><br>9a Mixture-SFA ξ</td>
<td align="center"><img src="mnist_paper/fig9b_z_posterior_tsne3d_mixture.png" width="250"><br>9b Mixture-SFA z</td></tr>
</table>

Our ξ histograms show a single bar per image. That is how the authors' published plotting code works: `generate()` histograms the argmax of the logits over 100 repeats, which is the same value each time. For Mixture-SFA the bar is on different components for different digits. The paper's histograms have several bars per image, so they were produced by different code, probably one that samples ξ.

### Figure 3b: 1-D PCA of sampled trajectories

<table>
<tr><td align="center"><img src="paper_figures/paper_fig3b_trajectories.png" width="250"><br>Paper</td>
<td align="center"><img src="mnist_paper/fig3b_trajectories_sfa.png" width="330"><br>Ours, SFA</td>
<td align="center"><img src="mnist_paper/fig3b_trajectories_mixture.png" width="330"><br>Ours, Mixture-SFA</td></tr>
</table>

The repository has no code for this figure. Ours is a reconstruction (`fig3b()` in `reproduce_mnist.py`): ODE paths from shared noise, and posterior paths along the training interpolation.

The paper's paths look different:
- the observation paths stay at a constant, class-specific level instead of moving from shared noise to the data;
- the latent paths look like random walks.

So the paper used some other, unspecified procedure. Our SFA latent paths do diverge smoothly from shared noise and separate by digit, as the caption describes.

## 7. Why the results differ

1. **The published code is not the code that produced the paper's numbers.**
   - The published Mixture-SFA crashed on its first batch.
   - The MNIST evaluation and `--figure` commands couldn't run.
   - SFA's reported log p(z|x) of 793.26 needs a posterior std floor of 1e-6; the published code has 1e-5, which caps it at 646.
   - The repository's own configuration has a 32-dimensional latent and 3.27M parameters, against the paper's 64 and 2.4M.
   - The learning-rate and clipping bugs are fixed in the pendulum runner but not in the MNIST runners.
   - The paper's figures (ξ histograms, Fig. 3b) and its SSIM values need code that isn't in the repository.
   - The history is one commit ("consolidated the change"), and App. C says an LLM was used to refine the architecture code.
2. **Stated hyperparameters and training bugs (measured).**
   - Going from the repository's settings to the paper config moved SFA from SSIM 0.225 / NMI 0.173 to 0.538 / 0.533.
   - Of the settings we ablated, the penalty on z² accounts for most of it: putting it back gives 0.263 / 0.441. The training fixes account for a smaller part, and only on SSIM: without them, 0.512 / 0.536.
   - Latent dimension and activation, which the paper states, weren't ablated.
3. **The decoder is trained on a different z than it is given at generation time.** This follows the paper (Eq. 6 and Fig. 3a use q_t(z_t|x_t)); it isn't an implementation error.
   - In training, the decoder gets z_t ~ q_t(z | x_t), computed from the same noisy image x_t that it also receives directly. At generation time it gets a single z from the clean image, for all t.
   - How the decoder behaves on that clean-image z is therefore not fixed by the training objective. It depends on details the paper doesn't give, which makes the results sensitive to them.
   - **Mixture-SFA:** the published posterior is a linear map of the raw noisy x_t (its input normalization is commented out in the code), so z_t drifts strongly with t: a relative change of 0.83 ± 0.01 at t = 0.5. At t = 0.5, feeding the decoder the clean-image z gives a *higher* loss than another image's z_t (3.15 ± 0.05 against 1.64 ± 0.01; its own z_t gives 1.31). Generation fails in all nine paper-config Mixture-SFA runs here (three seeds each of the main model and its two ablations), with or without the settings we changed.
   - **SFA:** its posterior normalizes its input (a LayerNorm), drifts much less (0.18 ± 0.00 at t = 0.5), and its generation works. For its decoder the clean-image z is nearly as good as its own z_t (loss 1.55 ± 0.01 against 1.31) and far better than another image's (4.73 ± 0.09).
   - These measurements are from the evaluated checkpoints of the three paper-config runs of each model (mean ± std over seeds), on 300 held-out images, using the posterior mean.
4. **The estimator the paper prescribes pushes the posterior to be deterministic.**
   - Eqs. 6 and 8 put the expectation over q(z|x) *inside* the squared norm. §3.1 approximates it with one sample ("as commonly used in VAE"), and the code does the same. That single-sample loss equals the paper's objective *plus* a penalty on how much the velocity field varies with z.
   - The penalty vanishes when the posterior becomes deterministic, and that is what we observe: SFA's posterior std ends near the floor, and the stochastic and deterministic variants give the same results.
   - The paper's own log p(z|x) for SFA is exactly the value of a posterior on its floor. Its reported difference between stochastic and deterministic SFA (Vendi 25.6 against 10.2) is therefore not something this code can produce.
5. **The CNF posterior in the repository is a linear encoder.** See Section 2a. It fails with and without the penalty on z², in the paper config and in the repository's. The paper doesn't say what vector field its CNF row used; it describes the family as "maximally expressive", which the repository's one-layer field isn't.
6. **Underspecified baselines.**
   - The paper describes the VAE and Mixture-SVAE only loosely. Unstated choices (variance floors, prior placement, β, training length, checkpoint) decide whether they collapse.
   - The paper's VAE had collapsed onto its prior: its log p(z|x) of −85.4 is close to the −90.8 of an exact standard normal, and its Vendi is that of prior samples. Our VAE, trained identically to SFA, doesn't collapse.
   - Our Mixture-SVAE collapses onto one component in one seed of three. With the fixed prior's components far apart (not the setting used here) it collapsed within 5 epochs.
7. **Metric and protocol issues.**
   - The paper's Vendi is computed on latents, and its VAE value fits 3000 images, the size of the repository's `--sample` set of *training* images. The captions say "held-out set of size 1000". We report both sets; only the VAE's value pins the size, so for the other rows this is an inference.
   - softNMI as implemented rewards collapse and depends on the number of images (Section 3).
   - The log-likelihood space is unstated, and so is the SSIM protocol.
   - The paper's figures mix decoder means and samples.
   - Checkpoints are chosen by a noisy validation loss, on images from the same split as the evaluation set.

## 8. Not yet reproduced, and next steps

- **SFA w/CNF Posterior with a nonlinear vector field** (one or more hidden layers) hasn't been tried. It would depart from the repository, but it is closer to the paper's description of the family, and it is the remaining candidate for why the CNF row doesn't reproduce. Each run takes 5 to 9 hours on this machine, more with hidden layers.
- **Baselines not implemented:** VampVAE, Latent FM, Latent FM (VAE). Neither the repository nor the paper specifies them in enough detail.
- **Possible further checks, not from the paper:**
  - give Mixture-SFA's posterior the input normalization that SFA's posterior has, to test whether that is what breaks its generation;
  - ablate the latent dimension and the activation;
  - an intermediate `alpha` for SFA (the loss class's own default is 0.001): the paper's SFA Vendi of 25.6 lies between our 16.7 (`alpha` 0) and 34.6 (`alpha` 0.01).

## Appendix: reproducing this report

```bash
# paper config (default); --seed 123 is the default, 124 and 125 give the other seeds
python reproduce_mnist.py train --variant {sfa,sfa_det,sfa_cnf,mixture,vae,mixture_svae} [--seed 124]
# ablations (same seeds)
python reproduce_mnist.py train --variant {sfa_floor1e5,sfa_alpha,sfa_w800,sfa_legacy,sfa_cnf_alpha,mixture_alpha0,mixture_logits} [--seed 124]
python reproduce_mnist.py eval [--train-seed 124]      # -> exp/reproduction/mnist_paper[_s124]/
# CNF on its final weights
python reproduce_mnist.py eval --variants sfa_cnf --checkpoint last --out exp/reproduction/mnist_paper_cnf_last[_s124] [--train-seed 124]
python reproduce_mnist.py aggregate --train-seeds 123 124 125   # -> exp/reproduction/mnist_paper_seeds/
# original config
python reproduce_mnist.py eval --config mnist.yml --variants sfa sfa_cnf mixture   # -> exp/reproduction/mnist/
# measurements of Section 2a (CNF posterior) and Section 7, point 3 (posterior drift); they need the trained runs
python diagnostics/cnf_posterior.py
python diagnostics/posterior_drift.py
```

Runs are saved in `exp/logs/<config>_<variant>[_s<seed>]/`, each with the code it was trained from (`code_state.patch`). Each results folder contains the tables (`table2_mnist_continuous.*`, `table3_mnist_mixture.*`, `table_train_protocol.*`, `table_ablations.*`), the figures, the raw outputs of the authors' functions (`raw/<variant>/`), per-run `metrics.json`, and a README with the full protocol. `mnist_paper_seeds/` has the mean ± std tables and the per-seed values. The paper's figures were cropped from the PDF into `paper_figures/` for comparison.

In the git repository: the run folders with the checkpoints (5.2 GB) are not included, so `eval` and the diagnostics need the runs to be trained first. In `raw/<variant>/` only `metrics.json` is included, because its figures are copies of the ones in the results folder. Of the final-checkpoint folders (`mnist_paper_cnf_last*`) only the metrics and two tables are included. The compressed MNIST and EMNIST files the code reads are in `data/` and `data_test/`.

Three ablation runs reuse runs trained before the 2026-11-17 version: `sfa_floor1e5` seeds 123 and 124 (the former main SFA runs) and `mixture_alpha0` seed 123 (the former main Mixture-SFA run). They have no `code_state.patch`.
