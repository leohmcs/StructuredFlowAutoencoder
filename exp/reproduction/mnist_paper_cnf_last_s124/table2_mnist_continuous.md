| Model | Source | log p(x|z) ↑ | log p(z|x) ↑ | Vendi ↑ | SSIM ↑ | NMI (OOD) ↑ | ARI (OOD) ↑ | Vendi x ↑ | SSIM (code protocol) ↑ | checkpoint epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| VAE | paper | -453.648 | -85.448 | 63.286 | 0.419 | 0.039 (0.033) | 0.017 (0.012) |  |  |  |
| VAE | reproduced | not evaluated |  |  |  |  |  |  |  |  |
| VampVAE | paper | -584.845 | 155.820 | 1.140 | 0.866 | 0.006 (0.006) | 0.000 (0.000) |  |  |  |
| VampVAE | reproduced | not in repo | not in repo | not in repo | not in repo | not in repo | not in repo |  |  |  |
| Latent FM | paper | – | – | 8.380 | 0.980 | 0.488 (0.392) | 0.381 (0.194) |  |  |  |
| Latent FM | reproduced | not in repo | not in repo | not in repo | not in repo | not in repo | not in repo |  |  |  |
| Latent FM (VAE) | paper | -910.925 | -11.192 | 19.631 | 0.697 | 0.309 (0.152) | 0.205 (0.073) |  |  |  |
| Latent FM (VAE) | reproduced | not in repo | not in repo | not in repo | not in repo | not in repo | not in repo |  |  |  |
| SFA | paper | -916.901 | 793.262 | 25.589 | 0.716 | 0.490 (0.394) | 0.356 (0.208) |  |  |  |
| SFA | reproduced | not evaluated |  |  |  |  |  |  |  |  |
| w/Deterministic | paper | -858.385 | – | 10.189 | 0.732 | 0.501 (0.333) | 0.379 (0.155) |  |  |  |
| w/Deterministic | reproduced | not evaluated |  |  |  |  |  |  |  |  |
| w/CNF Posterior | paper | -907.998 | 356.141 | 23.166 | 0.654 | 0.485 (0.325) | 0.355 (0.118) |  |  |  |
| w/CNF Posterior | reproduced | -1189.639 | -40.077 | 1.479 | 0.160 | 0.167 (0.295) | 0.106 (0.136) | 24.480 | 0.158 | 200 (last) |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **Vendi ↑**: higher = more diverse latent samples z ~ q(z|x) (effective number of distinct directions; at most the latent dimension). This is what the repository's MNIST test code reports as Vendi (`vendi.score_dual(z, normalize=True)`), and what the paper's Table 2 reports: its VAE value, 63.29, is what 3000 draws from a 64-dim standard normal give (63.33). The value depends on the number of images.
- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **NMI (OOD) ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of the latent clustering with the labels: KMeans on z with k given (k = 10; 20 for OOD). First value MNIST, in parentheses EMNIST (out of distribution).
- **ARI (OOD) ↑**: higher is better (0 = chance level, can be negative; 1 = perfect). First value MNIST, in parentheses EMNIST (out of distribution).
- **Vendi x ↑**: diversity of the posterior-predictive images (higher = more diverse). Not a quantity the paper's tables report. Noisy or unrelated samples also score high, so read it together with SSIM.
- **SSIM (code protocol) ↑**: higher is better (0 to 1). SSIM as returned by the authors' generate_img: first test image of each digit, 7 posterior-predictive samples each.
- **checkpoint epoch**: epoch of the evaluated checkpoint; best = lowest validation loss.
