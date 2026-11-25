| Model | Run | Setting changed | SSIM ↑ | NMI ↑ | ARI ↑ | softNMI ↑ | Vendi ↑ | log p(z|x) ↑ | log p(x|z) ↑ | checkpoint epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| SFA | sfa | none (paper config) | 0.536 | 0.549 | 0.403 | – | 16.618 | 734.433 | -861.564 | 197 (best) |
| SFA | sfa_floor1e5 | posterior std floor 1e-5 (published code) instead of 1e-6 | 0.536 | 0.552 | 0.405 | – | 16.636 | 628.057 | -859.283 | 197 (best) |
| SFA | sfa_alpha | alpha 0.01 (repository's z^2 penalty) instead of 0 | 0.272 | 0.472 | 0.300 | – | 34.977 | 561.235 | -941.201 | 197 (best) |
| SFA | sfa_w800 | width 800 (repository; 3.35M parameters) instead of 618 (2.40M) | 0.541 | 0.556 | 0.389 | – | 16.132 | 719.329 | -841.683 | 198 (best) |
| SFA | sfa_legacy | training loop as published (flat lr 5e-4, no effective clipping) | 0.487 | 0.537 | 0.400 | – | 18.726 | 613.490 | -831.050 | 197 (best) |
| w/CNF Posterior | sfa_cnf | none (paper config) | 0.160 | 0.175 | 0.095 | – | 1.488 | -40.084 | -1199.381 | 187 (best) |
| w/CNF Posterior | sfa_cnf_alpha | alpha 0.01 (repository's z^2 penalty) instead of 0 | 0.217 | 0.079 | 0.027 | – | 1.644 | 205.220 | -878.285 | 187 (best) |
| Mixture-SFA | mixture | none (paper config) | 0.050 | 0.522 | 0.376 | 0.632 | – | 1087.869 | -1295.583 | 199 (best) |
| Mixture-SFA | mixture_alpha0 | no batch-entropy term (alpha_mixture 0) instead of 0.01 | 0.050 | 0.143 | 0.021 | 0.775 | – | 1087.803 | -1300.179 | 199 (best) |
| Mixture-SFA | mixture_logits | z conditioned on the sorted logits (repository) instead of the xi sample | 0.058 | 0.327 | 0.199 | 0.594 | – | 1087.869 | -1254.988 | 199 (best) |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **Vendi ↑**: higher = more diverse latent samples z ~ q(z|x) (effective number of distinct directions; at most the latent dimension). This is what the repository's MNIST test code reports as Vendi (`vendi.score_dual(z, normalize=True)`), and what the paper's Table 2 reports: its VAE value, 63.29, is what 3000 draws from a 64-dim standard normal give (63.33). The value depends on the number of images.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **checkpoint epoch**: epoch of the evaluated checkpoint; best = lowest validation loss.
