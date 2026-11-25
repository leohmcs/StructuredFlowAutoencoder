| Model | Run | Setting changed | SSIM ↑ | NMI ↑ | ARI ↑ | softNMI ↑ | Vendi ↑ | log p(z|x) ↑ | log p(x|z) ↑ | checkpoint epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| SFA | sfa | none (paper config) | 0.541 | 0.519 | 0.355 | – | 16.793 | 731.431 | -862.324 | 180 (best) |
| SFA | sfa_floor1e5 | posterior std floor 1e-5 (published code) instead of 1e-6 | 0.541 | 0.542 | 0.373 | – | 16.841 | 626.794 | -863.294 | 180 (best) |
| SFA | sfa_alpha | alpha 0.01 (repository's z^2 penalty) instead of 0 | 0.267 | 0.414 | 0.271 | – | 33.546 | 566.050 | -936.226 | 180 (best) |
| SFA | sfa_w800 | width 800 (repository; 3.35M parameters) instead of 618 (2.40M) | 0.552 | 0.512 | 0.372 | – | 16.454 | 718.892 | -836.745 | 196 (best) |
| SFA | sfa_legacy | training loop as published (flat lr 5e-4, no effective clipping) | 0.539 | 0.511 | 0.359 | – | 18.832 | 617.280 | -823.842 | 192 (best) |
| w/CNF Posterior | sfa_cnf | none (paper config) | 0.173 | 0.134 | 0.066 | – | 1.438 | -40.723 | -1177.928 | 116 (best) |
| w/CNF Posterior | sfa_cnf_alpha | alpha 0.01 (repository's z^2 penalty) instead of 0 | 0.203 | 0.081 | 0.040 | – | 2.257 | 170.893 | -893.784 | 116 (best) |
| Mixture-SFA | mixture | none (paper config) | 0.059 | 0.589 | 0.454 | 0.649 | – | 1087.870 | -1259.308 | 168 (best) |
| Mixture-SFA | mixture_alpha0 | no batch-entropy term (alpha_mixture 0) instead of 0.01 | 0.057 | 0.316 | 0.092 | 0.752 | – | 1087.869 | -1279.380 | 168 (best) |
| Mixture-SFA | mixture_logits | z conditioned on the sorted logits (repository) instead of the xi sample | 0.061 | 0.292 | 0.159 | 0.587 | – | 1087.883 | -1244.820 | 168 (best) |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **Vendi ↑**: higher = more diverse latent samples z ~ q(z|x) (effective number of distinct directions; at most the latent dimension). This is what the repository's MNIST test code reports as Vendi (`vendi.score_dual(z, normalize=True)`), and what the paper's Table 2 reports: its VAE value, 63.29, is what 3000 draws from a 64-dim standard normal give (63.33). The value depends on the number of images.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **checkpoint epoch**: epoch of the evaluated checkpoint; best = lowest validation loss.
