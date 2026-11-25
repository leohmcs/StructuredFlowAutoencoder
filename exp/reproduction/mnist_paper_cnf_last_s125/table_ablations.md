| Model | Run | Setting changed | SSIM ↑ | NMI ↑ | ARI ↑ | softNMI ↑ | Vendi ↑ | log p(z|x) ↑ | log p(x|z) ↑ | checkpoint epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| w/CNF Posterior | sfa_cnf | none (paper config) | 0.166 | 0.166 | 0.103 | – | 1.498 | -42.705 | -1179.813 | 200 (last) |
| w/CNF Posterior | sfa_cnf_alpha | alpha 0.01 (repository's z^2 penalty) instead of 0 | 0.219 | 0.081 | 0.030 | – | 1.705 | 203.879 | -876.203 | 200 (last) |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **Vendi ↑**: higher = more diverse latent samples z ~ q(z|x) (effective number of distinct directions; at most the latent dimension). This is what the repository's MNIST test code reports as Vendi (`vendi.score_dual(z, normalize=True)`), and what the paper's Table 2 reports: its VAE value, 63.29, is what 3000 draws from a 64-dim standard normal give (63.33). The value depends on the number of images.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **checkpoint epoch**: epoch of the evaluated checkpoint; best = lowest validation loss.
