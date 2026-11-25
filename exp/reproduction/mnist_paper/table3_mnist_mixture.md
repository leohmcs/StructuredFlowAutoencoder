| Model | Source | log p(x|z) ↑ | log p(z|x) ↑ | SSIM ↑ | softNMI ↑ | NMI ↑ | ARI ↑ | Vendi x ↑ | checkpoint epoch |
|---|---|---|---|---|---|---|---|---|---|
| Mixture-SVAE | paper | -1133 | 0.667 | 0.634 | 0.698 | 0.161 | 0.072 |  |  |
| Mixture-SVAE | reproduced | 290.878 | -23.838 | 0.513 | 0.800 | 0.000 | 0.000 | 35.017 | 113 (best) |
| Mixture-SFA | paper | -906 | 725 | 0.779 | 0.728 | 0.489 | 0.332 |  |  |
| Mixture-SFA | reproduced | -1295.583 | 1087.869 | 0.050 | 0.632 | 0.522 | 0.376 | 18.193 | 199 (best) |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **Vendi x ↑**: diversity of the posterior-predictive images (higher = more diverse). Not a quantity the paper's tables report. Noisy or unrelated samples also score high, so read it together with SSIM.
- **checkpoint epoch**: epoch of the evaluated checkpoint; best = lowest validation loss.
