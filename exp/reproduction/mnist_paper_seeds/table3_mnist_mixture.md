| Model | seeds (n) | log p(x|z) ↑ | log p(z|x) ↑ | SSIM ↑ | softNMI ↑ | NMI ↑ | ARI ↑ | Vendi x ↑ | NMI (train set) ↑ | ARI (train set) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Mixture-SFA | 3 (123, 124, 125) | -1281.182 ± 19.257 | 1087.872 ± 0.005 | 0.054 ± 0.004 | 0.635 ± 0.014 | 0.529 ± 0.057 | 0.391 ± 0.056 | 17.585 ± 1.261 | 0.509 ± 0.058 | 0.386 ± 0.059 |
| Mixture-SVAE | 3 (123, 124, 125) | 317.198 ± 134.135 | -24.789 ± 3.416 | 0.512 ± 0.010 | 0.672 ± 0.111 | 0.110 ± 0.096 | 0.034 ± 0.030 | 35.248 ± 0.647 | 0.118 ± 0.102 | 0.043 ± 0.037 |
| Mixture-SFA [mixture_alpha0: no batch-entropy term (alpha_mixture 0) instead of 0.01] | 3 (123, 124, 125) | -1292.050 ± 11.118 | 1087.788 ± 0.089 | 0.054 ± 0.003 | 0.768 ± 0.014 | 0.205 ± 0.096 | 0.044 ± 0.041 | 17.619 ± 1.082 | 0.200 ± 0.077 | 0.042 ± 0.034 |
| Mixture-SFA [mixture_logits: z conditioned on the sorted logits (repository) instead of the xi sample] | 3 (123, 124, 125) | -1248.315 ± 5.781 | 1087.878 ± 0.008 | 0.059 ± 0.002 | 0.597 ± 0.012 | 0.341 ± 0.058 | 0.199 ± 0.040 | 17.126 ± 0.909 | 0.325 ± 0.044 | 0.199 ± 0.037 |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **Vendi x ↑**: diversity of the posterior-predictive images (higher = more diverse). Not a quantity the paper's tables report. Noisy or unrelated samples also score high, so read it together with SSIM.
