| Model | seed | checkpoint epoch | log p(x|z) ↑ | log p(z|x) ↑ | SSIM ↑ | softNMI ↑ | NMI ↑ | ARI ↑ | Vendi x ↑ | NMI (train set) ↑ | ARI (train set) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mixture-SFA | 123 | 199 (best) | -1295.583 | 1087.869 | 0.050 | 0.632 | 0.522 | 0.376 | 18.193 | 0.512 | 0.382 |
| Mixture-SFA | 124 | 168 (best) | -1259.308 | 1087.870 | 0.059 | 0.649 | 0.589 | 0.454 | 18.427 | 0.567 | 0.447 |
| Mixture-SFA | 125 | 173 (best) | -1288.654 | 1087.878 | 0.054 | 0.622 | 0.475 | 0.345 | 16.135 | 0.450 | 0.330 |
| Mixture-SVAE | 123 | 113 (best) | 290.878 | -23.838 | 0.513 | 0.800 | 0.000 | 0.000 | 35.017 | 0.000 | 0.000 |
| Mixture-SVAE | 124 | 102 (best) | 198.175 | -28.580 | 0.501 | 0.606 | 0.180 | 0.055 | 35.979 | 0.181 | 0.062 |
| Mixture-SVAE | 125 | 127 (best) | 462.543 | -21.950 | 0.521 | 0.610 | 0.150 | 0.047 | 34.749 | 0.171 | 0.066 |
| Mixture-SFA [mixture_alpha0: no batch-entropy term (alpha_mixture 0) instead of 0.01] | 123 | 199 (best) | -1300.179 | 1087.803 | 0.050 | 0.775 | 0.143 | 0.021 | 17.998 | 0.145 | 0.022 |
| Mixture-SFA [mixture_alpha0: no batch-entropy term (alpha_mixture 0) instead of 0.01] | 124 | 168 (best) | -1279.380 | 1087.869 | 0.057 | 0.752 | 0.316 | 0.092 | 18.460 | 0.288 | 0.082 |
| Mixture-SFA [mixture_alpha0: no batch-entropy term (alpha_mixture 0) instead of 0.01] | 125 | 173 (best) | -1296.590 | 1087.693 | 0.055 | 0.777 | 0.156 | 0.020 | 16.399 | 0.167 | 0.023 |
| Mixture-SFA [mixture_logits: z conditioned on the sorted logits (repository) instead of the xi sample] | 123 | 199 (best) | -1254.988 | 1087.869 | 0.058 | 0.594 | 0.327 | 0.199 | 16.099 | 0.338 | 0.215 |
| Mixture-SFA [mixture_logits: z conditioned on the sorted logits (repository) instead of the xi sample] | 124 | 168 (best) | -1244.820 | 1087.883 | 0.061 | 0.587 | 0.292 | 0.159 | 17.827 | 0.277 | 0.157 |
| Mixture-SFA [mixture_logits: z conditioned on the sorted logits (repository) instead of the xi sample] | 125 | 173 (best) | -1245.137 | 1087.883 | 0.058 | 0.610 | 0.405 | 0.238 | 17.452 | 0.361 | 0.225 |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **checkpoint epoch**: epoch of the evaluated checkpoint; best = lowest validation loss.
- **log p(x|z) ↑**: higher is better. Log-likelihood (nats) of the test image under the decoder, given z ~ q(z|x), in the logit-transformed pixel space the code trains in. Compare only within one model family: a Gaussian decoder (VAE, Mixture-SVAE) can score very high from tight variances on the constant background pixels.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
- **SSIM ↑**: higher is better (0 to 1): structural similarity between each test image and its posterior-predictive sample.
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **Vendi x ↑**: diversity of the posterior-predictive images (higher = more diverse). Not a quantity the paper's tables report. Noisy or unrelated samples also score high, so read it together with SSIM.
