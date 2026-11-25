| Model | Source | NMI ↑ | ARI ↑ | Vendi ↑ | softNMI ↑ | log p(z|x) ↑ |
|---|---|---|---|---|---|---|
| VAE | paper | 0.039 | 0.017 | 63.286 |  | -85.448 |
| VAE | reproduced | 0.419 | 0.310 | 10.186 |  | -25.337 |
| SFA | paper | 0.490 | 0.356 | 25.589 |  | 793.262 |
| SFA | reproduced | 0.508 | 0.363 | 17.027 |  | 723.329 |
| w/Deterministic | paper | 0.501 | 0.379 | 10.189 |  | – |
| w/Deterministic | reproduced | 0.495 | 0.367 | 16.800 |  | – |
| w/CNF Posterior | paper | 0.485 | 0.355 | 23.166 |  | 356.141 |
| w/CNF Posterior | reproduced | 0.146 | 0.093 | 1.508 |  | -42.307 |
| Mixture-SVAE | paper | 0.161 | 0.072 |  | 0.698 |  |
| Mixture-SVAE | reproduced | 0.171 | 0.066 |  | 0.673 |  |
| Mixture-SFA | paper | 0.489 | 0.332 |  | 0.728 |  |
| Mixture-SFA | reproduced | 0.450 | 0.330 |  | 0.689 |  |

↑ = higher is better (as marked in the paper). No metric here is lower-is-better.

- **NMI ↑**: higher is better (0 = no relation to the labels, 1 = perfect). Agreement of argmax q(xi|x) with the digit labels.
- **ARI ↑**: higher is better (0 = chance level, can be negative; 1 = perfect).
- **Vendi ↑**: higher = more diverse latent samples z ~ q(z|x) (effective number of distinct directions; at most the latent dimension). This is what the repository's MNIST test code reports as Vendi (`vendi.score_dual(z, normalize=True)`), and what the paper's Table 2 reports: its VAE value, 63.29, is what 3000 draws from a 64-dim standard normal give (63.33). The value depends on the number of images.
- **softNMI ↑**: the paper marks it ↑, but with the authors' soft_nmi higher is NOT reliably better: a perfect clustering scores 0.750, an uninformative uniform q(xi|x) 0.600, and all images in one component 0.800. Use NMI / ARI to judge clustering.
- **log p(z|x) ↑**: higher = more concentrated posterior (the paper marks it ↑). Log density (nats) of q(z|x) at its own sample. Capped by each model's std floor (per latent dimension: about 10.1 nats for a 1e-5 floor, 12.4 for 1e-6, 17.0 for 1e-8), and a posterior collapsed onto that floor scores highest, so higher is not automatically better. The paper's 793.26 for SFA is 64 x 12.4: a 64-dim posterior sitting on a 1e-6 floor.
