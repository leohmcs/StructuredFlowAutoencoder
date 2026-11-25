| Method | runs | repo eval: W1_hat as coded | repo eval: /10 | repo eval: floor /10 | n=5000: W1 | n=5000: floor | n=5000: raw W1 | paper |
|---|---|---|---|---|---|---|---|---|
| SFA, CNF posterior (shipped config) | 3 | 0.4028 +/- 0.0140 | 0.0403 +/- 0.0014 | 0.0340 +/- 0.0014 | 0.0231 +/- 0.0059 | 0.0080 | 0.676 +/- 0.170 | 0.024 |
| SFA, Gaussian posterior (paper App. B.1) | 3 | 0.3895 +/- 0.0097 | 0.0390 +/- 0.0010 | 0.0343 +/- 0.0011 | 0.0205 +/- 0.0047 | 0.0080 | 0.604 +/- 0.134 | 0.024 |
| Mixture-SFA | 3 | 0.2458 +/- 0.0670 | 0.0246 +/- 0.0067 | 0.0148 +/- 0.0040 | 0.0220 +/- 0.0114 | 0.0080 | 0.648 +/- 0.329 | 0.046 |
| Mixture-SVAE (our baseline) | 3 | 0.3767 +/- 0.0473 | 0.0377 +/- 0.0047 | 0.0114 +/- 0.0027 | 0.0317 +/- 0.0028 | 0.0080 | 0.954 +/- 0.090 | 0.457 |
| FM (our baseline) | 3 | 0.3264 +/- 0.0246 | 0.0326 +/- 0.0025 | 0.0341 +/- 0.0007 | 0.0088 +/- 0.0010 | 0.0080 | 0.259 +/- 0.027 | 0.025 |

Checkpoint: best (FM has one final model). W1 values except 'as coded' use a cost normalised to max 1, the paper's scale. Floors are real-vs-real scores, i.e. a perfect model.

Latent structure (scripts/pinwheel_latent_eval.py, on the Figure 1 samples):

| Method | runs | arm from z (chance ~0.20) | cluster NMI | cluster ARI | clusters used (of 5), per seed |
|---|---|---|---|---|---|
| SFA, CNF posterior (shipped config) | 3 | 0.80 +/- 0.08 | - | - | - |
| SFA, Gaussian posterior (paper App. B.1) | 3 | 0.28 +/- 0.12 | - | - | - |
| Mixture-SFA | 3 | 0.85 +/- 0.04 | 0.83 +/- 0.07 | 0.72 +/- 0.09 | 5, 4, 4 |
| Mixture-SVAE (our baseline) | 3 | 0.99 +/- 0.00 | 0.81 +/- 0.02 | 0.78 +/- 0.03 | 5, 5, 5 |
