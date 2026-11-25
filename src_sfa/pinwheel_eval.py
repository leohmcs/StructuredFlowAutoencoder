"""Shared helpers for the Pinwheel experiments (Table 1 / Figure 1).

- find_checkpoint: locate the checkpoint to evaluate or resume from in a run folder.
- w1_hat: the authors' W1 estimate, exactly as in runner_continuous.LightningModule.test_step.
- w1_raw: the same optimal-transport distance without the max-distance normalisation.
- true_pinwheel: fresh ground-truth draws, used for the data-vs-data reference (noise floor).
"""
import glob
import json
import os
import re

import numpy as np
import ot

from dataloader.dataloader_pinwheel import make_pinwheel


def find_checkpoint(log_path, which="best"):
    """Return the checkpoint path for `which` in {'best', 'last', <path to .ckpt>}.

    'best' picks the lowest val_loss parsed from the ModelCheckpoint filenames
    (ties broken by the later epoch).
    """
    if which not in ("best", "last"):
        if not os.path.isfile(which):
            raise FileNotFoundError(f"Checkpoint not found: {which}")
        return which

    if which == "last":
        path = os.path.join(log_path, "last.ckpt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No last.ckpt in {log_path}")
        return path

    pattern = re.compile(r"epoch=(\d+)-val_loss=(-?[\d.]+)\.ckpt$")
    candidates = []
    for path in glob.glob(os.path.join(log_path, "best-checkpoint-*.ckpt")):
        m = pattern.search(os.path.basename(path))
        if m:
            candidates.append((float(m.group(2)), -int(m.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No best-checkpoint-*.ckpt files in {log_path} (train first)")
    return min(candidates)[2]


def ckpt_tag(which):
    """Suffix for output files: '' for best, '_last' for last, '_<name>' for an explicit path."""
    if which == "best":
        return ""
    if which == "last":
        return "_last"
    return "_" + os.path.splitext(os.path.basename(which))[0]


def _emd(x_true, x_gen, normalise):
    x_true = np.asarray(x_true, dtype=np.float64)
    x_gen = np.asarray(x_gen, dtype=np.float64)
    w_true = np.full(len(x_true), 1.0 / len(x_true))
    w_gen = np.full(len(x_gen), 1.0 / len(x_gen))
    M = ot.dist(x_true, x_gen, "euclidean")
    if normalise:
        M /= M.max() * 0.1
    return float(ot.emd2(w_true, w_gen, M))


def w1_hat(x_true, x_gen):
    """Authors' metric: Euclidean OT cost with the cost matrix rescaled so its max is 10."""
    return _emd(x_true, x_gen, normalise=True)


def w1_raw(x_true, x_gen):
    """Plain empirical W1 (Euclidean cost, data units)."""
    return _emd(x_true, x_gen, normalise=False)


def true_pinwheel(n, num_classes):
    """n fresh i.i.d. ground-truth points (labels dropped), drawn from numpy's global RNG.

    The number of points per arm is random (multinomial), just as in a shuffled test batch or in a
    model's samples. Exactly n/num_classes per arm would make the data-vs-data reference too low:
    0.025 instead of 0.033 at n=500 (cost normalised to max 1), because W1 charges for moving mass
    between arms.
    """
    data = make_pinwheel(num_classes, n)[:, :2]  # n points per arm, stored arm by arm
    counts = np.random.multinomial(n, [1.0 / num_classes] * num_classes)
    x = np.concatenate([data[k * n:k * n + c] for k, c in enumerate(counts)])
    return x[np.random.permutation(n)]


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
