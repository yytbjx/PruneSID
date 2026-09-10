"""Regression checks for faithful DGSM-CDKM Numba acceleration."""

from __future__ import annotations

import numpy as np
import torch

from prunesid.clustering.dgsm_cdkm import (
    batch_dgsm_cdkm,
    dgsm_cdkm,
    kmeans_plusplus_init,
    warmup_dgsm_cdkm,
    _batch_dgsm_torch,
)


def _sse(data: np.ndarray, labels: np.ndarray, k: int) -> float:
    data64 = data.astype(np.float64, copy=False)
    total = 0.0
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            continue
        mu = data64[mask].mean(axis=0)
        total += float(((data64[mask] - mu) ** 2).sum())
    return total


def test_determinism_labels_sse_soft():
    warmup_dgsm_cdkm(32, 32, 4)
    rng = np.random.default_rng(1)
    data = rng.normal(size=(80, 16)).astype(np.float32)
    init = kmeans_plusplus_init(data, 4, np.random.default_rng(7))
    lab1, _, soft1 = dgsm_cdkm(data, k=4, init_indices=init.copy(), rng=np.random.default_rng(0))
    lab2, _, soft2 = dgsm_cdkm(data, k=4, init_indices=init.copy(), rng=np.random.default_rng(0))
    assert np.array_equal(lab1, lab2)
    assert np.array_equal(soft1, soft2)
    assert abs(_sse(data, lab1, 4) - _sse(data, lab2, 4)) < 1e-8


def test_batch_shapes_and_alias():
    feat = torch.randn(2, 65, 32)
    soft, belong = batch_dgsm_cdkm(feat, min_components=8, drop_cls=True, seed=0)
    assert soft.shape == (2, 64, 8)
    assert belong.shape == (2, 64)
    soft2, belong2 = _batch_dgsm_torch(feat[:, 1:, :], k=8, seed=0, max_iters=50)
    assert soft2.shape == soft.shape
    assert belong2.shape == belong.shape


if __name__ == "__main__":
    test_determinism_labels_sse_soft()
    test_batch_shapes_and_alias()
    print("ok")
