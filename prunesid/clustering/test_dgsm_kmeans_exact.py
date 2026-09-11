"""Smoke checks for DGSM-KM+."""

from __future__ import annotations

import torch

from prunesid.clustering.dgsm_kmeans import (
    _gram_merge_pair,
    _per_image_top_variance_dims,
    batch_dgsm_kmeans,
)


def test_batch_dgsm_kmeans_plus():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 577, 64, device=device)
    imp = torch.rand(2, 576, device=device)
    soft, lab = batch_dgsm_kmeans(
        x, min_components=8, seed=0, token_importance=imp, sse_rel_tol=1e-3
    )
    assert soft.shape[0] == 2 and lab.shape == (2, 576)
    soft2, lab2 = batch_dgsm_kmeans(
        x, min_components=8, seed=0, token_importance=imp, sse_rel_tol=1e-3
    )
    assert torch.equal(lab, lab2)


def test_fast_knobs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 577, 64, device=device)
    soft, lab = batch_dgsm_kmeans(
        x,
        min_components=8,
        seed=1,
        oversplit_ratio=1.125,
        split_num=2,
        sse_rel_tol=1e-3,
    )
    assert lab.unique().numel() <= 8


def test_per_image_dims():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.zeros(2, 32, 16, device=device)
    data[0, :, 0] = torch.linspace(-1, 1, 32, device=device)
    data[1, :, 7] = torch.linspace(-1, 1, 32, device=device)
    dims = _per_image_top_variance_dims(data, 4)
    assert int(dims[0, 0]) == 0 and int(dims[1, 0]) == 7


if __name__ == "__main__":
    test_per_image_dims()
    test_batch_dgsm_kmeans_plus()
    test_fast_knobs()
    print("ok")
