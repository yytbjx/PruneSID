"""Smoke / determinism checks for Exact dgsm_kmeans."""

from __future__ import annotations

import torch

from prunesid.clustering.dgsm_kmeans import (
    _gram_merge_pair,
    _kmeans_plusplus,
    batch_dgsm_kmeans,
)


def test_gram_matches_sst():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, K, D = 2, 10, 64
    torch.manual_seed(0)
    counts = torch.randint(1, 6, (B, K), device=device).float()
    sums = torch.randn(B, K, D, device=device)
    gram = torch.bmm(sums, sums.transpose(1, 2))
    sn = torch.diagonal(gram, dim1=1, dim2=2).clone()
    a = torch.tensor([0, 2], device=device)
    b = torch.tensor([1, 4], device=device)
    _gram_merge_pair(counts, sums, gram, sn, a, b)
    ref = torch.bmm(sums, sums.transpose(1, 2))
    assert torch.allclose(gram, ref, atol=1e-4, rtol=1e-4)


def test_kpp_incremental_runs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.randn(2, 64, 32, device=device)
    c = _kmeans_plusplus(data, 8, seed=0)
    assert c.shape == (2, 8, 32)


def test_batch_dgsm_kmeans_deterministic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 577, 128, device=device)
    s1, l1 = batch_dgsm_kmeans(x, min_components=8, seed=7)
    s2, l2 = batch_dgsm_kmeans(x, min_components=8, seed=7)
    assert torch.equal(l1, l2)
    assert torch.equal(s1, s2)
    assert l1.unique().numel() <= 8


if __name__ == "__main__":
    test_gram_matches_sst()
    test_kpp_incremental_runs()
    test_batch_dgsm_kmeans_deterministic()
    print("ok")
