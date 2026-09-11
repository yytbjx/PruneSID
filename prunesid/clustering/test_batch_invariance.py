"""Batch-invariance tests for DGSM-KM / DGSM-KM-Att."""

from __future__ import annotations

import torch

from prunesid.clustering.dgsm_kmeans import _kmeans_plusplus, batch_dgsm_kmeans
from prunesid.clustering.dgsm_km_att import batch_dgsm_km_att


def test_kpp_batch_invariant():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(8, 64, 32, device=device)
    c8 = _kmeans_plusplus(x, k=8, seed=42)
    c1 = _kmeans_plusplus(x[3:4], k=8, seed=42)
    assert torch.equal(c1[0], c8[3])


def test_dgsm_kmeans_batch_invariant():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(8, 577, 64, device=device)
    imp = torch.rand(8, 576, device=device)
    _, lab8 = batch_dgsm_kmeans(x, min_components=8, seed=7, token_importance=imp)
    _, lab1 = batch_dgsm_kmeans(
        x[2:3], min_components=8, seed=7, token_importance=imp[2:3]
    )
    assert torch.equal(lab1[0], lab8[2]), (
        f"dgsm_kmeans B-invariant fail: mismatch={(lab1[0] != lab8[2]).sum().item()}"
    )


def test_dgsm_km_att_batch_invariant():
    """Same content at batch slot 5 vs B=1 must match (Att path)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    x = torch.randn(8, 577, 64, device=device)
    imp = torch.rand(8, 576, device=device)
    _, lab8 = batch_dgsm_km_att(x, min_components=8, seed=3, token_importance=imp)
    _, lab1 = batch_dgsm_km_att(
        x[5:6], min_components=8, seed=3, token_importance=imp[5:6]
    )
    assert torch.equal(lab1[0], lab8[5]), (
        f"dgsm_km_att B-invariant fail: mismatch={(lab1[0] != lab8[5]).sum().item()}"
    )


if __name__ == "__main__":
    test_kpp_batch_invariant()
    test_dgsm_kmeans_batch_invariant()
    test_dgsm_km_att_batch_invariant()
    print("ok")
