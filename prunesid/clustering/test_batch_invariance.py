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


def test_final_refine_none_keeps_merge_labels():
    """final_refine=none must not reassign via Euclidean argmin after merge."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2)
    x = torch.randn(2, 577, 32, device=device)
    imp = torch.rand(2, 576, device=device)
    soft_l, lab_l = batch_dgsm_km_att(
        x, min_components=8, seed=11, token_importance=imp, final_refine="lloyd"
    )
    soft_n, lab_n = batch_dgsm_km_att(
        x, min_components=8, seed=11, token_importance=imp, final_refine="none"
    )
    assert soft_l.shape == soft_n.shape == (2, 576, 8)
    assert lab_l.shape == lab_n.shape == (2, 576)
    # Soft still defined; labels may differ when Lloyd moves boundaries.
    # At minimum, none path must be batch-invariant itself.
    _, lab_n1 = batch_dgsm_km_att(
        x[0:1], min_components=8, seed=11, token_importance=imp[0:1], final_refine="none"
    )
    assert torch.equal(lab_n1[0], lab_n[0])


def test_dgsm_cd_att_smoke():
    """Early-stop CD + Att merge returns valid [B,N,K] soft and labels."""
    from prunesid.clustering.dgsm_cd_att import batch_dgsm_cd_att
    from prunesid.clustering.dgsm_cdkm import cdk_refine_early_stop
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3)
    x = torch.randn(2, 577, 32, device=device)
    imp = torch.rand(2, 576, device=device)
    soft, lab = batch_dgsm_cd_att(
        x,
        min_components=8,
        seed=5,
        token_importance=imp,
        cd_sse_rel_tol=0.05,
        cd_min_iters=2,
        cd_max_iters=10,
    )
    assert soft.shape == (2, 576, 8)
    assert lab.shape == (2, 576)
    assert int(lab.min()) >= 0 and int(lab.max()) < 8

    # Early-stop respects dual condition on a single sample.
    data = torch.sigmoid(x[0, 1:]).detach().cpu().numpy().astype(np.float32)
    labels, n_iters, sse = cdk_refine_early_stop(
        data, k=8, rng=np.random.default_rng(0), sse_rel_tol=0.05, min_cd_iters=2, max_cd_iters=10
    )
    assert labels.shape == (576,)
    assert n_iters >= 2
    assert sse >= 0.0


def test_spatial_nms_far_tokens_not_suppressed():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "prunesid_llava" / "clip_encoder.py"
    spec = importlib.util.spec_from_file_location("clip_encoder_nms_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    N, K = 16, 2  # 4x4 grid
    # Two identical features at opposite corners of the same group.
    feats = torch.zeros(1, N, 4)
    feats[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    feats[0, 15] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    feats[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    sim = torch.bmm(
        torch.nn.functional.normalize(feats, dim=-1),
        torch.nn.functional.normalize(feats, dim=-1).transpose(1, 2),
    )
    scores = torch.zeros(1, N, K)
    scores[0, 0, 0] = 10.0
    scores[0, 15, 0] = 9.0
    scores[0, 1, 1] = 8.0
    thr = torch.tensor([0.5])
    xy = mod._patch_xy_normalized(N, torch.device("cpu"), torch.float32)

    # Feature-only: token 15 suppressed by token 0 (sim≈1).
    keep0, _ = mod.batch_similarity_nms(sim, scores, thr, spatial_radius=None)
    assert int(keep0[0, 0].item()) == 1

    # Spatial r=0.25: corners are far (Chebyshev=1) → both kept in group 0.
    keep1, _ = mod.batch_similarity_nms(
        sim, scores, thr, spatial_radius=0.25, token_xy=xy
    )
    assert int(keep1[0, 0].item()) == 2


if __name__ == "__main__":
    test_kpp_batch_invariant()
    test_dgsm_kmeans_batch_invariant()
    test_dgsm_km_att_batch_invariant()
    test_final_refine_none_keeps_merge_labels()
    test_dgsm_cd_att_smoke()
    test_spatial_nms_far_tokens_not_suppressed()
    print("ok")
