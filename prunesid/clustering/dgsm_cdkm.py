"""
DGSM-CDKM clustering used as a drop-in replacement for PruneSID's PSCA grouping.

Accuracy-preserving acceleration:
  - Python/Torch wrapper for device I/O and k-means++ seeding
  - Numba FP64 core for CD / split / merge (see dgsm_cdkm_numba.py)
  - One host transfer per batch; batch axis may use prange
  - Decision path keeps strict '>' tie-breaks (no parallel reduce on tokens)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from . import dgsm_cdkm_numba as _nb

_WARM = False


def _ensure_warm() -> None:
    global _WARM
    if not _WARM and _nb._HAS_NUMBA:
        _nb.warmup(n=32, m=64, k=8)
        _WARM = True


def kmeans_plusplus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Select k initial center indices via k-means++. data: [n, m]."""
    n = data.shape[0]
    k = min(k, n)
    centers_idx = np.empty(k, dtype=np.int64)
    centers_idx[0] = int(rng.integers(0, n))
    closest_dist_sq = np.full(n, np.inf, dtype=np.float64)
    data64 = data.astype(np.float64, copy=False)

    for c in range(1, k):
        last = data64[centers_idx[c - 1]]
        dist_sq = np.sum((data64 - last) ** 2, axis=1)
        np.minimum(closest_dist_sq, dist_sq, out=closest_dist_sq)
        total = float(closest_dist_sq.sum())
        if total <= 0 or not np.isfinite(total):
            centers_idx[c:] = rng.choice(n, size=k - c, replace=False)
            break
        probs = closest_dist_sq / total
        centers_idx[c] = int(rng.choice(n, p=probs))
    return centers_idx


def dgsm_cdkm(
    data: np.ndarray,
    k: int,
    init_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_cd_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DGSM-CDKM on one sample.

    Args:
        data: [n, m] points
        k: number of clusters (aligned with PSCA's K)
        max_cd_iters: cap on final CD; stops early when a pass moves 0 points
        do_split_merge: if False, skip oversplit/merge
    """
    if not _nb._HAS_NUMBA:
        raise RuntimeError(
            "DGSM-CDKM requires numba. Install numba or use --group_method psca."
        )
    _ensure_warm()
    if rng is None:
        rng = np.random.default_rng(0)

    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    data64 = np.ascontiguousarray(data_f32, dtype=np.float64)
    n, _m = data64.shape
    if n == 0:
        raise ValueError("empty data")
    k = int(max(1, min(k, n)))

    if init_indices is None:
        init_indices = kmeans_plusplus_init(data_f32, k, rng)
    else:
        init_indices = np.asarray(init_indices, dtype=np.int64)[:k]
    init_indices = np.ascontiguousarray(init_indices, dtype=np.int64)

    empty_center_idx = int(rng.integers(0, n))
    labels, centers, soft = _nb.run_single(
        data64,
        k,
        init_indices,
        int(max_cd_iters),
        bool(do_split_merge),
        empty_center_idx,
    )
    return labels, centers, soft


def batch_dgsm_cdkm(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_cd_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PSCA-compatible batch wrapper — real DGSM-CDKM per sample (Numba core).

    Returns:
        soft_scores: [B, T_patch, K]
        belong_components: [B, T_patch]
    """
    if not _nb._HAS_NUMBA:
        raise RuntimeError(
            "DGSM-CDKM requires numba. Install numba or use --group_method psca."
        )
    _ensure_warm()

    device = features.device
    dtype = torch.float32
    x = torch.sigmoid(features.to(dtype))

    if drop_cls:
        if x.shape[1] < 2:
            raise ValueError("features must include CLS + patches when drop_cls=True")
        tokens = x[:, 1:, :]
    else:
        tokens = x

    B, T, D = tokens.shape
    k = max(1, min(int(min_components), T))

    # One host transfer for the whole batch.
    tokens_np = (
        tokens.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
    )
    data_b = np.ascontiguousarray(tokens_np.astype(np.float64, copy=False))

    init_indices_b = np.empty((B, k), dtype=np.int64)
    empty_center_idx_b = np.empty(B, dtype=np.int64)
    for b in range(B):
        rng = np.random.default_rng(seed + b)
        init_indices_b[b] = kmeans_plusplus_init(tokens_np[b], k, rng)
        empty_center_idx_b[b] = int(rng.integers(0, T))

    if B == 1:
        labels, _centers, soft = _nb.run_single(
            data_b[0],
            k,
            init_indices_b[0],
            int(max_cd_iters),
            bool(do_split_merge),
            int(empty_center_idx_b[0]),
        )
        soft_scores = torch.from_numpy(np.ascontiguousarray(soft[:, :k]))
        belong = torch.from_numpy(np.ascontiguousarray(labels))
        soft_scores = soft_scores.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
        belong = belong.unsqueeze(0).to(device=device, dtype=torch.long, non_blocking=True)
        return soft_scores, belong

    labels_b, _centers_b, soft_b = _nb.run_batch(
        data_b,
        k,
        init_indices_b,
        int(max_cd_iters),
        bool(do_split_merge),
        empty_center_idx_b,
    )
    soft_scores = torch.from_numpy(np.ascontiguousarray(soft_b[:, :, :k])).to(
        device=device, dtype=dtype, non_blocking=True
    )
    belong = torch.from_numpy(np.ascontiguousarray(labels_b)).to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return soft_scores, belong


def warmup_dgsm_cdkm(n: int = 64, m: int = 64, k: int = 8) -> None:
    """Compile Numba kernels once (call before timed evaluation)."""
    global _WARM
    _nb.warmup(n=n, m=m, k=k)
    _WARM = True


# Back-compat alias expected by cdkm_aism dgsm-init path.
def _batch_dgsm_torch(
    data: torch.Tensor,
    k: int,
    seed: int = 0,
    max_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Device-resident entry used by CDKM-AISM init_mode='dgsm'.

    data: [B, T, D] already without CLS (caller responsibility).
    Returns soft_scores [B,T,K], labels [B,T].
    """
    soft, belong = batch_dgsm_cdkm(
        data,
        min_components=k,
        seed=seed,
        drop_cls=False,
        max_cd_iters=max_iters,
        do_split_merge=do_split_merge,
    )
    return soft, belong


__all__ = [
    "batch_dgsm_cdkm",
    "dgsm_cdkm",
    "kmeans_plusplus_init",
    "warmup_dgsm_cdkm",
    "_batch_dgsm_torch",
]
