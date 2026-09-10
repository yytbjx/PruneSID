"""
CDKM-AISM clustering for PruneSID Stage-1 token grouping.

Faithful Numba/FP64 port of 源.cpp (cd / splitCachedCandidate), structured
like dgsm_cdkm.py:

  - Python/Torch wrapper for device I/O and k-means++ / DGSM seeding
  - Numba core for CD + AISM (see cdkm_aism_numba.py)
  - One host transfer per batch; batch axis may use prange
  - No Torch approximate fallback (numba required for eval fidelity)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import torch

from . import cdkm_aism_numba as _nb
from .dgsm_cdkm import kmeans_plusplus_init

InitMode = Literal["kmeans++", "dgsm"]

_WARM = False


def _ensure_warm() -> None:
    global _WARM
    if not _WARM and _nb._HAS_NUMBA:
        _nb.warmup(n=32, m=64, k=8)
        _WARM = True


def cdkm_aism(
    data: np.ndarray,
    k: int,
    init_indices: Optional[np.ndarray] = None,
    init_labels: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    initial_cd_passes: int = 10,
    max_big_loops: int = 5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run CDKM-AISM on one sample.

    Returns:
        labels: [n] int64
        soft_scores: [n, k] float32 (= -squared distance to final centers)
    """
    if not _nb._HAS_NUMBA:
        raise RuntimeError(
            "CDKM-AISM requires numba. Install numba or use --group_method psca."
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

    if init_labels is not None:
        labels0 = np.ascontiguousarray(init_labels, dtype=np.int64)
        init_idx = np.zeros(k, dtype=np.int64)
        use_indices = 0
    else:
        if init_indices is None:
            init_indices = kmeans_plusplus_init(data_f32, k, rng)
        init_idx = np.ascontiguousarray(init_indices, dtype=np.int64)[:k]
        labels0 = np.zeros(n, dtype=np.int64)
        use_indices = 1

    labels, dists = _nb.run_single(
        data64,
        k,
        labels0,
        init_idx,
        use_indices,
        int(initial_cd_passes),
        int(max_big_loops),
        float(eps),
    )
    soft = np.ascontiguousarray((-dists).astype(np.float32, copy=False))
    return labels, soft


def batch_cdkm_aism(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    init_mode: InitMode = "kmeans++",
    initial_cd_passes: int = 10,
    max_big_loops: int = 5,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PSCA-compatible batch wrapper — real CDKM-AISM per sample (Numba core).

    init_mode="kmeans++": pure CDKM-AISM (C++-style algorithm + k-means++ seed).
    init_mode="dgsm":     real DGSM-CDKM partition, then AISM CD + refine.

    Returns:
        soft_scores: [B, T_patch, K]
        belong_components: [B, T_patch]
    """
    if not _nb._HAS_NUMBA:
        raise RuntimeError(
            "CDKM-AISM requires numba. Install numba or use --group_method psca."
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

    B, T, _D = tokens.shape
    k = max(1, min(int(min_components), T))

    # One host transfer for the whole batch.
    tokens_np = (
        tokens.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
    )
    data_b = np.ascontiguousarray(tokens_np.astype(np.float64, copy=False))

    init_indices_b = np.zeros((B, k), dtype=np.int64)
    init_labels_b = np.zeros((B, T), dtype=np.int64)
    use_indices_b = np.ones(B, dtype=np.int64)

    if init_mode == "kmeans++":
        for b in range(B):
            rng = np.random.default_rng(seed + b)
            init_indices_b[b] = kmeans_plusplus_init(tokens_np[b], k, rng)
            use_indices_b[b] = 1
    elif init_mode == "dgsm":
        # Real DGSM-CDKM (same path as --group_method dgsm), then AISM refine.
        from .dgsm_cdkm import batch_dgsm_cdkm

        _soft_dgsm, belong_dgsm = batch_dgsm_cdkm(
            features,
            min_components=k,
            seed=seed,
            drop_cls=drop_cls,
            max_cd_iters=100,
            do_split_merge=True,
        )
        init_labels_b = (
            belong_dgsm.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        init_labels_b = np.ascontiguousarray(init_labels_b)
        # Fallback empty-cluster batches to k-means++ (AISM assumes non-empty K).
        for b in range(B):
            counts = np.bincount(init_labels_b[b], minlength=k)
            if counts.shape[0] < k or np.any(counts[:k] <= 0):
                rng = np.random.default_rng(seed + b)
                init_indices_b[b] = kmeans_plusplus_init(tokens_np[b], k, rng)
                use_indices_b[b] = 1
            else:
                use_indices_b[b] = 0
    else:
        raise ValueError(f"Unknown init_mode={init_mode!r}; expected 'kmeans++' or 'dgsm'")

    if B == 1:
        labels, dists = _nb.run_single(
            data_b[0],
            k,
            init_labels_b[0],
            init_indices_b[0],
            int(use_indices_b[0]),
            int(initial_cd_passes),
            int(max_big_loops),
            float(eps),
        )
        soft = np.ascontiguousarray((-dists[:, :k]).astype(np.float32, copy=False))
        soft_scores = torch.from_numpy(soft).unsqueeze(0).to(
            device=device, dtype=dtype, non_blocking=True
        )
        belong = torch.from_numpy(np.ascontiguousarray(labels)).unsqueeze(0).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        return soft_scores, belong

    labels_b, dists_b = _nb.run_batch(
        data_b,
        k,
        init_labels_b,
        init_indices_b,
        use_indices_b,
        int(initial_cd_passes),
        int(max_big_loops),
        float(eps),
    )
    soft_b = np.ascontiguousarray((-dists_b[:, :, :k]).astype(np.float32, copy=False))
    soft_scores = torch.from_numpy(soft_b).to(
        device=device, dtype=dtype, non_blocking=True
    )
    belong = torch.from_numpy(np.ascontiguousarray(labels_b)).to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return soft_scores, belong


def warmup_cdkm_aism(n: int = 64, m: int = 64, k: int = 8) -> None:
    """Compile Numba kernels once (call before timed evaluation)."""
    global _WARM
    _nb.warmup(n=n, m=m, k=k)
    _WARM = True


__all__ = ["batch_cdkm_aism", "cdkm_aism", "warmup_cdkm_aism"]
