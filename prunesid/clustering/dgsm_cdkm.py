"""
DGSM-CDKM clustering used as a drop-in replacement for PruneSID's PSCA grouping.

Hybrid accuracy-preserving acceleration:
  - NumPy/BLAS for distance matrices, reductions, final soft scores
  - Numba only for sequential CD / hierarchical merge / merge-back
  - One host transfer per batch; optional thread pool over images
  - Strict '>' tie-breaks; deterministic top-k dim selection
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import torch

from . import dgsm_cdkm_numba as _nb

_WARM = False


def _ensure_warm() -> None:
    global _WARM
    if not _WARM and _nb._HAS_NUMBA:
        # Compile kernels at realistic vision sizes once.
        _nb.warmup(n=64, m=128, k=8)
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


def _cluster_ave_var_from_F(F, temp0, temp1, point_sq, c) -> float:
    np_c = temp1[c]
    if np_c <= 1:
        return 0.0
    sum_x2 = float(point_sq @ F[:, c])
    sumX_dot = float(np.dot(temp0[:, c], temp0[:, c]))
    return sum_x2 - sumX_dot / np_c


def _select_top_dims(dim_scores: np.ndarray, true_split_num: int) -> np.ndarray:
    """Deterministic top-k by (-score, +index)."""
    m = dim_scores.shape[0]
    ksel = min(int(true_split_num), m)
    # lexsort: last key is primary. Want higher score first, then smaller index.
    order = np.lexsort((np.arange(m), -dim_scores))
    return order[:ksel].astype(np.int64, copy=False)


def _sync_stats_from_F(data64, F, temp0, temp1, temp2, k) -> None:
    for c in range(k):
        members = F[:, c] > 0
        temp1[c] = float(members.sum())
        if temp1[c] > 0:
            temp0[:, c] = data64[members].sum(axis=0)
            temp2[c] = float(np.dot(temp0[:, c], temp0[:, c]))
        else:
            temp0[:, c] = 0.0
            temp2[c] = 0.0


def _sse_from_cd_stats(
    temp1: np.ndarray, temp2: np.ndarray, point_sq: np.ndarray, k: int
) -> float:
    """SSE = Σ||x||² - Σ_c ||S_c||² / n_c."""
    total = float(point_sq.sum())
    term = 0.0
    for c in range(k):
        n_c = float(temp1[c])
        if n_c > 0.0:
            term += float(temp2[c]) / n_c
    return max(0.0, total - term)


def cdk_refine_early_stop(
    data: np.ndarray,
    k: int,
    init_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    sse_rel_tol: float = 0.05,
    min_cd_iters: int = 2,
    max_cd_iters: int = 10,
) -> Tuple[np.ndarray, int, float]:
    """
    CD-only structure refinement (no split/merge, no final argmin).

    Pipeline:
      k-means++ → assign → coordinate descent until
        (iter >= min_cd_iters) and (ΔSSE / SSE_prev < sse_rel_tol)
      or moved==0, or max_cd_iters.

    Returns:
      labels [N], n_iters used, final SSE
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
    n, m = data64.shape
    if n == 0:
        raise ValueError("empty data")
    k = int(max(1, min(k, n)))
    min_cd_iters = max(1, int(min_cd_iters))
    max_cd_iters = max(min_cd_iters, int(max_cd_iters))
    sse_rel_tol = float(sse_rel_tol)

    if init_indices is None:
        init_indices = kmeans_plusplus_init(data_f32, k, rng)
    else:
        init_indices = np.asarray(init_indices, dtype=np.int64)[:k]

    centers = data64[init_indices]
    point_sq = np.sum(data64 * data64, axis=1)
    cen_sq = np.sum(centers * centers, axis=1)
    dists = point_sq[:, None] + cen_sq[None, :] - 2.0 * (data64 @ centers.T)
    for j, idx in enumerate(init_indices):
        dists[idx, :] = np.inf
        dists[idx, j] = 0.0
    labels0 = np.argmin(dists, axis=1)

    F = np.zeros((n, k), dtype=np.float64)
    F[np.arange(n), labels0] = 1.0
    temp0 = np.zeros((m, k), dtype=np.float64)
    temp1 = np.zeros(k, dtype=np.float64)
    temp2 = np.zeros(k, dtype=np.float64)
    _sync_stats_from_F(data64, F, temp0, temp1, temp2, k)

    prev_sse = _sse_from_cd_stats(temp1, temp2, point_sq, k)
    n_iters = 0
    for it in range(1, max_cd_iters + 1):
        moved = int(_nb.one_cd(data64, F, temp0, temp1, temp2, point_sq, k))
        n_iters = it
        sse = _sse_from_cd_stats(temp1, temp2, point_sq, k)
        if prev_sse > 0.0:
            gain = (prev_sse - sse) / prev_sse
        else:
            gain = 0.0
        # Dual stop: enough iters AND relative SSE gain below α (or converged).
        if it >= min_cd_iters and (gain < sse_rel_tol or moved == 0):
            prev_sse = sse
            break
        if moved == 0 and it >= min_cd_iters:
            prev_sse = sse
            break
        prev_sse = sse

    labels = np.argmax(F[:, :k], axis=1).astype(np.int64)
    return labels, n_iters, float(prev_sse)


def _cdk_refine_one_sample(
    data_f32: np.ndarray,
    k: int,
    seed: int,
    sse_rel_tol: float,
    min_cd_iters: int,
    max_cd_iters: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels, _, _ = cdk_refine_early_stop(
        data_f32,
        k=k,
        rng=rng,
        sse_rel_tol=sse_rel_tol,
        min_cd_iters=min_cd_iters,
        max_cd_iters=max_cd_iters,
    )
    return labels


def batch_cdk_refine_early_stop(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    *,
    sse_rel_tol: float = 0.05,
    min_cd_iters: int = 2,
    max_cd_iters: int = 10,
) -> torch.Tensor:
    """
    Batch CD-only early-stop refine. Returns labels [B, T_patch].

    No split/merge and no final Euclidean reassignment — intended as the
    structure stage before SSE-split + attention-merge.
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

    B, T, _D = tokens.shape
    k = max(1, min(int(min_components), T))
    tokens_np = np.ascontiguousarray(
        tokens.detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32
    )

    prev_numba_threads = None
    try:
        from numba import get_num_threads, set_num_threads

        prev_numba_threads = get_num_threads()
        set_num_threads(1)
    except Exception:
        set_num_threads = None  # type: ignore

    max_workers = int(os.environ.get("DGSM_BATCH_WORKERS", "0"))
    if max_workers <= 0:
        max_workers = min(B, max(1, min(4, (os.cpu_count() or 4) // 2)))

    belong_list = [None] * B
    try:
        if B == 1 or max_workers == 1:
            for b in range(B):
                belong_list[b] = _cdk_refine_one_sample(
                    tokens_np[b],
                    k,
                    seed + b,
                    sse_rel_tol,
                    min_cd_iters,
                    max_cd_iters,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        _cdk_refine_one_sample,
                        tokens_np[b],
                        k,
                        seed + b,
                        sse_rel_tol,
                        min_cd_iters,
                        max_cd_iters,
                    ): b
                    for b in range(B)
                }
                for fut, b in futs.items():
                    belong_list[b] = fut.result()
    finally:
        if set_num_threads is not None and prev_numba_threads is not None:
            set_num_threads(prev_numba_threads)

    return torch.from_numpy(np.stack(belong_list, axis=0)).to(
        device=device, dtype=torch.long, non_blocking=True
    )


def dgsm_cdkm(
    data: np.ndarray,
    k: int,
    init_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_cd_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DGSM-CDKM on one sample (NumPy/BLAS + Numba CD/merge).
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
    n, m = data64.shape
    if n == 0:
        raise ValueError("empty data")
    k = int(max(1, min(k, n)))

    if init_indices is None:
        init_indices = kmeans_plusplus_init(data_f32, k, rng)
    else:
        init_indices = np.asarray(init_indices, dtype=np.int64)[:k]

    max_k = max(k, int(1.5 * k)) if do_split_merge else k
    max_k = min(max_k, n)
    split_num = min(3, m)

    centers = data64[init_indices]
    point_sq = np.sum(data64 * data64, axis=1)
    cen_sq = np.sum(centers * centers, axis=1)
    # BLAS GEMM for initial assignment
    dists = point_sq[:, None] + cen_sq[None, :] - 2.0 * (data64 @ centers.T)
    for j, idx in enumerate(init_indices):
        dists[idx, :] = np.inf
        dists[idx, j] = 0.0
    labels0 = np.argmin(dists, axis=1)

    F = np.zeros((n, max_k), dtype=np.float64)
    F[np.arange(n), labels0] = 1.0

    temp0 = np.zeros((m, max_k), dtype=np.float64)
    temp1 = np.zeros(max_k, dtype=np.float64)
    temp2 = np.zeros(max_k, dtype=np.float64)
    _sync_stats_from_F(data64, F, temp0, temp1, temp2, k)

    _nb.one_cd(data64, F, temp0, temp1, temp2, point_sq, k)
    _nb.one_cd(data64, F, temp0, temp1, temp2, point_sq, k)

    clu_num = k
    if do_split_merge and max_k > k:
        clu_ave_var = np.zeros(max_k, dtype=np.float64)
        for c in range(k):
            clu_ave_var[c] = _cluster_ave_var_from_F(F, temp0, temp1, point_sq, c)

        while clu_num < max_k:
            split_clu_no = int(np.argmax(clu_ave_var[:clu_num]))
            if clu_ave_var[split_clu_no] == 0.0 or temp1[split_clu_no] <= 1.0:
                break

            split_locs = np.flatnonzero(F[:, split_clu_no] > 0).astype(np.int64, copy=False)
            clu_split_dot_num = int(split_locs.shape[0])
            if clu_split_dot_num < 2:
                break

            true_split_clu_num = 1 << split_num
            true_split_num = split_num
            while clu_split_dot_num < true_split_clu_num:
                true_split_clu_num >>= 1
                true_split_num -= 1
            if true_split_num <= 0:
                true_split_num = 1
                true_split_clu_num = 2

            mean = temp0[:, split_clu_no] / clu_split_dot_num
            vals = data64[split_locs]
            dim_scores = np.abs(vals - mean[None, :]).sum(axis=0)
            chosen_dims = _select_top_dims(dim_scores, true_split_num)

            split_signs = np.zeros(clu_split_dot_num, dtype=np.int64)
            for dim in chosen_dims:
                center = mean[dim]
                ge = (data64[split_locs, dim] >= center).astype(np.int64)
                split_signs = (split_signs << 1) | ge

            _nb.hierarchical_merge_to_two(
                data64,
                F,
                temp0,
                temp1,
                temp2,
                split_clu_no,
                clu_num,
                split_locs,
                split_signs,
                true_split_clu_num,
            )
            clu_ave_var[split_clu_no] = _cluster_ave_var_from_F(
                F, temp0, temp1, point_sq, split_clu_no
            )
            clu_ave_var[clu_num] = _cluster_ave_var_from_F(
                F, temp0, temp1, point_sq, clu_num
            )
            clu_num += 1

        _nb.one_cd(data64, F, temp0, temp1, temp2, point_sq, clu_num)
        _nb.one_cd(data64, F, temp0, temp1, temp2, point_sq, clu_num)
        _nb.merge_back_to_k(F, temp0, temp1, temp2, clu_num, k, max_k)

    _sync_stats_from_F(data64, F, temp0, temp1, temp2, k)

    for _ in range(max_cd_iters):
        moved = int(_nb.one_cd(data64, F, temp0, temp1, temp2, point_sq, k))
        if moved == 0:
            break

    labels = np.argmax(F[:, :k], axis=1).astype(np.int64)
    centers_out = np.zeros((k, m), dtype=np.float64)
    for c in range(k):
        members = labels == c
        if members.any():
            centers_out[c] = data64[members].mean(axis=0)
        else:
            centers_out[c] = data64[int(rng.integers(0, n))]

    cen_sq = np.sum(centers_out * centers_out, axis=1)
    dists = point_sq[:, None] + cen_sq[None, :] - 2.0 * (data64 @ centers_out.T)
    labels = np.argmin(dists, axis=1).astype(np.int64)
    for c in range(k):
        members = labels == c
        if members.any():
            centers_out[c] = data64[members].mean(axis=0)

    soft_scores = (-dists).astype(np.float32)
    return labels, centers_out.astype(np.float32), soft_scores


def _cluster_one_sample(
    data_f32: np.ndarray,
    k: int,
    seed: int,
    max_cd_iters: int,
    do_split_merge: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels, _, soft = dgsm_cdkm(
        data_f32,
        k=k,
        rng=rng,
        max_cd_iters=max_cd_iters,
        do_split_merge=do_split_merge,
    )
    return labels, soft


def batch_dgsm_cdkm(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_cd_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PSCA-compatible batch wrapper.

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
    # Keep sigmoid on-device; one contiguous host copy afterwards.
    x = torch.sigmoid(features.to(dtype))

    if drop_cls:
        if x.shape[1] < 2:
            raise ValueError("features must include CLS + patches when drop_cls=True")
        tokens = x[:, 1:, :]
    else:
        tokens = x

    B, T, _D = tokens.shape
    k = max(1, min(int(min_components), T))

    tokens_np = np.ascontiguousarray(
        tokens.detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32
    )

    # Avoid OpenMP oversubscription: clustering threads x Numba threads.
    # Default: 1 Numba thread per worker; workers = min(B, cpu, 4).
    prev_numba_threads = None
    try:
        from numba import get_num_threads, set_num_threads

        prev_numba_threads = get_num_threads()
        set_num_threads(1)
    except Exception:
        set_num_threads = None  # type: ignore

    max_workers = int(os.environ.get("DGSM_BATCH_WORKERS", "0"))
    if max_workers <= 0:
        max_workers = min(B, max(1, min(4, (os.cpu_count() or 4) // 2)))

    soft_list = [None] * B
    belong_list = [None] * B

    try:
        if B == 1 or max_workers == 1:
            for b in range(B):
                labels, soft = _cluster_one_sample(
                    tokens_np[b], k, seed + b, max_cd_iters, do_split_merge
                )
                soft_list[b] = soft[:, :k]
                belong_list[b] = labels
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        _cluster_one_sample,
                        tokens_np[b],
                        k,
                        seed + b,
                        max_cd_iters,
                        do_split_merge,
                    ): b
                    for b in range(B)
                }
                for fut, b in futs.items():
                    labels, soft = fut.result()
                    soft_list[b] = soft[:, :k]
                    belong_list[b] = labels
    finally:
        if set_num_threads is not None and prev_numba_threads is not None:
            set_num_threads(prev_numba_threads)

    soft_scores = torch.from_numpy(np.stack(soft_list, axis=0)).to(
        device=device, dtype=dtype, non_blocking=True
    )
    belong = torch.from_numpy(np.stack(belong_list, axis=0)).to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return soft_scores, belong


def warmup_dgsm_cdkm(n: int = 64, m: int = 64, k: int = 8) -> None:
    """Compile Numba kernels once (call before timed evaluation)."""
    global _WARM
    _nb.warmup(n=n, m=m, k=k)
    _WARM = True


def _batch_dgsm_torch(
    data: torch.Tensor,
    k: int,
    seed: int = 0,
    max_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Entry used by CDKM-AISM init_mode='dgsm'. data: [B,T,D] without CLS."""
    return batch_dgsm_cdkm(
        data,
        min_components=k,
        seed=seed,
        drop_cls=False,
        max_cd_iters=max_iters,
        do_split_merge=do_split_merge,
    )


__all__ = [
    "batch_dgsm_cdkm",
    "dgsm_cdkm",
    "kmeans_plusplus_init",
    "warmup_dgsm_cdkm",
    "_batch_dgsm_torch",
    "cdk_refine_early_stop",
    "batch_cdk_refine_early_stop",
]
