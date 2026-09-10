"""
DGSM-CDKM clustering used as a drop-in replacement for PruneSID's PSCA grouping.

Accuracy-preserving acceleration of the original Python port:
  - Numba / vectorized coordinate descent (main speedup)
  - float32 features, float64 decision statistics (temp*, merge, variance)
  - CD runs until convergence (moved==0), capped by max_cd_iters
  - batch path calls real DGSM-CDKM (not Lloyd approx)
"""

from __future__ import annotations

import heapq
from typing import Optional, Tuple

import numpy as np
import torch


def kmeans_plusplus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Select k initial center indices via k-means++. data: [n, m]."""
    n = data.shape[0]
    k = min(k, n)
    centers_idx = np.empty(k, dtype=np.int64)
    centers_idx[0] = int(rng.integers(0, n))
    # float64 distances for stable sampling probabilities
    closest_dist_sq = np.full(n, np.inf, dtype=np.float64)

    for c in range(1, k):
        last = data[centers_idx[c - 1]]
        dist_sq = np.sum((data.astype(np.float64, copy=False) - last.astype(np.float64)) ** 2, axis=1)
        np.minimum(closest_dist_sq, dist_sq, out=closest_dist_sq)
        total = float(closest_dist_sq.sum())
        if total <= 0 or not np.isfinite(total):
            centers_idx[c:] = rng.choice(n, size=k - c, replace=False)
            break
        probs = closest_dist_sq / total
        centers_idx[c] = int(rng.choice(n, p=probs))
    return centers_idx


def _cluster_ave_var_from_F(
    F: np.ndarray, temp0: np.ndarray, temp1: np.ndarray, point_sq: np.ndarray, c: int
) -> float:
    np_c = temp1[c]
    if np_c <= 1:
        return 0.0
    sum_x2 = float(point_sq @ F[:, c])
    sumX_dot = float(np.dot(temp0[:, c], temp0[:, c]))
    return sum_x2 - sumX_dot / np_c


def _merge_tar_diff(temp0: np.ndarray, temp1: np.ndarray, i: int, j: int) -> float:
    ti = float(temp1[i])
    tj = float(temp1[j])
    if ti <= 0 or tj <= 0:
        return -np.inf
    # decision path: float64
    t0 = temp0.astype(np.float64, copy=False)
    numerator = -np.sum((t0[:, i] * tj - t0[:, j] * ti) ** 2)
    denom = ti * tj * (ti + tj)
    return float(numerator / denom)


def _one_cd_numpy(
    data: np.ndarray,
    F: np.ndarray,
    temp0: np.ndarray,
    temp1: np.ndarray,
    temp2: np.ndarray,
    point_sq: np.ndarray,
    active_k: int,
) -> int:
    """One CD pass (NumPy fallback). Returns #moves."""
    n = data.shape[0]
    moved = 0
    # Mirror one-hot F; updated in lockstep — equivalent to argmax(F[i]) per point
    labels = np.argmax(F[:, :active_k], axis=1)

    for i in range(n):
        p = int(labels[i])
        np_p = float(temp1[p])
        if np_p <= 0:
            continue
        xi = data[i].astype(np.float64, copy=False)
        xi_sq = float(point_sq[i])
        dots = temp0[:, :active_k].T @ xi

        m2 = float(temp2[p]) - 2.0 * float(dots[p]) + xi_sq
        if np_p == 1.0 or m2 == 0.0:
            m3 = 0.0
        else:
            m3 = m2 / (np_p - 1.0)
        best_M = float(temp2[p]) / np_p - m3
        best_q = p

        for j in range(active_k):
            if j == p:
                continue
            nj = float(temp1[j])
            if nj <= 0:
                continue
            m4 = float(temp2[j]) + 2.0 * float(dots[j]) + xi_sq
            Mj = m4 / (nj + 1.0) - float(temp2[j]) / nj
            if Mj > best_M:
                best_M = Mj
                best_q = j

        if best_q != p:
            q = best_q
            F[i, p] = 0.0
            F[i, q] = 1.0
            labels[i] = q
            p1 = temp0[:, p].copy()
            p2 = temp0[:, q].copy()
            temp0[:, p] -= xi
            temp0[:, q] += xi
            temp1[p] -= 1.0
            temp1[q] += 1.0
            temp2[p] = float(temp2[p] - 2.0 * np.dot(xi, p1) + xi_sq)
            temp2[q] = float(temp2[q] + 2.0 * np.dot(xi, p2) + xi_sq)
            moved += 1
    return moved


def _try_build_numba_one_cd():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=False)
    def _one_cd_numba(data, F, temp0, temp1, temp2, point_sq, active_k):
        n, m = data.shape
        moved = 0
        labels = np.empty(n, dtype=np.int64)
        for i in range(n):
            p = 0
            for j in range(active_k):
                if F[i, j] != 0.0:
                    p = j
                    break
            labels[i] = p

        for i in range(n):
            p = labels[i]
            np_p = temp1[p]
            if np_p <= 0:
                continue
            xi_sq = point_sq[i]

            dot_p = 0.0
            for d in range(m):
                dot_p += data[i, d] * temp0[d, p]
            m2 = temp2[p] - 2.0 * dot_p + xi_sq
            if np_p == 1.0 or m2 == 0.0:
                m3 = 0.0
            else:
                m3 = m2 / (np_p - 1.0)
            best_M = temp2[p] / np_p - m3
            best_q = p

            for j in range(active_k):
                if j == p:
                    continue
                nj = temp1[j]
                if nj <= 0:
                    continue
                dot_j = 0.0
                for d in range(m):
                    dot_j += data[i, d] * temp0[d, j]
                m4 = temp2[j] + 2.0 * dot_j + xi_sq
                Mj = m4 / (nj + 1.0) - temp2[j] / nj
                if Mj > best_M:
                    best_M = Mj
                    best_q = j

            if best_q != p:
                q = best_q
                old_dot_p = 0.0
                old_dot_q = 0.0
                for d in range(m):
                    old_dot_p += data[i, d] * temp0[d, p]
                    old_dot_q += data[i, d] * temp0[d, q]
                F[i, p] = 0.0
                F[i, q] = 1.0
                labels[i] = q
                for d in range(m):
                    temp0[d, p] -= data[i, d]
                    temp0[d, q] += data[i, d]
                temp2[p] = temp2[p] - 2.0 * old_dot_p + xi_sq
                temp2[q] = temp2[q] + 2.0 * old_dot_q + xi_sq
                temp1[p] -= 1.0
                temp1[q] += 1.0
                moved += 1
        return moved

    return _one_cd_numba


_one_cd_numba_impl = _try_build_numba_one_cd()


def _one_cd(data, F, temp0, temp1, temp2, point_sq, active_k):
    if _one_cd_numba_impl is not None:
        try:
            return int(
                _one_cd_numba_impl(data, F, temp0, temp1, temp2, point_sq, active_k)
            )
        except Exception:
            pass
    return _one_cd_numpy(data, F, temp0, temp1, temp2, point_sq, active_k)


def _hierarchical_merge_to_two_v2(
    X: np.ndarray,
    F: np.ndarray,
    temp0: np.ndarray,
    temp1: np.ndarray,
    temp2: np.ndarray,
    split_clu_no: int,
    new_clu: int,
    split_locs: np.ndarray,
    split_signs: np.ndarray,
    true_split_clu_num: int,
) -> None:
    """Faithful port of hierarchicalClusteringMerge from DGSM-CDKM.cpp."""
    clu_split_dot_num = len(split_locs)
    if clu_split_dot_num == 0:
        return

    # float64 decision statistics for merge ranking
    initial_temp0 = np.zeros((X.shape[1], true_split_clu_num), dtype=np.float64)
    initial_temp1 = np.zeros(true_split_clu_num, dtype=np.float64)
    for j in range(clu_split_dot_num):
        s = int(split_signs[j])
        s = max(0, min(s, true_split_clu_num - 1))
        initial_temp1[s] += 1.0
        initial_temp0[:, s] += X[split_locs[j]].astype(np.float64, copy=False)

    active = initial_temp1 > 0
    final_cluster_index = np.arange(true_split_clu_num, dtype=np.int64)
    cluster_count = int(active.sum())

    do_merge = clu_split_dot_num != 2
    while do_merge and cluster_count > 2:
        active_indices = np.where(active)[0]
        best_val = -np.inf
        best_a = best_b = -1
        for ii, a in enumerate(active_indices):
            for b in active_indices[ii + 1 :]:
                a_i = int(a)
                b_i = int(b)
                if initial_temp1[a_i] == 0 or initial_temp1[b_i] == 0:
                    continue
                val = _merge_tar_diff(initial_temp0, initial_temp1, a_i, b_i)
                if val > best_val:
                    best_val = val
                    best_a, best_b = a_i, b_i
        if best_a < 0:
            break
        initial_temp0[:, best_a] += initial_temp0[:, best_b]
        initial_temp1[best_a] += initial_temp1[best_b]
        for t in range(true_split_clu_num):
            if final_cluster_index[t] == best_b:
                final_cluster_index[t] = best_a
        active[best_b] = False
        cluster_count -= 1

    if do_merge:
        finals = np.where(active)[0]
        if len(finals) < 2:
            return
        f0, f1 = int(finals[0]), int(finals[1])
        for j in range(clu_split_dot_num):
            root = int(final_cluster_index[int(split_signs[j])])
            if root != f0:
                F[split_locs[j], split_clu_no] = 0.0
                F[split_locs[j], new_clu] = 1.0
        temp0[:, split_clu_no] = initial_temp0[:, f0]
        temp0[:, new_clu] = initial_temp0[:, f1]
        temp1[split_clu_no] = initial_temp1[f0]
        temp1[new_clu] = initial_temp1[f1]
        temp2[split_clu_no] = float(np.dot(temp0[:, split_clu_no], temp0[:, split_clu_no]))
        temp2[new_clu] = float(np.dot(temp0[:, new_clu], temp0[:, new_clu]))
    else:
        F[split_locs[1], split_clu_no] = 0.0
        F[split_locs[1], new_clu] = 1.0
        temp0[:, split_clu_no] = X[split_locs[0]].astype(np.float64, copy=False)
        temp0[:, new_clu] = X[split_locs[1]].astype(np.float64, copy=False)
        temp1[split_clu_no] = 1.0
        temp1[new_clu] = 1.0
        temp2[split_clu_no] = float(np.dot(temp0[:, split_clu_no], temp0[:, split_clu_no]))
        temp2[new_clu] = float(np.dot(temp0[:, new_clu], temp0[:, new_clu]))


def dgsm_cdkm(
    data: np.ndarray,
    k: int,
    init_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_cd_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DGSM-CDKM.

    Args:
        data: [n, m] points
        k: number of clusters (aligned with PSCA's K)
        max_cd_iters: cap on final CD; stops early when a pass moves 0 points
        do_split_merge: if False, skip oversplit/merge (faster CDKM-only mode)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Features: float32; CD / split-merge state: float64
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
    split_num = min(3, m)

    centers = data64[init_indices].copy()
    data_sq = np.sum(data64 * data64, axis=1, keepdims=True)
    cen_sq = np.sum(centers * centers, axis=1)
    dists = data_sq + cen_sq[None, :] - 2.0 * (data64 @ centers.T)
    for j, idx in enumerate(init_indices):
        dists[idx, :] = np.inf
        dists[idx, j] = 0.0
    labels0 = np.argmin(dists, axis=1)

    F = np.zeros((n, max_k), dtype=np.float64)
    F[np.arange(n), labels0] = 1.0

    temp0 = np.zeros((m, max_k), dtype=np.float64)
    temp1 = np.zeros(max_k, dtype=np.float64)
    temp2 = np.zeros(max_k, dtype=np.float64)
    point_sq = np.sum(data64 * data64, axis=1)

    for c in range(k):
        members = F[:, c] > 0
        temp1[c] = float(members.sum())
        if temp1[c] > 0:
            temp0[:, c] = data64[members].sum(axis=0)
            temp2[c] = float(np.dot(temp0[:, c], temp0[:, c]))

    for _ in range(2):
        _one_cd(data64, F, temp0, temp1, temp2, point_sq, k)

    clu_num = k
    if do_split_merge and max_k > k:
        clu_ave_var = np.zeros(max_k, dtype=np.float64)
        for c in range(k):
            clu_ave_var[c] = _cluster_ave_var_from_F(F, temp0, temp1, point_sq, c)

        while clu_num < max_k:
            split_clu_no = int(np.argmax(clu_ave_var[:clu_num]))
            if clu_ave_var[split_clu_no] == 0 or temp1[split_clu_no] <= 1:
                break

            split_locs = np.where(F[:, split_clu_no] > 0)[0]
            clu_split_dot_num = len(split_locs)
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
            chosen_dims = np.argpartition(-dim_scores, true_split_num - 1)[:true_split_num]

            split_signs = np.zeros(clu_split_dot_num, dtype=np.int64)
            for dim in chosen_dims:
                center = mean[dim]
                ge = data64[split_locs, dim] >= center
                split_signs = (split_signs << 1) | ge.astype(np.int64)

            _hierarchical_merge_to_two_v2(
                data64, F, temp0, temp1, temp2, split_clu_no, clu_num, split_locs, split_signs, true_split_clu_num
            )
            clu_ave_var[split_clu_no] = _cluster_ave_var_from_F(F, temp0, temp1, point_sq, split_clu_no)
            clu_ave_var[clu_num] = _cluster_ave_var_from_F(F, temp0, temp1, point_sq, clu_num)
            clu_num += 1

        for _ in range(2):
            _one_cd(data64, F, temp0, temp1, temp2, point_sq, clu_num)

        # Merge back to k
        clu_sign = np.ones(clu_num, dtype=np.int64)
        heap = []
        for i in range(clu_num - 1):
            for j in range(i + 1, clu_num):
                if temp1[i] > 0 and temp1[j] > 0:
                    diff = _merge_tar_diff(temp0, temp1, i, j)
                    heapq.heappush(heap, (-diff, i, j, int(clu_sign[i]), int(clu_sign[j])))

        while clu_num > k and heap:
            while heap:
                neg_diff, i, j, si, sj = heapq.heappop(heap)
                if temp1[i] > 0 and temp1[j] > 0 and clu_sign[i] == si and clu_sign[j] == sj:
                    break
            else:
                break
            F[:, i] = np.maximum(F[:, i], F[:, j])
            F[:, j] = 0.0
            temp0[:, i] = temp0[:, i] + temp0[:, j]
            temp1[i] = temp1[i] + temp1[j]
            temp2[i] = float(np.dot(temp0[:, i], temp0[:, i]))
            temp1[j] = 0.0
            clu_sign[i] += 1
            for jj in range(max_k):
                if temp1[jj] == 0 or jj == i:
                    continue
                if temp1[jj] > 0:
                    diff = _merge_tar_diff(temp0, temp1, i, jj)
                    heapq.heappush(heap, (-diff, i, jj, int(clu_sign[i]), int(clu_sign[jj])))
            clu_num -= 1

        # Compact non-empty clusters into first k columns
        write = 0
        for c in range(max_k):
            if temp1[c] > 0:
                if write != c:
                    F[:, write] = F[:, c]
                    temp0[:, write] = temp0[:, c]
                    temp1[write] = temp1[c]
                    temp2[write] = temp2[c]
                    F[:, c] = 0.0
                    temp1[c] = 0.0
                write += 1
                if write >= k:
                    break
        for c in range(k, max_k):
            F[:, c] = 0.0
            temp1[c] = 0.0

    # Re-sync temp from F for first k
    for c in range(k):
        members = F[:, c] > 0
        temp1[c] = float(members.sum())
        if temp1[c] > 0:
            temp0[:, c] = data64[members].sum(axis=0)
            temp2[c] = float(np.dot(temp0[:, c], temp0[:, c]))
        else:
            temp0[:, c] = 0.0
            temp2[c] = 0.0

    # Final CD until convergence (same intent as original F_old equality check)
    for _ in range(max_cd_iters):
        moved = _one_cd(data64, F, temp0, temp1, temp2, point_sq, k)
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


def batch_dgsm_cdkm(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_cd_iters: int = 100,
    do_split_merge: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PSCA-compatible batch wrapper — real DGSM-CDKM per sample (Numba-accelerated).

    Args:
        features: [B, T(+1), D] vision tokens (LLaVA includes CLS at index 0)
        min_components: K, same as PSCA (= need_token_num // 4)
        seed: RNG seed for k-means++
        drop_cls: if True, cluster tokens[..., 1:, :] (LLaVA); if False, use all tokens (Qwen)
        max_cd_iters: final CD iteration cap (early-stops on convergence)
        do_split_merge: enable DGSM oversplit/merge

    Returns:
        soft_scores: [B, T_patch, K]
        belong_components: [B, T_patch]
    """
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

    soft_list = []
    belong_list = []
    for b in range(B):
        # Keep feature transfer cheap (float32); dgsm_cdkm promotes decision state to float64
        data = tokens[b].detach().cpu().numpy().astype(np.float32, copy=False)
        rng = np.random.default_rng(seed + b)
        labels, _, soft = dgsm_cdkm(
            data, k=k, rng=rng, max_cd_iters=max_cd_iters, do_split_merge=do_split_merge
        )
        soft_list.append(torch.from_numpy(soft[:, :k]))
        belong_list.append(torch.from_numpy(labels))

    soft_scores = torch.stack(soft_list, dim=0).to(device=device, dtype=dtype)
    belong = torch.stack(belong_list, dim=0).to(device=device, dtype=torch.long)
    return soft_scores, belong
