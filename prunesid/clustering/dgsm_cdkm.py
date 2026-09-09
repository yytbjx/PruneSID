"""
DGSM-CDKM clustering used as a drop-in replacement for PruneSID's PSCA grouping.

Optimized port of DGSM-CDKM.cpp for VLM token grouping (n~576, d~1024, k~16).
Hot path: vectorized coordinate descent + float32 (original Python loops were ~1s+/image).
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
    closest_dist_sq = np.full(n, np.inf, dtype=np.float32)

    for c in range(1, k):
        last = data[centers_idx[c - 1]]
        dist_sq = np.sum((data - last) ** 2, axis=1)
        np.minimum(closest_dist_sq, dist_sq, out=closest_dist_sq)
        total = float(closest_dist_sq.sum())
        if total <= 0 or not np.isfinite(total):
            centers_idx[c:] = rng.choice(n, size=k - c, replace=False)
            break
        probs = closest_dist_sq / total
        centers_idx[c] = int(rng.choice(n, p=probs))
    return centers_idx


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """Min-max normalize attention to [0, 1]."""
    attention = np.asarray(attention, dtype=np.float32).reshape(-1)
    amin = float(attention.min())
    amax = float(attention.max())
    return (attention - amin) / (amax - amin + 1e-8)


def semantic_kmeans_plusplus_init(
    data: np.ndarray,
    k: int,
    attention: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Semantic-aware k-means++ initialization.

    First center ~ attention; later centers ~ D_i^2 * (1 + alpha * A_i).
    alpha=0 reduces to (almost) classic k-means++ after the first pick.
    """
    n = data.shape[0]
    k = min(k, n)
    attn = normalize_attention(attention)
    if attn.shape[0] != n:
        raise ValueError(f"attention length {attn.shape[0]} != n={n}")

    centers_idx = np.empty(k, dtype=np.int64)
    # First center: attention-weighted
    prob0 = attn + 1e-6
    prob0 = prob0 / prob0.sum()
    centers_idx[0] = int(rng.choice(n, p=prob0))

    closest_dist_sq = np.full(n, np.inf, dtype=np.float32)
    alpha = float(alpha)

    for c in range(1, k):
        last = data[centers_idx[c - 1]]
        dist_sq = np.sum((data - last) ** 2, axis=1)
        np.minimum(closest_dist_sq, dist_sq, out=closest_dist_sq)

        semantic_score = closest_dist_sq * (1.0 + alpha * attn)
        total = float(semantic_score.sum())
        if total <= 0 or not np.isfinite(total):
            centers_idx[c:] = rng.choice(n, size=k - c, replace=False)
            break
        probs = semantic_score / total
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
    ti = temp1[i]
    tj = temp1[j]
    if ti <= 0 or tj <= 0:
        return -np.inf
    numerator = -np.sum((temp0[:, i] * tj - temp0[:, j] * ti) ** 2)
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
    labels = np.argmax(F[:, :active_k], axis=1)

    for i in range(n):
        p = int(labels[i])
        np_p = temp1[p]
        if np_p <= 0:
            continue
        xi = data[i]
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
            nj = temp1[j]
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

            # M[p]
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
                # old dots for temp2
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

    initial_temp0 = np.zeros((X.shape[1], true_split_clu_num), dtype=np.float32)
    initial_temp1 = np.zeros(true_split_clu_num, dtype=np.float32)
    for j in range(clu_split_dot_num):
        s = int(split_signs[j])
        s = max(0, min(s, true_split_clu_num - 1))
        initial_temp1[s] += 1.0
        initial_temp0[:, s] += X[split_locs[j]]

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
        temp0[:, split_clu_no] = X[split_locs[0]]
        temp0[:, new_clu] = X[split_locs[1]]
        temp1[split_clu_no] = 1.0
        temp1[new_clu] = 1.0
        temp2[split_clu_no] = float(np.dot(temp0[:, split_clu_no], temp0[:, split_clu_no]))
        temp2[new_clu] = float(np.dot(temp0[:, new_clu], temp0[:, new_clu]))


def dgsm_cdkm(
    data: np.ndarray,
    k: int,
    init_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_cd_iters: int = 8,
    do_split_merge: bool = True,
    attention: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    init_method: str = "kpp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DGSM-CDKM (core CD / split-merge unchanged).

    Init only:
      init_method='kpp'      -> classic k-means++
      init_method='semantic' -> semantic k-means++ (needs attention)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    data = np.ascontiguousarray(data, dtype=np.float32)
    n, m = data.shape
    if n == 0:
        raise ValueError("empty data")
    k = int(max(1, min(k, n)))

    if init_indices is None:
        use_semantic = (
            init_method in ("semantic", "semantic_kpp", "skpp")
            and attention is not None
        )
        if use_semantic:
            init_indices = semantic_kmeans_plusplus_init(
                data, k, attention, rng, alpha=alpha
            )
        else:
            init_indices = kmeans_plusplus_init(data, k, rng)
    else:
        init_indices = np.asarray(init_indices, dtype=np.int64)[:k]

    max_k = max(k, int(1.5 * k)) if do_split_merge else k
    split_num = min(3, m)

    centers = data[init_indices].copy()
    # dists via (a-b)^2 = a^2 + b^2 - 2ab
    data_sq = np.sum(data * data, axis=1, keepdims=True)
    cen_sq = np.sum(centers * centers, axis=1)
    dists = data_sq + cen_sq[None, :] - 2.0 * (data @ centers.T)
    for j, idx in enumerate(init_indices):
        dists[idx, :] = np.inf
        dists[idx, j] = 0.0
    labels0 = np.argmin(dists, axis=1)

    F = np.zeros((n, max_k), dtype=np.float32)
    F[np.arange(n), labels0] = 1.0

    temp0 = np.zeros((m, max_k), dtype=np.float32)
    temp1 = np.zeros(max_k, dtype=np.float32)
    temp2 = np.zeros(max_k, dtype=np.float32)
    point_sq = np.sum(data * data, axis=1)

    for c in range(k):
        members = F[:, c] > 0
        temp1[c] = float(members.sum())
        if temp1[c] > 0:
            temp0[:, c] = data[members].sum(axis=0)
            temp2[c] = float(np.dot(temp0[:, c], temp0[:, c]))

    for _ in range(2):
        _one_cd(data, F, temp0, temp1, temp2, point_sq, k)

    clu_num = k
    if do_split_merge and max_k > k:
        clu_ave_var = np.zeros(max_k, dtype=np.float32)
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
            vals = data[split_locs]  # [s, m]
            dim_scores = np.abs(vals - mean[None, :]).sum(axis=0)
            chosen_dims = np.argpartition(-dim_scores, true_split_num - 1)[:true_split_num]

            split_signs = np.zeros(clu_split_dot_num, dtype=np.int64)
            for dim in chosen_dims:
                center = mean[dim]
                ge = data[split_locs, dim] >= center
                split_signs = (split_signs << 1) | ge.astype(np.int64)

            _hierarchical_merge_to_two_v2(
                data, F, temp0, temp1, temp2, split_clu_no, clu_num, split_locs, split_signs, true_split_clu_num
            )
            clu_ave_var[split_clu_no] = _cluster_ave_var_from_F(F, temp0, temp1, point_sq, split_clu_no)
            clu_ave_var[clu_num] = _cluster_ave_var_from_F(F, temp0, temp1, point_sq, clu_num)
            clu_num += 1

        for _ in range(2):
            _one_cd(data, F, temp0, temp1, temp2, point_sq, clu_num)

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
            temp0[:, c] = data[members].sum(axis=0)
            temp2[c] = float(np.dot(temp0[:, c], temp0[:, c]))
        else:
            temp0[:, c] = 0.0
            temp2[c] = 0.0

    for _ in range(max_cd_iters):
        moved = _one_cd(data, F, temp0, temp1, temp2, point_sq, k)
        if moved == 0:
            break

    labels = np.argmax(F[:, :k], axis=1).astype(np.int64)
    centers_out = np.zeros((k, m), dtype=np.float32)
    for c in range(k):
        members = labels == c
        if members.any():
            centers_out[c] = data[members].mean(axis=0)
        else:
            centers_out[c] = data[int(rng.integers(0, n))]

    cen_sq = np.sum(centers_out * centers_out, axis=1)
    dists = point_sq[:, None] + cen_sq[None, :] - 2.0 * (data @ centers_out.T)
    labels = np.argmin(dists, axis=1).astype(np.int64)
    for c in range(k):
        members = labels == c
        if members.any():
            centers_out[c] = data[members].mean(axis=0)

    soft_scores = (-dists).astype(np.float32)
    return labels, centers_out, soft_scores


def batch_dgsm_cdkm(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_cd_iters: int = 8,
    do_split_merge: bool = True,
    attention: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    init_method: str = "kpp",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PSCA-compatible batch wrapper — stays on `features.device`, truly batched.

    attention: optional [B, T_patch] (or [B, T_patch+1] if drop_cls and CLS included);
               used only when init_method='semantic'.
    """
    x = torch.sigmoid(features.float())

    if drop_cls:
        if x.shape[1] < 2:
            raise ValueError("features must include CLS + patches when drop_cls=True")
        tokens = x[:, 1:, :]
        attn = None
        if attention is not None:
            attn = attention.float()
            if attn.shape[-1] == features.shape[1]:
                attn = attn[:, 1:]
            elif attn.shape[-1] != tokens.shape[1]:
                raise ValueError(
                    f"attention last dim {attn.shape[-1]} incompatible with tokens {tokens.shape[1]}"
                )
    else:
        tokens = x
        attn = attention.float() if attention is not None else None

    B, T, D = tokens.shape
    k = max(1, min(int(min_components), T))
    return _batch_dgsm_torch(
        tokens,
        k=k,
        seed=seed,
        max_iters=max_cd_iters,
        do_split_merge=do_split_merge,
        attention=attn,
        alpha=alpha,
        init_method=init_method,
    )


def _normalize_attention_torch(attention: torch.Tensor) -> torch.Tensor:
    """Min-max normalize last dim to [0,1]. attention: [B, N]."""
    amin = attention.amin(dim=-1, keepdim=True)
    amax = attention.amax(dim=-1, keepdim=True)
    return (attention - amin) / (amax - amin + 1e-8)


def _batch_kmeans_plusplus(
    data: torch.Tensor,
    k: int,
    seed: int,
    attention: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    init_method: str = "kpp",
) -> torch.Tensor:
    """
    Batched k-means++ / semantic k-means++.
    data: [B, N, D] -> centers [B, K, D]
    attention: optional [B, N]
    """
    B, N, D = data.shape
    device, dtype = data.device, data.dtype
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    use_semantic = (
        init_method in ("semantic", "semantic_kpp", "skpp")
        and attention is not None
    )
    attn = _normalize_attention_torch(attention.to(device=device, dtype=dtype)) if use_semantic else None
    alpha = float(alpha)

    centers = torch.empty(B, k, D, device=device, dtype=dtype)
    batch_idx = torch.arange(B, device=device)

    if use_semantic:
        prob0 = attn + 1e-6
        prob0 = prob0 / prob0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        first = torch.multinomial(prob0, num_samples=1, generator=g).squeeze(-1)
    else:
        first = torch.randint(0, N, (B,), generator=g, device=device)
    centers[:, 0] = data[batch_idx, first]

    closest = torch.full((B, N), float("inf"), device=device, dtype=dtype)

    for c in range(1, k):
        last = centers[:, c - 1 : c]
        dist_sq = ((data - last) ** 2).sum(dim=-1)
        closest = torch.minimum(closest, dist_sq)
        if use_semantic:
            score = closest * (1.0 + alpha * attn)
        else:
            score = closest
        total = score.sum(dim=-1).clamp_min(1e-12)
        probs = score / total.unsqueeze(-1)
        choice = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)
        centers[:, c] = data[batch_idx, choice]
        closest[batch_idx, choice] = 0.0

    return centers


def _batch_lloyd(
    data: torch.Tensor,
    centers: torch.Tensor,
    n_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched Lloyd iterations.
    data [B,N,D], centers [B,K,D]
    returns labels [B,N], centers [B,K,D], dists [B,N,K] (squared L2)
    """
    B, N, D = data.shape
    K = centers.shape[1]
    for _ in range(max(1, n_iters)):
        # squared distances via cdist^2
        dists = torch.cdist(data, centers, p=2).pow(2)  # [B,N,K]
        labels = dists.argmin(dim=-1)  # [B,N]
        onehot = torch.nn.functional.one_hot(labels, K).to(dtype=data.dtype)  # [B,N,K]
        counts = onehot.sum(dim=1).clamp_min(1.0)  # [B,K]
        new_centers = torch.einsum("bnd,bnk->bkd", data, onehot) / counts.unsqueeze(-1)
        # keep empty clusters at old location (counts were clamped; detect empties)
        empty = onehot.sum(dim=1) == 0  # [B,K]
        if empty.any():
            new_centers = torch.where(empty.unsqueeze(-1), centers, new_centers)
        centers = new_centers

    dists = torch.cdist(data, centers, p=2).pow(2)
    labels = dists.argmin(dim=-1)
    return labels, centers, dists


def _batch_cluster_vars(
    data: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, K: int
) -> torch.Tensor:
    """Within-cluster sum of squared errors. returns [B,K]."""
    B, N, D = data.shape
    assigned = centers[torch.arange(B, device=data.device).unsqueeze(1), labels]  # [B,N,D]
    sse = ((data - assigned) ** 2).sum(dim=-1)  # [B,N]
    vars_ = torch.zeros(B, K, device=data.device, dtype=data.dtype)
    vars_.scatter_add_(1, labels, sse)
    counts = torch.zeros(B, K, device=data.device, dtype=data.dtype)
    counts.scatter_add_(1, labels, torch.ones_like(sse))
    # zero variance for singleton / empty
    vars_ = torch.where(counts <= 1, torch.zeros_like(vars_), vars_)
    return vars_


def _batch_split_once(
    data: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    live_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Split the highest-variance cluster in each batch item along its max-MAD dimension.
    Returns labels, centers with shape [B, live_k+1, D], and new live_k.
    """
    B, N, D = data.shape
    device, dtype = data.device, data.dtype
    vars_ = _batch_cluster_vars(data, labels, centers[:, :live_k], live_k)
    split_id = vars_.argmax(dim=-1)  # [B]
    batch_idx = torch.arange(B, device=device)

    split_mask = labels == split_id.unsqueeze(1)  # [B,N]
    mean = centers[batch_idx, split_id]  # [B,D]

    filled = torch.where(split_mask.unsqueeze(-1), data, mean.unsqueeze(1))
    mad_sum = ((filled - mean.unsqueeze(1)).abs() * split_mask.unsqueeze(-1).to(dtype)).sum(dim=1)
    split_dim = mad_sum.argmax(dim=-1)  # [B]
    thresh = mean[batch_idx, split_dim]  # [B]
    vals = data[batch_idx[:, None], torch.arange(N, device=device)[None, :], split_dim[:, None]]
    go_right = (vals >= thresh.unsqueeze(1)) & split_mask

    new_clu = live_k
    new_labels = torch.where(go_right, torch.full_like(labels, new_clu), labels)

    left_mask = split_mask & ~go_right
    right_mask = go_right
    left_counts = left_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype)
    right_counts = right_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype)
    left_c = (data * left_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / left_counts
    right_c = (data * right_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / right_counts

    new_centers = torch.zeros(B, live_k + 1, D, device=device, dtype=dtype)
    new_centers[:, :live_k] = centers[:, :live_k]
    new_centers[batch_idx, split_id] = left_c
    new_centers[:, new_clu] = right_c

    bad = (left_mask.sum(dim=1) == 0) | (right_mask.sum(dim=1) == 0) | (vars_[batch_idx, split_id] <= 0)
    if bad.any():
        new_labels[bad] = labels[bad]
        new_centers[bad, :live_k] = centers[bad, :live_k]
        # keep appended column as duplicate of split center so lloyd can absorb it
        new_centers[bad, new_clu] = centers[bad, split_id[bad]]

    return new_labels, new_centers, live_k + 1


def _batch_merge_to_k(
    data: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    live_k: int,
    target_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy merge of nearest centers down to target_k."""
    B, N, D = data.shape
    device = data.device
    dtype = data.dtype
    cur_k = live_k
    cur_centers = centers[:, :cur_k].clone()
    cur_labels = labels.clamp(max=cur_k - 1).clone()

    while cur_k > target_k:
        d = torch.cdist(cur_centers, cur_centers, p=2).pow(2)
        eye = torch.eye(cur_k, device=device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, float("inf"))
        counts = torch.zeros(B, cur_k, device=device, dtype=dtype)
        counts.scatter_add_(1, cur_labels, torch.ones(B, N, device=device, dtype=dtype))
        empty = counts <= 0
        d = d.masked_fill(empty.unsqueeze(1) | empty.unsqueeze(2), float("inf"))

        min_idx = d.view(B, -1).argmin(dim=-1)
        i0 = min_idx // cur_k
        j0 = min_idx % cur_k
        i = torch.minimum(i0, j0)
        j = torch.maximum(i0, j0)

        # Vectorized remap: merge j->i then compact ids > j (j differs per batch)
        # Build mapping table [B, cur_k]
        arange_k = torch.arange(cur_k, device=device).view(1, -1).expand(B, -1)
        mapped = torch.where(arange_k == j.unsqueeze(1), i.unsqueeze(1), arange_k)
        # compact: subtract 1 from ids strictly greater than j
        mapped = torch.where(mapped > j.unsqueeze(1), mapped - 1, mapped)
        cur_labels = mapped.gather(1, cur_labels)

        cur_k -= 1
        onehot = torch.nn.functional.one_hot(cur_labels, cur_k).to(dtype)
        cnt = onehot.sum(dim=1).clamp_min(1.0)
        cur_centers = torch.einsum("bnd,bnk->bkd", data, onehot) / cnt.unsqueeze(-1)

    return cur_labels, cur_centers


def _batch_dgsm_torch(
    tokens: torch.Tensor,
    k: int,
    seed: int = 0,
    max_iters: int = 8,
    do_split_merge: bool = True,
    attention: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    init_method: str = "kpp",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Device-resident, batched DGSM-style grouping.
    tokens: [B, N, D] on CPU or CUDA — never moved across for clustering itself.
    Core Lloyd/split-merge unchanged; only initialization can use attention.
    """
    data = tokens.contiguous()
    B, N, D = data.shape
    k = max(1, min(k, N))

    centers = _batch_kmeans_plusplus(
        data,
        k,
        seed,
        attention=attention,
        alpha=alpha,
        init_method=init_method,
    )
    labels, centers, dists = _batch_lloyd(data, centers, max_iters)

    if do_split_merge and k >= 2:
        max_k = max(k, int(1.5 * k))
        live_k = k
        while live_k < max_k:
            labels, centers, live_k = _batch_split_once(data, labels, centers, live_k)
            labels, centers, dists = _batch_lloyd(data, centers[:, :live_k], 2)
        if live_k > k:
            labels, centers = _batch_merge_to_k(data, labels, centers, live_k, k)
            labels, centers, dists = _batch_lloyd(data, centers, max_iters)

    soft = (-dists).to(dtype=torch.float32)
    return soft, labels.long()
