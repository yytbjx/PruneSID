"""Numba hot kernels for DGSM-CDKM (CD + hierarchical merge + merge-back).

BLAS-heavy steps stay in NumPy. No fastmath; FP64 decision path; strict '>' ties.
Token axis is sequential; do not prange over tokens.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def merge_tar_diff(temp0, temp1, i, j):
        ti = temp1[i]
        tj = temp1[j]
        if ti <= 0.0 or tj <= 0.0:
            return -np.inf
        m = temp0.shape[0]
        numerator = 0.0
        for d in range(m):
            v = temp0[d, i] * tj - temp0[d, j] * ti
            numerator -= v * v
        return numerator / (ti * tj * (ti + tj))

    @njit(cache=True)
    def one_cd(data, F, temp0, temp1, temp2, point_sq, active_k):
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
            if np_p <= 0.0:
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
                if nj <= 0.0:
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

    @njit(cache=True)
    def hierarchical_merge_to_two(
        X,
        F,
        temp0,
        temp1,
        temp2,
        split_clu_no,
        new_clu,
        split_locs,
        split_signs,
        true_split_clu_num,
    ):
        clu_split_dot_num = split_locs.shape[0]
        if clu_split_dot_num == 0:
            return

        m = X.shape[1]
        initial_temp0 = np.zeros((m, true_split_clu_num), dtype=np.float64)
        initial_temp1 = np.zeros(true_split_clu_num, dtype=np.float64)
        for j in range(clu_split_dot_num):
            s = split_signs[j]
            if s < 0:
                s = 0
            if s >= true_split_clu_num:
                s = true_split_clu_num - 1
            initial_temp1[s] += 1.0
            for d in range(m):
                initial_temp0[d, s] += X[split_locs[j], d]

        active = np.zeros(true_split_clu_num, dtype=np.uint8)
        final_cluster_index = np.empty(true_split_clu_num, dtype=np.int64)
        cluster_count = 0
        for s in range(true_split_clu_num):
            final_cluster_index[s] = s
            if initial_temp1[s] > 0.0:
                active[s] = 1
                cluster_count += 1

        do_merge = clu_split_dot_num != 2
        while do_merge and cluster_count > 2:
            best_val = -np.inf
            best_a = -1
            best_b = -1
            for a in range(true_split_clu_num):
                if active[a] == 0 or initial_temp1[a] == 0.0:
                    continue
                for b in range(a + 1, true_split_clu_num):
                    if active[b] == 0 or initial_temp1[b] == 0.0:
                        continue
                    val = merge_tar_diff(initial_temp0, initial_temp1, a, b)
                    if val > best_val:
                        best_val = val
                        best_a = a
                        best_b = b
            if best_a < 0:
                break
            for d in range(m):
                initial_temp0[d, best_a] += initial_temp0[d, best_b]
            initial_temp1[best_a] += initial_temp1[best_b]
            for t in range(true_split_clu_num):
                if final_cluster_index[t] == best_b:
                    final_cluster_index[t] = best_a
            active[best_b] = 0
            cluster_count -= 1

        if do_merge:
            f0 = -1
            f1 = -1
            for s in range(true_split_clu_num):
                if active[s] != 0:
                    if f0 < 0:
                        f0 = s
                    elif f1 < 0:
                        f1 = s
                        break
            if f0 < 0 or f1 < 0:
                return
            for j in range(clu_split_dot_num):
                root = final_cluster_index[split_signs[j]]
                if root != f0:
                    F[split_locs[j], split_clu_no] = 0.0
                    F[split_locs[j], new_clu] = 1.0
            for d in range(m):
                temp0[d, split_clu_no] = initial_temp0[d, f0]
                temp0[d, new_clu] = initial_temp0[d, f1]
            temp1[split_clu_no] = initial_temp1[f0]
            temp1[new_clu] = initial_temp1[f1]
            ss0 = 0.0
            ss1 = 0.0
            for d in range(m):
                ss0 += temp0[d, split_clu_no] * temp0[d, split_clu_no]
                ss1 += temp0[d, new_clu] * temp0[d, new_clu]
            temp2[split_clu_no] = ss0
            temp2[new_clu] = ss1
        else:
            F[split_locs[1], split_clu_no] = 0.0
            F[split_locs[1], new_clu] = 1.0
            for d in range(m):
                temp0[d, split_clu_no] = X[split_locs[0], d]
                temp0[d, new_clu] = X[split_locs[1], d]
            temp1[split_clu_no] = 1.0
            temp1[new_clu] = 1.0
            ss0 = 0.0
            ss1 = 0.0
            for d in range(m):
                ss0 += temp0[d, split_clu_no] * temp0[d, split_clu_no]
                ss1 += temp0[d, new_clu] * temp0[d, new_clu]
            temp2[split_clu_no] = ss0
            temp2[new_clu] = ss1

    @njit(cache=True)
    def merge_back_to_k(F, temp0, temp1, temp2, clu_num, k, max_k):
        while clu_num > k:
            best_diff = -np.inf
            best_i = -1
            best_j = -1
            for i in range(max_k):
                if temp1[i] <= 0.0:
                    continue
                for j in range(i + 1, max_k):
                    if temp1[j] <= 0.0:
                        continue
                    diff = merge_tar_diff(temp0, temp1, i, j)
                    if diff > best_diff:
                        best_diff = diff
                        best_i = i
                        best_j = j
            if best_i < 0:
                break
            n = F.shape[0]
            m = temp0.shape[0]
            for t in range(n):
                if F[t, best_j] != 0.0:
                    F[t, best_i] = 1.0
                    F[t, best_j] = 0.0
            for d in range(m):
                temp0[d, best_i] = temp0[d, best_i] + temp0[d, best_j]
                temp0[d, best_j] = 0.0
            temp1[best_i] = temp1[best_i] + temp1[best_j]
            temp1[best_j] = 0.0
            ss = 0.0
            for d in range(m):
                ss += temp0[d, best_i] * temp0[d, best_i]
            temp2[best_i] = ss
            temp2[best_j] = 0.0
            clu_num -= 1

        write = 0
        n = F.shape[0]
        m = temp0.shape[0]
        for c in range(max_k):
            if temp1[c] > 0.0:
                if write != c:
                    for t in range(n):
                        F[t, write] = F[t, c]
                        F[t, c] = 0.0
                    for d in range(m):
                        temp0[d, write] = temp0[d, c]
                        temp0[d, c] = 0.0
                    temp1[write] = temp1[c]
                    temp2[write] = temp2[c]
                    temp1[c] = 0.0
                    temp2[c] = 0.0
                write += 1
                if write >= k:
                    break
        for c in range(k, max_k):
            for t in range(n):
                F[t, c] = 0.0
            temp1[c] = 0.0
            temp2[c] = 0.0

else:  # pragma: no cover

    def one_cd(*args, **kwargs):
        raise RuntimeError("numba is required")

    def hierarchical_merge_to_two(*args, **kwargs):
        raise RuntimeError("numba is required")

    def merge_back_to_k(*args, **kwargs):
        raise RuntimeError("numba is required")

    def merge_tar_diff(*args, **kwargs):
        raise RuntimeError("numba is required")


def warmup(n: int = 64, m: int = 64, k: int = 8) -> None:
    if not _HAS_NUMBA:
        return
    rng = np.random.default_rng(0)
    data = np.ascontiguousarray(rng.normal(size=(n, m)), dtype=np.float64)
    F = np.zeros((n, k), dtype=np.float64)
    labels = rng.integers(0, k, size=n)
    F[np.arange(n), labels] = 1.0
    temp0 = np.zeros((m, k), dtype=np.float64)
    temp1 = np.zeros(k, dtype=np.float64)
    temp2 = np.zeros(k, dtype=np.float64)
    point_sq = np.sum(data * data, axis=1)
    for c in range(k):
        members = F[:, c] > 0
        temp1[c] = float(members.sum())
        if temp1[c] > 0:
            temp0[:, c] = data[members].sum(axis=0)
            temp2[c] = float(np.dot(temp0[:, c], temp0[:, c]))
    one_cd(data, F, temp0, temp1, temp2, point_sq, k)
    locs = np.arange(min(8, n), dtype=np.int64)
    signs = np.zeros(locs.shape[0], dtype=np.int64)
    hierarchical_merge_to_two(data, F, temp0, temp1, temp2, 0, min(1, k - 1), locs, signs, 2)
    merge_back_to_k(F, temp0, temp1, temp2, k, max(1, k // 2), k)


__all__ = [
    "one_cd",
    "hierarchical_merge_to_two",
    "merge_back_to_k",
    "merge_tar_diff",
    "warmup",
    "_HAS_NUMBA",
]
