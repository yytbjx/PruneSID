"""Numba backend for DGSM-CDKM.

Kept in a separate top-level module so Numba's on-disk cache can be reused
across lmms-eval processes. No fastmath. Decision statistics stay FP64.

Token-order coordinate descent is sequential within each sample; only the
batch axis may use prange.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def _merge_tar_diff(temp0, temp1, i, j):
        ti = temp1[i]
        tj = temp1[j]
        if ti <= 0.0 or tj <= 0.0:
            return -np.inf
        m = temp0.shape[0]
        numerator = 0.0
        for d in range(m):
            v = temp0[d, i] * tj - temp0[d, j] * ti
            numerator -= v * v
        denom = ti * tj * (ti + tj)
        return numerator / denom

    @njit(cache=True)
    def _cluster_ave_var(F, temp0, temp1, point_sq, c):
        np_c = temp1[c]
        if np_c <= 1.0:
            return 0.0
        n = F.shape[0]
        sum_x2 = 0.0
        for i in range(n):
            if F[i, c] != 0.0:
                sum_x2 += point_sq[i]
        sumX_dot = 0.0
        m = temp0.shape[0]
        for d in range(m):
            sumX_dot += temp0[d, c] * temp0[d, c]
        return sum_x2 - sumX_dot / np_c

    @njit(cache=True)
    def _one_cd(data, F, temp0, temp1, temp2, point_sq, active_k):
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
                # Strict > preserves first-best on ties (same as prior port).
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
    def _hierarchical_merge_to_two(
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
                    val = _merge_tar_diff(initial_temp0, initial_temp1, a, b)
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
    def _select_top_dims(dim_scores, true_split_num):
        """Deterministic top-k: sort by (-score, +index), take first k."""
        m = dim_scores.shape[0]
        ksel = true_split_num
        if ksel > m:
            ksel = m
        order = np.empty(m, dtype=np.int64)
        for i in range(m):
            order[i] = i
        # Insertion sort: higher score first; on tie, smaller index first.
        for i in range(1, m):
            key = order[i]
            key_score = dim_scores[key]
            j = i - 1
            while j >= 0:
                cur = order[j]
                cur_score = dim_scores[cur]
                # cur should stay before key?
                if cur_score > key_score or (cur_score == key_score and cur < key):
                    break
                order[j + 1] = cur
                j -= 1
            order[j + 1] = key
        out = np.empty(ksel, dtype=np.int64)
        for i in range(ksel):
            out[i] = order[i]
        return out

    @njit(cache=True)
    def _sync_stats_from_F(data, F, temp0, temp1, temp2, k):
        n, m = data.shape
        for c in range(k):
            cnt = 0.0
            for d in range(m):
                temp0[d, c] = 0.0
            for i in range(n):
                if F[i, c] != 0.0:
                    cnt += 1.0
                    for d in range(m):
                        temp0[d, c] += data[i, d]
            temp1[c] = cnt
            if cnt > 0.0:
                ss = 0.0
                for d in range(m):
                    ss += temp0[d, c] * temp0[d, c]
                temp2[c] = ss
            else:
                temp2[c] = 0.0

    @njit(cache=True)
    def _merge_back_to_k(F, temp0, temp1, temp2, clu_num, k, max_k):
        """Rescan best-pair merge (strict >). Equivalent to lazy heap with same keys."""
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
                    diff = _merge_tar_diff(temp0, temp1, i, j)
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

        # Compact non-empty clusters into first k columns.
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

    @njit(cache=True)
    def run_single(
        data,
        k,
        init_indices,
        max_cd_iters,
        do_split_merge,
        empty_center_idx,
    ):
        """
        Faithful DGSM-CDKM on one sample.

        data: float64 [n, m], contiguous
        init_indices: int64 [k]
        Returns labels int64 [n], centers float32 [k, m], soft float32 [n, k]
        """
        n, m = data.shape
        if do_split_merge:
            max_k = k
            grown = int(1.5 * k)
            if grown > max_k:
                max_k = grown
        else:
            max_k = k
        if max_k > n:
            max_k = n
        split_num = 3
        if m < split_num:
            split_num = m

        point_sq = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for d in range(m):
                s += data[i, d] * data[i, d]
            point_sq[i] = s

        # Initial assignment to k-means++ centers (force init points onto themselves).
        labels0 = np.empty(n, dtype=np.int64)
        for i in range(n):
            forced = -1
            for c in range(k):
                if init_indices[c] == i:
                    forced = c
                    break
            if forced >= 0:
                labels0[i] = forced
                continue
            best_c = 0
            best_d = np.inf
            for c in range(k):
                idx = init_indices[c]
                d2 = 0.0
                for d in range(m):
                    z = data[i, d] - data[idx, d]
                    d2 += z * z
                if d2 < best_d:
                    best_d = d2
                    best_c = c
            labels0[i] = best_c

        F = np.zeros((n, max_k), dtype=np.float64)
        for i in range(n):
            F[i, labels0[i]] = 1.0

        temp0 = np.zeros((m, max_k), dtype=np.float64)
        temp1 = np.zeros(max_k, dtype=np.float64)
        temp2 = np.zeros(max_k, dtype=np.float64)
        _sync_stats_from_F(data, F, temp0, temp1, temp2, k)

        _one_cd(data, F, temp0, temp1, temp2, point_sq, k)
        _one_cd(data, F, temp0, temp1, temp2, point_sq, k)

        clu_num = k
        if do_split_merge and max_k > k:
            clu_ave_var = np.zeros(max_k, dtype=np.float64)
            for c in range(k):
                clu_ave_var[c] = _cluster_ave_var(F, temp0, temp1, point_sq, c)

            while clu_num < max_k:
                split_clu_no = 0
                best_var = clu_ave_var[0]
                for c in range(1, clu_num):
                    if clu_ave_var[c] > best_var:
                        best_var = clu_ave_var[c]
                        split_clu_no = c
                if clu_ave_var[split_clu_no] == 0.0 or temp1[split_clu_no] <= 1.0:
                    break

                # Collect members (stable ascending index order, same as np.where).
                clu_split_dot_num = 0
                for i in range(n):
                    if F[i, split_clu_no] != 0.0:
                        clu_split_dot_num += 1
                if clu_split_dot_num < 2:
                    break
                split_locs = np.empty(clu_split_dot_num, dtype=np.int64)
                p = 0
                for i in range(n):
                    if F[i, split_clu_no] != 0.0:
                        split_locs[p] = i
                        p += 1

                true_split_clu_num = 1
                for _ in range(split_num):
                    true_split_clu_num *= 2
                true_split_num = split_num
                while clu_split_dot_num < true_split_clu_num:
                    true_split_clu_num //= 2
                    true_split_num -= 1
                if true_split_num <= 0:
                    true_split_num = 1
                    true_split_clu_num = 2

                mean = np.empty(m, dtype=np.float64)
                for d in range(m):
                    mean[d] = temp0[d, split_clu_no] / clu_split_dot_num

                dim_scores = np.zeros(m, dtype=np.float64)
                for j in range(clu_split_dot_num):
                    idx = split_locs[j]
                    for d in range(m):
                        v = data[idx, d] - mean[d]
                        if v < 0.0:
                            v = -v
                        dim_scores[d] += v

                chosen_dims = _select_top_dims(dim_scores, true_split_num)
                split_signs = np.zeros(clu_split_dot_num, dtype=np.int64)
                for t in range(true_split_num):
                    dim = chosen_dims[t]
                    center = mean[dim]
                    for j in range(clu_split_dot_num):
                        ge = 1 if data[split_locs[j], dim] >= center else 0
                        split_signs[j] = (split_signs[j] << 1) | ge

                _hierarchical_merge_to_two(
                    data,
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
                clu_ave_var[split_clu_no] = _cluster_ave_var(
                    F, temp0, temp1, point_sq, split_clu_no
                )
                clu_ave_var[clu_num] = _cluster_ave_var(
                    F, temp0, temp1, point_sq, clu_num
                )
                clu_num += 1

            _one_cd(data, F, temp0, temp1, temp2, point_sq, clu_num)
            _one_cd(data, F, temp0, temp1, temp2, point_sq, clu_num)
            _merge_back_to_k(F, temp0, temp1, temp2, clu_num, k, max_k)

        _sync_stats_from_F(data, F, temp0, temp1, temp2, k)

        for _ in range(max_cd_iters):
            moved = _one_cd(data, F, temp0, temp1, temp2, point_sq, k)
            if moved == 0:
                break

        labels = np.empty(n, dtype=np.int64)
        for i in range(n):
            p = 0
            for j in range(k):
                if F[i, j] != 0.0:
                    p = j
                    break
            labels[i] = p

        centers_out = np.zeros((k, m), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            c = labels[i]
            counts[c] += 1
            for d in range(m):
                centers_out[c, d] += data[i, d]
        for c in range(k):
            if counts[c] > 0:
                inv = 1.0 / counts[c]
                for d in range(m):
                    centers_out[c, d] *= inv
            else:
                idx = empty_center_idx
                if idx < 0 or idx >= n:
                    idx = 0
                for d in range(m):
                    centers_out[c, d] = data[idx, d]

        # Soft scores from dists to these centers; then reassign; then refresh means.
        # (Matches prior Python quirk: soft uses pre-refresh distances.)
        soft = np.empty((n, k), dtype=np.float32)
        dists = np.empty((n, k), dtype=np.float64)
        cen_sq = np.empty(k, dtype=np.float64)
        for c in range(k):
            s = 0.0
            for d in range(m):
                s += centers_out[c, d] * centers_out[c, d]
            cen_sq[c] = s
        for i in range(n):
            best_c = 0
            best_d = np.inf
            for c in range(k):
                cross = 0.0
                for d in range(m):
                    cross += data[i, d] * centers_out[c, d]
                d2 = point_sq[i] + cen_sq[c] - 2.0 * cross
                dists[i, c] = d2
                soft[i, c] = np.float32(-d2)
                if d2 < best_d:
                    best_d = d2
                    best_c = c
            labels[i] = best_c

        for c in range(k):
            for d in range(m):
                centers_out[c, d] = 0.0
            counts[c] = 0
        for i in range(n):
            c = labels[i]
            counts[c] += 1
            for d in range(m):
                centers_out[c, d] += data[i, d]
        centers_f32 = np.empty((k, m), dtype=np.float32)
        for c in range(k):
            if counts[c] > 0:
                inv = 1.0 / counts[c]
                for d in range(m):
                    centers_f32[c, d] = np.float32(centers_out[c, d] * inv)
            else:
                idx = empty_center_idx
                if idx < 0 or idx >= n:
                    idx = 0
                for d in range(m):
                    centers_f32[c, d] = np.float32(data[idx, d])

        return labels, centers_f32, soft

    @njit(cache=True, parallel=True)
    def run_batch(data_b, k, init_indices_b, max_cd_iters, do_split_merge, empty_center_idx_b):
        """
        Independent samples on batch axis only.

        data_b: [B, n, m] float64
        init_indices_b: [B, k] int64
        empty_center_idx_b: [B] int64
        """
        B = data_b.shape[0]
        n = data_b.shape[1]
        m = data_b.shape[2]
        labels_b = np.empty((B, n), dtype=np.int64)
        centers_b = np.empty((B, k, m), dtype=np.float32)
        soft_b = np.empty((B, n, k), dtype=np.float32)
        for b in prange(B):
            labels, centers, soft = run_single(
                data_b[b],
                k,
                init_indices_b[b],
                max_cd_iters,
                do_split_merge,
                empty_center_idx_b[b],
            )
            labels_b[b] = labels
            centers_b[b] = centers
            soft_b[b] = soft
        return labels_b, centers_b, soft_b

else:  # pragma: no cover

    def run_single(*args, **kwargs):
        raise RuntimeError("numba is required for dgsm_cdkm_numba")

    def run_batch(*args, **kwargs):
        raise RuntimeError("numba is required for dgsm_cdkm_numba")


def warmup(n: int = 64, m: int = 64, k: int = 8) -> None:
    """Compile and cache kernels once before evaluation."""
    if not _HAS_NUMBA:
        return
    rng = np.random.default_rng(0)
    data = np.ascontiguousarray(rng.normal(size=(n, m)), dtype=np.float64)
    init = np.ascontiguousarray(rng.choice(n, size=k, replace=False), dtype=np.int64)
    run_single(data, k, init, 10, True, 0)
    data_b = np.ascontiguousarray(rng.normal(size=(2, n, m)), dtype=np.float64)
    init_b = np.stack([init, init], axis=0)
    empty_b = np.zeros(2, dtype=np.int64)
    run_batch(data_b, k, init_b, 10, True, empty_b)


__all__ = ["run_single", "run_batch", "warmup", "_HAS_NUMBA"]
