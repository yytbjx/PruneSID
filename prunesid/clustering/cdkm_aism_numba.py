"""Numba backend for CDKM-AISM (faithful to 源.cpp cd / splitCachedCandidate).

Kept as a top-level module so Numba's on-disk cache can be reused across
lmms-eval processes. No fastmath. All AISM decisions use FP64.

Within one sample, token-order CD is sequential. Independent split candidates
and the batch axis may use prange (lossless parallelism).
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
    def _initial_labels(data, init_indices, k):
        """Match C++: force init centers onto their own clusters, else nearest."""
        n, m = data.shape
        labels = np.empty(n, dtype=np.int64)
        for i in range(n):
            forced = -1
            for c in range(k):
                if init_indices[c] == i:
                    forced = c
                    break
            if forced >= 0:
                labels[i] = forced
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
            labels[i] = best_c
        return labels

    @njit(cache=True)
    def _stats(data, labels, k):
        n, m = data.shape
        counts = np.zeros(k, dtype=np.int64)
        sums = np.zeros((k, m), dtype=np.float64)
        for i in range(n):
            c = labels[i]
            counts[c] += 1
            for d in range(m):
                sums[c, d] += data[i, d]
        norm2 = np.zeros(k, dtype=np.float64)
        obj = np.zeros(k, dtype=np.float64)
        total = 0.0
        for c in range(k):
            ss = 0.0
            for d in range(m):
                ss += sums[c, d] * sums[c, d]
            norm2[c] = ss
            if counts[c] > 0:
                obj[c] = ss / counts[c]
                total += obj[c]
        return counts, sums, norm2, obj, total

    @njit(cache=True)
    def _cd(data, point_sq, labels, k, max_passes):
        """
        Coordinate descent matching C++ OneCd selection:
        build full M[0..k), pick first max via strict '>', ties -> lowest index.
        Empty non-current clusters score -inf (C++ yields NaN and never wins '>').
        """
        n, m = data.shape
        counts, sums, norm2, _, _ = _stats(data, labels, k)
        passes = 0
        while max_passes < 0 or passes < max_passes:
            moved = 0
            for i in range(n):
                p = labels[i]
                np_p = counts[p]
                if np_p <= 0:
                    continue

                # Stay score for current cluster p.
                dot_p = 0.0
                for d in range(m):
                    dot_p += data[i, d] * sums[p, d]
                m2 = norm2[p] - 2.0 * dot_p + point_sq[i]
                if np_p == 1 or m2 == 0.0:
                    m3 = 0.0
                else:
                    m3 = m2 / (np_p - 1)
                stay = norm2[p] / np_p - m3

                # C++: q=0, l1=M[0]; for j if M[j] > l1 update.
                best_q = 0
                if 0 == p:
                    best_m = stay
                elif counts[0] <= 0:
                    best_m = -np.inf
                else:
                    dot0 = 0.0
                    for d in range(m):
                        dot0 += data[i, d] * sums[0, d]
                    best_m = (norm2[0] + 2.0 * dot0 + point_sq[i]) / (counts[0] + 1) - norm2[0] / counts[0]

                for c in range(1, k):
                    if c == p:
                        score = stay
                    elif counts[c] <= 0:
                        score = -np.inf
                    else:
                        dot_c = 0.0
                        for d in range(m):
                            dot_c += data[i, d] * sums[c, d]
                        score = (
                            (norm2[c] + 2.0 * dot_c + point_sq[i]) / (counts[c] + 1)
                            - norm2[c] / counts[c]
                        )
                    if score > best_m:
                        best_m = score
                        best_q = c

                if best_q != p:
                    q = best_q
                    old_dot_p = 0.0
                    old_dot_q = 0.0
                    for d in range(m):
                        old_dot_p += data[i, d] * sums[p, d]
                        old_dot_q += data[i, d] * sums[q, d]
                    for d in range(m):
                        sums[p, d] -= data[i, d]
                        sums[q, d] += data[i, d]
                    counts[p] -= 1
                    counts[q] += 1
                    norm2[p] = norm2[p] - 2.0 * old_dot_p + point_sq[i]
                    norm2[q] = norm2[q] + 2.0 * old_dot_q + point_sq[i]
                    labels[i] = q
                    moved += 1
            passes += 1
            if moved == 0:
                break
        return labels

    @njit(cache=True)
    def _split_one(data, point_sq, points, count, out_labels):
        """Faithful splitCachedCandidate (2-means CD on a point subset)."""
        m = data.shape[1]
        if count <= 1:
            return -np.inf, 0

        for ii in range(count):
            out_labels[ii] = 0
        out_labels[0] = 1

        s0 = np.zeros(m, dtype=np.float64)
        s1 = np.zeros(m, dtype=np.float64)
        n0 = 0
        n1 = 0
        for ii in range(count):
            idx = points[ii]
            if out_labels[ii] == 0:
                n0 += 1
                for d in range(m):
                    s0[d] += data[idx, d]
            else:
                n1 += 1
                for d in range(m):
                    s1[d] += data[idx, d]

        norm0 = 0.0
        norm1 = 0.0
        for d in range(m):
            norm0 += s0[d] * s0[d]
            norm1 += s1[d] * s1[d]

        while True:
            changed = 0
            for ii in range(count):
                idx = points[ii]
                p = out_labels[ii]
                dot0 = 0.0
                dot1 = 0.0
                for d in range(m):
                    dot0 += data[idx, d] * s0[d]
                    dot1 += data[idx, d] * s1[d]
                xsq = point_sq[idx]

                if p == 0:
                    m2 = norm0 - 2.0 * dot0 + xsq
                    m3 = 0.0 if (n0 == 1 or m2 == 0.0) else m2 / (n0 - 1)
                    score0 = norm0 / n0 - m3
                    score1 = (norm1 + 2.0 * dot1 + xsq) / (n1 + 1) - norm1 / n1
                else:
                    score0 = (norm0 + 2.0 * dot0 + xsq) / (n0 + 1) - norm0 / n0
                    m2 = norm1 - 2.0 * dot1 + xsq
                    m3 = 0.0 if (n1 == 1 or m2 == 0.0) else m2 / (n1 - 1)
                    score1 = norm1 / n1 - m3

                # C++: q=0, then update only on strict >
                q = 0
                best = score0
                if score1 > best:
                    best = score1
                    q = 1

                if q != p:
                    if p == 0:
                        norm0 = norm0 - 2.0 * dot0 + xsq
                        norm1 = norm1 + 2.0 * dot1 + xsq
                        n0 -= 1
                        n1 += 1
                        for d in range(m):
                            s0[d] -= data[idx, d]
                            s1[d] += data[idx, d]
                    else:
                        norm1 = norm1 - 2.0 * dot1 + xsq
                        norm0 = norm0 + 2.0 * dot0 + xsq
                        n1 -= 1
                        n0 += 1
                        for d in range(m):
                            s1[d] -= data[idx, d]
                            s0[d] += data[idx, d]
                    out_labels[ii] = q
                    changed = 1
            if changed == 0:
                break

        if n0 == 0 or n1 == 0:
            return -np.inf, 0
        return norm0 / n0 + norm1 / n1, 1

    @njit(cache=True, parallel=True)
    def _split_all(
        data,
        point_sq,
        base_points,
        base_counts,
        merge_points,
        merge_counts,
        base_labels,
        base_obj,
        base_valid,
        merge_labels,
        merge_obj,
        merge_valid,
    ):
        """Independent splitCachedCandidate calls — lossless parallel."""
        k = base_points.shape[0]
        cnum = merge_points.shape[0]
        for c in prange(k):
            obj, valid = _split_one(
                data, point_sq, base_points[c], base_counts[c], base_labels[c]
            )
            base_obj[c] = obj
            base_valid[c] = valid
        for c in prange(cnum):
            obj, valid = _split_one(
                data, point_sq, merge_points[c], merge_counts[c], merge_labels[c]
            )
            merge_obj[c] = obj
            merge_valid[c] = valid

    @njit(cache=True)
    def run_single(
        data,
        k,
        init_labels,
        init_indices,
        use_indices,
        initial_cd_passes,
        max_big_loops,
        eps,
    ):
        """
        Full CDKM-AISM for one sample.

        data: [n, m] float64 contiguous
        use_indices: 1 -> assign from init_indices (k-means++); 0 -> use init_labels
        """
        n, m = data.shape
        point_sq = np.empty(n, dtype=np.float64)
        for i in range(n):
            ss = 0.0
            for d in range(m):
                ss += data[i, d] * data[i, d]
            point_sq[i] = ss

        if use_indices == 1:
            labels = _initial_labels(data, init_indices, k)
        else:
            labels = init_labels.copy()

        labels = _cd(data, point_sq, labels, k, initial_cd_passes)
        cnum = k // 2 + 1
        if cnum > k:
            cnum = k

        for _big in range(max_big_loops):
            counts, sums, norm2, arr, myuan = _stats(data, labels, k)

            # findTopkf2Indices
            top = np.full(cnum, -1, dtype=np.int64)
            for i in range(k):
                for j in range(cnum):
                    if top[j] == -1 or arr[i] > arr[top[j]]:
                        for z in range(cnum - 1, j, -1):
                            top[z] = top[z - 1]
                        top[j] = i
                        break

            merge1 = np.full(cnum, -1, dtype=np.int64)
            merge2 = np.full(cnum, -1, dtype=np.int64)
            merge_ok = np.zeros(cnum, dtype=np.uint8)
            merge_object = np.zeros(cnum, dtype=np.float64)

            for mi in range(cnum):
                c1 = top[mi]
                if c1 < 0 or counts[c1] == 0:
                    continue
                best_cost = np.inf
                best_p = -1
                for p in range(k):
                    if p == c1 or counts[p] == 0:
                        continue
                    dot = 0.0
                    for d in range(m):
                        dot += sums[c1, d] * sums[p, d]
                    merged_norm = norm2[c1] + norm2[p] + 2.0 * dot
                    merged = merged_norm / (counts[c1] + counts[p])
                    cost = arr[c1] + arr[p] - merged
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p
                if best_p >= 0:
                    he_norm = 0.0
                    for d in range(m):
                        z = sums[c1, d] + sums[best_p, d]
                        he_norm += z * z
                    merge1[mi] = c1
                    merge2[mi] = best_p
                    merge_object[mi] = he_norm / (counts[c1] + counts[best_p])
                    merge_ok[mi] = 1

            base_points = np.zeros((k, n), dtype=np.int64)
            base_counts = np.zeros(k, dtype=np.int64)
            for i in range(n):
                c = labels[i]
                base_points[c, base_counts[c]] = i
                base_counts[c] += 1

            merge_points = np.zeros((cnum, n), dtype=np.int64)
            merge_counts = np.zeros(cnum, dtype=np.int64)
            for mi in range(cnum):
                if merge_ok[mi] == 0:
                    continue
                h1 = merge1[mi]
                h2 = merge2[mi]
                for i in range(n):
                    if labels[i] == h1 or labels[i] == h2:
                        merge_points[mi, merge_counts[mi]] = i
                        merge_counts[mi] += 1

            base_split_labels = np.zeros((k, n), dtype=np.int64)
            base_split_obj = np.full(k, -np.inf, dtype=np.float64)
            base_split_valid = np.zeros(k, dtype=np.uint8)
            merge_split_labels = np.zeros((cnum, n), dtype=np.int64)
            merge_split_obj = np.full(cnum, -np.inf, dtype=np.float64)
            merge_split_valid = np.zeros(cnum, dtype=np.uint8)
            _split_all(
                data,
                point_sq,
                base_points,
                base_counts,
                merge_points,
                merge_counts,
                base_split_labels,
                base_split_obj,
                base_split_valid,
                merge_split_labels,
                merge_split_obj,
                merge_split_valid,
            )

            used = np.zeros(k, dtype=np.uint8)
            accepted_any = 0
            split_merge_times = 0
            while split_merge_times < cnum:
                best_obj = myuan
                best_split = -1
                best_h1 = -1
                best_h2 = -1
                best_mi = -1

                for mi in range(cnum):
                    if merge_ok[mi] == 0:
                        continue
                    h1 = merge1[mi]
                    h2 = merge2[mi]
                    if used[h1] == 1 or used[h2] == 1:
                        continue
                    for oo in range(k):
                        if oo == h1 or used[oo] == 1:
                            continue
                        if oo == h2:
                            if merge_split_valid[mi] == 0:
                                continue
                            cand = myuan - arr[h1] - arr[h2] + merge_split_obj[mi]
                        else:
                            if base_split_valid[oo] == 0:
                                continue
                            cand = (
                                myuan
                                - arr[h1]
                                - arr[h2]
                                - arr[oo]
                                + merge_object[mi]
                                + base_split_obj[oo]
                            )
                        if cand > best_obj + eps:
                            best_obj = cand
                            best_split = oo
                            best_h1 = h1
                            best_h2 = h2
                            best_mi = mi

                if best_split < 0:
                    break

                for i in range(n):
                    if labels[i] == best_h1:
                        labels[i] = best_h2

                if best_split == best_h2:
                    for ii in range(merge_counts[best_mi]):
                        idx = merge_points[best_mi, ii]
                        if merge_split_labels[best_mi, ii] == 0:
                            labels[idx] = best_h2
                        else:
                            labels[idx] = best_h1
                else:
                    for ii in range(base_counts[best_split]):
                        idx = base_points[best_split, ii]
                        if base_split_labels[best_split, ii] == 0:
                            labels[idx] = best_split
                        else:
                            labels[idx] = best_h1

                myuan = best_obj
                accepted_any = 1
                split_merge_times += 1
                used[best_split] = 1
                used[best_h1] = 1
                used[best_h2] = 1

            if accepted_any == 0:
                break
            # Post-accept CD to convergence (C++ while(1)).
            labels = _cd(data, point_sq, labels, k, -1)

        # Finalize: centers from F, one nearest-center pass (no second mean refresh).
        counts, sums, _, _, _ = _stats(data, labels, k)
        centers = np.zeros((k, m), dtype=np.float64)
        for c in range(k):
            if counts[c] > 0:
                inv = 1.0 / counts[c]
                for d in range(m):
                    centers[c, d] = sums[c, d] * inv

        dists = np.empty((n, k), dtype=np.float64)
        final_labels = np.empty(n, dtype=np.int64)
        for i in range(n):
            best_d = np.inf
            best_c = 0
            for c in range(k):
                if counts[c] == 0:
                    d2 = np.inf
                else:
                    d2 = 0.0
                    for d in range(m):
                        z = data[i, d] - centers[c, d]
                        d2 += z * z
                dists[i, c] = d2
                if d2 < best_d:
                    best_d = d2
                    best_c = c
            final_labels[i] = best_c
        return final_labels, dists

    def run_batch(
        data_b,
        k,
        init_labels_b,
        init_indices_b,
        use_indices_b,
        initial_cd_passes,
        max_big_loops,
        eps,
    ):
        """
        Independent samples on batch axis.

        Implemented in Python (not nested njit prange) so each sample can keep
        lossless parallel splitCachedCandidate inside run_single without Windows
        nested-parallel crashes.

        data_b: [B, n, m] float64
        """
        B = data_b.shape[0]
        n = data_b.shape[1]
        labels_b = np.empty((B, n), dtype=np.int64)
        dists_b = np.empty((B, n, k), dtype=np.float64)
        for b in range(B):
            labels, dists = run_single(
                data_b[b],
                k,
                init_labels_b[b],
                init_indices_b[b],
                int(use_indices_b[b]),
                initial_cd_passes,
                max_big_loops,
                eps,
            )
            labels_b[b] = labels
            dists_b[b] = dists
        return labels_b, dists_b

else:  # pragma: no cover

    def run_single(*args, **kwargs):
        raise RuntimeError("numba is required for cdkm_aism_numba")

    def run_batch(*args, **kwargs):
        raise RuntimeError("numba is required for cdkm_aism_numba")


def warmup(n: int = 64, m: int = 64, k: int = 8) -> None:
    """Compile and cache kernels once before evaluation."""
    if not _HAS_NUMBA:
        return
    rng = np.random.default_rng(0)
    data = np.ascontiguousarray(rng.normal(size=(n, m)), dtype=np.float64)
    init = np.ascontiguousarray(rng.choice(n, size=k, replace=False), dtype=np.int64)
    labels0 = np.zeros(n, dtype=np.int64)
    run_single(data, k, labels0, init, 1, 10, 5, 1e-9)
    data_b = np.ascontiguousarray(rng.normal(size=(2, n, m)), dtype=np.float64)
    init_b = np.stack([init, init], axis=0)
    labels_b = np.zeros((2, n), dtype=np.int64)
    use_b = np.ones(2, dtype=np.int64)
    run_batch(data_b, k, labels_b, init_b, use_b, 10, 5, 1e-9)


__all__ = ["run_single", "run_batch", "warmup", "_HAS_NUMBA"]
