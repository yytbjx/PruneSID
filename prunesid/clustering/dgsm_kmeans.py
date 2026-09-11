"""
DGSM-KMeans Exact: GPU flash Lloyd + DGSM 8-leaf split/merge.

NOT faithful DGSM-CDKM (no coordinate descent).

Approximation-free implementation optimizations (Algorithm-Exact / E1):
  - Lazy second k-means++ (empty clusters only)
  - Incremental k-means++ nearest-distance
  - Init Lloyd without unused full distance matrix
  - Compact gather (prefix-sum) + CPU hier merge for L=8 (fewer CUDA launches)
  - In-place split; moved-token aggregate (not per-token scatter)
  - Main merge: one Gram + row refresh; delayed label remap via parent
  - SSE from sufficient statistics; assign returns best_d
  - flash-kmeans only with init_centroids; else self Lloyd (no re-init)

Pipeline (algorithm frozen):
  1) k-means++ + flash Lloyd until relative SSE gain < 10%
  2) Oversplit ~1.25K: global top-16 dims, 8-leaf one-pass split
  3) Merge back to K (Gram incremental)
  4) flash Lloyd to label convergence (capped); soft = -dists
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from flash_kmeans import batch_kmeans_Euclid as _FLASH_BATCH_KMEANS

    _HAS_FLASH_KMEANS = True
except Exception:  # pragma: no cover
    _FLASH_BATCH_KMEANS = None
    _HAS_FLASH_KMEANS = False


# ---------------------------------------------------------------------------
# Distances / Lloyd
# ---------------------------------------------------------------------------


def _squared_dists(data: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x2 = (data * data).sum(dim=-1, keepdim=True)
    c2 = (centers * centers).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(data, centers.transpose(1, 2))
    return (x2 + c2 - 2.0 * cross).clamp_min(0.0)


def _flash_assign(
    data: torch.Tensor,
    centers: torch.Tensor,
    point_sq: torch.Tensor,
    chunk_k: int = 32,
    *,
    best_d: Optional[torch.Tensor] = None,
    best_c: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact nearest-center assignment. Returns (labels [B,N], best_d [B,N]).
    Small K (<= chunk_k): single bmm. Else chunked online argmin.
    """
    B, N, _D = data.shape
    K = centers.shape[1]
    device = data.device
    dtype = data.dtype
    x2 = point_sq

    if K <= chunk_k:
        c2 = (centers * centers).sum(dim=-1)
        cross = torch.bmm(data, centers.transpose(1, 2))
        d = (x2.unsqueeze(-1) + c2.unsqueeze(1) - 2.0 * cross).clamp_min(0.0)
        best_d_out, best_c_out = d.min(dim=-1)
        return best_c_out, best_d_out

    if best_d is None:
        best_d = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    else:
        best_d.fill_(float("inf"))
    if best_c is None:
        best_c = torch.zeros(B, N, dtype=torch.long, device=device)
    else:
        best_c.zero_()

    for k0 in range(0, K, chunk_k):
        k1 = min(K, k0 + chunk_k)
        cen = centers[:, k0:k1]
        c2 = (cen * cen).sum(dim=-1)
        cross = torch.bmm(data, cen.transpose(1, 2))
        d = (x2.unsqueeze(-1) + c2.unsqueeze(1) - 2.0 * cross).clamp_min(0.0)
        dmin, amin = d.min(dim=-1)
        better = dmin < best_d
        best_d = torch.where(better, dmin, best_d)
        best_c = torch.where(better, amin + k0, best_c)
    return best_c, best_d


def _onehot_centroid_update(
    data: torch.Tensor, labels: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact centroid update via one_hot^T @ X.
    Returns centers [B,K,D], counts [B,K], sums [B,K,D].
    """
    B, N, D = data.shape
    device = data.device
    dtype = data.dtype
    oh = F.one_hot(labels, k).to(dtype=dtype)  # [B,N,K]
    sums = torch.bmm(oh.transpose(1, 2), data)  # [B,K,D]
    counts = oh.sum(dim=1)  # [B,K]
    centers = sums / counts.clamp_min(1.0).unsqueeze(-1)
    empty = counts <= 0
    if bool(empty.any()):
        batch = torch.arange(B, device=device)
        rand_idx = torch.randint(0, N, (B,), device=device)
        fill = data[batch, rand_idx].unsqueeze(1).expand(-1, k, -1)
        centers = torch.where(empty.unsqueeze(-1), fill, centers)
    return centers, counts, sums


def _sse_from_stats(
    counts: torch.Tensor, sums: torch.Tensor, total_x2: torch.Tensor
) -> torch.Tensor:
    """Partition SSE = Σ||x||² - Σ_c ||S_c||² / n_c. total_x2: [B]."""
    sn = (sums * sums).sum(dim=-1)
    term = torch.where(counts > 0, sn / counts.clamp_min(1.0), torch.zeros_like(sn))
    return (total_x2 - term.sum(dim=-1)).clamp_min(0.0)


def _lloyd(
    data: torch.Tensor,
    centers: torch.Tensor,
    max_iters: int,
    point_sq: torch.Tensor,
    *,
    stop: str = "fixed",
    sse_rel_tol: float = 0.1,
    return_full_dists: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact flash Lloyd.

    stop: "fixed" | "sse_tol" | "converge"
    return_full_dists: if False, third return is None (skip unused [B,N,K]).
    """
    K = centers.shape[1]
    device = data.device
    B, N, _D = data.shape
    total_x2 = point_sq.sum(dim=-1)

    # Optional Triton path only for fixed-iter AND with init centroids.
    if (
        _HAS_FLASH_KMEANS
        and data.is_cuda
        and stop == "fixed"
        and max_iters > 0
    ):
        try:
            _ids, cen, _info = _FLASH_BATCH_KMEANS(
                data.contiguous(),
                n_clusters=K,
                max_iters=max_iters,
                tol=0.0,
                init_centroids=centers.contiguous(),
            )
            centers = cen.to(dtype=data.dtype)
            if return_full_dists:
                dists = _squared_dists(data, centers)
                labels = dists.argmin(dim=-1)
                return labels, centers, dists
            labels, _bd = _flash_assign(data, centers, point_sq)
            return labels, centers, None
        except Exception:
            # Exact: never fall back to package re-init without our centers.
            pass

    prev_labels: Optional[torch.Tensor] = None
    labels = torch.zeros(B, N, dtype=torch.long, device=device)
    prev_sse = None

    for _ in range(max(1, max_iters)):
        new_labels, _best_d = _flash_assign(data, centers, point_sq)
        centers, counts, sums = _onehot_centroid_update(data, new_labels, K)

        if stop == "converge":
            if prev_labels is not None and torch.equal(new_labels, prev_labels):
                labels = new_labels
                break
            prev_labels = new_labels
            labels = new_labels
            continue

        labels = new_labels
        if stop == "sse_tol":
            sse = _sse_from_stats(counts, sums, total_x2)
            if prev_sse is not None:
                prev = prev_sse.clamp_min(torch.finfo(data.dtype).tiny)
                rel = (prev_sse - sse) / prev
                if bool((rel < sse_rel_tol).all()):
                    break
            prev_sse = sse

    if return_full_dists:
        dists = _squared_dists(data, centers)
        labels = dists.argmin(dim=-1)
        return labels, centers, dists
    labels, _bd = _flash_assign(data, centers, point_sq)
    return labels, centers, None


# ---------------------------------------------------------------------------
# Init / stats
# ---------------------------------------------------------------------------


def _global_top_variance_dims(data: torch.Tensor, n_dims: int = 16) -> torch.Tensor:
    """Batch-shared top variance dims (same as prior Exact behavior)."""
    n_dims = min(n_dims, data.shape[-1])
    mean = data.mean(dim=1, keepdim=True)
    var = ((data - mean) ** 2).mean(dim=1)  # [B,D]
    var_mean = var.mean(dim=0)
    return torch.argsort(var_mean, descending=True, stable=True)[:n_dims]


def _kmeans_plusplus(data: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Incremental nearest-distance k-means++ (O(NDK) distances)."""
    B, N, D = data.shape
    device = data.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    batch = torch.arange(B, device=device)
    centers = torch.empty(B, k, D, dtype=data.dtype, device=device)
    chosen = torch.zeros(B, N, dtype=torch.bool, device=device)

    first = torch.randint(0, N, (B,), generator=g, device=device)
    centers[:, 0] = data[batch, first]
    chosen[batch, first] = True
    closest = torch.full((B, N), float("inf"), dtype=data.dtype, device=device)

    for c in range(1, k):
        # Only distance to the center added in the previous step.
        new_c = centers[:, c - 1 : c]
        dist = _squared_dists(data, new_c).squeeze(-1)
        closest = torch.minimum(closest, dist)
        closest = closest.masked_fill(chosen, 0.0)
        total = closest.sum(dim=-1)
        probs = closest / total.clamp_min(torch.finfo(data.dtype).tiny).unsqueeze(-1)
        fallback = (~chosen).to(data.dtype)
        fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = torch.where((total > 0).unsqueeze(-1), probs, fallback)
        nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1)
        centers[:, c] = data[batch, nxt]
        chosen[batch, nxt] = True
    return centers


def _stats_from_labels(
    data: torch.Tensor, labels: torch.Tensor, k: int, point_sq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster stats via one_hot / bmm (GEMM-friendly, no scatter atomics)."""
    B, N, D = data.shape
    dtype = data.dtype
    oh = F.one_hot(labels.clamp(0, k - 1), k).to(dtype=dtype)  # [B,N,K]
    counts = oh.sum(dim=1)
    sums = torch.bmm(oh.transpose(1, 2), data)
    sum_x2 = torch.bmm(oh.transpose(1, 2), point_sq.unsqueeze(-1)).squeeze(-1)
    return counts, sums, sum_x2


def _sse(counts: torch.Tensor, sums: torch.Tensor, sum_x2: torch.Tensor) -> torch.Tensor:
    sum_norm2 = (sums * sums).sum(dim=-1)
    return torch.where(
        counts > 1,
        sum_x2 - sum_norm2 / counts.clamp_min(1.0),
        torch.zeros_like(sum_x2),
    )


# ---------------------------------------------------------------------------
# Gather / 8-leaf split
# ---------------------------------------------------------------------------


def _gather_cluster_padded(
    data: torch.Tensor, member: torch.Tensor, n_mem: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prefix-sum compact members into [B,M,D] (M=max n_mem).
    Preserves original token order among members.
    """
    B, N, D = data.shape
    device = data.device
    dtype = data.dtype
    M = int(n_mem.max().item()) if B > 0 else 0
    M = max(M, 1)

    pos = member.to(torch.long).cumsum(dim=1) - 1
    packed = torch.zeros(B, M, D, dtype=dtype, device=device)
    packed_idx = torch.full((B, M), -1, dtype=torch.long, device=device)
    valid = torch.arange(M, device=device).view(1, M) < n_mem.unsqueeze(1).clamp_min(0)

    flat_b = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    tok = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    if member.any():
        b_m = flat_b[member]
        t_m = tok[member]
        p_m = pos[member]
        # clamp for safety if any batch has fewer members
        p_m = p_m.clamp(0, M - 1)
        packed[b_m, p_m] = data[b_m, t_m]
        packed_idx[b_m, p_m] = t_m
    packed = packed * valid.unsqueeze(-1).to(dtype)
    packed_idx = torch.where(valid, packed_idx, torch.full_like(packed_idx, -1))
    return packed, packed_idx, valid


def _hier_merge_leaves_cpu(
    leaf_counts: torch.Tensor, leaf_sums: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hierarchical merge L->2 on CPU (L<=8: fewer launches than GPU kernels).
    Same merge score as Gram path.
    """
    import numpy as np

    device = leaf_counts.device
    lc = leaf_counts.detach().float().cpu().numpy()
    ls = leaf_sums.detach().float().cpu().numpy()
    B, L, _D = ls.shape
    parents = np.empty((B, L), dtype=np.int64)
    f0s = np.empty(B, dtype=np.int64)

    for b in range(B):
        counts = lc[b].copy()
        sums = ls[b].copy()
        active = counts > 0
        parent = np.arange(L, dtype=np.int64)
        n_active = int(active.sum())
        while n_active > 2:
            best_val = -np.inf
            best_a = best_b = -1
            ids = np.nonzero(active)[0]
            for ii, a in enumerate(ids):
                ti = float(counts[a])
                if ti <= 0:
                    continue
                for bb in ids[ii + 1 :]:
                    tj = float(counts[bb])
                    if tj <= 0:
                        continue
                    diff = sums[a] * tj - sums[bb] * ti
                    val = -float(np.dot(diff, diff)) / (ti * tj * (ti + tj))
                    if val > best_val:
                        best_val = val
                        best_a, best_b = int(a), int(bb)
            if best_a < 0:
                break
            sums[best_a] = sums[best_a] + sums[best_b]
            counts[best_a] = counts[best_a] + counts[best_b]
            counts[best_b] = 0.0
            active[best_b] = False
            parent[parent == best_b] = best_a
            for t in range(L):
                if parent[t] == best_b:
                    parent[t] = best_a
            n_active -= 1

        for t in range(L):
            r = t
            while parent[r] != r:
                r = parent[r]
            root = r
            r = t
            while parent[r] != r:
                nxt = parent[r]
                parent[r] = root
                r = nxt

        act = np.nonzero(active)[0]
        f0s[b] = int(act[0]) if act.size else 0
        parents[b] = parent

    parent_t = torch.from_numpy(parents).to(device=device, dtype=torch.long)
    f0_t = torch.from_numpy(f0s).to(device=device, dtype=torch.long)
    return parent_t, f0_t


def _eight_leaf_split_packed(
    packed: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    split_num: int,
    candidate_dims: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-pass 8-leaf on packed members; CPU hier merge L->2."""
    B, M, D = packed.shape
    device = packed.device
    dtype = packed.dtype
    L = 1 << split_num

    if candidate_dims is None:
        abs_dev = (packed - mean.unsqueeze(1)).abs() * valid.unsqueeze(-1).to(dtype)
        dim_scores = abs_dev.sum(dim=1)
        order = torch.argsort(dim_scores, dim=-1, descending=True, stable=True)
        chosen = order[:, :split_num]
    else:
        C = int(candidate_dims.numel())
        sub = packed.index_select(2, candidate_dims)
        mean_sub = mean.index_select(1, candidate_dims)
        abs_dev = (sub - mean_sub.unsqueeze(1)).abs() * valid.unsqueeze(-1).to(dtype)
        dim_scores = abs_dev.sum(dim=1)
        order_local = torch.argsort(dim_scores, dim=-1, descending=True, stable=True)
        take = min(split_num, C)
        local = order_local[:, :take]
        chosen = candidate_dims[local]
        if take < split_num:
            pad = chosen[:, :1].expand(B, split_num - take)
            chosen = torch.cat([chosen, pad], dim=1)

    signs = torch.zeros(B, M, dtype=torch.long, device=device)
    for t in range(split_num):
        dim_t = chosen[:, t]
        vals = torch.gather(packed, 2, dim_t.view(B, 1, 1).expand(B, M, 1)).squeeze(-1)
        thr = torch.gather(mean, 1, dim_t.unsqueeze(1)).squeeze(1)
        ge = (vals >= thr.unsqueeze(1)) & valid
        signs = (signs << 1) | ge.to(torch.long)
    signs = torch.where(valid, signs.clamp(0, L - 1), torch.zeros_like(signs))

    ones = valid.to(dtype)
    leaf_counts = torch.zeros(B, L, dtype=dtype, device=device)
    leaf_sums = torch.zeros(B, L, D, dtype=dtype, device=device)
    leaf_counts.scatter_add_(1, signs, ones)
    leaf_sums.scatter_add_(
        1, signs.unsqueeze(-1).expand(B, M, D), packed * ones.unsqueeze(-1)
    )

    parent, f0 = _hier_merge_leaves_cpu(leaf_counts, leaf_sums)
    point_root = torch.gather(parent, 1, signs)
    return valid & (point_root != f0.unsqueeze(1))


def _split_one_round(
    data: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    sum_x2: torch.Tensor,
    clu_num: int,
    new_clu: int,
    split_num: int,
    point_sq: torch.Tensor,
    candidate_dims: Optional[torch.Tensor] = None,
) -> None:
    """In-place max-SSE split; moved tokens aggregated once (no per-token scatter)."""
    B, N, D = data.shape
    device = data.device
    dtype = data.dtype
    batch = torch.arange(B, device=device)

    sse = _sse(counts[:, :clu_num], sums[:, :clu_num], sum_x2[:, :clu_num])
    sse = sse.masked_fill(counts[:, :clu_num] <= 1, float("-inf"))
    split_clu = sse.argmax(dim=-1)

    member = labels == split_clu.unsqueeze(1)
    n_mem = member.sum(dim=-1)
    can = n_mem >= 2

    mean = sums[batch, split_clu] / n_mem.clamp_min(1).to(dtype).unsqueeze(-1)

    packed, packed_idx, valid = _gather_cluster_padded(data, member, n_mem)
    move_local = _eight_leaf_split_packed(
        packed, valid, mean, split_num, candidate_dims=candidate_dims
    )
    move_local = move_local & can.unsqueeze(1)

    move_f = move_local.to(dtype)
    moved_count = move_f.sum(dim=1)
    moved_sum = torch.bmm(move_f.unsqueeze(1), packed).squeeze(1)
    # point_sq in packed layout
    ps_pack = torch.zeros(B, packed.shape[1], dtype=dtype, device=device)
    ok = packed_idx >= 0
    if ok.any():
        bb = batch.unsqueeze(1).expand_as(packed_idx)[ok]
        pp = torch.arange(packed.shape[1], device=device).view(1, -1).expand(B, -1)[ok]
        ti = packed_idx[ok]
        ps_pack[bb, pp] = point_sq[bb, ti]
    moved_q = (move_f * ps_pack).sum(dim=1)

    do = (moved_count > 0) & can
    vf = do.to(dtype)
    v = vf.unsqueeze(-1)
    old_c = split_clu
    counts[batch, old_c] = counts[batch, old_c] - moved_count * vf
    counts[batch, new_clu] = counts[batch, new_clu] + moved_count * vf
    sums[batch, old_c] = sums[batch, old_c] - moved_sum * v
    sums[batch, new_clu] = sums[batch, new_clu] + moved_sum * v
    sum_x2[batch, old_c] = sum_x2[batch, old_c] - moved_q * vf
    sum_x2[batch, new_clu] = sum_x2[batch, new_clu] + moved_q * vf

    write = move_local & (packed_idx >= 0)
    if write.any():
        bb = batch.unsqueeze(1).expand_as(packed_idx)[write]
        ti = packed_idx[write]
        labels[bb, ti] = new_clu


def _merge_cost_from_gram(
    counts: torch.Tensor, gram: torch.Tensor, sn: torch.Tensor
) -> torch.Tensor:
    """Pairwise merge score from Gram and ||S||². Higher is better."""
    _B, K = counts.shape
    device = counts.device
    ti = counts.unsqueeze(2)
    tj = counts.unsqueeze(1)
    numer = -(
        tj * tj * sn.unsqueeze(2) + ti * ti * sn.unsqueeze(1) - 2.0 * ti * tj * gram
    )
    denom = ti * tj * (ti + tj)
    cost = numer / denom.clamp_min(torch.finfo(counts.dtype).tiny)
    invalid = (counts.unsqueeze(2) <= 0) | (counts.unsqueeze(1) <= 0)
    cost = cost.masked_fill(invalid, float("-inf"))
    eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
    cost = cost.masked_fill(eye, float("-inf"))
    return cost


def _gram_merge_pair(
    counts: torch.Tensor,
    sums: torch.Tensor,
    gram: torch.Tensor,
    sn: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    active: Optional[torch.Tensor] = None,
) -> None:
    """
    In-place merge b -> a for each batch (a,b are [B] indices).
    Updates counts, sums, then refreshes Gram row/col a via S_a·S^T (O(KD)),
    zeros row/col b. Algebraically same as full SS^T with updated sums.
    """
    B, K, _D = sums.shape
    device = sums.device
    dtype = sums.dtype
    batch = torch.arange(B, device=device)
    do = a != b
    if active is not None:
        do = do & active[batch, a] & active[batch, b]

    if not bool(do.any()):
        return

    sa = sums[batch, a]
    sb = sums[batch, b]
    ca = counts[batch, a]
    cb = counts[batch, b]
    v = do.to(dtype).unsqueeze(-1)
    vf = do.to(dtype)

    sa_new = sa + sb * v
    ca_new = ca + cb * vf

    sums[batch, a] = torch.where(do.unsqueeze(-1), sa_new, sa)
    counts[batch, a] = torch.where(do, ca_new, ca)
    sums[batch, b] = torch.where(do.unsqueeze(-1), torch.zeros_like(sb), sb)
    counts[batch, b] = torch.where(do, torch.zeros_like(cb), cb)

    # Refresh row/col a from updated sums: g_a = S_a @ S^T
    g_a = torch.bmm(sums[batch, a].unsqueeze(1), sums.transpose(1, 2)).squeeze(1)
    zero_k = torch.zeros(B, K, dtype=dtype, device=device)
    old_row_a = gram[batch, a, :].clone()
    old_col_a = gram[batch, :, a].clone()
    old_row_b = gram[batch, b, :].clone()
    old_col_b = gram[batch, :, b].clone()

    gram[batch, a, :] = torch.where(do.unsqueeze(-1), g_a, old_row_a)
    gram[batch, :, a] = torch.where(do.unsqueeze(-1), g_a, old_col_a)
    gram[batch, b, :] = torch.where(do.unsqueeze(-1), zero_k, old_row_b)
    gram[batch, :, b] = torch.where(do.unsqueeze(-1), zero_k, old_col_b)

    sn[batch, a] = torch.where(do, (sa_new * sa_new).sum(dim=-1), sn[batch, a])
    sn[batch, b] = torch.where(do, torch.zeros_like(cb), sn[batch, b])

    if active is not None:
        active[batch, b] = active[batch, b] & ~do


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------


def _merge_back_gpu(
    data: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    max_k: int,
    point_sq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedy merge max_k -> k with one Gram + incremental updates;
    delay label remap via parent until the end.
    """
    B, N, D = data.shape
    device = data.device
    dtype = data.dtype
    batch = torch.arange(B, device=device)

    counts, sums, sum_x2 = _stats_from_labels(data, labels, max_k, point_sq)
    gram = torch.bmm(sums, sums.transpose(1, 2))
    sn = torch.diagonal(gram, dim1=1, dim2=2).clone()

    # Union-find parent over cluster ids
    parent = torch.arange(max_k, device=device, dtype=torch.long).view(1, max_k).expand(B, max_k).clone()

    triu = torch.triu(torch.ones(max_k, max_k, dtype=torch.bool, device=device), diagonal=1)
    eye = torch.eye(max_k, dtype=torch.bool, device=device)

    for _ in range(max_k - k):
        cost = _merge_cost_from_gram(counts, gram, sn)
        cost = cost.masked_fill(eye.unsqueeze(0), float("-inf"))
        cost = cost.masked_fill(~triu.unsqueeze(0), float("-inf"))
        nonempty = (counts > 0).sum(dim=-1)
        need = nonempty > k
        cost = cost.masked_fill(~need.view(B, 1, 1), float("-inf"))

        flat = cost.reshape(B, -1)
        best = flat.argmax(dim=-1)
        bi = best // max_k
        bj = best % max_k
        best_val = flat[batch, best]
        valid_pair = need & torch.isfinite(best_val)
        b_m = torch.where(valid_pair, bj, bi)

        # Stats + gram
        # Also update sum_x2
        vf = valid_pair.to(dtype)
        sum_x2[batch, bi] = sum_x2[batch, bi] + sum_x2[batch, bj] * vf
        sum_x2[batch, bj] = torch.where(
            valid_pair, torch.zeros_like(sum_x2[batch, bj]), sum_x2[batch, bj]
        )

        _gram_merge_pair(counts, sums, gram, sn, bi, b_m, active=None)
        # _gram_merge_pair already zeros b when bi!=b_m; when invalid bi==b_m no-op

        # parent: map bj -> bi
        match = (parent == bj.unsqueeze(1)) & valid_pair.unsqueeze(1)
        parent = torch.where(match, bi.unsqueeze(1).expand(B, max_k), parent)

    # Compress parent
    for _ in range(max_k):
        parent = torch.gather(parent, 1, parent)

    # Remap labels once
    labels = torch.gather(parent, 1, labels.clamp(0, max_k - 1))

    # Compact cluster ids to 0..k-1
    nonempty_mask = counts > 0
    ranks = nonempty_mask.to(torch.long).cumsum(dim=-1) - 1
    map_ids = torch.where(nonempty_mask, ranks, torch.zeros_like(ranks)).clamp(max=k - 1)
    labels = torch.gather(map_ids, 1, labels.clamp(0, max_k - 1))
    counts, sums, sum_x2 = _stats_from_labels(data, labels, k, point_sq)
    return labels, counts, sums, sum_x2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def batch_dgsm_kmeans(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_lloyd_iters: int = 16,
    do_split_merge: bool = True,
    *,
    oversplit_ratio: float = 1.25,
    global_var_dims: int = 16,
    init_sse_rel_tol: float = 0.1,
    final_max_iters: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact-speed DGSM-Lloyd (algorithm-frozen defaults + implementation opts).
    """
    data = torch.sigmoid(features.to(torch.float32))
    if drop_cls:
        if data.shape[1] < 2:
            raise ValueError("features must include CLS + patches when drop_cls=True")
        data = data[:, 1:, :].contiguous()
    else:
        data = data.contiguous()

    B, N, D = data.shape
    k = max(1, min(int(min_components), N))
    point_sq = (data * data).sum(dim=-1)
    final_cap = (
        int(final_max_iters) if final_max_iters is not None else max(32, max_lloyd_iters * 4)
    )

    centers = _kmeans_plusplus(data, k, seed)
    labels, centers, _ = _lloyd(
        data,
        centers,
        max_lloyd_iters,
        point_sq,
        stop="sse_tol",
        sse_rel_tol=init_sse_rel_tol,
        return_full_dists=False,
    )

    max_k = max(k, min(N, int(round(oversplit_ratio * k))))
    dists: Optional[torch.Tensor] = None
    if do_split_merge and max_k > k:
        split_num = min(3, D)
        cand = _global_top_variance_dims(data, n_dims=min(global_var_dims, D))
        counts, sums, sum_x2 = _stats_from_labels(data, labels, max_k, point_sq)
        clu_num = k
        while clu_num < max_k:
            _split_one_round(
                data,
                labels,
                counts,
                sums,
                sum_x2,
                clu_num=clu_num,
                new_clu=clu_num,
                split_num=split_num,
                point_sq=point_sq,
                candidate_dims=cand,
            )
            clu_num += 1

        labels, counts, sums, _ = _merge_back_gpu(
            data, labels, k=k, max_k=max_k, point_sq=point_sq
        )
        nonempty = counts > 0
        centers = sums / counts.clamp_min(1.0).unsqueeze(-1)
        if bool((~nonempty).any()):
            fill = _kmeans_plusplus(data, k, seed + 1)
            centers = torch.where(nonempty.unsqueeze(-1), centers, fill)

        labels, centers, dists = _lloyd(
            data,
            centers,
            final_cap,
            point_sq,
            stop="converge",
            return_full_dists=True,
        )
    else:
        dists = _squared_dists(data, centers)
        labels = dists.argmin(dim=-1)

    soft = (-dists).to(torch.float32)
    return soft, labels


__all__ = ["batch_dgsm_kmeans", "_HAS_FLASH_KMEANS"]
