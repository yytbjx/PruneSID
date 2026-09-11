"""
DGSM-KM+: GPU Lloyd + optional spatial/importance DGSM split/merge.

Batch-invariance (same image ⇒ same clustering for any B):
  - Per-image KMeans++ RNG streams (seed + b * 1000003)
  - Per-sample Lloyd early-stop (freeze finished images)
  - Per-image seeded empty-cluster fill
  - Per-image top variance dims
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
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


def _squared_dists(
    data: torch.Tensor,
    centers: torch.Tensor,
    point_sq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x2 = point_sq.unsqueeze(-1) if point_sq is not None else (data * data).sum(dim=-1, keepdim=True)
    c2 = (centers * centers).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(data, centers.transpose(1, 2))
    return (x2 + c2 - 2.0 * cross).clamp_min(0.0)


def _flash_assign(
    data: torch.Tensor,
    centers: torch.Tensor,
    point_sq: torch.Tensor,
    chunk_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _D = data.shape
    K = centers.shape[1]
    x2 = point_sq

    if K <= chunk_k:
        c2 = (centers * centers).sum(dim=-1)
        cross = torch.bmm(data, centers.transpose(1, 2))
        d = (x2.unsqueeze(-1) + c2.unsqueeze(1) - 2.0 * cross).clamp_min(0.0)
        best_d, best_c = d.min(dim=-1)
        return best_c, best_d

    device, dtype = data.device, data.dtype
    best_d = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    best_c = torch.zeros(B, N, dtype=torch.long, device=device)
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
    data: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    point_sq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact centroid update via one_hot^T @ X.
    Empty clusters filled deterministically with max-||x|| token (batch-invariant).
    """
    B, N, D = data.shape
    device, dtype = data.device, data.dtype
    oh = F.one_hot(labels, k).to(dtype=dtype)
    sums = torch.bmm(oh.transpose(1, 2), data)
    counts = oh.sum(dim=1)
    centers = sums / counts.clamp_min(1.0).unsqueeze(-1)
    empty = counts <= 0
    if bool(empty.any()):
        batch = torch.arange(B, device=device)
        if point_sq is None:
            point_sq = (data * data).sum(dim=-1)
        fill_idx = point_sq.argmax(dim=-1)
        fill = data[batch, fill_idx].unsqueeze(1).expand(-1, k, -1)
        centers = torch.where(empty.unsqueeze(-1), fill, centers)
    return centers, counts, sums


def _sse_from_stats(
    counts: torch.Tensor, sums: torch.Tensor, total_x2: torch.Tensor
) -> torch.Tensor:
    sn = (sums * sums).sum(dim=-1)
    term = torch.where(counts > 0, sn / counts.clamp_min(1.0), torch.zeros_like(sn))
    return (total_x2 - term.sum(dim=-1)).clamp_min(0.0)


def _lloyd(
    data: torch.Tensor,
    centers: torch.Tensor,
    max_iters: int,
    point_sq: torch.Tensor,
    *,
    stop: str = "sse_tol",
    sse_rel_tol: float = 1e-3,
    return_full_dists: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact flash Lloyd with per-sample early stop (batch-invariant trajectories).

    stop:
      - "sse_tol": freeze each sample when relative SSE drop < sse_rel_tol
      - "fixed": run max_iters
      - "converge": freeze when labels unchanged
    """
    K = centers.shape[1]
    device = data.device
    B, N, _D = data.shape
    total_x2 = point_sq.sum(dim=-1)
    dtype = data.dtype
    _ = seed  # reserved; empty fills are deterministic from point_sq

    labels = torch.zeros(B, N, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    prev_sse: Optional[torch.Tensor] = None
    prev_labels: Optional[torch.Tensor] = None

    for it in range(max(1, max_iters)):
        if not bool(active.any()):
            break

        new_labels, _bd = _flash_assign(data, centers, point_sq)
        new_centers, _counts, _sums = _onehot_centroid_update(
            data, new_labels, K, point_sq=point_sq
        )

        a1 = active.view(B, 1)
        a2 = active.view(B, 1, 1)
        labels = torch.where(a1, new_labels, labels)
        centers = torch.where(a2, new_centers, centers)

        oh = F.one_hot(labels, K).to(dtype=dtype)
        k_sums = torch.bmm(oh.transpose(1, 2), data)
        k_counts = oh.sum(dim=1)
        sse = _sse_from_stats(k_counts, k_sums, total_x2)

        if stop == "converge":
            if prev_labels is not None:
                unchanged = (labels == prev_labels).all(dim=-1)
                active = active & ~unchanged
            prev_labels = labels.clone()
            continue

        if stop == "sse_tol":
            if prev_sse is not None:
                prev = prev_sse.clamp_min(torch.finfo(dtype).tiny)
                rel = (prev_sse - sse) / prev
                newly_done = active & (rel < sse_rel_tol)
                active = active & ~newly_done
            prev_sse = torch.where(
                active,
                sse,
                prev_sse if prev_sse is not None else sse,
            )

    # Final assign from frozen centers (per-image; no cross-batch state).
    if return_full_dists:
        dists = _squared_dists(data, centers, point_sq)
        labels = dists.argmin(dim=-1)
        return labels, centers, dists
    labels, _ = _flash_assign(data, centers, point_sq)
    return labels, centers, None


# ---------------------------------------------------------------------------
# Init / stats / coords / importance
# ---------------------------------------------------------------------------


def _per_image_top_variance_dims(data: torch.Tensor, n_dims: int = 16) -> torch.Tensor:
    n_dims = min(n_dims, data.shape[-1])
    mean = data.mean(dim=1, keepdim=True)
    var = ((data - mean) ** 2).mean(dim=1)
    return torch.argsort(var, dim=-1, descending=True, stable=True)[:, :n_dims]


_global_top_variance_dims = _per_image_top_variance_dims


def _patch_xy(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized (x,y) for N patches. Square grid when possible."""
    side = int(math.sqrt(n))
    if side * side == n:
        yy, xx = torch.meshgrid(
            torch.arange(side, device=device, dtype=dtype),
            torch.arange(side, device=device, dtype=dtype),
            indexing="ij",
        )
        denom = max(side - 1, 1)
        return torch.stack([xx.reshape(-1) / denom, yy.reshape(-1) / denom], dim=-1)
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    return torch.stack([t, t], dim=-1)


def _kmeans_plusplus_one(
    data: torch.Tensor, k: int, seed: int, point_sq: torch.Tensor
) -> torch.Tensor:
    """
    Deterministic k-means++-style init (batch-position invariant).

    Uses furthest-point selection with seed-only tie jitter — no Generator, so
    the same image features always get the same centers regardless of B or slot.
    """
    N, D = data.shape
    device = data.device
    dtype = data.dtype
    centers = torch.empty(k, D, dtype=dtype, device=device)
    chosen = torch.zeros(N, dtype=torch.bool, device=device)

    # Tiny seed-only jitter for ties; depends on token index + seed, not batch slot.
    idx = torch.arange(N, device=device, dtype=dtype)
    jitter = 1e-6 * torch.sin((idx + 1.0) * (float(seed) * 0.6180339887 + 1.0))

    # First center: max ||x||^2 (with jitter)
    first = int((point_sq + jitter).argmax().item())
    centers[0] = data[first]
    chosen[first] = True
    closest = torch.full((N,), float("inf"), dtype=dtype, device=device)

    for c in range(1, k):
        dist = ((data - centers[c - 1]) ** 2).sum(dim=-1)
        closest = torch.minimum(closest, dist)
        score = closest + jitter
        score = score.masked_fill(chosen, float("-inf"))
        nxt = int(score.argmax().item())
        centers[c] = data[nxt]
        chosen[nxt] = True
    return centers


def _kmeans_plusplus(
    data: torch.Tensor, k: int, seed: int, point_sq: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Per-image deterministic init; identical for the same features at any batch slot."""
    B, N, D = data.shape
    if point_sq is None:
        point_sq = (data * data).sum(dim=-1)
    out = torch.empty(B, k, D, dtype=data.dtype, device=data.device)
    for b in range(B):
        out[b] = _kmeans_plusplus_one(data[b], k, seed=seed, point_sq=point_sq[b])
    return out


def _stats_from_labels(
    data: torch.Tensor, labels: torch.Tensor, k: int, point_sq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = data.dtype
    oh = F.one_hot(labels.clamp(0, k - 1), k).to(dtype=dtype)
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


def _cluster_importance(
    labels: torch.Tensor, token_imp: torch.Tensor, k: int
) -> torch.Tensor:
    """I_c = sum(attn)/sqrt(n_c). token_imp [B,N] -> [B,K]."""
    dtype = token_imp.dtype
    oh = F.one_hot(labels.clamp(0, k - 1), k).to(dtype=dtype)
    attn_sum = torch.bmm(oh.transpose(1, 2), token_imp.unsqueeze(-1)).squeeze(-1)
    counts = oh.sum(dim=1)
    return attn_sum / counts.clamp_min(1.0).sqrt()


# ---------------------------------------------------------------------------
# Gather / leaf split
# ---------------------------------------------------------------------------


def _gather_cluster_fixed(
    data: torch.Tensor,
    member: torch.Tensor,
    n_mem: torch.Tensor,
    extra: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Prefix compact into fixed [B,N,*] workspace (no .item() host sync).
    Optionally pack extra [B,N,E] the same way.
    """
    B, N, D = data.shape
    device, dtype = data.device, data.dtype
    pos = member.to(torch.long).cumsum(dim=1) - 1
    packed = torch.zeros(B, N, D, dtype=dtype, device=device)
    packed_idx = torch.full((B, N), -1, dtype=torch.long, device=device)
    valid = torch.arange(N, device=device).view(1, N) < n_mem.unsqueeze(1).clamp_min(0)

    flat_b = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    tok = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    b_m = flat_b[member]
    t_m = tok[member]
    p_m = pos[member].clamp(0, N - 1)
    if b_m.numel() > 0:
        packed[b_m, p_m] = data[b_m, t_m]
        packed_idx[b_m, p_m] = t_m
    packed = packed * valid.unsqueeze(-1).to(dtype)
    packed_idx = torch.where(valid, packed_idx, torch.full_like(packed_idx, -1))

    packed_extra = None
    if extra is not None:
        E = extra.shape[-1]
        packed_extra = torch.zeros(B, N, E, dtype=extra.dtype, device=device)
        if b_m.numel() > 0:
            packed_extra[b_m, p_m] = extra[b_m, t_m]
        packed_extra = packed_extra * valid.unsqueeze(-1).to(extra.dtype)
    return packed, packed_idx, valid, packed_extra


def _hier_merge_leaves_gram_cpu(
    leaf_counts: torch.Tensor, leaf_gram: torch.Tensor, leaf_sn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L->2 merge on CPU using only [B,L] counts + [B,L,L] Gram (tiny transfer).
    """
    device = leaf_counts.device
    lc = leaf_counts.detach().float().cpu().numpy()
    lg = leaf_gram.detach().float().cpu().numpy()
    ls = leaf_sn.detach().float().cpu().numpy()
    B, L = lc.shape
    parents = np.empty((B, L), dtype=np.int64)
    f0s = np.empty(B, dtype=np.int64)

    for b in range(B):
        counts = lc[b].copy()
        gram = lg[b].copy()
        sn = ls[b].copy()
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
                    # -||Sa*tj - Sb*ti||^2 / (ti*tj*(ti+tj))
                    # = -(tj^2*||Sa||^2 + ti^2*||Sb||^2 - 2*ti*tj*Sa·Sb) / denom
                    numer = -(tj * tj * sn[a] + ti * ti * sn[bb] - 2.0 * ti * tj * gram[a, bb])
                    denom = ti * tj * (ti + tj)
                    val = numer / denom if denom > 0 else -np.inf
                    if val > best_val:
                        best_val = val
                        best_a, best_b = int(a), int(bb)
            if best_a < 0:
                break
            # merge b -> a in gram/counts/sn
            sn[best_a] = sn[best_a] + sn[best_b] + 2.0 * gram[best_a, best_b]
            gram[best_a, :] = gram[best_a, :] + gram[best_b, :]
            gram[:, best_a] = gram[:, best_a] + gram[:, best_b]
            gram[best_a, best_a] = sn[best_a]
            gram[best_b, :] = 0.0
            gram[:, best_b] = 0.0
            sn[best_b] = 0.0
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

    return (
        torch.from_numpy(parents).to(device=device, dtype=torch.long),
        torch.from_numpy(f0s).to(device=device, dtype=torch.long),
    )


def _eight_leaf_split_packed(
    packed: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    split_num: int,
    candidate_dims: Optional[torch.Tensor] = None,
    packed_xy: Optional[torch.Tensor] = None,
    use_spatial_bit: bool = True,
) -> torch.Tensor:
    """
    Multi-bit split → leaf id → Gram-8x8 hier merge → move mask.
    If use_spatial_bit and packed_xy given: last bit is x or y threshold.
    """
    B, M, D = packed.shape
    device, dtype = packed.device, packed.dtype
    L = 1 << split_num
    n_spatial = 1 if (use_spatial_bit and packed_xy is not None and split_num >= 1) else 0
    n_feat = split_num - n_spatial

    if n_feat <= 0:
        chosen = torch.zeros(B, 0, dtype=torch.long, device=device)
    elif candidate_dims is None:
        abs_dev = (packed - mean.unsqueeze(1)).abs() * valid.unsqueeze(-1).to(dtype)
        dim_scores = abs_dev.sum(dim=1)
        order = torch.argsort(dim_scores, dim=-1, descending=True, stable=True)
        chosen = order[:, :n_feat]
    else:
        cand = (
            candidate_dims.view(1, -1).expand(B, -1)
            if candidate_dims.ndim == 1
            else candidate_dims
        )
        C = cand.shape[1]
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, C)
        m_idx = torch.arange(M, device=device).view(1, M, 1).expand(B, M, C)
        d_idx = cand.unsqueeze(1).expand(B, M, C)
        sub = packed[b_idx, m_idx, d_idx]
        mean_sub = torch.gather(mean, 1, cand)
        abs_dev = (sub - mean_sub.unsqueeze(1)).abs() * valid.unsqueeze(-1).to(dtype)
        dim_scores = abs_dev.sum(dim=1)
        order_local = torch.argsort(dim_scores, dim=-1, descending=True, stable=True)
        take = min(n_feat, C)
        chosen = torch.gather(cand, 1, order_local[:, :take])
        if take < n_feat:
            chosen = torch.cat([chosen, chosen[:, :1].expand(B, n_feat - take)], dim=1)

    signs = torch.zeros(B, M, dtype=torch.long, device=device)
    for t in range(n_feat):
        dim_t = chosen[:, t]
        vals = torch.gather(packed, 2, dim_t.view(B, 1, 1).expand(B, M, 1)).squeeze(-1)
        thr = torch.gather(mean, 1, dim_t.unsqueeze(1)).squeeze(1)
        ge = (vals >= thr.unsqueeze(1)) & valid
        signs = (signs << 1) | ge.to(torch.long)

    if n_spatial:
        # Pick axis with larger member spread; threshold at member mean.
        xy = packed_xy
        mem_f = valid.to(dtype)
        n = mem_f.sum(dim=1).clamp_min(1.0)
        mu = (xy * mem_f.unsqueeze(-1)).sum(dim=1) / n.unsqueeze(-1)
        var = ((xy - mu.unsqueeze(1)) ** 2 * mem_f.unsqueeze(-1)).sum(dim=1) / n.unsqueeze(-1)
        axis = (var[:, 1] > var[:, 0]).long()  # 0=x, 1=y
        ax = axis.view(B, 1, 1).expand(B, M, 1)
        vals = torch.gather(xy, 2, ax).squeeze(-1)
        thr = torch.gather(mu, 1, axis.unsqueeze(1)).squeeze(1)
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
    # S4: Gram on GPU, transfer only LxL
    leaf_gram = torch.bmm(leaf_sums, leaf_sums.transpose(1, 2))
    leaf_sn = torch.diagonal(leaf_gram, dim1=1, dim2=2).contiguous()
    parent, f0 = _hier_merge_leaves_gram_cpu(leaf_counts, leaf_gram, leaf_sn)
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
    token_imp: Optional[torch.Tensor] = None,
    split_imp_alpha: float = 1.0,
    xy: Optional[torch.Tensor] = None,
    use_spatial_bit: bool = True,
) -> None:
    B, N, D = data.shape
    device, dtype = data.device, data.dtype
    batch = torch.arange(B, device=device)

    sse = _sse(counts[:, :clu_num], sums[:, :clu_num], sum_x2[:, :clu_num])
    sse = sse.masked_fill(counts[:, :clu_num] <= 1, float("-inf"))
    # P4: SplitScore = SSE * (1 + alpha * I_c)
    if token_imp is not None and split_imp_alpha > 0:
        imp = _cluster_importance(labels, token_imp, clu_num)
        score = sse * (1.0 + split_imp_alpha * imp)
        score = score.masked_fill(counts[:, :clu_num] <= 1, float("-inf"))
        split_clu = score.argmax(dim=-1)
    else:
        split_clu = sse.argmax(dim=-1)

    member = labels == split_clu.unsqueeze(1)
    n_mem = member.sum(dim=-1)
    can = n_mem >= 2
    mean = sums[batch, split_clu] / n_mem.clamp_min(1).to(dtype).unsqueeze(-1)

    xy_b = xy.unsqueeze(0).expand(B, -1, -1) if xy is not None else None
    packed, packed_idx, valid, packed_xy = _gather_cluster_fixed(
        data, member, n_mem, extra=xy_b
    )
    move_local = _eight_leaf_split_packed(
        packed,
        valid,
        mean,
        split_num,
        candidate_dims=candidate_dims,
        packed_xy=packed_xy,
        use_spatial_bit=use_spatial_bit and packed_xy is not None,
    )
    move_local = move_local & can.unsqueeze(1)

    move_f = move_local.to(dtype)
    moved_count = move_f.sum(dim=1)
    moved_sum = torch.bmm(move_f.unsqueeze(1), packed).squeeze(1)
    ps_pack = torch.zeros(B, N, dtype=dtype, device=device)
    ok = packed_idx >= 0
    if bool(ok.any()):
        bb = batch.unsqueeze(1).expand_as(packed_idx)[ok]
        pp = torch.arange(N, device=device).view(1, N).expand(B, N)[ok]
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
    if bool(write.any()):
        bb = batch.unsqueeze(1).expand_as(packed_idx)[write]
        ti = packed_idx[write]
        labels[bb, ti] = new_clu


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------


def _pair_importance(
    cluster_imp: torch.Tensor,
    imp_pair: str,
    merge_imp_alpha: float,
) -> torch.Tensor:
    ii = cluster_imp.unsqueeze(2)
    jj = cluster_imp.unsqueeze(1)
    if imp_pair == "max":
        return torch.maximum(ii, jj)
    if imp_pair == "soft":
        a_max = torch.maximum(ii, jj)
        a_mean = 0.5 * (ii + jj)
        return merge_imp_alpha * a_max + (1.0 - merge_imp_alpha) * a_mean
    return 0.5 * (ii + jj)


def _merge_cost_from_gram(
    counts: torch.Tensor,
    gram: torch.Tensor,
    sn: torch.Tensor,
    cluster_imp: Optional[torch.Tensor] = None,
    merge_imp_lambda: float = 0.1,
    imp_pair: str = "mean",
    merge_imp_alpha: float = 0.7,
    merge_size_gamma: float = 0.0,
    *,
    xy_sum: Optional[torch.Tensor] = None,
    merge_spatial_gamma: float = 0.0,
    merge_cost_normalize: bool = False,
) -> torch.Tensor:
    """
    Pairwise merge score (higher = merge first).

    Default (legacy):
      Score = G_DGSM - λ A_ij + γ_size log(n_i + n_j)

    Optional scale-calibrated spatial merge (V3 ablation):
      C_ij = -G_DGSM   (Ward SSE increase, smaller better)
      J_ij = C/s_C + λ̃ A/s_A + γ D_spatial
      Score = -J
      D_spatial = ||μ^p_i - μ^p_j||_2^2 / 2,  μ^p in [0,1]^2
      s_C / s_A are per-image finite-pair medians / nonempty means.
    """
    _B, K = counts.shape
    device = counts.device
    dtype = counts.dtype
    ti = counts.unsqueeze(2)
    tj = counts.unsqueeze(1)
    numer = -(
        tj * tj * sn.unsqueeze(2) + ti * ti * sn.unsqueeze(1) - 2.0 * ti * tj * gram
    )
    denom = ti * tj * (ti + tj)
    g_dgsm = numer / denom.clamp_min(torch.finfo(dtype).tiny)

    invalid = (counts.unsqueeze(2) <= 0) | (counts.unsqueeze(1) <= 0)
    eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pair = (~invalid) & (~eye)

    a_ij = None
    if cluster_imp is not None and merge_imp_lambda > 0:
        a_ij = _pair_importance(cluster_imp, imp_pair, merge_imp_alpha)

    d_spatial = None
    if (
        merge_spatial_gamma > 0
        and xy_sum is not None
        and counts is not None
    ):
        mu = xy_sum / counts.clamp_min(1.0).unsqueeze(-1)
        diff = mu.unsqueeze(2) - mu.unsqueeze(1)
        d_spatial = (diff * diff).sum(dim=-1) * 0.5

    if merge_cost_normalize:
        # Lower-is-better J, then flip for argmax.
        c_ward = (-g_dgsm).clamp_min(0.0)
        # Per-image scale: median of valid pairwise C.
        c_flat = c_ward.masked_fill(~valid_pair, float("nan"))
        s_c = torch.nanmedian(c_flat.reshape(_B, -1), dim=-1).values
        s_c = s_c.clamp_min(torch.finfo(dtype).tiny).view(_B, 1, 1)
        j = c_ward / s_c
        if a_ij is not None:
            nonempty = counts > 0
            a_mean = (cluster_imp * nonempty.to(dtype)).sum(dim=-1) / nonempty.to(
                dtype
            ).sum(dim=-1).clamp_min(1.0)
            s_a = a_mean.clamp_min(torch.finfo(dtype).tiny).view(_B, 1, 1)
            j = j + merge_imp_lambda * (a_ij / s_a)
        if d_spatial is not None:
            j = j + merge_spatial_gamma * d_spatial
        if merge_size_gamma > 0:
            # Prefer keeping large regions → slightly lower J when merging large pairs.
            j = j - merge_size_gamma * torch.log((ti + tj).clamp_min(1.0))
        cost = -j
    else:
        cost = g_dgsm
        if merge_size_gamma > 0:
            cost = cost + merge_size_gamma * torch.log((ti + tj).clamp_min(1.0))
        if a_ij is not None:
            cost = cost - merge_imp_lambda * a_ij
        if d_spatial is not None:
            # Unnormalized soft spatial penalty (prefer V3 normalize path).
            cost = cost - merge_spatial_gamma * d_spatial

    cost = cost.masked_fill(~valid_pair, float("-inf"))
    return cost


def _gram_merge_pair(
    counts: torch.Tensor,
    sums: torch.Tensor,
    gram: torch.Tensor,
    sn: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    att_sum: Optional[torch.Tensor] = None,
) -> None:
    B, K, _D = sums.shape
    device, dtype = sums.device, sums.dtype
    batch = torch.arange(B, device=device)
    do = a != b
    if not bool(do.any()):
        return

    sa, sb = sums[batch, a], sums[batch, b]
    ca, cb = counts[batch, a], counts[batch, b]
    v = do.to(dtype).unsqueeze(-1)
    vf = do.to(dtype)
    sa_new = sa + sb * v
    ca_new = ca + cb * vf

    sums[batch, a] = torch.where(do.unsqueeze(-1), sa_new, sa)
    counts[batch, a] = torch.where(do, ca_new, ca)
    sums[batch, b] = torch.where(do.unsqueeze(-1), torch.zeros_like(sb), sb)
    counts[batch, b] = torch.where(do, torch.zeros_like(cb), cb)

    if att_sum is not None:
        aa, ab = att_sum[batch, a], att_sum[batch, b]
        att_sum[batch, a] = torch.where(do, aa + ab, aa)
        att_sum[batch, b] = torch.where(do, torch.zeros_like(ab), ab)

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


def _att_sum_from_labels(
    labels: torch.Tensor, token_imp: torch.Tensor, k: int
) -> torch.Tensor:
    """Σ a_i per cluster. [B,N] -> [B,K]."""
    dtype = token_imp.dtype
    oh = F.one_hot(labels.clamp(0, k - 1), k).to(dtype=dtype)
    return torch.bmm(oh.transpose(1, 2), token_imp.unsqueeze(-1)).squeeze(-1)


def _imp_from_att_sum(att_sum: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """I_c = (Σ a) / sqrt(n_c)."""
    return att_sum / counts.clamp_min(1.0).sqrt()


def _init_cluster_xy_stats(
    labels: torch.Tensor, xy: torch.Tensor, k: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (xy_sum, xy_sumsq) each [B,K,2]."""
    B = labels.shape[0]
    if xy.ndim == 2:
        xy_b = xy.unsqueeze(0).expand(B, -1, -1)
    else:
        xy_b = xy
    oh = F.one_hot(labels.clamp(0, k - 1), k).to(dtype=dtype)
    xy_sum = torch.bmm(oh.transpose(1, 2), xy_b)
    xy_sumsq = torch.bmm(oh.transpose(1, 2), xy_b * xy_b)
    return xy_sum, xy_sumsq


def _spatial_var_from_stats(
    xy_sum: torch.Tensor, xy_sumsq: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    n = counts.clamp_min(1.0).unsqueeze(-1)
    return (xy_sumsq / n - (xy_sum / n) ** 2).clamp_min(0.0).sum(dim=-1)


def _cluster_merge_importance(
    att_sum: Optional[torch.Tensor],
    counts: torch.Tensor,
    xy_sum: Optional[torch.Tensor],
    xy_sumsq: Optional[torch.Tensor],
    spatial_beta: float,
) -> Optional[torch.Tensor]:
    """I_c = att/√n + β · (Var(x)+Var(y))."""
    if att_sum is None and (xy_sum is None or spatial_beta <= 0):
        return None
    if att_sum is not None:
        imp = _imp_from_att_sum(att_sum, counts)
    else:
        imp = torch.zeros_like(counts)
    if xy_sum is not None and xy_sumsq is not None and spatial_beta > 0:
        imp = imp + spatial_beta * _spatial_var_from_stats(xy_sum, xy_sumsq, counts)
    return imp


def _merge_back_gpu(
    data: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    max_k: int,
    point_sq: torch.Tensor,
    token_imp: Optional[torch.Tensor] = None,
    merge_imp_lambda: float = 0.1,
    imp_pair: str = "mean",
    merge_imp_alpha: float = 0.7,
    merge_size_gamma: float = 0.0,
    xy: Optional[torch.Tensor] = None,
    merge_spatial_beta: float = 0.0,
    merge_spatial_gamma: float = 0.0,
    merge_cost_normalize: bool = False,
    debug_merge_attn: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, _D = data.shape
    device, dtype = data.device, data.dtype
    batch = torch.arange(B, device=device)

    counts, sums, sum_x2 = _stats_from_labels(data, labels, max_k, point_sq)
    gram = torch.bmm(sums, sums.transpose(1, 2))
    sn = torch.diagonal(gram, dim1=1, dim2=2).clone()
    att_sum = (
        _att_sum_from_labels(labels, token_imp, max_k) if token_imp is not None else None
    )
    need_xy_stats = xy is not None and (
        merge_spatial_beta > 0 or merge_spatial_gamma > 0
    )
    xy_sum = xy_sumsq = None
    if need_xy_stats:
        xy_sum, xy_sumsq = _init_cluster_xy_stats(labels, xy, max_k, dtype)

    parent = (
        torch.arange(max_k, device=device, dtype=torch.long).view(1, max_k).expand(B, max_k).clone()
    )
    triu = torch.triu(torch.ones(max_k, max_k, dtype=torch.bool, device=device), diagonal=1)
    eye = torch.eye(max_k, dtype=torch.bool, device=device)

    dbg_changed = 0
    dbg_total = 0

    for _ in range(max_k - k):
        cluster_imp = _cluster_merge_importance(
            att_sum, counts, xy_sum, xy_sumsq, merge_spatial_beta
        )
        cost = _merge_cost_from_gram(
            counts,
            gram,
            sn,
            cluster_imp=cluster_imp,
            merge_imp_lambda=merge_imp_lambda,
            imp_pair=imp_pair,
            merge_imp_alpha=merge_imp_alpha,
            merge_size_gamma=merge_size_gamma,
            xy_sum=xy_sum,
            merge_spatial_gamma=merge_spatial_gamma,
            merge_cost_normalize=merge_cost_normalize,
        )
        cost = cost.masked_fill(eye.unsqueeze(0), float("-inf"))
        cost = cost.masked_fill(~triu.unsqueeze(0), float("-inf"))
        nonempty = (counts > 0).sum(dim=-1)
        need = nonempty > k
        cost = cost.masked_fill(~need.view(B, 1, 1), float("-inf"))

        if debug_merge_attn and cluster_imp is not None and merge_imp_lambda > 0:
            cost_sse = _merge_cost_from_gram(
                counts,
                gram,
                sn,
                cluster_imp=None,
                merge_imp_lambda=0.0,
                merge_size_gamma=merge_size_gamma,
                xy_sum=xy_sum,
                merge_spatial_gamma=merge_spatial_gamma,
                merge_cost_normalize=merge_cost_normalize,
            )
            cost_sse = cost_sse.masked_fill(eye.unsqueeze(0), float("-inf"))
            cost_sse = cost_sse.masked_fill(~triu.unsqueeze(0), float("-inf"))
            cost_sse = cost_sse.masked_fill(~need.view(B, 1, 1), float("-inf"))
            pure = cost_sse.reshape(B, -1).argmax(dim=-1)
            with_att = cost.reshape(B, -1).argmax(dim=-1)
            dbg_changed += int((pure != with_att).sum().item())
            dbg_total += int(need.sum().item())

        flat = cost.reshape(B, -1)
        best = flat.argmax(dim=-1)
        bi = best // max_k
        bj = best % max_k
        best_val = flat[batch, best]
        valid_pair = need & torch.isfinite(best_val)
        b_m = torch.where(valid_pair, bj, bi)

        vf = valid_pair.to(dtype)
        sum_x2[batch, bi] = sum_x2[batch, bi] + sum_x2[batch, bj] * vf
        sum_x2[batch, bj] = torch.where(
            valid_pair, torch.zeros_like(sum_x2[batch, bj]), sum_x2[batch, bj]
        )
        if xy_sum is not None and xy_sumsq is not None:
            xy_sum[batch, bi] = xy_sum[batch, bi] + xy_sum[batch, bj] * vf.unsqueeze(-1)
            xy_sumsq[batch, bi] = xy_sumsq[batch, bi] + xy_sumsq[batch, bj] * vf.unsqueeze(-1)
            xy_sum[batch, bj] = torch.where(
                valid_pair.unsqueeze(-1), torch.zeros_like(xy_sum[batch, bj]), xy_sum[batch, bj]
            )
            xy_sumsq[batch, bj] = torch.where(
                valid_pair.unsqueeze(-1),
                torch.zeros_like(xy_sumsq[batch, bj]),
                xy_sumsq[batch, bj],
            )
        _gram_merge_pair(counts, sums, gram, sn, bi, b_m, att_sum=att_sum)

        match = (parent == bj.unsqueeze(1)) & valid_pair.unsqueeze(1)
        parent = torch.where(match, bi.unsqueeze(1).expand(B, max_k), parent)

    if debug_merge_attn and dbg_total > 0:
        print(
            f"[dgsm_merge] attn changed pair {dbg_changed}/{dbg_total} "
            f"({100.0 * dbg_changed / dbg_total:.1f}%) "
            f"token_imp={'set' if token_imp is not None else 'None'} "
            f"λ={merge_imp_lambda}"
        )

    for _ in range(max_k):
        parent = torch.gather(parent, 1, parent)
    labels = torch.gather(parent, 1, labels.clamp(0, max_k - 1))

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
    max_lloyd_iters: int = 64,
    do_split_merge: bool = True,
    *,
    oversplit_ratio: float = 1.25,
    global_var_dims: int = 16,
    sse_rel_tol: float = 1e-3,
    final_max_iters: Optional[int] = None,
    split_num: int = 3,
    token_importance: Optional[torch.Tensor] = None,
    use_spatial_split: bool = True,
    split_imp_alpha: float = 1.0,
    merge_imp_lambda: float = 0.1,
    merge_imp_pair: str = "mean",
    merge_imp_alpha: float = 0.7,
    merge_size_gamma: float = 0.0,
    merge_spatial_beta: float = 0.0,
    merge_spatial_gamma: float = 0.0,
    merge_cost_normalize: bool = False,
    final_refine: str = "lloyd",
    debug_merge_attn: bool = False,
    init_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DGSM-KM+ clustering.

    token_importance: optional [B, N_patches] (e.g. CLS attention). If features
    include CLS and drop_cls=True, pass importance for patches only (length N-1
    or aligned after drop). If None, importance terms are skipped.

    Merge score: G - λ A + γ log(n_i+n_j), with I_c = att/√n + β Var(x,y).

    final_refine:
      - "lloyd": final Euclidean Lloyd after merge (default baseline)
      - "none": keep merge labels; still compute soft scores from merge centers
                (ablation: does final Lloyd undo attention/spatial protection?)

    init_labels: optional [B, N_patches] to skip KMeans++/Lloyd (e.g. CD early-stop).
    """
    if final_refine not in ("lloyd", "none"):
        raise ValueError(f"final_refine must be 'lloyd' or 'none', got {final_refine!r}")

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
    final_cap = int(final_max_iters) if final_max_iters is not None else max_lloyd_iters

    # Align importance to patches
    tok_imp = None
    if token_importance is not None:
        ti = token_importance.to(device=data.device, dtype=data.dtype)
        if ti.ndim == 1:
            ti = ti.unsqueeze(0).expand(B, -1)
        if ti.shape[-1] == N + 1:
            ti = ti[:, 1:]
        if ti.shape[-1] != N:
            raise ValueError(f"token_importance length {ti.shape[-1]} != N={N}")
        # normalize per image for scale-stable λ
        ti = ti / ti.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(ti.dtype).tiny)
        tok_imp = ti

    need_xy = (
        use_spatial_split
        or (merge_spatial_beta > 0)
        or (merge_spatial_gamma > 0)
    )
    xy = _patch_xy(N, data.device, data.dtype) if need_xy else None

    if init_labels is not None:
        labels = init_labels.to(device=data.device, dtype=torch.long)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.shape != (B, N):
            raise ValueError(
                f"init_labels shape {tuple(labels.shape)} != expected {(B, N)}"
            )
        labels = labels.clamp(0, k - 1).contiguous()
        centers, _counts, _sums = _onehot_centroid_update(
            data, labels, k, point_sq=point_sq
        )
    else:
        centers = _kmeans_plusplus(data, k, seed, point_sq=point_sq)
        labels, centers, _ = _lloyd(
            data,
            centers,
            max_lloyd_iters,
            point_sq,
            stop="sse_tol",
            sse_rel_tol=sse_rel_tol,
            return_full_dists=False,
            seed=seed,
        )

    max_k = max(k, min(N, int(round(oversplit_ratio * k))))
    dists: Optional[torch.Tensor] = None
    if do_split_merge and max_k > k:
        sn = min(int(split_num), max(1, D))
        cand = _per_image_top_variance_dims(data, n_dims=min(global_var_dims, D))
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
                split_num=sn,
                point_sq=point_sq,
                candidate_dims=cand,
                token_imp=tok_imp,
                split_imp_alpha=split_imp_alpha,
                xy=xy,
                use_spatial_bit=use_spatial_split,
            )
            clu_num += 1

        labels, counts, sums, _ = _merge_back_gpu(
            data,
            labels,
            k=k,
            max_k=max_k,
            point_sq=point_sq,
            token_imp=tok_imp,
            merge_imp_lambda=merge_imp_lambda,
            imp_pair=merge_imp_pair,
            merge_imp_alpha=merge_imp_alpha,
            merge_size_gamma=merge_size_gamma,
            xy=xy,
            merge_spatial_beta=merge_spatial_beta,
            merge_spatial_gamma=merge_spatial_gamma,
            merge_cost_normalize=merge_cost_normalize,
            debug_merge_attn=debug_merge_attn,
        )
        nonempty = counts > 0
        centers = sums / counts.clamp_min(1.0).unsqueeze(-1)
        if bool((~nonempty).any()):
            fill = _kmeans_plusplus(data, k, seed + 1, point_sq=point_sq)
            centers = torch.where(nonempty.unsqueeze(-1), centers, fill)

        if final_refine == "none":
            # Ablation: keep merge labels. Soft scores still from merge centers.
            # Do NOT reassign via argmin — that would undo merge protection.
            dists = _squared_dists(data, centers, point_sq)
        else:
            labels, centers, dists = _lloyd(
                data,
                centers,
                final_cap,
                point_sq,
                stop="sse_tol",
                sse_rel_tol=sse_rel_tol,
                return_full_dists=True,
                seed=seed + 17,
            )
    else:
        dists = _squared_dists(data, centers, point_sq)
        if init_labels is None:
            labels = dists.argmin(dim=-1)
        # else: keep CD / provided labels; soft from current centers

    soft = (-dists).to(torch.float32)
    return soft, labels


__all__ = ["batch_dgsm_kmeans", "_HAS_FLASH_KMEANS", "_patch_xy"]
