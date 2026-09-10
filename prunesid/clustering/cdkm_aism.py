"""
GPU/batched CDKM-AISM for PruneSID token grouping.

The implementation follows the objective and acceptance logic of the supplied
CDKM-AISM C++ code while accelerating only algebraically equivalent work:

* cluster sufficient statistics are cached (count, sum, squared sum-norm);
* all merge losses are evaluated by batched matrix products;
* base-split and merge-split candidates are solved in parallel;
* each individual split keeps the original sequential point-update order;
* AISM accepts a move only when J_new > J_old + eps;
* no iteration is removed from the original 10-pass initial CD / 5 AISM loops,
  and post-AISM CD still runs to convergence.

The optimized objective is
    J = sum_c ||sum_{x in c} x||^2 / |c|,
which is equivalent (for fixed data) to minimizing k-means SSE.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


InitMode = Literal["kmeans++", "dgsm"]


def _squared_l2(data: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance using GEMM. data [B,N,D], centers [B,K,D]."""
    data_sq = (data * data).sum(dim=-1, keepdim=True)
    center_sq = (centers * centers).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(data, centers.transpose(1, 2))
    return data_sq + center_sq - 2.0 * cross


def _kmeans_plusplus_with_indices(
    data: torch.Tensor, k: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched k-means++ returning both centers and source-token indices."""
    B, N, D = data.shape
    device = data.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    batch = torch.arange(B, device=device)
    indices = torch.empty(B, k, dtype=torch.long, device=device)
    centers = torch.empty(B, k, D, dtype=data.dtype, device=device)

    first = torch.randint(0, N, (B,), generator=generator, device=device)
    indices[:, 0] = first
    centers[:, 0] = data[batch, first]

    closest = torch.full((B, N), float("inf"), dtype=data.dtype, device=device)
    chosen = torch.zeros((B, N), dtype=torch.bool, device=device)
    chosen[batch, first] = True

    for c in range(1, k):
        last = centers[:, c - 1 : c]
        dist_sq = ((data - last) ** 2).sum(dim=-1)
        closest = torch.minimum(closest, dist_sq)
        closest = closest.masked_fill(chosen, 0.0)

        total = closest.sum(dim=-1)
        valid = total > 0
        probs = closest / total.clamp_min(torch.finfo(data.dtype).tiny).unsqueeze(-1)

        # torch.multinomial requires non-zero row mass. Build the degenerate
        # fallback unconditionally so the k-means++ hot loop never synchronizes
        # CUDA just to test whether a fallback row exists.
        fallback = (~chosen).to(data.dtype)
        fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = torch.where(valid.unsqueeze(-1), probs, fallback)

        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        indices[:, c] = nxt
        centers[:, c] = data[batch, nxt]
        chosen[batch, nxt] = True

    return centers, indices


def _labels_from_initial_centers(
    data: torch.Tensor, centers: torch.Tensor, center_indices: torch.Tensor
) -> torch.Tensor:
    """C++-equivalent initial assignment, forcing each initial center to its own cluster."""
    B, _, _ = data.shape
    K = centers.shape[1]
    dists = _squared_l2(data, centers)
    batch = torch.arange(B, device=data.device)
    for c in range(K):
        idx = center_indices[:, c]
        dists[batch, idx, :] = float("inf")
        dists[batch, idx, c] = 0.0
    return dists.argmin(dim=-1)


def _cluster_stats(
    data: torch.Tensor, labels: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return counts [B,K], sums [B,K,D], and ||sums||^2 [B,K]."""
    B, N, D = data.shape
    counts = torch.zeros(B, k, dtype=data.dtype, device=data.device)
    counts.scatter_add_(1, labels, torch.ones(B, N, dtype=data.dtype, device=data.device))

    sums = torch.zeros(B, k, D, dtype=data.dtype, device=data.device)
    sums.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, D), data)
    sum_norm2 = (sums * sums).sum(dim=-1)
    return counts, sums, sum_norm2


def _coordinate_descent(
    data: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    max_passes: Optional[int],
    active_batches: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CDKM point-wise coordinate descent with the same sequential token order as C++.

    Different batch elements are processed in parallel, but point i+1 observes all
    accepted updates from point i, preserving the source algorithm's search path.
    """
    B, N, _ = data.shape
    device = data.device
    batch = torch.arange(B, device=device)
    point_sq = (data * data).sum(dim=-1)
    counts, sums, sum_norm2 = _cluster_stats(data, labels, k)

    if active_batches is None:
        active = torch.ones(B, dtype=torch.bool, device=device)
    else:
        active = active_batches.clone()

    pass_id = 0
    while bool(active.any()):
        if max_passes is not None and pass_id >= max_passes:
            break

        moved_in_pass = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(N):
            x = data[:, i, :]
            x_sq = point_sq[:, i]
            p = labels[:, i]

            dots = torch.bmm(sums, x.unsqueeze(-1)).squeeze(-1)
            safe_counts = counts.clamp_min(1.0)

            # Objective increment for adding x to every cluster.
            add_score = (sum_norm2 + 2.0 * dots + x_sq.unsqueeze(-1)) / (counts + 1.0)
            add_score = add_score - sum_norm2 / safe_counts
            add_score = add_score.masked_fill(counts <= 0, float("-inf"))

            p_count = counts[batch, p]
            p_norm = sum_norm2[batch, p]
            p_dot = dots[batch, p]
            m2 = p_norm - 2.0 * p_dot + x_sq
            m3 = torch.where(
                (p_count == 1.0) | (m2 == 0.0),
                torch.zeros_like(m2),
                m2 / (p_count - 1.0).clamp_min(1.0),
            )
            stay_score = p_norm / p_count.clamp_min(1.0) - m3

            scores = add_score
            scores[batch, p] = stay_score
            q = scores.argmax(dim=-1)

            move = active & (q != p)
            move_f = move.to(data.dtype)

            # Static masked updates avoid a CUDA synchronization/nonzero allocation
            # for every token while preserving the exact sequential CD dependency.
            p_hot = F.one_hot(p, k).to(data.dtype)
            q_hot = F.one_hot(q, k).to(data.dtype)
            delta = (q_hot - p_hot) * move_f.unsqueeze(-1)

            p_dot = dots[batch, p]
            q_dot = dots[batch, q]
            norm_delta = (
                p_hot * (move_f * (-2.0 * p_dot + x_sq)).unsqueeze(-1)
                + q_hot * (move_f * (2.0 * q_dot + x_sq)).unsqueeze(-1)
            )

            sums += delta.unsqueeze(-1) * x.unsqueeze(1)
            counts += delta
            sum_norm2 += norm_delta
            labels[:, i] = torch.where(move, q, p)
            moved_in_pass |= move

        active = active & moved_in_pass
        pass_id += 1

    return labels


def _objective_from_stats(counts: torch.Tensor, sum_norm2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    per_cluster = torch.where(
        counts > 0,
        sum_norm2 / counts.clamp_min(1.0),
        torch.zeros_like(sum_norm2),
    )
    return per_cluster, per_cluster.sum(dim=-1)


def _merge_candidates(
    counts: torch.Tensor,
    sums: torch.Tensor,
    sum_norm2: torch.Tensor,
    per_cluster_obj: torch.Tensor,
    candidate_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of the C++ top-cluster + minimum merge-cost search."""
    B, K = counts.shape
    device = counts.device

    # Stable descending sort matches findTopkf2Indices' strict-'>' tie behavior.
    top = torch.argsort(per_cluster_obj, dim=-1, descending=True, stable=True)[:, :candidate_count]

    gram = torch.bmm(sums, sums.transpose(1, 2))
    merged_norm = sum_norm2.unsqueeze(2) + sum_norm2.unsqueeze(1) + 2.0 * gram
    merged_count = counts.unsqueeze(2) + counts.unsqueeze(1)
    merged_obj_all = merged_norm / merged_count.clamp_min(1.0)
    merge_cost_all = per_cluster_obj.unsqueeze(2) + per_cluster_obj.unsqueeze(1) - merged_obj_all

    invalid = (counts.unsqueeze(2) <= 0) | (counts.unsqueeze(1) <= 0)
    eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
    merge_cost_all = merge_cost_all.masked_fill(invalid | eye, float("inf"))

    row_idx = top.unsqueeze(-1).expand(-1, -1, K)
    candidate_costs = merge_cost_all.gather(1, row_idx)
    partner = candidate_costs.argmin(dim=-1)
    min_cost = candidate_costs.gather(2, partner.unsqueeze(-1)).squeeze(-1)

    merged_obj_rows = merged_obj_all.gather(1, row_idx)
    merge_obj = merged_obj_rows.gather(2, partner.unsqueeze(-1)).squeeze(-1)
    valid = torch.isfinite(min_cost)
    return top, partner, valid, min_cost, merge_obj


def _split_candidates_parallel(
    data: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_sums: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve many splitCachedCandidate calls in parallel without changing any
    individual candidate's sequential coordinate-descent order.

    Returns:
        split_object [B,C]
        split_valid  [B,C]
        split_token_label [B,C,N], values 0/1 for candidate members
    """
    B, C, N = candidate_masks.shape
    device = data.device
    dtype = data.dtype
    batch = torch.arange(B, device=device).view(B, 1).expand(B, C)
    point_sq = (data * data).sum(dim=-1)

    counts_total = candidate_masks.sum(dim=-1)
    valid_candidate = counts_total > 1

    # Stable boolean sort compacts members to the front while retaining original
    # token order exactly (the C++ pointIndex order).
    point_order = torch.argsort((~candidate_masks).to(torch.int8), dim=-1, stable=True)
    first_idx = point_order[:, :, 0]
    first_x = data[batch, first_idx]

    n1 = torch.where(valid_candidate, torch.ones_like(counts_total, dtype=dtype), torch.ones_like(counts_total, dtype=dtype))
    n0 = (counts_total.to(dtype) - 1.0).clamp_min(1.0)
    s1 = first_x.clone()
    s0 = candidate_sums - first_x
    norm0 = (s0 * s0).sum(dim=-1)
    norm1 = (s1 * s1).sum(dim=-1)

    # Labels are indexed by position in point_order, exactly like splitLabel[ii].
    split_labels = torch.zeros(B, C, N, dtype=torch.long, device=device)
    split_labels[:, :, 0] = valid_candidate.long()

    unfinished = valid_candidate.clone()
    max_count = int(counts_total.max().item()) if counts_total.numel() else 0

    while bool(unfinished.any()):
        moved_in_pass = torch.zeros(B, C, dtype=torch.bool, device=device)

        for ii in range(max_count):
            pos_valid = unfinished & (ii < counts_total)

            idx = point_order[:, :, ii]
            x = data[batch, idx]
            x_sq = point_sq[batch, idx]
            p = split_labels[:, :, ii].clone()

            dot0 = (x * s0).sum(dim=-1)
            dot1 = (x * s1).sum(dim=-1)

            # Score for j=0.
            m2_0 = norm0 - 2.0 * dot0 + x_sq
            rem0 = torch.where(
                (n0 == 1.0) | (m2_0 == 0.0),
                torch.zeros_like(m2_0),
                m2_0 / (n0 - 1.0).clamp_min(1.0),
            )
            score0_leave = norm0 / n0.clamp_min(1.0) - rem0
            score0_join = (norm0 + 2.0 * dot0 + x_sq) / (n0 + 1.0) - norm0 / n0.clamp_min(1.0)
            score0 = torch.where(p == 0, score0_leave, score0_join)

            # Score for j=1.
            m2_1 = norm1 - 2.0 * dot1 + x_sq
            rem1 = torch.where(
                (n1 == 1.0) | (m2_1 == 0.0),
                torch.zeros_like(m2_1),
                m2_1 / (n1 - 1.0).clamp_min(1.0),
            )
            score1_leave = norm1 / n1.clamp_min(1.0) - rem1
            score1_join = (norm1 + 2.0 * dot1 + x_sq) / (n1 + 1.0) - norm1 / n1.clamp_min(1.0)
            score1 = torch.where(p == 1, score1_leave, score1_join)

            # C++ selects q=1 only on strict improvement over cluster 0.
            q = (score1 > score0).long()
            move = pos_valid & (q != p)
            split_labels[:, :, ii] = torch.where(pos_valid, q, split_labels[:, :, ii])

            move01 = move & (p == 0)
            move10 = move & (p == 1)
            f01 = move01.to(dtype)
            f10 = move10.to(dtype)

            # Use the same incremental norm formulas as the C++ implementation.
            norm0 = norm0 + f01 * (-2.0 * dot0 + x_sq) + f10 * (2.0 * dot0 + x_sq)
            norm1 = norm1 + f10 * (-2.0 * dot1 + x_sq) + f01 * (2.0 * dot1 + x_sq)
            n0 = n0 - f01 + f10
            n1 = n1 - f10 + f01
            s0 = s0 - f01.unsqueeze(-1) * x + f10.unsqueeze(-1) * x
            s1 = s1 - f10.unsqueeze(-1) * x + f01.unsqueeze(-1) * x
            moved_in_pass |= move

        # A candidate with a zero-move full pass is permanently converged.
        unfinished &= moved_in_pass

    split_valid = valid_candidate & (n0 > 0) & (n1 > 0)
    split_object = torch.where(
        split_valid,
        norm0 / n0.clamp_min(1.0) + norm1 / n1.clamp_min(1.0),
        torch.full_like(norm0, float("-inf")),
    )

    # Convert position-wise labels back to token-wise labels. N is a sentinel for
    # padded positions, avoiding collisions with any real token index.
    valid_pos = torch.arange(N, device=device).view(1, 1, N) < counts_total.unsqueeze(-1)
    scatter_idx = torch.where(valid_pos, point_order, torch.full_like(point_order, N))
    token_labels_pad = torch.zeros(B, C, N + 1, dtype=torch.long, device=device)
    token_labels_pad.scatter_(2, scatter_idx, torch.where(valid_pos, split_labels, torch.zeros_like(split_labels)))
    return split_object, split_valid, token_labels_pad[:, :, :N]


def _aism_refine(
    data: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    max_big_loops: int = 5,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Faithful, batched AISM merge+split neighborhood refinement."""
    if k < 2:
        return labels

    B, N, _ = data.shape
    device = data.device
    candidate_count = min(k // 2 + 1, k)
    cluster_ids = torch.arange(k, device=device).view(1, k, 1)
    outer_active = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_big_loops):
        if not bool(outer_active.any()):
            break

        # All cached candidates in this big loop are built from this partition.
        base_labels = labels.clone()
        counts, sums, sum_norm2 = _cluster_stats(data, base_labels, k)
        per_obj, current_obj = _objective_from_stats(counts, sum_norm2)

        h1, h2, merge_valid, _, merge_obj = _merge_candidates(
            counts, sums, sum_norm2, per_obj, candidate_count
        )
        merge_valid &= outer_active.unsqueeze(-1)

        base_masks = base_labels.unsqueeze(1) == cluster_ids  # [B,K,N]
        h1_masks = base_masks.gather(1, h1.unsqueeze(-1).expand(-1, -1, N))
        h2_masks = base_masks.gather(1, h2.unsqueeze(-1).expand(-1, -1, N))
        merge_masks = (h1_masks | h2_masks) & merge_valid.unsqueeze(-1)

        merge_sums = sums.gather(1, h1.unsqueeze(-1).expand(-1, -1, sums.shape[-1]))
        merge_sums = merge_sums + sums.gather(1, h2.unsqueeze(-1).expand(-1, -1, sums.shape[-1]))

        all_masks = torch.cat([base_masks, merge_masks], dim=1)
        all_sums = torch.cat([sums, merge_sums], dim=1)
        split_obj, split_valid, split_token_labels = _split_candidates_parallel(data, all_masks, all_sums)
        base_split_obj = split_obj[:, :k]
        base_split_valid = split_valid[:, :k]
        base_split_token = split_token_labels[:, :k]
        merge_split_obj = split_obj[:, k:]
        merge_split_valid = split_valid[:, k:] & merge_valid
        merge_split_token = split_token_labels[:, k:]

        # A split of (h1 union h2) can occasionally reproduce the exact same two
        # partitions with only their numeric cluster IDs exchanged. In exact
        # arithmetic its gain is zero, so the C++ strict-improvement test rejects
        # it. Explicitly mask these no-op permutations to prevent float32 roundoff
        # from turning an exact zero into a tiny positive gain.
        same_partition = (
            ((~h1_masks) | (merge_split_token == 1)).all(dim=-1)
            & ((~h2_masks) | (merge_split_token == 0)).all(dim=-1)
        )
        swapped_partition = (
            ((~h1_masks) | (merge_split_token == 0)).all(dim=-1)
            & ((~h2_masks) | (merge_split_token == 1)).all(dim=-1)
        )
        merge_split_valid &= ~(same_partition | swapped_partition)

        used = torch.zeros(B, k, dtype=torch.bool, device=device)
        accepted_in_big_loop = torch.zeros(B, dtype=torch.bool, device=device)
        running_obj = current_obj.clone()
        split_axis = torch.arange(k, device=device).view(1, 1, k)

        h1_obj = per_obj.gather(1, h1)
        h2_obj = per_obj.gather(1, h2)

        for _step in range(candidate_count):
            h1_used = used.gather(1, h1)
            h2_used = used.gather(1, h2)
            common_valid = merge_valid & (~h1_used) & (~h2_used)

            split_used = used.unsqueeze(1).expand(-1, candidate_count, -1)
            not_h1 = split_axis != h1.unsqueeze(-1)
            special = split_axis == h2.unsqueeze(-1)

            regular_obj = (
                running_obj.view(B, 1, 1)
                - h1_obj.unsqueeze(-1)
                - h2_obj.unsqueeze(-1)
                - per_obj.unsqueeze(1)
                + merge_obj.unsqueeze(-1)
                + base_split_obj.unsqueeze(1)
            )
            special_obj = (
                running_obj.view(B, 1, 1)
                - h1_obj.unsqueeze(-1)
                - h2_obj.unsqueeze(-1)
                + merge_split_obj.unsqueeze(-1)
            )
            candidate_obj = torch.where(special, special_obj, regular_obj)

            valid_split = torch.where(
                special,
                merge_split_valid.unsqueeze(-1),
                base_split_valid.unsqueeze(1),
            )
            valid = common_valid.unsqueeze(-1) & (~split_used) & not_h1 & valid_split
            better = valid & (candidate_obj > running_obj.view(B, 1, 1) + eps)
            scored = candidate_obj.masked_fill(~better, float("-inf"))

            best_val, best_flat = scored.reshape(B, -1).max(dim=-1)
            accepted = torch.isfinite(best_val) & outer_active
            if not bool(accepted.any()):
                break

            best_mi = best_flat // k
            best_split = best_flat % k
            sel_h1 = h1.gather(1, best_mi.unsqueeze(-1)).squeeze(-1)
            sel_h2 = h2.gather(1, best_mi.unsqueeze(-1)).squeeze(-1)
            is_special = best_split == sel_h2

            # Merge bestMerge1 into bestMerge2 first, exactly as the C++ code.
            merged_labels = torch.where(
                accepted.unsqueeze(-1) & (labels == sel_h1.unsqueeze(-1)),
                sel_h2.unsqueeze(-1),
                labels,
            )

            merge_member = merge_masks.gather(
                1, best_mi.view(B, 1, 1).expand(-1, 1, N)
            ).squeeze(1)
            merge_bits = merge_split_token.gather(
                1, best_mi.view(B, 1, 1).expand(-1, 1, N)
            ).squeeze(1)
            special_new = torch.where(
                merge_bits == 0,
                sel_h2.unsqueeze(-1),
                sel_h1.unsqueeze(-1),
            )

            base_member = base_masks.gather(
                1, best_split.view(B, 1, 1).expand(-1, 1, N)
            ).squeeze(1)
            base_bits = base_split_token.gather(
                1, best_split.view(B, 1, 1).expand(-1, 1, N)
            ).squeeze(1)
            regular_new = torch.where(
                base_bits == 0,
                best_split.unsqueeze(-1),
                sel_h1.unsqueeze(-1),
            )

            special_apply = accepted.unsqueeze(-1) & is_special.unsqueeze(-1) & merge_member
            regular_apply = accepted.unsqueeze(-1) & (~is_special).unsqueeze(-1) & base_member
            labels = torch.where(special_apply, special_new, merged_labels)
            labels = torch.where(regular_apply, regular_new, labels)

            active_col = accepted.unsqueeze(-1)
            used |= F.one_hot(sel_h1, k).bool() & active_col
            used |= F.one_hot(sel_h2, k).bool() & active_col
            used |= F.one_hot(best_split, k).bool() & active_col
            running_obj = torch.where(accepted, best_val, running_obj)
            accepted_in_big_loop |= accepted

        if not bool(accepted_in_big_loop.any()):
            break

        # Source code re-runs coordinate descent to full convergence whenever the
        # big loop accepted at least one split/merge operation.
        labels = _coordinate_descent(
            data,
            labels,
            k,
            max_passes=None,
            active_batches=accepted_in_big_loop,
        )
        outer_active &= accepted_in_big_loop

    return labels


def _finalize(
    data: torch.Tensor, labels: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the C++ final center recomputation + nearest-center reassignment."""
    counts, sums, _ = _cluster_stats(data, labels, k)
    centers = sums / counts.clamp_min(1.0).unsqueeze(-1)

    # Empty clusters should not occur for the faithful CDKM-AISM path. If a DGSM
    # initializer ever creates one, keep it from winning nearest-center assignment.
    dists = _squared_l2(data, centers)
    empty = counts <= 0
    dists = dists.masked_fill(empty.unsqueeze(1), float("inf"))

    final_labels = dists.argmin(dim=-1)
    # Keep raw -distance^2 scores. Qwen's GPU NMS below handles signed scores
    # explicitly, so no monotonic-but-numerically-lossy score remapping is needed.
    soft_scores = (-dists).to(torch.float32)
    return soft_scores, final_labels.long()



try:
    from .cdkm_aism_numba import run_single as _NUMBA_AISM_SINGLE
except Exception:
    _NUMBA_AISM_SINGLE = None


def _numba_refine_batch(
    data: torch.Tensor,
    k: int,
    seed: int,
    init_mode: InitMode,
    initial_cd_passes: int,
    max_big_loops: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Numba/C++-style backend. Sequential decisions are FP64; split candidates run in parallel."""
    if _NUMBA_AISM_SINGLE is None:
        raise RuntimeError("Numba backend requested but numba is not available")

    import numpy as np

    B, _, _ = data.shape
    device = data.device

    init_indices_t = None
    init_labels_t = None
    fallback_to_kpp = torch.zeros(B, dtype=torch.bool, device=device)
    if init_mode == "kmeans++":
        _, init_indices_t = _kmeans_plusplus_with_indices(data, k, seed)
        fallback_to_kpp.fill_(True)
    elif init_mode == "dgsm":
        from .dgsm_cdkm import _batch_dgsm_torch

        _, init_labels_t = _batch_dgsm_torch(
            data, k=k, seed=seed, max_iters=8, do_split_merge=True
        )
        dgsm_counts, _, _ = _cluster_stats(data, init_labels_t, k)
        fallback_to_kpp = (dgsm_counts <= 0).any(dim=-1)
        if bool(fallback_to_kpp.any()):
            _, init_indices_t = _kmeans_plusplus_with_indices(data, k, seed)
    else:
        raise ValueError(f"Unknown init_mode={init_mode!r}")

    # One compact transfer per image. The source C++ algorithm uses double, so
    # accumulators/decisions stay FP64 without changing any split/merge criterion.
    data_np = data.detach().cpu().numpy().astype(np.float64, copy=False)
    if init_indices_t is not None:
        init_indices_np = init_indices_t.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        init_indices_np = np.zeros((B, k), dtype=np.int64)
    fallback_np = fallback_to_kpp.detach().cpu().numpy()
    if init_labels_t is not None:
        init_labels_np = init_labels_t.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        init_labels_np = np.zeros((B, data.shape[1]), dtype=np.int64)

    out_labels = []
    out_scores = []
    for b in range(B):
        labels_np, dists_np = _NUMBA_AISM_SINGLE(
            data_np[b],
            k,
            init_labels_np[b],
            init_indices_np[b],
            1 if fallback_np[b] else 0,
            initial_cd_passes,
            max_big_loops,
            eps,
        )
        out_labels.append(torch.from_numpy(labels_np))
        out_scores.append(torch.from_numpy((-dists_np).astype(np.float32, copy=False)))

    labels = torch.stack(out_labels, dim=0).to(device=device, non_blocking=True)
    scores = torch.stack(out_scores, dim=0).to(device=device, non_blocking=True)
    return scores, labels.long()

def batch_cdkm_aism(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    init_mode: InitMode = "kmeans++",
    initial_cd_passes: int = 10,
    max_big_loops: int = 5,
    eps: float = 1e-9,
    backend: Literal["auto", "numba", "torch"] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PruneSID-compatible batched CDKM-AISM.

    init_mode="kmeans++": pure CDKM-AISM initialized by k-means++ centers.
    init_mode="dgsm":     current fast DGSM result followed by unchanged CDKM-AISM
                           coordinate descent + AISM refinement.

    Returns:
        soft_scores [B,T_patch,K] = -squared distance to final centers
        belong_components [B,T_patch]
    """
    x = torch.sigmoid(features.to(torch.float32))
    if drop_cls:
        if x.shape[1] < 2:
            raise ValueError("features must include CLS + patches when drop_cls=True")
        data = x[:, 1:, :].contiguous()
    else:
        data = x.contiguous()

    B, N, _ = data.shape
    k = max(1, min(int(min_components), N))

    if backend == "auto":
        # CDKM-AISM is intrinsically sequential within each candidate. Numba keeps
        # those decisions in one compiled FP64 CPU kernel and parallelizes only
        # independent split candidates, avoiding thousands of tiny CUDA launches.
        backend = "numba" if _NUMBA_AISM_SINGLE is not None else "torch"
    if backend == "numba":
        return _numba_refine_batch(
            data, k, seed, init_mode, initial_cd_passes, max_big_loops, eps
        )
    if backend != "torch":
        raise ValueError(f"Unknown backend={backend!r}; expected 'auto', 'numba', or 'torch'")

    if init_mode == "dgsm":
        # Reuse the project's fast, device-resident DGSM only as initialization;
        # AISM's objective/acceptance logic below is unchanged.
        from .dgsm_cdkm import _batch_dgsm_torch

        _, labels = _batch_dgsm_torch(
            data,
            k=k,
            seed=seed,
            max_iters=8,
            do_split_merge=True,
        )

        # AISM assumes all K clusters are non-empty. Degenerate DGSM batches are
        # reinitialized by the same k-means++ path rather than changing AISM rules.
        counts, _, _ = _cluster_stats(data, labels, k)
        bad = (counts <= 0).any(dim=-1)
        if bool(bad.any()):
            centers, center_indices = _kmeans_plusplus_with_indices(data, k, seed)
            fallback = _labels_from_initial_centers(data, centers, center_indices)
            labels = torch.where(bad.unsqueeze(-1), fallback, labels)
    elif init_mode == "kmeans++":
        centers, center_indices = _kmeans_plusplus_with_indices(data, k, seed)
        labels = _labels_from_initial_centers(data, centers, center_indices)
    else:
        raise ValueError(f"Unknown init_mode={init_mode!r}; expected 'kmeans++' or 'dgsm'")

    # Source CDKM-AISM performs at most 10 initial coordinate-descent sweeps.
    labels = _coordinate_descent(data, labels, k, max_passes=initial_cd_passes)
    labels = _aism_refine(data, labels, k, max_big_loops=max_big_loops, eps=eps)
    return _finalize(data, labels, k)


__all__ = ["batch_cdkm_aism"]
