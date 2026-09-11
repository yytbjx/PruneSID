"""
DGSM-KM-Att (DGSM-KM-SA): SSE-guided split + attention-guided merge.

Design (v2 merge score):
  Split  : argmax SSE  (where more capacity is needed)
  Merge  : Score = G_DGSM - λ · A_ij + γ · log(n_i + n_j)
           I_c = (Σ a_i) / √n_c + β · (Var(x) + Var(y))
           A_ij = α · max(I_i, I_j) + (1-α) · mean(I_i, I_j)   (soft)
           → protect high-attention detail AND spatially spread regions
             (Landmark / Scene); size term prefers keeping large regions.

Ablation knobs (passed through to shared Exact backend):
  final_refine="lloyd"|"none"
      "none" keeps merge labels (no final Euclidean Lloyd reassignment).
  merge_spatial_gamma + merge_cost_normalize
      V3 scale-calibrated pairwise spatial merge penalty (off by default).

Implementation:
  Thin preset over ``batch_dgsm_kmeans`` (shared Exact GPU backend).

Batch invariance (same as dgsm_kmeans after the B1/B8 fix):
  - Deterministic furthest-point KMeans++ (seed-only tie jitter; no shared RNG)
  - Per-sample Lloyd early-stop (SSE rel < 0.1%)
  - Deterministic empty-cluster fill (max ||x||^2)
  - Per-image variance dims; per-image attention normalization
  → same image features (+ same token_importance) ⇒ same labels for any B / slot.

Does NOT use spatial split bits or SSE×importance split (kept pure SSE).
Spatial coords are used only in merge importance (β term) unless V3 γ is on.

Requires token_importance (e.g. CLS→patch attention already in the vision forward).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from .dgsm_kmeans import batch_dgsm_kmeans


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def batch_dgsm_km_att(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_lloyd_iters: int = 64,
    do_split_merge: bool = True,
    *,
    token_importance: Optional[torch.Tensor] = None,
    merge_imp_lambda: float = 0.05,
    merge_imp_alpha: float = 0.7,
    merge_size_gamma: float = 0.05,
    merge_spatial_beta: float = 0.3,
    merge_spatial_gamma: float = 0.0,
    merge_cost_normalize: bool = False,
    final_refine: str = "lloyd",
    oversplit_ratio: float = 1.25,
    global_var_dims: int = 16,
    sse_rel_tol: float = 1e-3,
    final_max_iters: Optional[int] = None,
    split_num: int = 3,
    debug_merge_attn: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSE split + attention-aware merge (batch-invariant via shared Exact backend).

    Defaults (DGSM-KM-SA v2 / V0 baseline):
      split_imp_alpha=0, use_spatial_split=False,
      merge_imp_pair=\"soft\", merge_imp_alpha=0.7,
      merge_imp_lambda=0.05, merge_size_gamma=0.05, merge_spatial_beta=0.3,
      final_refine=\"lloyd\"

    Env overrides (for server ablations without CLI rebuild):
      DGSM_FINAL_REFINE=lloyd|none
      DGSM_MERGE_SPATIAL_GAMMA=float
      DGSM_MERGE_COST_NORMALIZE=0|1
      DGSM_DEBUG_MERGE=0|1

    token_importance: [B, N_patches] (or [B, N+1] with CLS). Recommended:
    CLS attention sum already available in PruneSID LLaVA forward (no extra forward).
    If None, falls back to pure DGSM merge (λ unused; β/γ still apply if set).
    """
    env_refine = os.environ.get("DGSM_FINAL_REFINE")
    if env_refine:
        final_refine = env_refine.strip().lower()
    env_gamma = _env_float("DGSM_MERGE_SPATIAL_GAMMA")
    if env_gamma is not None:
        merge_spatial_gamma = env_gamma
    if os.environ.get("DGSM_MERGE_COST_NORMALIZE") is not None:
        merge_cost_normalize = _env_bool("DGSM_MERGE_COST_NORMALIZE", False)
    if debug_merge_attn is None:
        debug_merge_attn = _env_bool("DGSM_DEBUG_MERGE", False)

    return batch_dgsm_kmeans(
        features,
        min_components=min_components,
        seed=seed,
        drop_cls=drop_cls,
        max_lloyd_iters=max_lloyd_iters,
        do_split_merge=do_split_merge,
        oversplit_ratio=oversplit_ratio,
        global_var_dims=global_var_dims,
        sse_rel_tol=sse_rel_tol,
        final_max_iters=final_max_iters,
        split_num=split_num,
        token_importance=token_importance,
        # SA v2 recipe:
        use_spatial_split=False,
        split_imp_alpha=0.0,
        merge_imp_lambda=merge_imp_lambda,
        merge_imp_pair="soft",
        merge_imp_alpha=merge_imp_alpha,
        merge_size_gamma=merge_size_gamma,
        merge_spatial_beta=merge_spatial_beta,
        merge_spatial_gamma=merge_spatial_gamma,
        merge_cost_normalize=merge_cost_normalize,
        final_refine=final_refine,
        debug_merge_attn=bool(debug_merge_attn),
    )


__all__ = ["batch_dgsm_km_att"]
