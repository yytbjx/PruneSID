"""
DGSM-CD-Att: Early-stop CDKM structure refine + SSE split + attention merge.

Pipeline (method B):
  KMeans++ → CD (no split/merge) until ΔSSE/SSE_prev < α and iter≥min
  → SSE Split → Attention Merge → Stop  (no final Lloyd / CD)

Role separation:
  - CDKM early-stop: better semantic cluster structure than Lloyd
  - Attention merge: decide which evidence must survive token budget
  - Explicitly NOT full CDKM (no split↔CD↔merge↔CD loop)

Compare:
  A) dgsm_km_att  : Lloyd → SSE split → Att merge → (optional) final Lloyd
  B) dgsm_cd_att  : this module
  C) dgsm         : full CDKM upper bound
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from .dgsm_cdkm import batch_cdk_refine_early_stop
from .dgsm_kmeans import batch_dgsm_kmeans


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def batch_dgsm_cd_att(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    *,
    token_importance: Optional[torch.Tensor] = None,
    # CD early-stop (structure stage)
    cd_sse_rel_tol: float = 0.05,
    cd_min_iters: int = 2,
    cd_max_iters: int = 10,
    # Attention merge (selection stage) — same defaults as dgsm_km_att V0
    merge_imp_lambda: float = 0.05,
    merge_imp_alpha: float = 0.7,
    merge_size_gamma: float = 0.05,
    merge_spatial_beta: float = 0.3,
    merge_spatial_gamma: float = 0.0,
    merge_cost_normalize: bool = False,
    oversplit_ratio: float = 1.25,
    global_var_dims: int = 16,
    split_num: int = 3,
    debug_merge_attn: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Early-stop CD refine + SSE split + attention merge; no final Lloyd/CD.

    Env overrides:
      DGSM_CD_SSE_REL_TOL, DGSM_CD_MIN_ITERS, DGSM_CD_MAX_ITERS
      DGSM_DEBUG_MERGE
    """
    env_tol = _env_float("DGSM_CD_SSE_REL_TOL")
    if env_tol is not None:
        cd_sse_rel_tol = env_tol
    env_min = _env_int("DGSM_CD_MIN_ITERS")
    if env_min is not None:
        cd_min_iters = env_min
    env_max = _env_int("DGSM_CD_MAX_ITERS")
    if env_max is not None:
        cd_max_iters = env_max
    if debug_merge_attn is None:
        debug_merge_attn = os.environ.get("DGSM_DEBUG_MERGE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    # Stage 1: CD-only structure (no split/merge inside CD).
    init_labels = batch_cdk_refine_early_stop(
        features,
        min_components=min_components,
        seed=seed,
        drop_cls=drop_cls,
        sse_rel_tol=cd_sse_rel_tol,
        min_cd_iters=cd_min_iters,
        max_cd_iters=cd_max_iters,
    )

    # Stage 2: SSE split + attention merge; stop (final_refine=none).
    return batch_dgsm_kmeans(
        features,
        min_components=min_components,
        seed=seed,
        drop_cls=drop_cls,
        do_split_merge=True,
        oversplit_ratio=oversplit_ratio,
        global_var_dims=global_var_dims,
        split_num=split_num,
        token_importance=token_importance,
        use_spatial_split=False,
        split_imp_alpha=0.0,
        merge_imp_lambda=merge_imp_lambda,
        merge_imp_pair="soft",
        merge_imp_alpha=merge_imp_alpha,
        merge_size_gamma=merge_size_gamma,
        merge_spatial_beta=merge_spatial_beta,
        merge_spatial_gamma=merge_spatial_gamma,
        merge_cost_normalize=merge_cost_normalize,
        final_refine="none",
        debug_merge_attn=bool(debug_merge_attn),
        init_labels=init_labels,
    )


__all__ = ["batch_dgsm_cd_att"]
