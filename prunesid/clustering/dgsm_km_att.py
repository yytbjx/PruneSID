"""
DGSM-KM-Att (DGSM-KM-SA): SSE-guided split + attention-guided merge.

Design:
  Split  : argmax SSE  (where more capacity is needed)
  Merge  : G' = G_DGSM - λ · A_ij
           I_c = (Σ a_i) / √n_c
           A_ij = (I_i + I_j) / 2
           → low-importance / redundant clusters merge first;
             high-attention fine detail (OCR / count / position) is protected.

Implementation:
  Thin preset over ``batch_dgsm_kmeans`` (shared Exact GPU backend).

Batch invariance (same as dgsm_kmeans after the B1/B8 fix):
  - Deterministic furthest-point KMeans++ (seed-only tie jitter; no shared RNG)
  - Per-sample Lloyd early-stop (SSE rel < 0.1%)
  - Deterministic empty-cluster fill (max ||x||^2)
  - Per-image variance dims; per-image attention normalization
  → same image features (+ same token_importance) ⇒ same labels for any B / slot.

Does NOT use spatial split bits or SSE×importance split (kept pure SSE).

Requires token_importance (e.g. CLS→patch attention already in the vision forward).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .dgsm_kmeans import batch_dgsm_kmeans


def batch_dgsm_km_att(
    features: torch.Tensor,
    min_components: int = 32,
    seed: int = 0,
    drop_cls: bool = True,
    max_lloyd_iters: int = 64,
    do_split_merge: bool = True,
    *,
    token_importance: Optional[torch.Tensor] = None,
    merge_imp_lambda: float = 0.15,
    oversplit_ratio: float = 1.25,
    global_var_dims: int = 16,
    sse_rel_tol: float = 1e-3,
    final_max_iters: Optional[int] = None,
    split_num: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSE split + attention-aware merge (batch-invariant via shared Exact backend).

    token_importance: [B, N_patches] (or [B, N+1] with CLS). Recommended:
    CLS attention sum already available in PruneSID LLaVA forward.
    If None, falls back to pure DGSM merge (λ effectively unused).
    """
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
        # SA recipe (no spatial / no attention-on-split):
        use_spatial_split=False,
        split_imp_alpha=0.0,
        merge_imp_lambda=merge_imp_lambda,
        merge_imp_pair="mean",
    )


__all__ = ["batch_dgsm_km_att"]
