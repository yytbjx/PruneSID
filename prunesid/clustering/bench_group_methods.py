"""Benchmark Stage-1 grouping: dgsm (lossless CDKM) vs dgsm_kmeans (GPU) vs psca."""

from __future__ import annotations

import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup=2, repeat=10):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync()
    return (time.perf_counter() - t0) / repeat


def main():
    from prunesid.clustering.dgsm_cdkm import batch_dgsm_cdkm, warmup_dgsm_cdkm
    from prunesid.clustering.dgsm_kmeans import batch_dgsm_kmeans

    def batch_pca(features, min_components=32):
        standard_features = torch.sigmoid(features.to(torch.float32)).transpose(2, 1)[:, :, 1:]
        _U, _S, V = torch.pca_lowrank(standard_features, q=min_components)
        V = torch.abs(V)
        belong = torch.argmax(V, dim=-1)
        return V, belong

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    print("warmup dgsm numba...")
    warmup_dgsm_cdkm(64, 128, 8)

    shapes = [
        (1, 577, 1024, 16),
        (8, 577, 1024, 16),
    ]
    for B, T, D, K in shapes:
        feat = torch.randn(B, T, D, device=device)
        print(f"\n=== B={B} T={T-1} D={D} K={K} ===")

        t_psca = _bench(lambda: batch_pca(feat, min_components=K))
        t_lloyd = _bench(
            lambda: batch_dgsm_kmeans(
                feat, min_components=K, drop_cls=True, do_split_merge=False
            )
        )
        t_km = _bench(lambda: batch_dgsm_kmeans(feat, min_components=K, drop_cls=True))
        t_dgsm = _bench(lambda: batch_dgsm_cdkm(feat, min_components=K, drop_cls=True))

        soft_km, bel_km = batch_dgsm_kmeans(feat, min_components=K, drop_cls=True)
        soft_dg, bel_dg = batch_dgsm_cdkm(feat, min_components=K, drop_cls=True)

        print(f"psca              {t_psca*1000:8.1f} ms")
        print(f"lloyd_gpu_only    {t_lloyd*1000:8.1f} ms   (flash exact Lloyd, no S/M)")
        print(f"dgsm_kmeans       {t_km*1000:8.1f} ms   (flash Lloyd + gather 8-leaf)")
        from prunesid.clustering.dgsm_kmeans import _HAS_FLASH_KMEANS
        print(f"  flash-kmeans pkg: {_HAS_FLASH_KMEANS}")
        print(f"dgsm (CDKM)       {t_dgsm*1000:8.1f} ms   (lossless CPU CDKM)")
        print(f"  uniq kmeans={bel_km[0].unique().numel()}  uniq dgsm={bel_dg[0].unique().numel()}")
        if t_dgsm > 0:
            print(f"speedup kmeans vs dgsm : {t_dgsm/max(t_km,1e-9):.2f}x")
            print(f"speedup lloyd  vs dgsm : {t_dgsm/max(t_lloyd,1e-9):.2f}x")
            print(f"overhead kmeans/lloyd  : {t_km/max(t_lloyd,1e-9):.2f}x")


if __name__ == "__main__":
    main()
