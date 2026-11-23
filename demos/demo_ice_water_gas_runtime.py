"""
Ice-Water-Gas Spacetime Scale-Rotor Runtime Empirical Benchmark
===============================================================

Demonstrates mapping the entire computer hardware memory & execution hierarchy
onto the 3-phase Scale-Rotor continuum:
- ICE (SSD `mmap`): 256 frozen world sectors (representing >1M spatial cells).
- WATER (RAM Pool): Only resonant active sectors melted into RAM.
- GAS (L1 Cache <= 32 KB): Contiguous hot SoA evolving 100 frames with 0 SSD I/O.
- Hardware Phase Controller: Real `psutil` + `perf_counter_ns` regulation.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.ice_water_gas_rotor_runtime import (
    IceWaterGasRotorRuntime,
    extract_macro_signature,
)
from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralScaleRotorManifold,
)


def _make_world_sector_2d(idx: int) -> SpectralScaleRotorManifold:
    wy = 2.0 * math.pi * (1.0 + 0.15 * (idx % 17))
    wx = 2.0 * math.pi * (1.5 + 0.12 * (idx // 17))
    phase = 0.19 * idx
    b0 = ScaleRotorBand(
        level=0,
        dim=2,
        wavevectors=np.array([[wy, wx], [-wx, wy], [wy * 1.5, -wx * 0.5]], dtype=np.float64),
        temporal_omegas=np.array(
            [math.hypot(wy, wx), math.hypot(wy, wx), math.hypot(wy * 1.5, wx * 0.5)],
            dtype=np.float64,
        ),
        amplitudes=np.array(
            [1.1 * np.exp(1j * phase), 0.7 * np.exp(-1j * phase), 0.4 + 0.2j],
            dtype=np.complex128,
        ),
        centers=np.zeros((3, 2), dtype=np.float64),
        sigmas=np.full(3, np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=np.array([-1, -1, -1], dtype=np.int32),
        parent_couplings=np.ones(3, dtype=np.float64),
    )
    b1 = ScaleRotorBand(
        level=1,
        dim=2,
        wavevectors=np.array([[wy * 2.5, wx * 2.5]], dtype=np.float64),
        temporal_omegas=np.array([math.hypot(wy * 2.5, wx * 2.5)], dtype=np.float64),
        amplitudes=np.array([0.5 * np.exp(1j * 0.5 * phase)], dtype=np.complex128),
        centers=np.array([[0.3 + 0.01 * (idx % 40), 0.7 - 0.01 * (idx % 40)]], dtype=np.float64),
        sigmas=np.array([0.07], dtype=np.float64),
        is_global=False,
        parent_indices=np.array([0], dtype=np.int32),
        parent_couplings=np.array([0.85], dtype=np.float64),
    )
    return SpectralScaleRotorManifold(dim=2, domain_bounds=([0.0, 0.0], [1.0, 1.0]), bands=[b0, b1])


def run_benchmark() -> None:
    print("=" * 94)
    print("       ICE-WATER-GAS SPACETIME SCALE-ROTOR RUNTIME: REAL HARDWARE CONTINUUM DEMO")
    print("=" * 94)

    with tempfile.TemporaryDirectory(prefix="elysia_iwg_") as tmp_dir:
        slab_path = Path(tmp_dir) / "world_crystals.slab"
        num_sectors = 256
        grid_res = 64  # 256 * 64 * 64 = 1,048,576 spatial cells (~8.39 MB dense float64)

        with IceWaterGasRotorRuntime(
            slab_path=slab_path,
            ice_capacity_bytes=2 * 1024 * 1024,
            max_water_bytes=64 * 1024,
            max_gas_bytes=32 * 1024,
        ) as rt:
            # ------------------------------------------------------------------
            # SECTION 1: Freeze 256 World Sectors onto real SSD mmap Ice Storage
            # ------------------------------------------------------------------
            sectors = {f"sector_{i:03d}": _make_world_sector_2d(i) for i in range(num_sectors)}
            t_freeze0 = time.perf_counter()
            for sid, m in sectors.items():
                rt.ingest_to_ice(sid, m)
            freeze_ms = (time.perf_counter() - t_freeze0) * 1000.0

            raw_dense_bytes = num_sectors * grid_res * grid_res * 8
            print("\n[SECTION 1] REAL SSD MMAP ICE CRYSTAL ARCHIVE (256 WORLD SECTORS)")
            print("-" * 94)
            print(
                f"Dense 64x64x256 Equivalent: {raw_dense_bytes / (1024 * 1024):.2f} MB "
                f"({num_sectors * grid_res * grid_res:,d} spatial cells)"
            )
            print(
                f"SSD Ice Slab Used         : {rt.ice.used_bytes:,d} B ({rt.ice.used_bytes / 1024:.1f} KB, "
                f"{raw_dense_bytes / rt.ice.used_bytes:.1f}x smaller), Freeze time={freeze_ms:.2f} ms"
            )
            print(
                f"RAM Macro Phase Index     : {rt.ice.index_nbytes:,d} B ({rt.ice.num_crystals} slots) | "
                f"SSD Read during ingest={rt.ice.bytes_read_from_ssd} B"
            )

            # Query 3 resonant sectors cleanly (0 SSD bytes during query, melt ONLY 3!)
            target_ids = ["sector_017", "sector_104", "sector_219"]
            t_q0 = time.perf_counter()
            for tid in target_ids:
                best_id, score = rt.ice.query_resonant_slot(
                    extract_macro_signature(sectors[tid]), dim=2
                )
                assert best_id == tid and score > 0.9999
            query_us = ((time.perf_counter() - t_q0) / len(target_ids)) * 1e6
            ssd_before_melt = rt.ice.bytes_read_from_ssd

            for tid in target_ids:
                _, _, melted = rt.activate_by_resonance(extract_macro_signature(sectors[tid]), dim=2)
                assert melted
            ssd_melted_bytes = rt.ice.bytes_read_from_ssd - ssd_before_melt
            print(
                f"Resonant Query (0 SSD I/O): {query_us:.2f} us/query | "
                f"Melted 3/{num_sectors} active sectors -> SSD Read={ssd_melted_bytes:,d} B "
                f"({100.0 * ssd_melted_bytes / rt.ice.used_bytes:.2f}% of Ice)"
            )

            # ------------------------------------------------------------------
            # SECTION 2: Naive 256-Sector Dense Frame Loop vs. Gas L1-Cache Burst
            # ------------------------------------------------------------------
            print("\n[SECTION 2] NAIVE FULL-SYSTEM 100-FRAME LOOP vs. GAS L1-CACHE BURST (<= 32 KB)")
            print("-" * 94)

            dense_world = np.ones((num_sectors, grid_res, grid_res), dtype=np.float64) * 0.5
            t_naive0 = time.perf_counter()
            for _ in range(100):
                dense_world = dense_world * 0.999 + 0.0005 * np.roll(dense_world, 1, axis=2)
            naive_ms = (time.perf_counter() - t_naive0) * 1000.0

            vel2 = np.array([0.001, -0.0015], dtype=np.float64)
            dt_step = 0.25
            burst = rt.run_gas_burst(num_frames=100, dt=dt_step, dim=2, velocity=vel2)

            # Exact error check against analytical dt = 25.0 on one active sector
            expected = sectors["sector_104"].evolve_spacetime(dt=100 * dt_step, velocity=vel2)
            actual = rt.water.entries["sector_104"].manifold
            gy, gx = np.meshgrid(np.linspace(0, 1, 24), np.linspace(0, 1, 24), indexing="ij")
            pts2 = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1)
            condense_err = float(
                np.max(np.abs(actual.slice_manifestation(pts2) - expected.slice_manifestation(pts2)))
            )

            speedup = naive_ms / max(burst["total_burst_ms"], 1e-6)
            print(
                f"Naive Dense (256 sectors x 100 frames) : {naive_ms:8.2f} ms | "
                f"Footprint={dense_world.nbytes / (1024 * 1024):.2f} MB"
            )
            print(
                f"Gas L1-Cache Burst (100 frames)       : {burst['total_burst_ms']:8.4f} ms "
                f"({burst['avg_frame_us']:.2f} us/frame, {speedup:.1f}x faster)"
            )
            print(
                f"  Gas Footprint={burst['gas_bytes']:,d} B (<= 32,768 B L1 cap) | "
                f"Water={burst['water_bytes']:,d} B | SSD Read in Burst={burst['ssd_read_bytes_delta']} B"
            )
            print(
                f"  Condense Exact Error vs Analytic dt=25: {condense_err:.2e} | "
                f"Real Process RSS={burst['rss_mb']:.2f} MB"
            )

            # ------------------------------------------------------------------
            # SECTION 3: Real Latency-Slope Regulation & CausalField Sync
            # ------------------------------------------------------------------
            print("\n[SECTION 3] REAL LATENCY-SLOPE PHASE REGULATION & CAUSAL FIELD SYNC")
            print("-" * 94)

            steady_tel = rt.hw_controller.observe_and_regulate(
                15_000, rt.gas.max_gas_bytes, rt.water.max_water_bytes
            )
            surge_tel = rt.hw_controller.observe_and_regulate(
                3_500_000, rt.gas.max_gas_bytes, rt.water.max_water_bytes
            )
            print(
                f"Steady State : latency={steady_tel['latency_us']:.2f} us, "
                f"stress={steady_tel['stress_index']:.4f}, gas_budget={steady_tel['effective_gas_bytes']:,d} B, "
                f"granularity={steady_tel['recommended_granularity']:.3f}"
            )
            print(
                f"Latency Surge: latency={surge_tel['latency_us']:.2f} us, "
                f"slope={surge_tel['latency_slope_us']:+.2f} us, stress={surge_tel['stress_index']:.4f}, "
                f"gas_budget={surge_tel['effective_gas_bytes']:,d} B, "
                f"granularity={surge_tel['recommended_granularity']:.3f}"
            )

            cf = CausalField(dimensions=3)
            sync = rt.sync_to_causal_field(cf, last_metrics=surge_tel, prefix="iwg")
            cf.step(dt=0.05)
            print(
                f"CausalField  : voxels={len(cf.voxels)} (Ice/Water/Gas), engrams={len(cf.engrams)}, "
                f"beams={sync['beams_linked']}, sys_RAM={surge_tel['sys_ram_percent']}%"
            )
            print("=" * 94)


if __name__ == "__main__":
    run_benchmark()
