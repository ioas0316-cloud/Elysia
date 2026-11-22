r"""
test_agamotto_causal_rotor_engram_engine.py

Zero-mock empirical proof that relational causal crystallization eliminates multi-second
sensory recomputation bottlenecks permanently:
  - Session 1: Performs one-time raw `.mp4` transduction & 5-sense Apple Rotor synthesis,
    compressing `>30 MB` of raw video arrays into a compact `<25 KB` relational engram file.
  - Session 2 (Brand-New Engine, ZERO MP4/FFT/LLM access): Cold-reloads only the JSON
    engram, runs 100 forward/reverse Eye-of-Agamotto scrubs (`theta: 0 <-> 1`), verifies
    sub-0.5ms per-query latency (`>1,000x` speedup), exact reversible S^3 round-trip,
    5-sense coupling, and continuous Distance-Scale spectral modulation (`0.15m <-> 8.0m`).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.sensory.agamotto_causal_rotor_engram_engine import (
    DEFAULT_ENGRAM_STORE_PATH,
    AgamottoCausalRotorEngramEngine,
)

REAL_MP4_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
DASHBOARD_OUT = Path(
    r"c:\Elysia\docs\assets\elysia_agamotto_rotor_engram_dashboard.png"
)


def test_first_pass_crystallization_then_cold_reload_o1_agamotto_scrub() -> None:
    # ------------------------------------------------------------------
    # Session 1: One-Time First-Pass Sensory Crystallization
    # ------------------------------------------------------------------
    eng1 = AgamottoCausalRotorEngramEngine()
    mp4_engram, key_frames = eng1.crystallize_real_mp4_to_engram(
        REAL_MP4_PATH, sample_frames=24, engram_id="real_mp4_flower"
    )
    apple_engram = eng1.crystallize_embodied_apple_engram("embodied_5sense_apple")
    assert "real_mp4_flower" in eng1.causal_field.voxels
    assert "embodied_5sense_apple" in eng1.causal_field.voxels
    assert len(eng1.causal_field.beams) >= 1

    saved_path = eng1.save_engrams(DEFAULT_ENGRAM_STORE_PATH)
    disk_bytes = int(saved_path.stat().st_size)
    assert disk_bytes < 25000
    assert mp4_engram.raw_source_bytes > 30_000_000
    compression_ratio = mp4_engram.raw_source_bytes / disk_bytes
    assert compression_ratio > 1000.0

    # ------------------------------------------------------------------
    # Session 2: Brand-New Instance Cold-Reloads ONLY Engram JSON
    # (No `.mp4` read, No FFT, No PyTorch, No LLM!)
    # ------------------------------------------------------------------
    eng2 = AgamottoCausalRotorEngramEngine()
    t_load0 = time.perf_counter()
    loaded_cnt = eng2.load_engrams(DEFAULT_ENGRAM_STORE_PATH)
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    assert loaded_cnt == 2
    assert load_ms < 50.0

    # Benchmark 100 consecutive O(1) scrubs + Eye-of-Agamotto reverse origin traces
    thetas = np.linspace(0.0, 1.0, 50)
    t_q0 = time.perf_counter()
    for th in thetas:
        eng2.scrub_engram_phase("real_mp4_flower", float(th), distance_m=1.0)
        eng2.reverse_trace_causal_origin("embodied_5sense_apple", float(th))
    mean_recall_ms = ((time.perf_counter() - t_q0) * 1000.0) / 100.0
    speedup = mp4_engram.first_pass_crystallization_ms / max(mean_recall_ms, 1e-6)

    print(
        f"\n[1st-Pass MP4 Transduction] {mp4_engram.first_pass_crystallization_ms:.2f} ms | "
        f"raw={mp4_engram.raw_source_bytes/1e6:.2f} MB"
    )
    print(
        f"[Cold-Reload Engram File ] {disk_bytes} bytes ({compression_ratio:,.0f}x compression) | "
        f"load={load_ms:.2f} ms"
    )
    print(
        f"[O(1) Agamotto Recall    ] {mean_recall_ms:.4f} ms/query ({speedup:,.0f}x faster than 1st pass)"
    )

    assert mean_recall_ms < 0.5
    assert speedup > 1000.0

    # Eye-of-Agamotto Exact Reversibility Verification
    rev_mp4 = eng2.reverse_trace_causal_origin("real_mp4_flower", observed_theta=1.0)
    rev_apple = eng2.reverse_trace_causal_origin("embodied_5sense_apple", observed_theta=1.0)
    assert rev_mp4.reversible_round_trip_error < 1e-5
    assert rev_apple.reversible_round_trip_error < 1e-5
    assert rev_mp4.causal_delta_by_channel["red_petal_chroma"] > 0.10
    assert rev_mp4.causal_delta_by_channel["bloom_area"] > 0.05

    # Embodied 5-Sense Apple Rotor Coherence (Fresh theta=0 -> Rotten theta=1 -> Rewound theta=0)
    fresh = eng2.scrub_engram_phase("embodied_5sense_apple", theta=0.0, distance_m=1.0)
    rotten = eng2.scrub_engram_phase("embodied_5sense_apple", theta=1.0, distance_m=1.0)
    rewound = eng2.scrub_engram_phase(
        "embodied_5sense_apple", theta=rev_apple.traced_origin_theta, distance_m=1.0
    )
    assert fresh.modality_channels["vision_red_chroma"] > 0.85
    assert fresh.modality_channels["touch_firmness_resistance"] > 0.85
    assert fresh.modality_channels["hearing_bite_crispness"] > 0.85
    assert rotten.modality_channels["vision_red_chroma"] < 0.25
    assert rotten.modality_channels["touch_firmness_resistance"] < 0.25
    assert rotten.modality_channels["hearing_bite_crispness"] < 0.25
    assert rewound.modality_channels["touch_firmness_resistance"] == fresh.modality_channels["touch_firmness_resistance"]
    assert fresh.order_negentropy > rotten.order_negentropy

    # Distance-Scale Perception Net over the SAME Fresh Apple Engram
    near_st = eng2.scrub_engram_phase("embodied_5sense_apple", theta=0.0, distance_m=0.15)
    far_st = eng2.scrub_engram_phase("embodied_5sense_apple", theta=0.0, distance_m=8.00)
    near_short_energy = float(np.sum(near_st.wavelength_spectrum_8d[5:]))
    far_short_energy = float(np.sum(far_st.wavelength_spectrum_8d[5:]))
    far_long_energy = float(np.sum(far_st.wavelength_spectrum_8d[:3]))
    print(
        f"[Distance Scale 0.15m vs 8.0m] short_micro: {near_short_energy:.3f} vs {far_short_energy:.3f} | "
        f"far_long_macro: {far_long_energy:.3f}"
    )
    assert near_short_energy > far_short_energy * 2.0
    assert far_long_energy > 0.90

    # Render Dashboard
    out_png = eng1.render_agamotto_dashboard(
        mp4_engram, apple_engram, key_frames, mean_recall_ms, disk_bytes, DASHBOARD_OUT
    )
    assert out_png.exists() and out_png.stat().st_size > 25000
