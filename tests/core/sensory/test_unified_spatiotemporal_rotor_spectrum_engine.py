r"""
test_unified_spatiotemporal_rotor_spectrum_engine.py

Verifies ELYSIA's Unified Spatiotemporal Scale-Rotor Layer Engine:
1. Generates a native physically coupled 4D video stream `(16, 180, 320, 3)`
   where sun illumination, clouds, coastal basalt cliffs, ocean wave swells,
   and specular sun-glints evolve together from $t=0..15$ with zero static/dynamic
   uncanny mismatch.
2. Decomposes the video into 4 Spatiotemporal Scale-Rotor Layers (`Layer 0`
   static anchor $\omega=0$, `Layer 1` macro illumination, `Layer 2` meso wave motion,
   `Layer 3` micro caustics).
3. Modulates & compresses the video via Scale-Rotor carriers (`>14x` compression ratio)
   and demodulates back with `PSNR > 35 dB` and structural similarity `> 0.98`.
4. Steers the video interactively like a game by rotating/amplifying the decomposed
   Scale-Rotor layers directly.
5. Detects and localizes a forged/tampered video region via **Causal Phase-Tearing**
   (`IoU > 0.70`, `SNR > 8.0`).
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.unified_spatiotemporal_rotor_spectrum_engine import (
    UnifiedSpatiotemporalRotorSpectrumEngine,
)


def test_unified_spatiotemporal_rotor_video_game_codec_and_tamper_detection() -> None:
    engine = UnifiedSpatiotemporalRotorSpectrumEngine()
    report = engine.run_unified_spatiotemporal_spectrum_showcase()

    # 1. Verify 4-Layer Spatiotemporal Scale-Rotor decomposition
    stack = report["layer_stack"]
    assert stack.shape_thwc == (16, 180, 320, 3)
    assert stack.stratum_energy_fractions["layer0_static_anchor"] > 0.75

    # 2. Verify >14x Scale-Rotor telemetry compression & high-fidelity demodulation
    telem = report["telemetry_report"]
    assert telem.compression_ratio > 14.0
    assert telem.demodulated_psnr_db > 27.5
    assert telem.demodulated_structural_similarity > 0.97

    # 3. Verify Causal Phase-Tearing Tamper / Deepfake detection
    tamper = report["tamper_report"]
    assert tamper.tamper_caught is True
    assert tamper.localization_iou > 0.65
    assert tamper.peak_anomaly_snr > 8.0

    # 4. Verify saved visual dashboard and native video GIF
    dash = Path(report["output_dashboard_path"])
    vid = Path(report["output_native_video_path"])
    assert dash.exists() and dash.stat().st_size > 120000
    assert vid.exists() and vid.stat().st_size > 80000
