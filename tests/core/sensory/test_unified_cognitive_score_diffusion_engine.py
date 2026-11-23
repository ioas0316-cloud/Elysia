"""
test_unified_cognitive_score_diffusion_engine.py

Physics & cross-modal integration tests for UnifiedCognitiveScoreDiffusionEngine
strictly on real photographs + 1D acoustic waveforms + Korean/English prompts
(zero external VLMs, zero LLMs, zero neural diffusion checkpoints).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.sensory.unified_cognitive_score_diffusion_engine import (
    UnifiedCognitiveScoreDiffusionEngine,
    synthesize_physical_audio_probe,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
YOSEMITE_PATH = ASSETS_DIR / "genuine_optical_4k_yosemite.jpg"
FACE_PATH = ASSETS_DIR / "real_human_face_hd.jpg"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to load image: {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_reverse_diffusion_convergence_and_cross_modal_steering() -> None:
    if not YOSEMITE_PATH.exists() or not FACE_PATH.exists():
        pytest.skip("Missing real visual assets")

    yose_rgb = _load_rgb(YOSEMITE_PATH)
    engine = UnifiedCognitiveScoreDiffusionEngine()
    yose_fact = engine.gen.factorize(yose_rgb, source_name="yosemite_4k", max_width=220)

    # 1. Cold Dawn + 1500Hz Crisp Wave on Yosemite Mold
    dawn_wave = synthesize_physical_audio_probe(
        freq_hz=1500.0, sharpness_q=12.0, decay_rate=6.0
    )
    dawn_attr = engine.encode_cross_modal_attractor(
        language_prompt="차가운 새벽 안개와 선명한 고산 봉우리",
        audio_wave=dawn_wave,
    )
    c_dawn = engine.crystallize_via_score_diffusion(
        structure_fact=yose_fact,
        attractor=dawn_attr,
        creation_name="dawn_test",
        n_steps=12,
        seed=11,
    )

    # 2. Warm Sunset + 220Hz Resonant Wave on the EXACT SAME Yosemite Mold
    sunset_wave = synthesize_physical_audio_probe(
        freq_hz=220.0, sharpness_q=4.0, decay_rate=2.4
    )
    sunset_attr = engine.encode_cross_modal_attractor(
        language_prompt="따뜻한 노을빛 붉은 숲과 부드러운 울림",
        audio_wave=sunset_wave,
    )
    c_sunset = engine.crystallize_via_score_diffusion(
        structure_fact=yose_fact,
        attractor=sunset_attr,
        creation_name="sunset_test",
        n_steps=12,
        seed=22,
    )

    # Reverse diffusion entropy decreases strictly and structure converges cleanly
    for res in (c_dawn, c_sunset):
        entropies = [st.entropy_level for st in res.trajectory]
        for i in range(1, len(entropies)):
            assert entropies[i] < entropies[i - 1], (
                f"Entropy failed strict monotonic descent at step {i}: {entropies}"
            )
        assert res.final_structural_corr >= 0.80
        assert res.trajectory[-1].structural_corr_to_mold > res.trajectory[0].structural_corr_to_mold

    # Cross-modal steering contrast on the EXACT SAME visual mold
    valid_d = c_dawn.reobserved_obs.l0.valid_mask
    valid_s = c_sunset.reobserved_obs.l0.valid_mask
    dawn_rg = float(np.mean(c_dawn.reobserved_obs.l1.rg_opponent[valid_d]))
    sunset_rg = float(np.mean(c_sunset.reobserved_obs.l1.rg_opponent[valid_s]))
    dawn_by = float(np.mean(c_dawn.reobserved_obs.l1.by_opponent[valid_d]))
    sunset_by = float(np.mean(c_sunset.reobserved_obs.l1.by_opponent[valid_s]))

    assert sunset_rg > dawn_rg, f"Expected Sunset warmer RG than Dawn ({sunset_rg:.3f} vs {dawn_rg:.3f})"
    assert dawn_by > sunset_by, f"Expected Dawn cooler BY than Sunset ({dawn_by:.3f} vs {sunset_by:.3f})"
    assert (
        c_dawn.synthesized_audio_obs.peak_frequency_hz
        > c_sunset.synthesized_audio_obs.peak_frequency_hz
    ), "Expected Dawn closed-loop audio response Hz higher than Sunset"

    # Unified CausalField(6D) binding across Vision, Language, Audio, Creation
    expected_voxels = {
        yose_fact.voxel_id,
        dawn_attr.language_voxel_id,
        dawn_attr.audio_voxel_id,
        c_dawn.voxel_id,
        sunset_attr.language_voxel_id,
        sunset_attr.audio_voxel_id,
        c_sunset.voxel_id,
    }
    assert expected_voxels.issubset(set(engine.causal_field.voxels.keys()))
    dawn_parents = {
        b.source_id for b in engine.causal_field.beams if b.target_id == c_dawn.voxel_id
    }
    assert yose_fact.voxel_id in dawn_parents
    assert dawn_attr.language_voxel_id in dawn_parents
    assert dawn_attr.audio_voxel_id in dawn_parents
