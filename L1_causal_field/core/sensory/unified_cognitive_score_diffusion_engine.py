"""
unified_cognitive_score_diffusion_engine.py

ELYSIA Unified Multi-Scale Score-Flow Diffusion & Cross-Modal Cognitive Integration
====================================================================================
Implements native reverse-diffusion generative physics coupled with Vision (Ocular),
Acoustic (Mechano-Acoustic Resonance), and Conceptual Language Axes inside CausalField(6D)
— zero external VLMs, zero LLMs, zero pretrained neural diffusion weights:

1. Cross-Modal Attractor Encoding (Language + Acoustic Wave -> Physical Axis Forces):
   - AcousticResonanceOrgan extracts peak Hz, centroid, Q-factor, and decay rate from raw PCM audio.
   - Conceptual language descriptors project onto continuous temperature (RG/BY), illumination (A),
     and spatial frequency (micro-texture vs macro-smooth) axes inside CausalField(6D).
2. Coarse-to-Fine Reverse Score-Flow Diffusion (Entropy -> Gradient Flux -> Crystalline Order):
   - Starts at high-entropy Gaussian retinal noise (t=1.0).
   - Integrates coarse-to-fine DoG multi-scale score forces alongside Perona-Malik
     anisotropic edge-preserving diffusion and annealed Langevin fluctuations down to t=0.0.
3. Closed-Loop Multi-Modal Closure (Vision -> Acoustic Response & Grounded Self-Observation):
   - Re-observes the crystallized image with OcularScaleOrgan, synthesizes an emergent physical
     acoustic resonance waveform from its measured optical state, re-listens via AcousticResonanceOrgan,
     and records all modality voxels + ConnectivityBeams in CausalField(6D).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.active_ocular_mechanics_engine import _extract_optical_descriptor
from core.sensory.mechano_acoustic_organ import AcousticObservation, AcousticResonanceOrgan
from core.sensory.ocular_generative_spectrum_engine import (
    OcularCognitiveFactorization,
    OcularGenerativeSpectrumEngine,
    _compute_persistence_correlation,
    _recompose_telescoping_bands,
)
from core.sensory.ocular_scale_organ import OcularScaleOrgan, OpticalSceneObservation


@dataclass
class CrossModalAttractor:
    """Unified physical steering potential encoded from conceptual language + acoustic physics."""

    language_prompt: str
    audio_obs: AcousticObservation
    luma_shift: float
    rg_shift: float
    by_shift: float
    chroma_rot_deg: float
    fine_energy_gain: float
    macro_smooth_gain: float
    language_voxel_id: str
    audio_voxel_id: str
    combined_6d: np.ndarray


@dataclass
class DiffusionTrajectoryStep:
    """Single step telemetry along the reverse score-flow trajectory (t: 1.0 -> 0.0)."""

    step_idx: int
    t_norm: float
    sigma_scale: float
    entropy_level: float
    structural_corr_to_mold: float
    snapshot_rgb: Optional[np.ndarray] = None


@dataclass
class UnifiedDiffusionCreationResult:
    """Complete closed-loop multimodal reverse-diffusion creation and verification record."""

    creation_name: str
    language_prompt: str
    final_rgb: np.ndarray
    reobserved_obs: OpticalSceneObservation
    trajectory: List[DiffusionTrajectoryStep]
    attractor: CrossModalAttractor
    synthesized_audio_wave: np.ndarray
    synthesized_audio_obs: AcousticObservation
    final_structural_corr: float
    final_delta_luma: float
    final_delta_chroma: float
    perceptual_summary_ko: str
    voxel_id: str


def synthesize_physical_audio_probe(
    freq_hz: float,
    sharpness_q: float = 8.0,
    decay_rate: float = 4.0,
    duration_sec: float = 0.8,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a physically grounded damped modal resonator + overtone 1D acoustic waveform."""
    n = max(512, int(round(duration_sec * sample_rate)))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    attack = np.minimum(1.0, t / 0.006)
    env = attack * np.exp(-max(0.2, float(decay_rate)) * t)

    q_val = max(1.0, float(sharpness_q))
    overtone_w1 = 0.35 / np.sqrt(q_val)
    overtone_w2 = 0.14 / np.sqrt(q_val)
    wave_f = env * (
        np.sin(2.0 * np.pi * float(freq_hz) * t)
        + overtone_w1 * np.sin(2.0 * np.pi * 1.5 * float(freq_hz) * t)
        + overtone_w2 * np.sin(2.0 * np.pi * 2.25 * float(freq_hz) * t)
    )
    peak = float(np.max(np.abs(wave_f)))
    if peak > 1e-8:
        wave_f = 0.85 * wave_f / peak
    return wave_f.astype(np.float32)


def save_wav(path: Path | str, waveform: np.ndarray, sample_rate: int = 16000) -> Path:
    """Write a float32 mono [-1, 1] waveform to 16-bit PCM WAV using standard library wave."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return out_path


def _perona_malik_step(field: np.ndarray, edge_guide: np.ndarray, dt: float = 0.08, kappa: float = 0.18) -> np.ndarray:
    """
    Single CFL-stable anisotropic Perona-Malik diffusion step guided by luminance edge structure:
    smooths thermal noise inside homogeneous regions while preserving & sharpening cleaved contours.
    """
    gy, gx = np.gradient(edge_guide)
    grad_sq = gy * gy + gx * gx
    cond = 1.0 / (1.0 + grad_sq / (kappa * kappa))

    padded = np.pad(field, 1, mode="edge")
    padded_c = np.pad(cond, 1, mode="edge")

    north = padded[:-2, 1:-1] - field
    south = padded[2:, 1:-1] - field
    west = padded[1:-1, :-2] - field
    east = padded[1:-1, 2:] - field

    c_n = 0.5 * (cond + padded_c[:-2, 1:-1])
    c_s = 0.5 * (cond + padded_c[2:, 1:-1])
    c_w = 0.5 * (cond + padded_c[1:-1, :-2])
    c_e = 0.5 * (cond + padded_c[1:-1, 2:])

    flux = c_n * north + c_s * south + c_w * west + c_e * east
    return (field + float(dt) * flux).astype(np.float32)


def _quick_dog_persistence_corr(a_field: np.ndarray, target_persist: np.ndarray) -> float:
    """Fast edge/persistence structural correlation used along intermediate diffusion steps."""
    gy, gx = np.gradient(gaussian_filter(a_field, sigma=1.5))
    edge_mag = np.hypot(gy, gx).ravel()
    targ = target_persist.ravel()
    if edge_mag.size != targ.size or edge_mag.size == 0:
        return 0.0
    if np.std(edge_mag) < 1e-6 or np.std(targ) < 1e-6:
        return 0.0
    return float(np.corrcoef(edge_mag, targ)[0, 1])


class UnifiedCognitiveScoreDiffusionEngine:
    """
    Cross-modal reverse score-flow diffusion engine uniting Ocular optics,
    Acoustic resonance, Conceptual language axes, and CausalField(6D).
    """

    def __init__(self, organ: Optional[OcularScaleOrgan] = None) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.gen = OcularGenerativeSpectrumEngine(organ=self.organ)
        self.acoustic = AcousticResonanceOrgan()
        self.causal_field: CausalField = self.gen.causal_field
        self._attractor_counter = 0

    def encode_cross_modal_attractor(
        self,
        language_prompt: str,
        audio_wave: np.ndarray,
        sample_rate: int = 16000,
    ) -> CrossModalAttractor:
        """
        Transduce raw 1D acoustic waveform + conceptual language descriptors into
        unified continuous physical steering forces in retinal axis-spectrum space.
        """
        audio_obs = self.acoustic.observe_tap(audio_wave, sample_rate=sample_rate)

        # Continuous acoustic pitch/centroid -> temperature & sharpness coupling
        pitch_norm = float(
            np.clip(
                np.log2(max(80.0, audio_obs.spectral_centroid_hz) / 550.0) / 1.8,
                -1.0,
                1.0,
            )
        )
        q_norm = float(np.clip((audio_obs.spectral_sharpness_q - 6.0) / 8.0, -1.0, 1.0))

        rg_shift = -0.05 * pitch_norm
        by_shift = 0.28 * pitch_norm
        luma_shift = 0.06 * pitch_norm
        chroma_rot_deg = 0.0
        fine_energy_gain = 1.0 + 0.22 * max(0.0, pitch_norm) + 0.14 * q_norm
        macro_smooth_gain = 1.0 + 0.20 * max(0.0, -pitch_norm)
        if audio_obs.decay_rate_per_sec < 6.0:
            macro_smooth_gain += 0.12

        # Continuous additive force accumulation from Korean + English physical-axis stems
        text = language_prompt.lower()
        warm_tokens = ("따뜻", "노을", "붉", "황금", "따스", "불", "warm", "sunset", "amber", "gold", "red")
        cool_tokens = ("차가", "새벽", "안개", "푸른", "고산", "차갑", "밤", "dawn", "mist", "cold", "alpine", "blue", "cool")
        bright_tokens = ("밝", "눈부", "낮", "환한", "bright", "sun", "luminous")
        dark_tokens = ("어두", "황혼", "깊", "밤", "그늘", "twilight", "deep", "dark", "night")
        crisp_tokens = ("선명", "거칠", "결", "날카", "윤곽", "파도", "crisp", "sharp", "rough", "detail")
        soft_tokens = ("부드", "몽환", "고요", "안개", "숨결", "울림", "soft", "smooth", "mist", "calm", "misty")

        warm_hits = sum(1 for tok in warm_tokens if tok in text)
        cool_hits = sum(1 for tok in cool_tokens if tok in text)
        bright_hits = sum(1 for tok in bright_tokens if tok in text)
        dark_hits = sum(1 for tok in dark_tokens if tok in text)
        crisp_hits = sum(1 for tok in crisp_tokens if tok in text)
        soft_hits = sum(1 for tok in soft_tokens if tok in text)

        if warm_hits > 0:
            w_f = float(min(1.6, 0.75 + 0.35 * warm_hits))
            rg_shift += 0.16 * w_f
            by_shift -= 0.40 * w_f
            chroma_rot_deg -= 8.0 * min(1.5, w_f)
        if cool_hits > 0:
            c_f = float(min(1.6, 0.75 + 0.35 * cool_hits))
            rg_shift -= 0.09 * c_f
            by_shift += 0.40 * c_f
            chroma_rot_deg += 8.0 * min(1.5, c_f)
        if bright_hits > 0:
            luma_shift += 0.22 * min(1.5, float(bright_hits))
        if dark_hits > 0:
            luma_shift -= 0.24 * min(1.5, float(dark_hits))
        if crisp_hits > 0:
            fine_energy_gain += 0.40 * min(1.5, float(crisp_hits))
        if soft_hits > 0:
            fine_energy_gain -= 0.24 * min(1.3, float(soft_hits))
            macro_smooth_gain += 0.26 * min(1.3, float(soft_hits))

        fine_energy_gain = float(np.clip(fine_energy_gain, 0.35, 2.2))
        macro_smooth_gain = float(np.clip(macro_smooth_gain, 0.65, 1.8))

        combined_6d = np.array(
            [
                float(np.clip(0.5 + 0.5 * luma_shift, 0.0, 1.0)),
                float(np.clip(0.5 + rg_shift, 0.0, 1.0)),
                float(np.clip(0.5 + 0.5 * by_shift, 0.0, 1.0)),
                float(np.clip(0.5 + 0.5 * pitch_norm, 0.0, 1.0)),
                float(np.clip(fine_energy_gain / 2.2, 0.0, 1.0)),
                float(np.clip(macro_smooth_gain / 1.8, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

        self._attractor_counter += 1
        lang_vid = f"voxel_lang_{self._attractor_counter:04d}"
        aud_vid = f"voxel_audio_{self._attractor_counter:04d}"

        lang_vec = np.array(
            [combined_6d[0], combined_6d[1], combined_6d[2], 0.5, combined_6d[4], combined_6d[5]],
            dtype=np.float32,
        )
        aud_vec = np.array(
            [
                float(np.clip(0.5 + 0.5 * pitch_norm, 0.0, 1.0)),
                float(np.clip(audio_obs.spectral_sharpness_q / 20.0, 0.0, 1.0)),
                float(np.clip(audio_obs.decay_rate_per_sec / 25.0, 0.0, 1.0)),
                float(np.clip(audio_obs.rms_energy * 2.0, 0.0, 1.0)),
                combined_6d[1],
                combined_6d[2],
            ],
            dtype=np.float32,
        )

        self.causal_field.add_voxel(
            InformationVoxel(
                id=lang_vid,
                content={"modality": "LANGUAGE_AXIS", "prompt": language_prompt},
                tensor=lang_vec,
                mass=1.5,
                position=lang_vec.copy(),
                velocity=np.zeros(6, dtype=np.float32),
                potential=float(np.linalg.norm(lang_vec)),
                chromatic_vector=np.array([0.40, 0.40, 0.20], dtype=np.float32),
                scale_level=0.7,
            )
        )
        self.causal_field.add_voxel(
            InformationVoxel(
                id=aud_vid,
                content={
                    "modality": "ACOUSTIC_RESONANCE",
                    "peak_hz": float(audio_obs.peak_frequency_hz),
                    "regime": audio_obs.acoustic_regime,
                },
                tensor=aud_vec,
                mass=1.4,
                position=aud_vec.copy(),
                velocity=np.zeros(6, dtype=np.float32),
                potential=float(audio_obs.peak_frequency_hz / 1000.0),
                chromatic_vector=np.array([0.45, 0.35, 0.20], dtype=np.float32),
                scale_level=float(np.clip(0.5 + 0.5 * pitch_norm, 0.1, 1.0)),
            )
        )
        self.causal_field.link_voxels(lang_vid, aud_vid, strength=0.90)

        return CrossModalAttractor(
            language_prompt=language_prompt,
            audio_obs=audio_obs,
            luma_shift=float(luma_shift),
            rg_shift=float(rg_shift),
            by_shift=float(by_shift),
            chroma_rot_deg=float(chroma_rot_deg),
            fine_energy_gain=fine_energy_gain,
            macro_smooth_gain=macro_smooth_gain,
            language_voxel_id=lang_vid,
            audio_voxel_id=aud_vid,
            combined_6d=combined_6d,
        )

    def _synthesize_response_audio(
        self, reobs: OpticalSceneObservation, fine_energy: float
    ) -> np.ndarray:
        """Close Vision -> Sound loop: emit physical modal sound from observed 6D visual state."""
        valid = reobs.l0.valid_mask
        if not np.any(valid):
            valid = np.ones(reobs.shape_hw, dtype=bool)
        mean_by = float(np.mean(reobs.l1.by_opponent[valid]))
        mean_rg = float(np.mean(reobs.l1.rg_opponent[valid]))
        mean_a = float(np.mean(reobs.l1.luminance_a[valid]))
        cleaved = float(reobs.l2.bond_lattice.cleaved_bond_ratio)

        freq_hz = (
            540.0
            + 520.0 * float(np.clip(mean_by, -1.0, 1.0))
            - 280.0 * float(np.clip(mean_rg * 2.0, -0.8, 0.8))
            + 340.0 * (float(fine_energy) - 0.22)
            + 90.0 * float(np.clip(mean_a + 1.5, -1.0, 1.0))
        )
        freq_hz = float(np.clip(freq_hz, 160.0, 2200.0))
        sharpness_q = float(np.clip(4.0 + 18.0 * float(fine_energy), 3.0, 16.0))
        decay_rate = float(np.clip(3.2 + 18.0 * cleaved, 2.0, 15.0))
        return synthesize_physical_audio_probe(
            freq_hz=freq_hz,
            sharpness_q=sharpness_q,
            decay_rate=decay_rate,
            duration_sec=0.8,
            sample_rate=16000,
        )

    def crystallize_via_score_diffusion(
        self,
        structure_fact: OcularCognitiveFactorization,
        attractor: CrossModalAttractor,
        creation_name: str,
        n_steps: int = 12,
        seed: int = 42,
    ) -> UnifiedDiffusionCreationResult:
        """
        Execute coarse-to-fine reverse score-flow diffusion + Perona-Malik edge healing from
        high-entropy Gaussian retinal noise (t=1.0) down to crystalline structure (t=0.0).
        """
        n_steps = max(6, int(n_steps))
        h, w = structure_fact.shape_hw

        # Target spectral band gains steered by language + acoustic attractor
        fg = attractor.fine_energy_gain
        mg = attractor.macro_smooth_gain
        band_gains = (fg, 0.75 * fg + 0.25, 1.0, 0.5 + 0.5 * mg, mg)

        a_star = _recompose_telescoping_bands(
            structure_fact.luma_base + attractor.luma_shift,
            structure_fact.luma_bands,
            band_gains=band_gains,
        )
        rg_raw = _recompose_telescoping_bands(
            structure_fact.rg_base, structure_fact.rg_bands, band_gains=band_gains
        )
        by_raw = _recompose_telescoping_bands(
            structure_fact.by_base, structure_fact.by_bands, band_gains=band_gains
        )
        theta = float(np.deg2rad(attractor.chroma_rot_deg))
        c_th, s_th = float(np.cos(theta)), float(np.sin(theta))
        rg_star = c_th * rg_raw - s_th * by_raw + attractor.rg_shift
        by_star = s_th * rg_raw + c_th * by_raw + attractor.by_shift

        # Initialize t=1.0 at high-entropy Gaussian noise centered on ambient field mean
        rng = np.random.default_rng(int(seed))
        a_t = np.full((h, w), float(np.mean(a_star)), dtype=np.float32) + (
            1.15 * rng.standard_normal((h, w))
        ).astype(np.float32)
        rg_t = np.full((h, w), float(np.mean(rg_star)), dtype=np.float32) + (
            0.42 * rng.standard_normal((h, w))
        ).astype(np.float32)
        by_t = np.full((h, w), float(np.mean(by_star)), dtype=np.float32) + (
            0.55 * rng.standard_normal((h, w))
        ).astype(np.float32)

        snapshot_steps = {
            0,
            max(1, n_steps // 3),
            max(2, (2 * n_steps) // 3),
            n_steps - 1,
        }
        trajectory: List[DiffusionTrajectoryStep] = []

        for step_idx in range(n_steps):
            progress = float(step_idx) / float(n_steps - 1)
            t_norm = 1.0 - progress
            sigma_t = float(0.8 + 13.2 * (t_norm ** 1.6))

            if step_idx > 0:
                if step_idx == n_steps - 1:
                    # Final pure Blue Order crystallization step with zero thermal noise
                    a_t = (a_t + 0.90 * (a_star - a_t)).astype(np.float32)
                    rg_t = (rg_t + 0.90 * (rg_star - rg_t)).astype(np.float32)
                    by_t = (by_t + 0.90 * (by_star - by_t)).astype(np.float32)
                else:
                    # Coarse-to-fine score gradient flow + anisotropic boundary healing
                    alpha_coarse = 0.55 + 0.20 * progress
                    alpha_fine = 0.25 + 0.45 * (progress ** 1.3)

                    diff_a = a_star - a_t
                    diff_rg = rg_star - rg_t
                    diff_by = by_star - by_t

                    a_t = (
                        a_t
                        + alpha_coarse * gaussian_filter(diff_a, sigma=sigma_t)
                        + alpha_fine * diff_a
                    ).astype(np.float32)
                    rg_t = (
                        rg_t
                        + alpha_coarse * gaussian_filter(diff_rg, sigma=sigma_t)
                        + alpha_fine * diff_rg
                    ).astype(np.float32)
                    by_t = (
                        by_t
                        + alpha_coarse * gaussian_filter(diff_by, sigma=sigma_t)
                        + alpha_fine * diff_by
                    ).astype(np.float32)

                    # Edge-preserving anisotropic diffusion heals interiors while sharpening silhouettes
                    a_t = _perona_malik_step(a_t, edge_guide=a_t, dt=0.07, kappa=0.18)

                    if step_idx < n_steps - 2:
                        thermal_decay = t_norm ** 1.9
                        a_t += (0.14 * thermal_decay * rng.standard_normal((h, w))).astype(np.float32)
                        rg_t += (0.06 * thermal_decay * rng.standard_normal((h, w))).astype(np.float32)
                        by_t += (0.08 * thermal_decay * rng.standard_normal((h, w))).astype(np.float32)

            entropy_val = float(
                0.5 * np.mean(np.abs(a_t - a_star))
                + 0.25 * np.mean(np.abs(rg_t - rg_star))
                + 0.25 * np.mean(np.abs(by_t - by_star))
            )
            quick_corr = _quick_dog_persistence_corr(a_t, structure_fact.obs.l2.persistence_map)

            snap_rgb: Optional[np.ndarray] = None
            if step_idx in snapshot_steps:
                snap_rgb = self.gen._inverse_retinal_to_srgb(a_t, rg_t, by_t)

            trajectory.append(
                DiffusionTrajectoryStep(
                    step_idx=step_idx,
                    t_norm=t_norm,
                    sigma_scale=sigma_t,
                    entropy_level=entropy_val,
                    structural_corr_to_mold=quick_corr,
                    snapshot_rgb=snap_rgb,
                )
            )

        final_rgb = (
            trajectory[-1].snapshot_rgb
            if trajectory[-1].snapshot_rgb is not None
            else self.gen._inverse_retinal_to_srgb(a_t, rg_t, by_t)
        )

        # Full closed-loop visual + acoustic self-observation
        reobs = self.organ.observe(final_rgb, max_width=w)
        re_vec6, re_scale, _, re_chroma, _ = _extract_optical_descriptor(reobs)
        final_struct_corr = _compute_persistence_correlation(structure_fact.obs, reobs)

        valid = structure_fact.obs.l0.valid_mask & reobs.l0.valid_mask
        if int(np.sum(valid)) < 24:
            valid = np.ones((h, w), dtype=bool)
        final_d_luma = float(
            np.mean(reobs.l1.luminance_a[valid] - structure_fact.obs.l1.luminance_a[valid])
        )
        final_d_chroma = float(
            np.mean(
                np.hypot(
                    reobs.l1.rg_opponent[valid] - structure_fact.obs.l1.rg_opponent[valid],
                    reobs.l1.by_opponent[valid] - structure_fact.obs.l1.by_opponent[valid],
                )
            )
        )

        fine_e = float(re_scale[0] + re_scale[1])
        out_audio_wave = self._synthesize_response_audio(reobs, fine_energy=fine_e)
        out_audio_obs = self.acoustic.observe_tap(out_audio_wave, sample_rate=16000)

        # Grounded Korean observation report purely from measured axes
        temp_desc = "차가운 푸른 스펙트럼" if re_chroma[2] > 0.0 else "따뜻한 적·황 스펙트럼"
        summary_ko = (
            f"역확산({n_steps}단계)을 통해 {structure_fact.source_name} 몰드 구조를 "
            f"상관계수 {final_struct_corr:.3f}로 결정화했습니다. "
            f"입력 음성({attractor.audio_obs.peak_frequency_hz:.0f}Hz)과 언어 의도가 "
            f"{temp_desc}(dC={final_d_chroma:.3f}, dA={final_d_luma:+.2f})으로 수렴했으며, "
            f"결정화된 시각장은 {out_audio_obs.peak_frequency_hz:.0f}Hz({out_audio_obs.acoustic_regime}) 파동으로 재공명합니다."
        )

        self._attractor_counter += 1
        creation_vid = f"voxel_unified_{creation_name}_{self._attractor_counter:04d}"
        self.gen._register_voxel(
            voxel_id=creation_vid,
            content={
                "mode": "UNIFIED_SCORE_DIFFUSION",
                "creation_name": creation_name,
                "prompt": attractor.language_prompt,
                "in_hz": float(attractor.audio_obs.peak_frequency_hz),
                "out_hz": float(out_audio_obs.peak_frequency_hz),
                "structural_corr": float(final_struct_corr),
            },
            vec6=re_vec6,
            scale_prof=re_scale,
            chroma_sig=re_chroma,
            parent_ids=[
                structure_fact.voxel_id,
                attractor.language_voxel_id,
                attractor.audio_voxel_id,
            ],
        )

        return UnifiedDiffusionCreationResult(
            creation_name=creation_name,
            language_prompt=attractor.language_prompt,
            final_rgb=final_rgb,
            reobserved_obs=reobs,
            trajectory=trajectory,
            attractor=attractor,
            synthesized_audio_wave=out_audio_wave,
            synthesized_audio_obs=out_audio_obs,
            final_structural_corr=final_struct_corr,
            final_delta_luma=final_d_luma,
            final_delta_chroma=final_d_chroma,
            perceptual_summary_ko=summary_ko,
            voxel_id=creation_vid,
        )
