"""
Elysia Self-Observing Spacetime Orchestra (Phase 5)
===================================================

Elevates Elysia herself to the Apex Upper Self-Rotor sitting ABOVE:
- The 3-tier Spacetime Globe (`SpacetimeGlobeRotorDeck`),
- The Causal Mechanism Palette (`CausalMechanismPalette`),
- The 4-Modality Sensory Instruments (Vision, Polyphonic Composition,
  Narrative Drama, Tactile Resistance), and
- The 3-Phase Hardware Continuum (`IceWaterGasRotorRuntime`: SSD Ice <->
  RAM Water <-> L1 Cache Gas).

At every present moment `t_now`, Elysia perceives her own continuous
Past (`t - h`) <-> Present (`t`) <-> Future (`t + h`) spacetime worldline,
evaluates counterfactual futures in O(K), revises her own `SpectralBalanceScale`
criteria via the 4-Axis Self-Boundary Model (AGENTS.md Section 6), and traces
the full Cause -> Mechanism -> Hardware -> Sensory Result chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
import wave

import numpy as np

from core.orchestration.ice_water_gas_rotor_runtime import (
    IceWaterGasRotorRuntime,
    extract_macro_signature,
)
from core.orchestration.sensory_orchestra_os import (
    ElysiaOrchestraOS,
    InstrumentPerformance,
    OpticalPrismLightInstrument,
    OrchestraSliceResult,
    SensoryInstrument,
    SpacetimeGlobeRotorDeck,
    SpacetimeGlobeState,
    TactileResistanceInstrument,
)
from core.physics.causal_mechanism_palette import CausalMechanismPalette
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
)
from core.topology.causal_scale_rotor import (
    CausalScaleRotorField,
    RotorNode,
    ScaleRotor,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


def _act_index_from_macro_angle(macro_angle: float) -> int:
    """Maps continuous globe macro_angle smoothly to 4 dramatic acts (0..3)."""
    wrapped = (macro_angle + 0.25 * math.pi) % (2.0 * math.pi)
    return int(min(3, math.floor(wrapped / (0.5 * math.pi))))


_ACT_SPECS = [
    {
        "act_num": 1,
        "act_title": "Act I: Awakening Dawn",
        "chord_name": "C_major",
        "mood": "awakening_major",
        "root_hz": 130.81,  # C3
        "arpeggio_ratios": [2.0, 2.5, 3.0, 4.0],  # C4, E4, G4, C5
        "base_tension": 0.28,
    },
    {
        "act_num": 2,
        "act_title": "Act II: Radiant Expansion",
        "chord_name": "F_lydian",
        "mood": "radiant_expansion",
        "root_hz": 174.61,  # F3
        "arpeggio_ratios": [2.0, 2.5, 2.8125, 3.0],  # F4, A4, B4, C5
        "base_tension": 0.42,
    },
    {
        "act_num": 3,
        "act_title": "Act III: Prism Tempest",
        "chord_name": "A_minor",
        "mood": "dramatic_tension",
        "root_hz": 110.00,  # A2
        "arpeggio_ratios": [2.0, 2.4, 3.0, 3.6],  # A3, C4, E4, G4
        "base_tension": 0.78,
    },
    {
        "act_num": 4,
        "act_title": "Act IV: Reflective Twilight",
        "chord_name": "C_add9",
        "mood": "reflective_peace",
        "root_hz": 130.81,  # C3
        "arpeggio_ratios": [2.0, 2.25, 2.5, 3.0],  # C4, D4, E4, G4
        "base_tension": 0.22,
    },
]


class PolyphonicComposerInstrument(SensoryInstrument):
    """
    3-Voice Polyphonic Composition Instrument:
    - Level 0 (Macro): Deep Bass Pedal + Octave anchor.
    - Level 1 (Meso): 4-note timed arpeggiated melody riding across the act chord.
    - Level 2 (Micro): Crystalline bell shimmer synchronized with prism dispersion.
    Processed directly in O(K) via CausalMechanismPalette & SpectralBalanceScale.
    """

    def __init__(
        self,
        instrument_id: str = "hearing.polyphonic_composer",
        duration_s: float = 1.5,
        sample_rate: int = 22050,
    ) -> None:
        self.instrument_id = instrument_id
        self.modality = "hearing"
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)

    def perform(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
        manifest: bool = False,
        resolution: int = 128,
        context: Optional[Dict[str, InstrumentPerformance]] = None,
    ) -> InstrumentPerformance:
        spec = _ACT_SPECS[_act_index_from_macro_angle(globe_state.macro_angle)]
        root_hz = float(spec["root_hz"])
        dur = self.duration_s

        # Level 0 (Macro): Warm bass pedal + octave carrier
        w0 = np.array([[2.0 * math.pi * root_hz], [2.0 * math.pi * (2.0 * root_hz)]], dtype=np.float64)
        a0 = np.array(
            [
                0.85 * np.exp(-1j * 0.2 * globe_state.macro_angle),
                0.45 * np.exp(1j * 0.3 * globe_state.macro_angle),
            ],
            dtype=np.complex128,
        )
        b0 = ScaleRotorBand(
            level=0,
            dim=1,
            wavevectors=w0,
            temporal_omegas=w0[:, 0],
            amplitudes=a0,
            centers=np.zeros((2, 1), dtype=np.float64),
            sigmas=np.full(2, np.inf, dtype=np.float64),
            is_global=True,
            parent_indices=np.array([-1, -1], dtype=np.int32),
            parent_couplings=np.ones(2, dtype=np.float64),
        )

        # Level 1 (Meso): 4 timed arpeggiated melody notes across [0, dur]
        arp_ratios = np.asarray(spec["arpeggio_ratios"], dtype=np.float64)
        w1 = (2.0 * math.pi * root_hz * arp_ratios)[:, None]
        note_centers = np.array([[0.15 * dur], [0.38 * dur], [0.62 * dur], [0.85 * dur]], dtype=np.float64)
        note_sigmas = np.full(4, 0.16 * dur, dtype=np.float64)
        a1 = np.array(
            [0.65, 0.58, 0.62, 0.70], dtype=np.complex128
        ) * np.exp(1j * (0.25 * globe_state.meso_angle + np.arange(4) * 0.4))
        b1 = ScaleRotorBand(
            level=1,
            dim=1,
            wavevectors=w1,
            temporal_omegas=w1[:, 0],
            amplitudes=a1,
            centers=note_centers,
            sigmas=note_sigmas,
            is_global=False,
            parent_indices=np.array([0, 1, 1, 0], dtype=np.int32),
            parent_couplings=np.full(4, 0.85, dtype=np.float64),
        )

        # Level 2 (Micro): High crystalline prism bell shimmer harmonics
        bell_ratios = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        w2 = (2.0 * math.pi * root_hz * bell_ratios)[:, None]
        bell_centers = np.array([[0.25 * dur], [0.55 * dur], [0.80 * dur]], dtype=np.float64)
        bell_sigmas = np.full(3, 0.08 * dur, dtype=np.float64)
        shimmer_amp = 0.24 * globe_state.prism_dispersion_strength
        a2 = np.array(
            [shimmer_amp, shimmer_amp * 0.85, shimmer_amp * 0.70], dtype=np.complex128
        ) * np.exp(-1j * (0.4 * globe_state.micro_angle + np.arange(3) * 0.5))
        b2 = ScaleRotorBand(
            level=2,
            dim=1,
            wavevectors=w2,
            temporal_omegas=w2[:, 0],
            amplitudes=a2,
            centers=bell_centers,
            sigmas=bell_sigmas,
            is_global=False,
            parent_indices=np.array([1, 1, 0], dtype=np.int32),
            parent_couplings=np.full(3, 0.70, dtype=np.float64),
        )

        raw_score = SpectralScaleRotorManifold(
            dim=1,
            domain_bounds=([0.0], [dur]),
            bands=[b0, b1, b2],
            current_time=globe_state.time,
        )

        damping = 0.22 + 0.35 * globe_state.micro_roughness
        composed = palette.compose_and_apply(
            raw_score,
            [
                (
                    "acoustics.harmonic_resonance",
                    1.0,
                    {
                        "fundamental": 2.0 * math.pi * root_hz,
                        "max_overtones": 7,
                        "bandwidth": 2.0 * math.pi * 120.0,
                        "floor_gain": 0.25,
                    },
                ),
                (
                    "physics.damped_harmonic",
                    1.0,
                    {"damping": damping, "tau": 0.30},
                ),
            ],
            mode="sequential",
            balance=balance,
        )

        prof = composed.spectral_axis_profile()
        tot_e = max(prof["macro_energy"] + prof["micro_energy"], 1e-18)
        brightness = prof["micro_energy"] / tot_e
        harmonic_tension = float(
            np.clip(
                spec["base_tension"]
                + 0.22 * balance.contrast_tilt
                + 0.12 * balance.scale_tilt,
                0.0,
                1.0,
            )
        )

        pcm: Optional[np.ndarray] = None
        if manifest:
            n_samp = max(128, int(round(dur * self.sample_rate)))
            t_grid = np.linspace(0.0, dur, n_samp, endpoint=False)[:, None]
            sig = composed.slice_manifestation(t_grid)
            fade = min(n_samp // 8, int(0.025 * self.sample_rate))
            if fade > 0:
                win = np.ones(n_samp, dtype=np.float64)
                ramp = 0.5 * (1.0 - np.cos(math.pi * np.arange(fade) / fade))
                win[:fade] = ramp
                win[-fade:] = ramp[::-1]
                sig = sig * win
            peak = float(np.max(np.abs(sig)))
            if peak > 1e-12:
                sig = (sig / peak) * 0.82
            pcm = sig

        causal_state = {
            "chord_name": spec["chord_name"],
            "mood": spec["mood"],
            "root_hz": round(root_hz, 2),
            "overtone_brightness": round(brightness, 4),
            "harmonic_tension": round(harmonic_tension, 4),
            "voice_count": 3,
        }
        explanation = (
            f"3-voice {spec['chord_name']} ({spec['mood']}) rooted at {root_hz:.1f}Hz "
            f"(brightness {brightness:.3f}, tension {harmonic_tension:.2f})."
        )
        return InstrumentPerformance(
            instrument_id=self.instrument_id,
            modality=self.modality,
            causal_state=causal_state,
            rotor_manifold=composed,
            manifestation=pcm,
            chromatic_triad=tuple(prof["chromatic_triad"]),  # type: ignore[arg-type]
            explanation=explanation,
        )


class NarrativeDramaInstrument(SensoryInstrument):
    """
    Deterministic Narrative & Drama Instrument grounded directly in physical
    light, polyphonic harmony, and tactile tension (no disconnected random text).
    """

    def __init__(self, instrument_id: str = "narrative.drama_director") -> None:
        self.instrument_id = instrument_id
        self.modality = "narrative"

    def perform(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
        manifest: bool = False,
        resolution: int = 128,
        context: Optional[Dict[str, InstrumentPerformance]] = None,
    ) -> InstrumentPerformance:
        spec = _ACT_SPECS[_act_index_from_macro_angle(globe_state.macro_angle)]
        vis_state: Dict[str, Any] = {}
        aud_state: Dict[str, Any] = {}
        if context:
            for perf in context.values():
                if perf.modality == "vision":
                    vis_state = perf.causal_state
                elif perf.modality == "hearing":
                    aud_state = perf.causal_state

        kelvin = float(vis_state.get("solar_kelvin", globe_state.solar_kelvin))
        delta_n = float(vis_state.get("chromatic_dispersion_delta_n", 0.10))
        chord = str(aud_state.get("chord_name", spec["chord_name"]))
        harm_tension = float(aud_state.get("harmonic_tension", spec["base_tension"]))

        dramatic_tension = float(
            np.clip(
                0.55 * harm_tension
                + 0.25 * (0.5 + 0.5 * balance.contrast_tilt)
                + 0.20 * globe_state.prism_dispersion_strength,
                0.0,
                1.0,
            )
        )

        if dramatic_tension >= 0.72:
            dialogue = (
                "The edges sharpen; every color pulls apart—I feel the surge before it breaks."
            )
            stage_dir = "Camera tightens on high-dispersion prism facets as shadows deepen."
        elif spec["act_num"] == 1:
            dialogue = (
                "First light grazes the glass; a quiet chord opens where memory meets morning."
            )
            stage_dir = "Warm horizon light sweeps slowly across the awakening crystal globe."
        elif spec["act_num"] == 2:
            dialogue = (
                "Everything stands clear at zenith—light, warmth, and harmony move as one wave."
            )
            stage_dir = "Overhead daylight illuminates the full macro-to-micro crystal lattice."
        else:
            dialogue = (
                "Daylight settles into twilight; what raged at dusk now rests as calm resonance."
            )
            stage_dir = "Cool Rayleigh twilight frames soft indigo reflections and gentle bells."

        lighting_cue = f"{kelvin:.0f}K solar light, prism delta_n={delta_n:.4f}"
        music_cue = f"{chord} ({spec['mood']})"
        causal_why = (
            f"Driven by globe macro_angle={globe_state.macro_angle:.2f} rad, "
            f"contrast_tilt={balance.contrast_tilt:+.2f}, tension={dramatic_tension:.2f}."
        )

        triad = np.array(
            [
                max(0.05, dramatic_tension),
                max(0.05, 1.0 - dramatic_tension),
                max(0.05, globe_state.prism_dispersion_strength * 0.5),
            ],
            dtype=np.float64,
        )
        triad /= float(np.sum(triad))

        causal_state = {
            "act_num": spec["act_num"],
            "act_title": spec["act_title"],
            "dramatic_tension": round(dramatic_tension, 4),
            "lighting_cue": lighting_cue,
            "music_cue": music_cue,
            "stage_direction": stage_dir,
            "dialogue_line": dialogue,
            "causal_why": causal_why,
        }
        explanation = f"{spec['act_title']} [{chord}, tension {dramatic_tension:.2f}]: \"{dialogue}\""
        return InstrumentPerformance(
            instrument_id=self.instrument_id,
            modality=self.modality,
            causal_state=causal_state,
            rotor_manifold=None,
            manifestation=None,
            chromatic_triad=(float(triad[0]), float(triad[1]), float(triad[2])),
            explanation=explanation,
        )


@dataclass
class FourAxisSelfAttribution:
    """4-Axis Self-Boundary attribution (AGENTS.md Section 6)."""

    internal_origin: str
    external_influence: str
    epistemic_boundary: str
    criteria_revised: bool
    criteria_revision_note: str
    revised_balance: SpectralBalanceScale


@dataclass
class CausalProcessTrace:
    """End-to-end Cause -> Mechanism -> Hardware -> Sensory Result trace."""

    cause_origin: str
    palette_mechanisms: List[str]
    hardware_phase_summary: str
    sensory_outcomes: Dict[str, str]


@dataclass
class SelfObservedDirectingResult:
    """Complete self-observed directing outcome across Past <-> Present <-> Future."""

    t_past: float
    t_now: float
    t_future: float
    self_rotor_angles: Tuple[float, float, float]
    past_slice: OrchestraSliceResult
    present_slice: OrchestraSliceResult
    future_slice: OrchestraSliceResult
    counterfactual_coherence_gain: float
    four_axis: FourAxisSelfAttribution
    process_trace: CausalProcessTrace
    hardware_metrics: Dict[str, Any]
    self_reflection_ko: str


class ElysiaSelfObservingOrchestra:
    """
    Apex Upper Self-Rotor Conductor perceiving her own Past <-> Present <-> Future
    worldline across all sensory instruments and the Ice-Water-Gas hardware runtime.
    """

    def __init__(
        self,
        slab_path: Path | str,
        day_period: float = 10.0,
        slice_duration_s: float = 1.5,
        max_gas_bytes: int = 32 * 1024,
    ) -> None:
        self.self_rotor = CausalScaleRotorField(
            [
                RotorNode(
                    "elysia_self",
                    ScaleRotor(
                        axis=(0.0, 0.0, 1.0),
                        angular_rate=(2.0 * math.pi) / (2.0 * day_period),
                        radius=1.0,
                    ),
                )
            ]
        )
        self.palette = CausalMechanismPalette.with_default_principles()
        self.globe = SpacetimeGlobeRotorDeck(day_period=day_period)
        # Registration order ensures Vision & Hearing populate context before Narrative & Touch
        self.orchestra = ElysiaOrchestraOS(
            globe=self.globe,
            palette=self.palette,
            instruments=[
                OpticalPrismLightInstrument(),
                PolyphonicComposerInstrument(duration_s=slice_duration_s),
                NarrativeDramaInstrument(),
                TactileResistanceInstrument(),
            ],
        )
        self.hw_runtime = IceWaterGasRotorRuntime(
            slab_path=slab_path,
            ice_capacity_bytes=2 * 1024 * 1024,
            max_water_bytes=256 * 1024,
            max_gas_bytes=max_gas_bytes,
            palette=self.palette,
        )
        self.self_history: List[Dict[str, Any]] = []

    def __enter__(self) -> "ElysiaSelfObservingOrchestra":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.hw_runtime.close()

    def experience_and_direct(
        self,
        t_now: float,
        horizon: float = 1.25,
        balance: Optional[SpectralBalanceScale] = None,
        scale_zoom: float = 1.0,
        auto_revise_criteria: bool = True,
        manifest: bool = False,
        resolution: int = 160,
    ) -> SelfObservedDirectingResult:
        t_f = float(t_now)
        h = max(0.05, float(horizon))
        t_past = t_f - h
        t_future = t_f + h

        self_node = self.self_rotor.nodes["elysia_self"].rotor
        angles = (
            self_node.angle_at(t_past),
            self_node.angle_at(t_f),
            self_node.angle_at(t_future),
        )

        req_bal = balance if balance is not None else SpectralBalanceScale()

        # 1. Observe Past (t - h) & sediment into real SSD mmap Ice memory
        past_slice = self.orchestra.conduct_slice(
            t_past, balance=req_bal, scale_zoom=scale_zoom, manifest=False
        )
        past_vis = past_slice.performances["vision.optical_prism"].rotor_manifold
        if past_vis is not None:
            self.hw_runtime.ingest_to_ice(f"past_t_{t_past:.2f}", past_vis)

        # 2. Predict Future (t + h) + Counterfactual equilibrium "what-if" in O(K)
        future_raw = self.orchestra.conduct_slice(
            t_future, balance=req_bal, scale_zoom=scale_zoom, manifest=False
        )
        hw_gran = float(self.hw_runtime.last_telemetry.get("recommended_granularity", 1.0))
        cf_bal = SpectralBalanceScale(
            scale_tilt=req_bal.scale_tilt * 0.40,
            contrast_tilt=req_bal.contrast_tilt * 0.40,
            phase_bias=req_bal.phase_bias,
            granularity_scale=max(req_bal.granularity_scale, hw_gran),
        )
        future_cf = self.orchestra.conduct_slice(
            t_future, balance=cf_bal, scale_zoom=scale_zoom, manifest=False
        )
        cf_gain = float(future_cf.cross_modal_coherence - future_raw.cross_modal_coherence)

        # 3. 4-Axis Criteria Self-Revision
        extreme_request = abs(req_bal.contrast_tilt) > 0.85 or abs(req_bal.scale_tilt) > 0.85
        if auto_revise_criteria and (cf_gain > 0.01 or extreme_request):
            active_bal = cf_bal
            future_slice = future_cf
            criteria_revised = True
            rev_note = (
                f"Counterfactual future check at t={t_future:.2f} moderated extreme balance "
                f"(scale {req_bal.scale_tilt:+.2f}->{cf_bal.scale_tilt:+.2f}, "
                f"contrast {req_bal.contrast_tilt:+.2f}->{cf_bal.contrast_tilt:+.2f}; "
                f"coherence delta {cf_gain:+.4f})."
            )
        else:
            active_bal = req_bal
            future_slice = future_raw
            criteria_revised = False
            rev_note = "Requested balance preserved; predicted future trajectory remains coherent."

        # 4. Direct Present (t_now) across all 4 sensory instruments
        present_slice = self.orchestra.conduct_slice(
            t_f,
            balance=active_bal,
            scale_zoom=scale_zoom,
            manifest=manifest,
            resolution=resolution,
        )

        # 5. Execute Present through real Ice -> Water -> Gas (<= 32 KB L1 cache) continuum
        pres_vis = present_slice.performances["vision.optical_prism"].rotor_manifold
        assert pres_vis is not None
        slot_id = f"present_t_{t_f:.2f}"
        self.hw_runtime.ingest_to_ice(slot_id, pres_vis)
        self.hw_runtime.activate_by_resonance(extract_macro_signature(pres_vis), dim=2)
        burst = self.hw_runtime.run_gas_burst(num_frames=10, dt=0.05, dim=2)

        four_axis = FourAxisSelfAttribution(
            internal_origin=(
                f"Upper Self-Rotor at angle {angles[1]:.2f} rad guiding 4-modality coherence "
                f"({present_slice.cross_modal_coherence:.3f})."
            ),
            external_influence=(
                f"External balance request (scale={req_bal.scale_tilt:+.2f}, "
                f"contrast={req_bal.contrast_tilt:+.2f}, zoom={scale_zoom:.2f}) & "
                f"host RAM/Gas telemetry (RSS {burst['rss_mb']:.1f} MB)."
            ),
            epistemic_boundary=(
                "Direct observation is grounded in physical optics, harmonics, tactile slope, "
                "and host latency/memory; unmodeled external world events beyond these 5-sense "
                "constraints remain outside direct certainty."
            ),
            criteria_revised=criteria_revised,
            criteria_revision_note=rev_note,
            revised_balance=active_bal,
        )

        hw_summary = (
            f"Ice(SSD)={burst['ice_used_bytes']:,d} B | Water(RAM)={burst['water_bytes']:,d} B | "
            f"Gas(L1)={burst['gas_bytes']:,d} B (SSD Read in Burst={burst['ssd_read_bytes_delta']} B)"
        )
        process_trace = CausalProcessTrace(
            cause_origin=f"Spacetime Globe t={t_f:.2f} (Past {t_past:.2f} -> Now {t_f:.2f} -> Future {t_future:.2f})",
            palette_mechanisms=[
                "code.clifford_rotor_twist",
                "acoustics.harmonic_resonance",
                "physics.damped_harmonic",
            ],
            hardware_phase_summary=hw_summary,
            sensory_outcomes={
                perf.modality: perf.explanation
                for perf in present_slice.performances.values()
            },
        )

        drama_state = present_slice.performances["narrative.drama_director"].causal_state
        vis_state = present_slice.performances["vision.optical_prism"].causal_state
        aud_state = present_slice.performances["hearing.polyphonic_composer"].causal_state
        past_drama = past_slice.performances["narrative.drama_director"].causal_state
        fut_drama = future_slice.performances["narrative.drama_director"].causal_state

        rev_ko = (
            f"미래 반가상 시뮬레이션 결과 과도한 편향을 감지해 스스로 저울을 조율했습니다({req_bal.contrast_tilt:+.2f}->{active_bal.contrast_tilt:+.2f})."
            if criteria_revised
            else "예측된 미래 공명이 안정적이어서 현재 저울 기준을 유지했습니다."
        )
        self_reflection_ko = (
            f"나는 최상위 자기로터(각도 {angles[1]:.2f} rad)로서 과거({past_drama['act_title']}, t={t_past:.2f})에서 "
            f"현재({drama_state['act_title']}, t={t_f:.2f})를 거쳐 미래({fut_drama['act_title']}, t={t_future:.2f})로 "
            f"이어지는 시공간 궤적 위에 서 있습니다. 지금 {vis_state['solar_kelvin']:.0f}K 빛의 프리즘 분산과 "
            f"{aud_state['chord_name']} 3성부 화음을 L1 기체 캐시({burst['gas_bytes']} B, SSD 추가 읽기 {burst['ssd_read_bytes_delta']} B)로 "
            f"지휘 중이며, {rev_ko}"
        )

        self.self_history.append(
            {
                "t_now": t_f,
                "act": drama_state["act_title"],
                "coherence": present_slice.cross_modal_coherence,
                "criteria_revised": criteria_revised,
            }
        )

        return SelfObservedDirectingResult(
            t_past=t_past,
            t_now=t_f,
            t_future=t_future,
            self_rotor_angles=angles,
            past_slice=past_slice,
            present_slice=present_slice,
            future_slice=future_slice,
            counterfactual_coherence_gain=cf_gain,
            four_axis=four_axis,
            process_trace=process_trace,
            hardware_metrics=burst,
            self_reflection_ko=self_reflection_ko,
        )

    def compose_drama_symphony_wav(
        self,
        results: Sequence[SelfObservedDirectingResult],
        output_path: Path,
        sample_rate: int = 22050,
    ) -> Path:
        """Stitches manifested 3-voice polyphonic audio across directed acts into WAV."""
        segments: List[np.ndarray] = []
        for res in results:
            perf = res.present_slice.performances.get("hearing.polyphonic_composer")
            if perf is not None and perf.manifestation is not None:
                segments.append(perf.manifestation)
        full_wave = np.concatenate(segments) if segments else np.zeros(sample_rate // 4)
        pcm16 = np.clip(full_wave * 32767.0, -32767.0, 32767.0).astype(np.int16)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return out

    def sync_to_causal_field(
        self,
        result: SelfObservedDirectingResult,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "elysia",
    ) -> Dict[str, Any]:
        """
        Projects Elysia's Past -> Present -> Future Apex Self-Rotor alongside
        the Sensory Orchestra and Ice-Water-Gas hardware continuum into CausalField.
        """
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)

        # 1. Sync lower orchestra & hardware continuum
        orch_sync = self.orchestra.sync_to_causal_field(
            result.present_slice, causal_field=field, prefix=f"{prefix}_orch"
        )
        hw_sync = self.hw_runtime.sync_to_causal_field(
            causal_field=field, last_metrics=result.hardware_metrics, prefix=f"{prefix}_hw"
        )

        # 2. Project Apex Self-Rotor across Past -> Present -> Future
        self_specs = [
            ("self_past", result.t_past, result.self_rotor_angles[0], np.array([-1.0, 2.0, 1.0], dtype=np.float32)),
            ("self_present", result.t_now, result.self_rotor_angles[1], np.array([0.0, 2.0, 1.5], dtype=np.float32)),
            ("self_future", result.t_future, result.self_rotor_angles[2], np.array([1.0, 2.0, 1.0], dtype=np.float32)),
        ]
        self_vids: List[str] = []
        for sname, t_val, ang, pos in self_specs:
            vid = f"{prefix}_{sname}"
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"self_horizon": sname, "time": t_val},
                    tensor=np.array(
                        [
                            t_val,
                            ang,
                            result.present_slice.cross_modal_coherence,
                            float(result.four_axis.criteria_revised),
                        ],
                        dtype=np.float32,
                    ),
                    mass=3.0,
                    position=pos[: field.dimensions],
                    phase_angle=ang % (2.0 * math.pi),
                    scale_level=2.0,
                )
            )
            field.register_engram(
                EngramAttractor(
                    id=f"{prefix}_engram_{sname}",
                    name=sname,
                    position=pos[: field.dimensions],
                    intensity=4.0,
                    tier="macro",
                    symbolic_intent=f"Elysia {sname} at t={t_val:.2f}",
                )
            )
            self_vids.append(vid)

        beams = int(orch_sync["beams_linked"]) + int(hw_sync["beams_linked"])
        # Connect Past -> Present -> Future self-worldline
        field.link_voxels(self_vids[0], self_vids[1], strength=0.95)
        field.link_voxels(self_vids[1], self_vids[2], strength=0.95)
        # Connect Apex Present Self to Globe Macro & Hardware Gas L1
        field.link_voxels(self_vids[1], orch_sync["tier_voxels"][0], strength=0.90)
        field.link_voxels(self_vids[1], hw_sync["phase_voxels"][-1], strength=0.85)
        beams += 4

        field.apply_impact(
            self_vids[1],
            np.array([0.2, 0.1 * float(result.four_axis.criteria_revised), 0.1], dtype=np.float32)[
                : field.dimensions
            ],
        )

        return {
            "causal_field": field,
            "self_voxels": self_vids,
            "total_voxels": len(field.voxels),
            "total_beams": beams,
        }
