"""
ELYSIA Living Embodied Multi-Organ Causal Organism (Phase 10)
==============================================================

Unifies Phases 1-9 into a single embodied living organism where:
- Eye, Ear, Hand/Skin, Body Motor, Language (`HangulTopologicalEngine`),
  Trinitarian Least-Action Manifold (`TrinitarianScaleRotorSystem`), and
  Unit-1 Spectral Mirrors (`PlanetaryOrderGlobe`) operate as ONE shared organism.

Core Living Principles:
1. Sensory Vacuum Substitution ("진공처럼 공백으로 치환"):
   When an organ or sensory channel goes dark (blindness, deafness, numbness),
   its channels automatically become FREE vacuum slots in the organism's unified
   equality manifold. Intact sibling organs flow through physical laws under
   Least Action (dA = 0) to fill what they lawfully determine and synthesize
   missing optical/acoustic perceptions.
2. Honest Irreducible Vacuum (Epistemic Humility via SVD Nullspace):
   Without hardcoded heuristics, the column-normalized Jacobian right nullspace
   identifies variables that sibling senses genuinely cannot observe (such as
   surface paint hue when blind, `nullspace_dof == 1`) and refuses to
   hallucinate them.
3. Embodied Motor Parallax (`probe_by_body_motion`):
   When dual sensory loss leaves distance & power statically entangled,
   moving the body (`delta_x`) couples temporal trajectories across steps to
   collapse the vacuum to exact truth.
4. Cross-Organ Self-Calibration (`self_calibrate_eye_via_body`):
   Leave-one-organ-out consensus + proprioceptive motion isolates a distorted
   eye lens and solves for its hidden scale & offset bias autonomously without
   ground-truth supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.orchestration.sensory_orchestra_os import CIE_CONE_MATRIX, planck_spectrum
from core.physics.trinitarian_scale_binding_protocol import (
    CoupledQuantity,
    EqualityBond,
    PinMode,
    ScaleTier,
    TrinitarianScaleRotorSystem,
    UpperRotorVerdict,
)
from core.physics.unit_spectral_mirror_protocol import (
    LivingCausalSuperposition,
    PlanetaryOrderGlobe,
)
from core.topology.hangul_topological_engine import HangulTopologicalEngine

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


SPEED_OF_SOUND_MS: float = 343.0

ORGAN_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "eye": ("eye_apparent_dist", "eye_apparent_vel", "eye_color_warmth", "eye_hue"),
    "ear": ("ear_intensity", "ear_doppler_hz"),
    "hand": ("touch_vib_hz", "touch_radiant_flux", "touch_contact_force"),
}

ALL_SENSORY_CHANNELS: Tuple[str, ...] = (
    "eye_apparent_dist",
    "eye_apparent_vel",
    "eye_color_warmth",
    "eye_hue",
    "ear_intensity",
    "ear_doppler_hz",
    "touch_vib_hz",
    "touch_radiant_flux",
    "touch_contact_force",
)


class OrganismStatus(str, Enum):
    """Unified homeostatic verdict across all sensory & motor organs."""

    LIVING_HARMONY = "living_harmony"
    VACUUM_COMPENSATED = "vacuum_compensated"
    CROSS_ORGAN_CONFLICT = "cross_organ_conflict"


@dataclass
class EmbodiedWorldState:
    """Shared external physical source perceived simultaneously by Eye, Ear, Hand."""

    source_pos_x: float = 5.0
    source_vel_x: float = -15.0
    source_freq_hz: float = 220.0
    source_power_w: float = 40.0
    source_temp_k: float = 3200.0
    source_pigment_hue: float = 0.72

    def generate_sensory_impressions(
        self,
        body_pos_x: float = 0.0,
        body_vel_x: float = 0.0,
        eye_scale_bias: float = 1.0,
        eye_offset_bias: float = 0.0,
    ) -> Dict[str, float]:
        d = max(0.1, float(self.source_pos_x) - float(body_pos_x))
        v_rel = float(self.source_vel_x) - float(body_vel_x)
        return {
            "eye_apparent_dist": d * float(eye_scale_bias) + float(eye_offset_bias),
            "eye_apparent_vel": v_rel,
            "eye_color_warmth": 6500.0 / max(200.0, float(self.source_temp_k)),
            "eye_hue": float(self.source_pigment_hue),
            "ear_intensity": float(self.source_power_w) / (4.0 * math.pi * d * d),
            "ear_doppler_hz": float(self.source_freq_hz)
            * (SPEED_OF_SOUND_MS / max(1.0, SPEED_OF_SOUND_MS + v_rel)),
            "touch_vib_hz": float(self.source_freq_hz),
            "touch_radiant_flux": ((float(self.source_temp_k) / 1000.0) ** 4) / (d * d),
            "touch_contact_force": float(self.source_power_w) / (1.0 + d),
        }

    def synthesize_optical_surface(
        self,
        body_pos_x: float = 0.0,
        size: int = 32,
        include_pigment: bool = True,
    ) -> np.ndarray:
        """Generates a real Planck-illuminated 2D RGB optical surface at `size x size`."""
        d = max(0.2, float(self.source_pos_x) - float(body_pos_x))
        planck_rgb = planck_spectrum(self.source_temp_k) @ CIE_CONE_MATRIX
        planck_rgb = planck_rgb / max(float(np.max(planck_rgb)), 1e-9)

        if include_pigment:
            hue_ang = 2.0 * math.pi * float(self.source_pigment_hue)
            pigment_tint = np.array(
                [
                    1.0 + 0.15 * math.cos(hue_ang),
                    1.0 + 0.15 * math.cos(hue_ang - 2.0 * math.pi / 3.0),
                    1.0 + 0.15 * math.cos(hue_ang - 4.0 * math.pi / 3.0),
                ],
                dtype=np.float64,
            )
            rgb = planck_rgb * pigment_tint
            rgb = rgb / max(float(np.max(rgb)), 1e-9)
        else:
            rgb = planck_rgb

        coords = np.linspace(0.0, 1.0, size, endpoint=False)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        r2 = (yy - 0.5) ** 2 + (xx - 0.5) ** 2
        sigma = 0.35 / (1.0 + 0.25 * d)
        amp = 1.0 / (1.0 + 0.1 * d)
        envelope = 0.08 + amp * np.exp(-0.5 * r2 / max(sigma * sigma, 1e-6))
        img = envelope[:, :, None] * rgb[None, None, :]
        return np.clip(img, 0.0, 1.0).astype(np.float64)

    def synthesize_acoustic_wave(
        self,
        body_pos_x: float = 0.0,
        body_vel_x: float = 0.0,
        n_samples: int = 256,
    ) -> np.ndarray:
        """Generates the physical 1D Doppler-shifted acoustic waveform heard at the ear."""
        imp = self.generate_sensory_impressions(body_pos_x=body_pos_x, body_vel_x=body_vel_x)
        cycles = imp["ear_doppler_hz"] / 20.0
        amp = math.sqrt(max(imp["ear_intensity"], 1e-12))
        t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        return (amp * np.cos(2.0 * math.pi * cycles * t)).astype(np.float64)


@dataclass
class SensoryVacuumReport:
    """Diagnostic breakdown of missing channels, cross-filled values, & honest voids."""

    active_organs: List[str]
    missing_organs: List[str]
    missing_channels: List[str]
    vacuum_filled_variables: Dict[str, float]
    irreducible_vacuum_variables: Dict[str, float]
    vacuum_nullspace_dof: int
    filling_chains: Dict[str, List[str]]


@dataclass
class OrganismPerceptionState:
    """Full living state returned by one embodied perceptual / motor cycle."""

    status: OrganismStatus
    body_pos_x: float
    body_vel_x: float
    inferred_world: Dict[str, float]
    reconstructed_organs: Dict[str, float]
    vacuum_report: SensoryVacuumReport
    reconstructed_optical_surface: np.ndarray
    reconstructed_acoustic_wave: np.ndarray
    visual_mirror: LivingCausalSuperposition
    acoustic_mirror: LivingCausalSuperposition
    trinitarian_system: TrinitarianScaleRotorSystem
    trinitarian_verdict: UpperRotorVerdict
    language_utterance: str
    hangul_wave_coherence: float
    calibrated_biases: Dict[str, float]
    total_action: float

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "organism",
    ) -> Dict[str, Any]:
        """
        Projects the unified living organism (World Core, Eye, Ear, Hand, Motor,
        Vacuum Cavity, Language Self, Upper Self-Rotor) into `CausalField`
        with consistent `(4,)` float32 voxel tensors.
        """
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        missing_org_set = set(self.vacuum_report.missing_organs)
        has_vacuum = len(self.vacuum_report.missing_channels) > 0
        if has_vacuum:
            vacuum_curiosity = min(
                1.0,
                0.6
                + 0.2 * len(missing_org_set)
                + 0.1 * self.vacuum_report.vacuum_nullspace_dof,
            )
        else:
            vacuum_curiosity = 0.0

        conflict_penalty = 1.0 / (1.0 + 10.0 * self.total_action)
        d = max(0.1, self.inferred_world["source_pos_x"] - self.body_pos_x)

        specs = [
            (
                "world_core",
                [
                    self.inferred_world["source_pos_x"] / 10.0,
                    self.inferred_world["source_vel_x"] / 50.0,
                    self.inferred_world["source_power_w"] / 50.0,
                    self.inferred_world["source_temp_k"] / 6500.0,
                ],
                [0.0, 0.0, 0.0],
                0.0,
            ),
            (
                "eye_organ",
                [
                    0.0 if "eye" in missing_org_set else 1.0,
                    self.reconstructed_organs["eye_apparent_dist"] / 10.0,
                    self.reconstructed_organs["eye_color_warmth"] / 3.0,
                    self.reconstructed_organs["eye_hue"],
                ],
                [-1.2, 0.8, 0.5],
                0.0,
            ),
            (
                "ear_organ",
                [
                    0.0 if "ear" in missing_org_set else 1.0,
                    self.reconstructed_organs["ear_intensity"],
                    self.reconstructed_organs["ear_doppler_hz"] / 500.0,
                    d / 10.0,
                ],
                [1.2, 0.8, 0.5],
                0.0,
            ),
            (
                "hand_organ",
                [
                    0.0 if "hand" in missing_org_set else 1.0,
                    self.reconstructed_organs["touch_vib_hz"] / 500.0,
                    self.reconstructed_organs["touch_radiant_flux"] / 10.0,
                    self.reconstructed_organs["touch_contact_force"] / 20.0,
                ],
                [0.0, -1.2, 0.5],
                0.0,
            ),
            (
                "body_motor",
                [
                    self.body_pos_x / 10.0,
                    self.body_vel_x / 50.0,
                    self.calibrated_biases["eye_scale_bias"],
                    self.calibrated_biases["eye_offset_bias"],
                ],
                [0.0, 0.0, -1.0],
                0.0,
            ),
            (
                "vacuum_cavity",
                [
                    float(len(self.vacuum_report.missing_channels)) / float(len(ALL_SENSORY_CHANNELS)),
                    float(self.vacuum_report.vacuum_nullspace_dof),
                    float(vacuum_curiosity),
                    1.0 if has_vacuum else 0.0,
                ],
                [0.0, 1.4, 1.0],
                float(vacuum_curiosity),
            ),
            (
                "language_self",
                [
                    self.hangul_wave_coherence,
                    float(len(self.language_utterance)) / 100.0,
                    conflict_penalty,
                    1.0,
                ],
                [-0.8, -0.8, 1.5],
                0.0,
            ),
            (
                "upper_self_rotor",
                [
                    conflict_penalty,
                    1.0 if self.status == OrganismStatus.LIVING_HARMONY else 0.0,
                    1.0 if self.status == OrganismStatus.VACUUM_COMPENSATED else 0.0,
                    float(self.total_action),
                ],
                [0.0, 0.0, 2.2],
                0.0,
            ),
        ]

        vids: Dict[str, str] = {}
        for key, vec4, pos3, cur in specs:
            vid = f"{prefix}_{key}"
            vids[key] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"component": key, "status": self.status.value},
                    tensor=np.asarray(vec4, dtype=np.float32),
                    mass=1.8 if key in ("world_core", "upper_self_rotor") else 1.0,
                    position=np.asarray(pos3[: field.dimensions], dtype=np.float32),
                    curiosity_charge=float(cur),
                )
            )

        links: List[Tuple[str, str, float]] = [
            ("world_core", "eye_organ", 0.35 if "eye" in missing_org_set else conflict_penalty),
            ("world_core", "ear_organ", 0.35 if "ear" in missing_org_set else conflict_penalty),
            ("world_core", "hand_organ", 0.35 if "hand" in missing_org_set else conflict_penalty),
            ("world_core", "body_motor", 0.95),
            ("world_core", "upper_self_rotor", conflict_penalty),
            ("body_motor", "upper_self_rotor", 0.95),
            ("language_self", "upper_self_rotor", 0.90),
            ("eye_organ", "vacuum_cavity", 0.90 if "eye" in missing_org_set else 0.50),
            ("ear_organ", "vacuum_cavity", 0.90 if has_vacuum else 0.50),
            ("hand_organ", "vacuum_cavity", 0.90 if has_vacuum else 0.50),
            ("vacuum_cavity", "upper_self_rotor", 0.90 if has_vacuum else 0.80),
        ]
        for a, b, st in links:
            field.link_voxels(vids[a], vids[b], strength=float(max(0.05, min(1.0, st))))

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_self",
                name="EmbodiedSelfRotor",
                position=np.array([0.0, 0.0, 2.2][: field.dimensions], dtype=np.float32),
                intensity=2.5,
                tier="macro",
                symbolic_intent=f"Organism status={self.status.value}",
            )
        )

        return {
            "causal_field": field,
            "voxels": vids,
            "beams_linked": len(links),
            "vacuum_curiosity": float(vacuum_curiosity),
            "conflict_beam_strength": float(conflict_penalty),
        }


class LivingCausalOrganism:
    """
    A single embodied living system uniting Eye, Ear, Hand, Body Motor,
    Language, Trinitarian Equality Coupling, and Unit-1 Spectral Mirrors.
    """

    def __init__(
        self,
        name: str = "elysia_living_organism",
        body_pos_x: float = 0.0,
        body_vel_x: float = 0.0,
        eye_scale_bias: float = 1.0,
        eye_offset_bias: float = 0.0,
    ) -> None:
        self.name = name
        self.body_pos_x = float(body_pos_x)
        self.body_vel_x = float(body_vel_x)
        self.eye_scale_bias = float(eye_scale_bias)
        self.eye_offset_bias = float(eye_offset_bias)
        self.hangul_engine = HangulTopologicalEngine(target_dim=64)
        self.perception_history: List[OrganismPerceptionState] = []
        self.last_outlier_organ: Optional[str] = None

        neutral_ref = EmbodiedWorldState(
            source_pos_x=5.0,
            source_vel_x=0.0,
            source_freq_hz=220.0,
            source_power_w=40.0,
            source_temp_k=6500.0,
            source_pigment_hue=0.5,
        )
        self.visual_globe = PlanetaryOrderGlobe(
            name="organism_visual_field",
            modality="optics_2d",
            reference_signal=neutral_ref.synthesize_optical_surface(body_pos_x=0.0, size=32, include_pigment=False),
            support_rel_tol=1e-4,
        )
        self.acoustic_globe = PlanetaryOrderGlobe(
            name="organism_acoustic_field",
            modality="acoustics_1d",
            reference_signal=neutral_ref.synthesize_acoustic_wave(body_pos_x=0.0, body_vel_x=0.0, n_samples=256),
            support_rel_tol=1e-4,
        )

    @staticmethod
    def _neutral_channel_priors() -> Dict[str, float]:
        return {
            "eye_apparent_dist": 3.0,
            "eye_apparent_vel": 0.0,
            "eye_color_warmth": 1.3,
            "eye_hue": 0.5,
            "ear_intensity": 0.1,
            "ear_doppler_hz": 100.0,
            "touch_vib_hz": 100.0,
            "touch_radiant_flux": 5.0,
            "touch_contact_force": 3.0,
        }

    def _active_and_missing(
        self,
        active_organs: Sequence[str],
        disabled_channels: Sequence[str],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        active_org_list = [o for o in ("eye", "ear", "hand") if o in active_organs]
        missing_org_list = [o for o in ("eye", "ear", "hand") if o not in active_organs]
        disabled_set = set(disabled_channels)
        observed: List[str] = []
        missing: List[str] = []
        for org in ("eye", "ear", "hand"):
            for ch in ORGAN_CHANNELS[org]:
                if org in active_org_list and ch not in disabled_set:
                    observed.append(ch)
                else:
                    missing.append(ch)
        return active_org_list, missing_org_list, observed, missing

    def _build_coupled_nervous_system(
        self,
        steps: List[Tuple[float, float, Dict[str, float], Sequence[str]]],
        calibrate_eye: bool = False,
        initial_world_guess: Optional[Dict[str, float]] = None,
    ) -> TrinitarianScaleRotorSystem:
        """
        Constructs the organism's unified multi-modal Trinitarian manifold
        initialized strictly from neutral priors (never oracle truth).
        """
        sys = TrinitarianScaleRotorSystem(name=f"{self.name}_nervous_system")
        max_body_x = max(st[0] for st in steps)
        guess = dict(initial_world_guess or {})
        init_pos = max(max_body_x + 0.5, float(guess.get("source_pos_x", max_body_x + 3.0)))

        # Shared physical world state quantities (FREE)
        sys.add_quantity(CoupledQuantity("source_pos_x", init_pos, ScaleTier.MECHANISM, "m", "state", PinMode.FREE, max_body_x + 0.2, 500.0))
        sys.add_quantity(CoupledQuantity("source_vel_x", float(guess.get("source_vel_x", 0.0)), ScaleTier.MECHANISM, "m/s", "state", PinMode.FREE, -200.0, 200.0))
        sys.add_quantity(CoupledQuantity("source_freq_hz", float(guess.get("source_freq_hz", 100.0)), ScaleTier.MECHANISM, "Hz", "state", PinMode.FREE, 1.0, 20000.0))
        sys.add_quantity(CoupledQuantity("source_power_w", float(guess.get("source_power_w", 10.0)), ScaleTier.MECHANISM, "W", "state", PinMode.FREE, 0.01, 1e5))
        sys.add_quantity(CoupledQuantity("source_temp_k", float(guess.get("source_temp_k", 5000.0)), ScaleTier.MECHANISM, "K", "state", PinMode.FREE, 300.0, 20000.0))
        sys.add_quantity(CoupledQuantity("source_pigment_hue", float(guess.get("source_pigment_hue", 0.5)), ScaleTier.FORMULA, "hue", "state", PinMode.FREE, 0.0, 1.0))

        # Ocular calibration parameters (internal organ model)
        bias_mode = PinMode.FREE if calibrate_eye else PinMode.FIXED_BOUNDARY
        sys.add_quantity(CoupledQuantity("eye_scale_bias", self.eye_scale_bias, ScaleTier.UPPER_ROTOR, "x", "parameter", bias_mode, 0.1, 10.0))
        sys.add_quantity(CoupledQuantity("eye_offset_bias", self.eye_offset_bias, ScaleTier.UPPER_ROTOR, "m", "parameter", bias_mode, -50.0, 50.0))

        priors = self._neutral_channel_priors()
        bounds_map = {
            "eye_apparent_dist": (0.05, 1000.0),
            "eye_apparent_vel": (-300.0, 300.0),
            "eye_color_warmth": (0.05, 50.0),
            "eye_hue": (0.0, 1.0),
            "ear_intensity": (1e-6, 1e6),
            "ear_doppler_hz": (1.0, 25000.0),
            "touch_vib_hz": (1.0, 25000.0),
            "touch_radiant_flux": (1e-6, 1e6),
            "touch_contact_force": (1e-6, 1e6),
        }

        for idx, (bx, bv, obs_vals, obs_ch_set) in enumerate(steps):
            suf = "" if idx == 0 else f"_s{idx}"
            obs_set = set(obs_ch_set)

            sys.add_quantity(CoupledQuantity(f"body_pos_x{suf}", bx, ScaleTier.VALUE, "m", "cause", PinMode.FIXED_BOUNDARY, -500.0, 500.0))
            sys.add_quantity(CoupledQuantity(f"body_vel_x{suf}", bv, ScaleTier.VALUE, "m/s", "cause", PinMode.FIXED_BOUNDARY, -200.0, 200.0))

            for ch in ALL_SENSORY_CHANNELS:
                lo, hi = bounds_map[ch]
                is_obs = ch in obs_set
                val = float(obs_vals[ch]) if is_obs else float(priors[ch])
                mode = PinMode.DIAL if is_obs else PinMode.FREE
                tier = ScaleTier.FORMULA if ch.startswith("eye_") else ScaleTier.MECHANISM
                sys.add_quantity(CoupledQuantity(f"{ch}{suf}", val, tier, "", "dial" if is_obs else "outcome", mode, lo, hi))

            # Register the 9 smooth, Unit-1 normalized cross-modal physical laws
            b_pos = f"body_pos_x{suf}"
            b_vel = f"body_vel_x{suf}"
            e_dist = f"eye_apparent_dist{suf}"
            e_vel = f"eye_apparent_vel{suf}"
            e_warm = f"eye_color_warmth{suf}"
            e_hue = f"eye_hue{suf}"
            a_int = f"ear_intensity{suf}"
            a_dop = f"ear_doppler_hz{suf}"
            t_vib = f"touch_vib_hz{suf}"
            t_rad = f"touch_radiant_flux{suf}"
            t_frc = f"touch_contact_force{suf}"

            sys.add_bond(
                EqualityBond(
                    name=f"eye_dist_law{suf}",
                    tier=ScaleTier.FORMULA,
                    variables=(e_dist, "source_pos_x", b_pos, "eye_scale_bias", "eye_offset_bias"),
                    lhs_fn=lambda v, ed=e_dist: v[ed],
                    rhs_fn=lambda v, bp=b_pos: (v["source_pos_x"] - v[bp]) * v["eye_scale_bias"] + v["eye_offset_bias"],
                    lhs_expr=e_dist,
                    rhs_expr=f"(source_pos_x - {b_pos}) * eye_scale_bias + eye_offset_bias",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"eye_vel_law{suf}",
                    tier=ScaleTier.FORMULA,
                    variables=(e_vel, "source_vel_x", b_vel),
                    lhs_fn=lambda v, ev=e_vel: v[ev] / 10.0,
                    rhs_fn=lambda v, bv_k=b_vel: (v["source_vel_x"] - v[bv_k]) / 10.0,
                    lhs_expr=f"{e_vel} / 10",
                    rhs_expr=f"(source_vel_x - {b_vel}) / 10",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"eye_planck_warmth_law{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=(e_warm, "source_temp_k"),
                    lhs_fn=lambda v, ew=e_warm: (v[ew] * v["source_temp_k"]) / 6500.0,
                    rhs_fn=lambda _v: 1.0,
                    lhs_expr=f"{e_warm} * source_temp_k / 6500",
                    rhs_expr="1",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"eye_pigment_law{suf}",
                    tier=ScaleTier.FORMULA,
                    variables=(e_hue, "source_pigment_hue"),
                    lhs_fn=lambda v, eh=e_hue: v[eh],
                    rhs_fn=lambda v: v["source_pigment_hue"],
                    lhs_expr=e_hue,
                    rhs_expr="source_pigment_hue",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"acoustic_inverse_square{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=(a_int, "source_pos_x", b_pos, "source_power_w"),
                    lhs_fn=lambda v, ai=a_int, bp=b_pos: math.sqrt(max(1e-12, v[ai]) * 4.0 * math.pi) * (v["source_pos_x"] - v[bp]),
                    rhs_fn=lambda v: math.sqrt(max(1e-12, v["source_power_w"])),
                    lhs_expr=f"sqrt({a_int} * 4pi) * d",
                    rhs_expr="sqrt(source_power_w)",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"acoustic_doppler_law{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=(a_dop, "source_vel_x", b_vel, "source_freq_hz"),
                    lhs_fn=lambda v, ad=a_dop, bv_k=b_vel: (v[ad] * (SPEED_OF_SOUND_MS + v["source_vel_x"] - v[bv_k])) / 34300.0,
                    rhs_fn=lambda v: (v["source_freq_hz"] * SPEED_OF_SOUND_MS) / 34300.0,
                    lhs_expr=f"{a_dop} * (c + v_rel) / 34300",
                    rhs_expr="source_freq_hz * c / 34300",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"tactile_vibration_law{suf}",
                    tier=ScaleTier.FORMULA,
                    variables=(t_vib, "source_freq_hz"),
                    lhs_fn=lambda v, tv=t_vib: v[tv] / 100.0,
                    rhs_fn=lambda v: v["source_freq_hz"] / 100.0,
                    lhs_expr=f"{t_vib} / 100",
                    rhs_expr="source_freq_hz / 100",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"stefan_boltzmann_warmth_law{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=(t_rad, "source_pos_x", b_pos, "source_temp_k"),
                    lhs_fn=lambda v, tr=t_rad, bp=b_pos: math.sqrt(max(1e-12, v[tr])) * (v["source_pos_x"] - v[bp]),
                    rhs_fn=lambda v: (v["source_temp_k"] / 1000.0) ** 2,
                    lhs_expr=f"sqrt({t_rad}) * d",
                    rhs_expr="(source_temp_k / 1000)^2",
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"tactile_pressure_law{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=(t_frc, "source_pos_x", b_pos, "source_power_w"),
                    lhs_fn=lambda v, tf=t_frc, bp=b_pos: (v[tf] * (1.0 + (v["source_pos_x"] - v[bp]))) / 10.0,
                    rhs_fn=lambda v: v["source_power_w"] / 10.0,
                    lhs_expr=f"{t_frc} * (1 + d) / 10",
                    rhs_expr="source_power_w / 10",
                )
            )

        return sys

    def _analyze_vacuum_svd(
        self,
        sys: TrinitarianScaleRotorSystem,
        active_organs: List[str],
        missing_organs: List[str],
        observed_channels: List[str],
        missing_channels: List[str],
    ) -> SensoryVacuumReport:
        """
        Uses column-normalized Jacobian SVD right nullspace (zero hardcoded rules!)
        to divide missing channels into cross-modally determined vs irreducible voids.
        """
        eq_bonds = [
            (b, 1.0)
            for b in sys.bonds.values()
            if b.kind == "equality"
        ]
        free_names = [n for n, q in sys.quantities.items() if q.pin_mode == PinMode.FREE]
        vals = sys.values_dict()

        if not free_names or not eq_bonds:
            return SensoryVacuumReport(
                active_organs=active_organs,
                missing_organs=missing_organs,
                missing_channels=missing_channels,
                vacuum_filled_variables={},
                irreducible_vacuum_variables={},
                vacuum_nullspace_dof=0,
                filling_chains={},
            )

        jac = sys._jacobian_subset(vals, free_names, eq_bonds)
        col_norms = np.linalg.norm(jac, axis=0)
        jac_normed = jac.copy()
        for j, cn in enumerate(col_norms):
            if cn > 1e-12:
                jac_normed[:, j] /= cn

        _, s_vals, vt = np.linalg.svd(jac_normed, full_matrices=True)
        thresh = 1e-5 * max(1.0, float(s_vals[0]) if len(s_vals) else 1.0)
        rank = int(np.sum(s_vals > thresh))
        nullity = max(0, len(free_names) - rank)

        indeterminacy: Dict[str, float] = {}
        for j, name in enumerate(free_names):
            if col_norms[j] <= 1e-12:
                indeterminacy[name] = 1.0
            elif nullity > 0:
                indeterminacy[name] = float(np.linalg.norm(vt[rank:, j]))
            else:
                indeterminacy[name] = 0.0

        irreducible: Dict[str, float] = {}
        for n in free_names:
            if indeterminacy[n] >= 1e-3:
                irreducible[n] = float(vals[n])

        vacuum_filled: Dict[str, float] = {}
        for ch in missing_channels:
            if ch in free_names and indeterminacy.get(ch, 0.0) < 1e-3:
                vacuum_filled[ch] = float(vals[ch])

        # Trace causal pathways showing WHICH active sibling channels filled WHICH missing slots
        chains_list = sys.trace_causal_chains(observed_channels, list(vacuum_filled.keys()), max_chains=24)
        filling_chains: Dict[str, List[str]] = {}
        for path in chains_list:
            tgt = path[-1]
            if tgt not in filling_chains:
                filling_chains[tgt] = path

        return SensoryVacuumReport(
            active_organs=active_organs,
            missing_organs=missing_organs,
            missing_channels=missing_channels,
            vacuum_filled_variables=vacuum_filled,
            irreducible_vacuum_variables=irreducible,
            vacuum_nullspace_dof=nullity,
            filling_chains=filling_chains,
        )

    def _narrate_in_hangul(
        self,
        status: OrganismStatus,
        vacuum: SensoryVacuumReport,
    ) -> Tuple[str, float]:
        if status == OrganismStatus.CROSS_ORGAN_CONFLICT:
            utterance = "감각 기관 사이 불일치가 감지되어 눈의 왜곡을 보정해야 합니다."
        elif status == OrganismStatus.VACUUM_COMPENSATED:
            if "eye" in vacuum.missing_organs:
                utterance = "눈이 닫힌 빈자리를 귀와 손의 인과 흐름으로 채웠으나 색소는 공백으로 남깁니다."
            elif "ear" in vacuum.missing_organs:
                utterance = "귀가 닫힌 빈자리를 눈과 손의 공명으로 되살려 소리를 이어 냈습니다."
            else:
                utterance = "비어 있는 감각 채널을 남은 기관들의 연속 인과로 복원했습니다."
        else:
            utterance = "눈과 귀와 손과 몸이 하나의 세계 질서 안에서 조화롭게 공명합니다."

        _, metrics = self.hangul_engine.encode_text(utterance)
        coherence = float(np.mean([m["coherence"] for m in metrics])) if metrics else 0.0
        return utterance, coherence

    def _finalize_perception(
        self,
        sys: TrinitarianScaleRotorSystem,
        active_organs: List[str],
        missing_organs: List[str],
        observed_channels: List[str],
        missing_channels: List[str],
        action: float,
    ) -> OrganismPerceptionState:
        vals = sys.values_dict()
        verdict = sys.observe_and_discern()
        vacuum = self._analyze_vacuum_svd(
            sys=sys,
            active_organs=active_organs,
            missing_organs=missing_organs,
            observed_channels=observed_channels,
            missing_channels=missing_channels,
        )

        if action > 1e-4:
            status = OrganismStatus.CROSS_ORGAN_CONFLICT
        elif len(missing_channels) > 0:
            status = OrganismStatus.VACUUM_COMPENSATED
        else:
            status = OrganismStatus.LIVING_HARMONY

        inferred_world = {
            "source_pos_x": float(vals["source_pos_x"]),
            "source_vel_x": float(vals["source_vel_x"]),
            "source_freq_hz": float(vals["source_freq_hz"]),
            "source_power_w": float(vals["source_power_w"]),
            "source_temp_k": float(vals["source_temp_k"]),
            "source_pigment_hue": float(vals["source_pigment_hue"]),
        }
        reconstructed_organs = {ch: float(vals[ch]) for ch in ALL_SENSORY_CHANNELS}

        pigment_unknown = any(
            k.startswith("eye_hue") or k == "source_pigment_hue"
            for k in vacuum.irreducible_vacuum_variables
        )
        inferred_world_obj = EmbodiedWorldState(**inferred_world)
        recon_optical = inferred_world_obj.synthesize_optical_surface(
            body_pos_x=self.body_pos_x,
            size=32,
            include_pigment=not pigment_unknown,
        )
        recon_acoustic = inferred_world_obj.synthesize_acoustic_wave(
            body_pos_x=self.body_pos_x,
            body_vel_x=self.body_vel_x,
            n_samples=256,
        )

        visual_mirror = self.visual_globe.superpose(recon_optical)
        acoustic_mirror = self.acoustic_globe.superpose(recon_acoustic)
        utterance, h_coh = self._narrate_in_hangul(status, vacuum)

        state = OrganismPerceptionState(
            status=status,
            body_pos_x=self.body_pos_x,
            body_vel_x=self.body_vel_x,
            inferred_world=inferred_world,
            reconstructed_organs=reconstructed_organs,
            vacuum_report=vacuum,
            reconstructed_optical_surface=recon_optical,
            reconstructed_acoustic_wave=recon_acoustic,
            visual_mirror=visual_mirror,
            acoustic_mirror=acoustic_mirror,
            trinitarian_system=sys,
            trinitarian_verdict=verdict,
            language_utterance=utterance,
            hangul_wave_coherence=h_coh,
            calibrated_biases={
                "eye_scale_bias": float(vals["eye_scale_bias"]),
                "eye_offset_bias": float(vals["eye_offset_bias"]),
            },
            total_action=float(action),
        )
        self.perception_history.append(state)
        return state

    def perceive(
        self,
        world: EmbodiedWorldState,
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
        disabled_channels: Sequence[str] = (),
        true_eye_scale_bias: float = 1.0,
        true_eye_offset_bias: float = 0.0,
    ) -> OrganismPerceptionState:
        """
        Single unified perceptual pulse: active channels act as live boundary
        dials while missing organs/channels become FREE vacuum cavities filled
        by sibling organs via Principle of Least Action.
        """
        active_org_list, missing_org_list, obs_ch, miss_ch = self._active_and_missing(
            active_organs, disabled_channels
        )
        raw_imp = world.generate_sensory_impressions(
            body_pos_x=self.body_pos_x,
            body_vel_x=self.body_vel_x,
            eye_scale_bias=true_eye_scale_bias,
            eye_offset_bias=true_eye_offset_bias,
        )
        sys = self._build_coupled_nervous_system(
            steps=[(self.body_pos_x, self.body_vel_x, raw_imp, obs_ch)],
            calibrate_eye=False,
        )
        rep = sys.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
            max_iter=200,
            tol=1e-11,
        )
        return self._finalize_perception(
            sys=sys,
            active_organs=active_org_list,
            missing_organs=missing_org_list,
            observed_channels=obs_ch,
            missing_channels=miss_ch,
            action=rep.final_action,
        )

    def probe_by_body_motion(
        self,
        world: EmbodiedWorldState,
        delta_x: float = 1.5,
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
        disabled_channels: Sequence[str] = (),
        true_eye_scale_bias: float = 1.0,
        true_eye_offset_bias: float = 0.0,
    ) -> OrganismPerceptionState:
        """
        Couples perception with active motor displacement (`x0 -> x0 + delta_x`):
        temporal parallax across the two positions collapses static vacuum
        ambiguities (e.g., distance vs power) into unique physical reality.
        """
        active_org_list, missing_org_list, obs_ch, miss_ch = self._active_and_missing(
            active_organs, disabled_channels
        )
        x0 = self.body_pos_x
        v0 = self.body_vel_x
        imp0 = world.generate_sensory_impressions(x0, v0, true_eye_scale_bias, true_eye_offset_bias)

        x1 = x0 + float(delta_x)
        v1 = v0
        self.body_pos_x = x1
        imp1 = world.generate_sensory_impressions(x1, v1, true_eye_scale_bias, true_eye_offset_bias)

        sys = self._build_coupled_nervous_system(
            steps=[(x0, v0, imp0, obs_ch), (x1, v1, imp1, obs_ch)],
            calibrate_eye=False,
        )
        rep = sys.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
            max_iter=240,
            tol=1e-11,
        )
        return self._finalize_perception(
            sys=sys,
            active_organs=active_org_list,
            missing_organs=missing_org_list,
            observed_channels=obs_ch,
            missing_channels=miss_ch,
            action=rep.final_action,
        )

    def self_calibrate_eye_via_body(
        self,
        world: EmbodiedWorldState,
        delta_x: float = 1.0,
        true_eye_scale_bias: float = 1.6,
        true_eye_offset_bias: float = 1.2,
    ) -> Tuple[OrganismPerceptionState, OrganismPerceptionState]:
        """
        Detects cross-organ conflict, isolates the outlier organ by leave-one-out
        sub-manifold consensus, unlocks the Eye's internal scale & offset dials,
        and converges via two-step embodied motion back into LIVING_HARMONY.
        """
        before = self.perceive(
            world,
            active_organs=("eye", "ear", "hand"),
            true_eye_scale_bias=true_eye_scale_bias,
            true_eye_offset_bias=true_eye_offset_bias,
        )

        # Move body by proprioceptive delta_x so static power/distance degeneracies break,
        # then test leave-one-organ-out consensus across both positions (x0, x1).
        x0 = self.body_pos_x
        v0 = self.body_vel_x
        imp0 = world.generate_sensory_impressions(x0, v0, true_eye_scale_bias, true_eye_offset_bias)

        x1 = x0 + float(delta_x)
        v1 = v0
        self.body_pos_x = x1
        imp1 = world.generate_sensory_impressions(x1, v1, true_eye_scale_bias, true_eye_offset_bias)

        loo_actions: Dict[str, float] = {}
        for drop_org in ("eye", "ear", "hand"):
            pair = tuple(o for o in ("eye", "ear", "hand") if o != drop_org)
            _, _, pair_ch, _ = self._active_and_missing(pair, ())
            pair_sys = self._build_coupled_nervous_system(
                steps=[(x0, v0, imp0, pair_ch), (x1, v1, imp1, pair_ch)],
                calibrate_eye=False,
            )
            pair_rep = pair_sys.relax_least_action(
                active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
                max_iter=200,
                tol=1e-11,
            )
            loo_actions[drop_org] = float(pair_rep.final_action)

        outlier_organ = min(loo_actions, key=lambda k: loo_actions[k])
        self.last_outlier_organ = outlier_organ

        _, _, all_obs_ch, _ = self._active_and_missing(("eye", "ear", "hand"), ())
        sys_cal = self._build_coupled_nervous_system(
            steps=[(x0, v0, imp0, all_obs_ch), (x1, v1, imp1, all_obs_ch)],
            calibrate_eye=(outlier_organ == "eye"),
        )
        rep = sys_cal.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
            max_iter=240,
            tol=1e-11,
        )
        if rep.converged or rep.final_action < 1e-6:
            self.eye_scale_bias = float(sys_cal.quantities["eye_scale_bias"].value)
            self.eye_offset_bias = float(sys_cal.quantities["eye_offset_bias"].value)

        after = self.perceive(
            world,
            active_organs=("eye", "ear", "hand"),
            true_eye_scale_bias=true_eye_scale_bias,
            true_eye_offset_bias=true_eye_offset_bias,
        )
        return before, after
