"""
grounded_rotor_spectrum_layer.py

Grounded Axis-Spectrum-Rotor Layer for Multi-Scale Embodied Perception
=======================================================================
Connects OcularScaleOrgan (L0-L4 + 1:1 Fovea) and MechanoAcousticOrgan to upper
axis, continuous-spectrum, and nematic/phase rotor operators WITHOUT losing
sensorimotor provenance or using random axis perturbations.

Key Capabilities:
1. OpticalStructureRotor: Computes 4-band spatial-frequency spectrum and 2D local
   structure tensor orientation phi_opt + coherence from native 1:1 foveal crops.
2. MechanicalDirectionalRotor: Reconstructs continuum angular stiffness k(theta),
   mean stiffness, anisotropy index, strong fiber axis phi_mech, and weak cleavage
   plane phi_mech + 90 deg strictly from raw force-displacement directional probes.
3. Top-Down Rotor Accommodation & Zero-Shot Frame-Invariant Transfer: Predicts
   orthogonal strong/weak probe axes from optical-mechanical phase lock across
   rotations and linear irradiance scaling (confirming causal cleavage visually).
4. Deterministic Grounded Axis Sprouting & Criteria Revision: When counter-examples
   (e.g., deceptive surface veneer) violate expectations (surprise > 0.75), splits
   the prototype cleanly, deterministically sprouts a normalized discriminative axis
   from the exact physical residual vector, and revises its observation criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.mechano_acoustic_organ import (
    AcousticObservation,
    MechanicalObservation,
    MechanoAcousticOrgan,
)
from core.sensory.ocular_scale_organ import (
    LayerProvenance,
    OcularScaleOrgan,
    OpticalSceneObservation,
    TemporalVisualAnalysis,
    VisualRegion,
)
from core.sensory.visual_causal_reasoner import (
    FourDomainCausalAudit,
    VisualCausalReasoner,
)
from simulators.hidden_multiphysics_world import SensorimotorWorldProtocol

AXIS_FEATURE_NAMES: List[str] = [
    "k_mean_log10",
    "anisotropy_index",
    "optical_mech_phase_lock",
    "weak_yield_detected",
    "acoustic_log10_freq",
    "foveal_lap_hf",
]


def circular_axis_diff_deg(angle_a_deg: float, angle_b_deg: float) -> float:
    """Unsigned smallest angular separation between two undirected axes mod 180 deg."""
    d = abs((float(angle_a_deg) - float(angle_b_deg)) % 180.0)
    return float(min(d, 180.0 - d))


def compute_nematic_phase_lock(phi_a_rad: float, phi_b_rad: float) -> float:
    """Cosine of double-angle phase difference cos(2 * (phi_a - phi_b)) in [-1, 1]."""
    return float(np.cos(2.0 * (float(phi_a_rad) - float(phi_b_rad))))


@dataclass
class OpticalStructureRotor:
    """Foveal multi-band spatial-frequency spectrum & 2D structure-tensor orientation."""

    phi_opt_rad: float
    phi_opt_deg: float
    coherence: float
    rotor_vec: np.ndarray  # (cos(2*phi_opt), sin(2*phi_opt))
    spatial_freq_spectrum: np.ndarray  # [mean_a, dog_coarse_std, dog_fine_std, lap_hf]
    provenance: LayerProvenance


@dataclass
class DirectionalProbeSample:
    """Single directional mechanical push result at probe angle theta."""

    angle_rad: float
    angle_deg: float
    dir_yx: Tuple[float, float]
    mech_obs: MechanicalObservation


@dataclass
class MechanicalDirectionalRotor:
    """Reconstructed second-harmonic directional stiffness & bond-lattice rotor."""

    k_mean_n_m: float
    k_aniso_n_m: float
    anisotropy_index: float
    phi_mech_rad: float
    phi_mech_deg: float
    weak_cleavage_rad: float
    weak_cleavage_deg: float
    rotor_vec: np.ndarray  # (cos(2*phi_mech), sin(2*phi_mech))
    samples: List[DirectionalProbeSample]
    is_anisotropic: bool
    provenance: LayerProvenance


@dataclass
class GroundedDiscriminativeAxis:
    """Deterministic orthogonal/residual axis birthed from cross-modal discrepancy."""

    axis_id: str
    name: str
    feature_names: List[str]
    unit_direction: np.ndarray
    separating_threshold: float
    provenance_organs: List[str]
    genesis_step: int
    genesis_reason: str


@dataclass
class RotorSpectrumPrototype:
    """Rotation- & illumination-invariant multi-modal prototype."""

    prototype_id: str
    k_mean_log10: float
    anisotropy_index: float
    optical_mech_phase_lock: float
    weak_yield_detected: bool
    acoustic_log10_freq: float
    spectrum_centroid: np.ndarray
    confirmation_count: int
    member_regions: List[str] = field(default_factory=list)
    parent_prototype_id: Optional[str] = None
    learned_delta_phi_rad: Optional[float] = None
    strong_k_log10: float = 0.0
    weak_k_log10: float = 0.0


@dataclass
class CriteriaRevisionEvent:
    """Explicit record of how world friction revised an observation criterion."""

    step: int
    target_id: str
    previous_criterion: str
    revised_criterion: str
    causal_trigger: str
    sprouted_axis_id: Optional[str]


@dataclass
class RotorTransferConfirmation:
    """Record of zero-shot rotor-accommodated transfer onto a rotated/dimmed object."""

    step: int
    target_id: str
    source_prototype_id: str
    predicted_strong_deg: float
    predicted_weak_deg: float
    strong_stiffness_n_m: float
    weak_stiffness_n_m: float
    weak_yield_detected: bool
    strong_yield_detected: bool
    cleavage_flow_mag: float
    prediction_surprise: float


@dataclass
class RegionRotorRecord:
    """Complete sensorimotor & rotor trace for one explored interior object region."""

    step: int
    region_id: int
    target_id: str
    centroid_yx_norm: Tuple[float, float]
    resolution_priority: float
    used_rotor_transfer: bool
    optical_rotor: OpticalStructureRotor
    mechanical_rotor: MechanicalDirectionalRotor
    acoustic_obs: AcousticObservation
    strong_push_sample: Optional[DirectionalProbeSample]
    weak_cleavage_sample: Optional[DirectionalProbeSample]
    cleavage_transition: Optional[TemporalVisualAnalysis]
    cleavage_flow_mag: float
    cleavage_confirmed: bool
    optical_mech_phase_lock: float
    prediction_surprise: float
    matched_prototype_id: str


@dataclass
class RotorExplorationTrace:
    """End-to-end record produced by GroundedRotorExplorer."""

    periphery_obs: OpticalSceneObservation
    region_records: List[RegionRotorRecord]
    prototypes: List[RotorSpectrumPrototype]
    sprouted_axes: List[GroundedDiscriminativeAxis]
    criteria_revisions: List[CriteriaRevisionEvent]
    rotor_transfer_confirmations: List[RotorTransferConfirmation]
    final_causal_field: CausalField
    organ_audit: FourDomainCausalAudit


def extract_optical_structure_rotor(
    periph_reg: VisualRegion, fov_obs: OpticalSceneObservation
) -> OpticalStructureRotor:
    """Extract 4-band spatial spectrum and 2D structure-tensor orientation from 1:1 fovea."""
    a_full = fov_obs.l1.luminance_a
    if a_full.shape[0] >= 4 and a_full.shape[1] >= 4:
        lap = (
            np.abs(
                4.0 * a_full[1:-1, 1:-1]
                - a_full[:-2, 1:-1]
                - a_full[2:, 1:-1]
                - a_full[1:-1, :-2]
                - a_full[1:-1, 2:]
            )
            / 4.0
        )
        lap_hf = float(np.mean(lap))
    else:
        lap_hf = 0.0

    spectrum = np.array(
        [
            float(periph_reg.mean_a),
            float(np.std(fov_obs.l1.dog_coarse)),
            float(np.std(fov_obs.l1.dog_fine)),
            lap_hf,
        ],
        dtype=np.float32,
    )

    margin = 6 if min(a_full.shape[:2]) >= 20 else 1
    a = a_full[margin:-margin, margin:-margin]
    gx = 0.5 * (a[1:-1, 2:] - a[1:-1, :-2])
    gy = 0.5 * (a[2:, 1:-1] - a[:-2, 1:-1])

    jxx = float(np.mean(gx**2))
    jyy = float(np.mean(gy**2))
    jxy = float(np.mean(gx * gy))

    # For intensity modulated across normal (-sin(phi), cos(phi)), grad ~ (-sin, cos),
    # so Jyy - Jxx ~ cos(2*phi) and -2*Jxy ~ sin(2*phi).
    phi_opt_rad = float(0.5 * np.arctan2(-2.0 * jxy, jyy - jxx) % np.pi)
    phi_opt_deg = float(np.rad2deg(phi_opt_rad))
    coherence = float(np.hypot(jxx - jyy, 2.0 * jxy) / (jxx + jyy + 1e-8))
    rotor_vec = np.array(
        [float(np.cos(2.0 * phi_opt_rad)), float(np.sin(2.0 * phi_opt_rad))], dtype=np.float32
    )

    prov = LayerProvenance(
        layer_id="OPTICAL_ROTOR_L1",
        name="Foveal 2D Structure Tensor & Spatial Frequency Rotor",
        domain="Vision",
        causal_mechanism=(
            "Compute central luminance gradients (gx, gy) on native 1:1 foveal interior, "
            "form 2x2 structure tensor [Jxx, Jxy; Jxy, Jyy], extract nematic fiber tangent "
            "phi_opt = 0.5*atan2(-2*Jxy, Jyy-Jxx) and coherence ratio."
        ),
        why_necessary=(
            "Coarse periphery averages out fine striation phase; native foveal gradients isolate "
            "irradiance-invariant surface orientation without semantic labels."
        ),
        key_metrics={
            "phi_opt_deg": phi_opt_deg,
            "coherence": coherence,
            "lap_hf": lap_hf,
        },
    )
    return OpticalStructureRotor(
        phi_opt_rad=phi_opt_rad,
        phi_opt_deg=phi_opt_deg,
        coherence=coherence,
        rotor_vec=rotor_vec,
        spatial_freq_spectrum=spectrum,
        provenance=prov,
    )


def reconstruct_mechanical_rotor(
    samples: List[DirectionalProbeSample],
) -> MechanicalDirectionalRotor:
    """
    Reconstruct continuum stiffness k(theta) = k_mean + c2*cos(2*theta) + s2*sin(2*theta)
    from directional mechanical pushes. Handles both 2 orthogonal probes and >=3 sweep angles.
    """
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 directional samples, got {len(samples)}")

    angles = np.array([s.angle_rad for s in samples], dtype=np.float64)
    stiff = np.array([s.mech_obs.effective_stiffness_n_m for s in samples], dtype=np.float64)

    if len(samples) == 2:
        s0, s1 = samples[0], samples[1]
        k0 = float(s0.mech_obs.effective_stiffness_n_m)
        k1 = float(s1.mech_obs.effective_stiffness_n_m)
        k_mean = 0.5 * (k0 + k1)
        k_aniso = 0.5 * abs(k0 - k1)
        anisotropy_index = float(k_aniso / max(k_mean, 1.0))
        strong_s = s0 if k0 >= k1 else s1
        weak_s = s1 if k0 >= k1 else s0
        phi_mech_rad = float(strong_s.angle_rad % np.pi)
        weak_cleavage_rad = float(weak_s.angle_rad % np.pi)
    else:
        design = np.column_stack([np.ones_like(angles), np.cos(2.0 * angles), np.sin(2.0 * angles)])
        coeffs, _, _, _ = np.linalg.lstsq(design, stiff, rcond=None)
        k_mean = max(1.0, float(coeffs[0]))
        c2 = float(coeffs[1])
        s2 = float(coeffs[2])
        k_aniso = float(np.hypot(c2, s2))
        anisotropy_index = float(k_aniso / max(k_mean, 1.0))
        phi_mech_rad = float((0.5 * np.arctan2(s2, c2)) % np.pi)
        weak_cleavage_rad = float((phi_mech_rad + 0.5 * np.pi) % np.pi)

    is_aniso = anisotropy_index > 0.25
    rotor_vec = np.array(
        [float(np.cos(2.0 * phi_mech_rad)), float(np.sin(2.0 * phi_mech_rad))], dtype=np.float32
    )
    prov = LayerProvenance(
        layer_id="MECH_ROTOR_L1",
        name="Continuum Directional Stiffness Harmonic Rotor",
        domain="Mechanics",
        causal_mechanism=(
            "Fit second-harmonic tensor k(theta) = k_mean + c2*cos(2*theta) + s2*sin(2*theta) "
            "to recover principal strong fiber axis phi_mech and orthogonal cleavage plane."
        ),
        why_necessary=(
            "Photons only detect surface striation contrast; directional contact stress reveals "
            "whether internal cohesive bond anisotropy actually aligns or contradicts surface grain."
        ),
        key_metrics={
            "k_mean_n_m": float(k_mean),
            "k_aniso_n_m": float(k_aniso),
            "anisotropy_index": anisotropy_index,
            "phi_mech_deg": float(np.rad2deg(phi_mech_rad)),
        },
    )
    return MechanicalDirectionalRotor(
        k_mean_n_m=float(k_mean),
        k_aniso_n_m=float(k_aniso),
        anisotropy_index=anisotropy_index,
        phi_mech_rad=phi_mech_rad,
        phi_mech_deg=float(np.rad2deg(phi_mech_rad)),
        weak_cleavage_rad=weak_cleavage_rad,
        weak_cleavage_deg=float(np.rad2deg(weak_cleavage_rad)),
        rotor_vec=rotor_vec,
        samples=list(samples),
        is_anisotropic=is_aniso,
        provenance=prov,
    )


def _feature_vector_from_measurements(
    mech_rotor: MechanicalDirectionalRotor,
    phase_lock: float,
    weak_yield: bool,
    acoust: AcousticObservation,
    opt_rotor: OpticalStructureRotor,
) -> np.ndarray:
    return np.array(
        [
            float(np.log10(max(mech_rotor.k_mean_n_m, 1.0))),
            float(mech_rotor.anisotropy_index),
            float(phase_lock),
            1.0 if weak_yield else 0.0,
            float(np.log10(max(acoust.peak_frequency_hz, 20.0))),
            float(opt_rotor.spatial_freq_spectrum[3]),
        ],
        dtype=np.float32,
    )


def _deterministic_sprout_axis(
    axis_id: str,
    name: str,
    vec_pos: np.ndarray,
    vec_neg: np.ndarray,
    provenance_organs: List[str],
    genesis_step: int,
    genesis_reason: str,
) -> GroundedDiscriminativeAxis:
    delta = (np.asarray(vec_pos, dtype=np.float64) - np.asarray(vec_neg, dtype=np.float64)).ravel()
    norm = float(np.linalg.norm(delta))
    if norm > 1e-9:
        unit_dir = (delta / norm).astype(np.float32)
    else:
        unit_dir = np.zeros(len(AXIS_FEATURE_NAMES), dtype=np.float32)
        unit_dir[0] = 1.0
    midpoint = 0.5 * (np.asarray(vec_pos, dtype=np.float64) + np.asarray(vec_neg, dtype=np.float64))
    thresh = float(np.dot(unit_dir.astype(np.float64), midpoint))
    return GroundedDiscriminativeAxis(
        axis_id=axis_id,
        name=name,
        feature_names=list(AXIS_FEATURE_NAMES),
        unit_direction=unit_dir,
        separating_threshold=thresh,
        provenance_organs=provenance_organs,
        genesis_step=genesis_step,
        genesis_reason=genesis_reason,
    )


class GroundedRotorExplorer:
    """
    Self-directing explorer integrating 1:1 foveal structure tensors, continuum
    mechanical directional rotors, top-down rotor accommodation, and deterministic
    grounded discriminative axis sprouting.
    """

    SURPRISE_THRESHOLD: float = 0.75
    PROTOTYPE_MATCH_THRESHOLD: float = 0.55

    def __init__(
        self,
        eye: Optional[OcularScaleOrgan] = None,
        mechano_acoustic: Optional[MechanoAcousticOrgan] = None,
        reasoner: Optional[VisualCausalReasoner] = None,
    ) -> None:
        self.eye = eye if eye is not None else OcularScaleOrgan()
        self.mechano_acoustic = (
            mechano_acoustic if mechano_acoustic is not None else MechanoAcousticOrgan()
        )
        self.reasoner = reasoner if reasoner is not None else VisualCausalReasoner(self.eye)

    def _probe_angle(
        self,
        world: SensorimotorWorldProtocol,
        y_norm: float,
        x_norm: float,
        angle_rad: float,
        force_n: float = 18.0,
    ) -> Tuple[DirectionalProbeSample, np.ndarray]:
        dy = float(np.sin(angle_rad))
        dx = float(np.cos(angle_rad))
        f_t, x_t, post_rgb = world.mechanical_push(
            y_norm, x_norm, force_n=force_n, dir_yx=(dy, dx)
        )
        m_obs = self.mechano_acoustic.observe_push(f_t, x_t)
        sample = DirectionalProbeSample(
            angle_rad=float(angle_rad % np.pi),
            angle_deg=float(np.rad2deg(angle_rad % np.pi)),
            dir_yx=(dy, dx),
            mech_obs=m_obs,
        )
        return sample, post_rgb

    def _measure_cleavage_flow(
        self,
        periph_obs: OpticalSceneObservation,
        post_rgb: np.ndarray,
        region_id: int,
    ) -> Tuple[TemporalVisualAnalysis, float]:
        post_obs = self.eye.observe(post_rgb)
        temporal = self.eye.observe_transition(periph_obs, post_obs)
        mask = periph_obs.region_labels == region_id
        speed_map = np.hypot(temporal.flow_yx[..., 0], temporal.flow_yx[..., 1])
        if np.any(mask):
            flow_mag = float(np.percentile(speed_map[mask], 85))
        else:
            flow_mag = 0.0
        return temporal, flow_mag

    def _run_four_angle_sweep(
        self,
        world: SensorimotorWorldProtocol,
        y_norm: float,
        x_norm: float,
        force_n: float = 18.0,
    ) -> MechanicalDirectionalRotor:
        samples: List[DirectionalProbeSample] = []
        for deg in (0.0, 45.0, 90.0, 135.0):
            ang = float(np.deg2rad(deg))
            smp, _ = self._probe_angle(world, y_norm, x_norm, ang, force_n=force_n)
            samples.append(smp)
        return reconstruct_mechanical_rotor(samples)

    def _select_anisotropic_prior(
        self, prototypes: List[RotorSpectrumPrototype]
    ) -> Optional[RotorSpectrumPrototype]:
        aniso = [
            p
            for p in prototypes
            if p.anisotropy_index > 0.25 and p.learned_delta_phi_rad is not None
        ]
        if not aniso:
            return None
        # Prefer phase-aligned & higher-confirmation priors, breaking ties deterministically
        aniso.sort(key=lambda p: (-p.confirmation_count, -p.optical_mech_phase_lock, p.prototype_id))
        return aniso[0]

    def explore(self, world: SensorimotorWorldProtocol) -> RotorExplorationTrace:
        if not world.supports_physical_probing():
            raise ValueError("GroundedRotorExplorer requires physical probing capability.")

        periph_rgb = world.observe_periphery()
        periph_obs = self.eye.observe(periph_rgb)
        h, w = periph_obs.shape_hw
        scene_med_a = float(np.median(periph_obs.l1.luminance_a))

        # Select coherent non-backdrop interior regions scored purely from optical observables
        interior_regions = [r for r in periph_obs.regions if r.area_ratio <= 0.40]
        scored_regions: List[Tuple[float, VisualRegion]] = []
        for reg in interior_regions:
            prio = float(
                abs(reg.mean_a - scene_med_a)
                + 0.30 * reg.boundary_persistence
                + 0.25 * min(reg.fine_texture_energy, 1.5)
            )
            scored_regions.append((prio, reg))
        scored_regions.sort(key=lambda pair: (-pair[0], pair[1].region_id))

        prototypes: List[RotorSpectrumPrototype] = []
        sprouted_axes: List[GroundedDiscriminativeAxis] = []
        criteria_revisions: List[CriteriaRevisionEvent] = []
        transfer_confirmations: List[RotorTransferConfirmation] = []
        region_records: List[RegionRotorRecord] = []

        for step_idx, (prio, reg) in enumerate(scored_regions, 1):
            yn = float(reg.centroid_yx[0] / max(1.0, float(h)))
            xn = float(reg.centroid_yx[1] / max(1.0, float(w)))
            target_id = f"region_{reg.region_id}"

            # 1. Optical Foveation + Structure Tensor & Acoustic Resonance Tap
            fov_rgb = world.foveate(yn, xn, fov_frac=0.10)
            fov_obs = self.eye.observe(fov_rgb, max_width=240)
            opt_rotor = extract_optical_structure_rotor(reg, fov_obs)
            wave = world.acoustic_tap(yn, xn, impulse_n=10.0)
            acoust_obs = self.mechano_acoustic.observe_tap(wave)

            prior_proto = self._select_anisotropic_prior(prototypes)
            opt_coherent = opt_rotor.coherence > 0.25

            used_transfer = False
            surprise_err = 0.0
            strong_smp: Optional[DirectionalProbeSample] = None
            weak_smp: Optional[DirectionalProbeSample] = None
            cleavage_trans: Optional[TemporalVisualAnalysis] = None
            cleavage_flow_mag = 0.0
            cleavage_confirmed = False

            if opt_coherent and prior_proto is not None:
                # Top-Down Rotor Accommodation: predict strong fiber & weak cleavage axes
                assert prior_proto.learned_delta_phi_rad is not None
                theta_strong_pred = float((opt_rotor.phi_opt_rad + prior_proto.learned_delta_phi_rad) % np.pi)
                theta_weak_pred = float((theta_strong_pred + 0.5 * np.pi) % np.pi)

                smp_str, _ = self._probe_angle(world, yn, xn, theta_strong_pred, force_n=18.0)
                smp_wk, post_wk_rgb = self._probe_angle(world, yn, xn, theta_weak_pred, force_n=18.0)

                k_str_log = float(np.log10(max(smp_str.mech_obs.effective_stiffness_n_m, 1.0)))
                k_wk_log = float(np.log10(max(smp_wk.mech_obs.effective_stiffness_n_m, 1.0)))
                err_k = abs(k_str_log - prior_proto.strong_k_log10) + abs(
                    k_wk_log - prior_proto.weak_k_log10
                )
                err_yield = (
                    0.0
                    if (smp_wk.mech_obs.yield_detected and not smp_str.mech_obs.yield_detected)
                    else 1.5
                )
                surprise_err = float(err_k + err_yield)

                if (
                    surprise_err <= self.SURPRISE_THRESHOLD
                    and smp_wk.mech_obs.yield_detected
                    and not smp_str.mech_obs.yield_detected
                ):
                    # Zero-Shot Frame-Invariant Rotor Transfer Confirmed!
                    used_transfer = True
                    mech_rotor = reconstruct_mechanical_rotor([smp_str, smp_wk])
                    strong_smp = smp_str
                    weak_smp = smp_wk
                    cleavage_trans, cleavage_flow_mag = self._measure_cleavage_flow(
                        periph_obs, post_wk_rgb, reg.region_id
                    )
                    cleavage_confirmed = bool(
                        smp_wk.mech_obs.yield_detected
                        and not smp_str.mech_obs.yield_detected
                        and cleavage_flow_mag >= 0.8
                    )
                    transfer_confirmations.append(
                        RotorTransferConfirmation(
                            step=step_idx,
                            target_id=target_id,
                            source_prototype_id=prior_proto.prototype_id,
                            predicted_strong_deg=float(np.rad2deg(theta_strong_pred)),
                            predicted_weak_deg=float(np.rad2deg(theta_weak_pred)),
                            strong_stiffness_n_m=float(smp_str.mech_obs.effective_stiffness_n_m),
                            weak_stiffness_n_m=float(smp_wk.mech_obs.effective_stiffness_n_m),
                            weak_yield_detected=bool(smp_wk.mech_obs.yield_detected),
                            strong_yield_detected=bool(smp_str.mech_obs.yield_detected),
                            cleavage_flow_mag=cleavage_flow_mag,
                            prediction_surprise=surprise_err,
                        )
                    )
                else:
                    # Counter-example falsified naive prior -> full 4-angle mechanical sweep!
                    mech_rotor = self._run_four_angle_sweep(world, yn, xn, force_n=18.0)
                    strong_smp, _ = self._probe_angle(
                        world, yn, xn, mech_rotor.phi_mech_rad, force_n=18.0
                    )
                    weak_smp, post_wk_rgb = self._probe_angle(
                        world, yn, xn, mech_rotor.weak_cleavage_rad, force_n=18.0
                    )
                    cleavage_trans, cleavage_flow_mag = self._measure_cleavage_flow(
                        periph_obs, post_wk_rgb, reg.region_id
                    )
                    cleavage_confirmed = bool(
                        weak_smp.mech_obs.yield_detected
                        and not strong_smp.mech_obs.yield_detected
                        and cleavage_flow_mag >= 0.8
                    )
            else:
                # No anisotropic prior yet or optically isotropic -> full 4-angle rotor sweep
                mech_rotor = self._run_four_angle_sweep(world, yn, xn, force_n=18.0)
                if mech_rotor.is_anisotropic:
                    strong_smp, _ = self._probe_angle(
                        world, yn, xn, mech_rotor.phi_mech_rad, force_n=18.0
                    )
                    weak_smp, post_wk_rgb = self._probe_angle(
                        world, yn, xn, mech_rotor.weak_cleavage_rad, force_n=18.0
                    )
                    cleavage_trans, cleavage_flow_mag = self._measure_cleavage_flow(
                        periph_obs, post_wk_rgb, reg.region_id
                    )
                    cleavage_confirmed = bool(
                        weak_smp.mech_obs.yield_detected
                        and not strong_smp.mech_obs.yield_detected
                        and cleavage_flow_mag >= 0.8
                    )
                else:
                    strong_smp, post_strong = self._probe_angle(world, yn, xn, 0.0, force_n=18.0)
                    cleavage_trans, cleavage_flow_mag = self._measure_cleavage_flow(
                        periph_obs, post_strong, reg.region_id
                    )
                    cleavage_confirmed = False

            phase_lock = (
                compute_nematic_phase_lock(opt_rotor.phi_opt_rad, mech_rotor.phi_mech_rad)
                if (opt_coherent and mech_rotor.is_anisotropic)
                else 0.0
            )
            weak_yield_flag = bool(weak_smp is not None and weak_smp.mech_obs.yield_detected)
            feat_vec = _feature_vector_from_measurements(
                mech_rotor=mech_rotor,
                phase_lock=phase_lock,
                weak_yield=weak_yield_flag,
                acoust=acoust_obs,
                opt_rotor=opt_rotor,
            )

            # Match or branch RotorSpectrumPrototype in invariant space (never average counter-examples!)
            matched_proto: Optional[RotorSpectrumPrototype] = None
            best_dist = 1e9
            for p in prototypes:
                d = float(np.linalg.norm(feat_vec[:5] - p.spectrum_centroid[:5]))
                if d < best_dist:
                    best_dist = d
                    matched_proto = p

            strong_k_log_val = float(
                np.log10(
                    max(
                        strong_smp.mech_obs.effective_stiffness_n_m
                        if strong_smp is not None
                        else mech_rotor.k_mean_n_m,
                        1.0,
                    )
                )
            )
            weak_k_log_val = float(
                np.log10(
                    max(
                        weak_smp.mech_obs.effective_stiffness_n_m
                        if weak_smp is not None
                        else mech_rotor.k_mean_n_m,
                        1.0,
                    )
                )
            )
            delta_phi_val = (
                float((mech_rotor.phi_mech_rad - opt_rotor.phi_opt_rad) % np.pi)
                if (opt_coherent and mech_rotor.is_anisotropic)
                else None
            )

            if matched_proto is not None and best_dist <= self.PROTOTYPE_MATCH_THRESHOLD and surprise_err <= self.SURPRISE_THRESHOLD:
                n_prev = float(matched_proto.confirmation_count)
                matched_proto.spectrum_centroid = (
                    matched_proto.spectrum_centroid * n_prev + feat_vec
                ) / (n_prev + 1.0)
                matched_proto.k_mean_log10 = float(matched_proto.spectrum_centroid[0])
                matched_proto.anisotropy_index = float(matched_proto.spectrum_centroid[1])
                matched_proto.optical_mech_phase_lock = float(matched_proto.spectrum_centroid[2])
                matched_proto.acoustic_log10_freq = float(matched_proto.spectrum_centroid[4])
                matched_proto.confirmation_count += 1
                matched_proto.member_regions.append(target_id)
                assigned_proto_id = matched_proto.prototype_id
            else:
                is_split = surprise_err > self.SURPRISE_THRESHOLD and prior_proto is not None
                new_id = (
                    f"proto_{len(prototypes)}_split" if is_split else f"proto_{len(prototypes)}"
                )
                parent_id = prior_proto.prototype_id if is_split and prior_proto is not None else None
                new_proto = RotorSpectrumPrototype(
                    prototype_id=new_id,
                    k_mean_log10=float(feat_vec[0]),
                    anisotropy_index=float(feat_vec[1]),
                    optical_mech_phase_lock=float(feat_vec[2]),
                    weak_yield_detected=weak_yield_flag,
                    acoustic_log10_freq=float(feat_vec[4]),
                    spectrum_centroid=feat_vec.copy(),
                    confirmation_count=1,
                    member_regions=[target_id],
                    parent_prototype_id=parent_id,
                    learned_delta_phi_rad=delta_phi_val,
                    strong_k_log10=strong_k_log_val,
                    weak_k_log10=weak_k_log_val,
                )
                prototypes.append(new_proto)
                assigned_proto_id = new_id

                # Deterministic Grounded Axis Sprouting:
                # Case A: Phase anti-lock counter-example -> sprout optical-mechanical phase lock axis
                if is_split and prior_proto is not None:
                    axis_id = f"axis_{len(sprouted_axes)}_optical_mechanical_phase_lock"
                    axis = _deterministic_sprout_axis(
                        axis_id=axis_id,
                        name="Optical-Mechanical Nematic Phase-Lock Axis",
                        vec_pos=prior_proto.spectrum_centroid,
                        vec_neg=feat_vec,
                        provenance_organs=[
                            "OcularScaleOrgan.L1_FovealStructureTensor",
                            "MechanicalContactOrgan.DirectionalRotor",
                            "AcousticResonanceOrgan.TapFFT",
                        ],
                        genesis_step=step_idx,
                        genesis_reason=(
                            f"Accommodated optical grain prediction failed on {target_id} "
                            f"(surprise={surprise_err:.3f} > {self.SURPRISE_THRESHOLD}; "
                            f"rho_rotor={phase_lock:+.3f} vs prior={prior_proto.optical_mech_phase_lock:+.3f})."
                        ),
                    )
                    sprouted_axes.append(axis)
                    criteria_revisions.append(
                        CriteriaRevisionEvent(
                            step=step_idx,
                            target_id=target_id,
                            previous_criterion=(
                                "Unconditional Optical Grain Rule: assume phi_mech == phi_opt "
                                "whenever surface striations exhibit coherence > 0.25."
                            ),
                            revised_criterion=(
                                "Defeasible Optical-Mechanical Phase-Lock Prior: project expected "
                                "phi_mech from phi_opt, ALWAYS test strong/weak orthogonal probes "
                                "via top-down accommodation, and decouple interior structure along "
                                f"{axis_id} when surprise > {self.SURPRISE_THRESHOLD:.2f}."
                            ),
                            causal_trigger=(
                                f"Counter-example {target_id} yielded along optically strong grain "
                                f"(phi_opt={opt_rotor.phi_opt_deg:.1f} deg, phi_mech={mech_rotor.phi_mech_deg:.1f} deg, "
                                f"rho_rotor={phase_lock:+.3f})."
                            ),
                            sprouted_axis_id=axis_id,
                        )
                    )
                # Case B: Structural Anisotropy vs Isotropic contrast discovered for the first time
                else:
                    existing_axis_names = {a.name for a in sprouted_axes}
                    if "Structural Anisotropy & Cleavage Axis" not in existing_axis_names:
                        contrast_proto = next(
                            (
                                p
                                for p in prototypes[:-1]
                                if (p.anisotropy_index > 0.25) != mech_rotor.is_anisotropic
                            ),
                            None,
                        )
                        if contrast_proto is not None:
                            axis_id = f"axis_{len(sprouted_axes)}_structural_anisotropy_cleavage"
                            vec_a = feat_vec if mech_rotor.is_anisotropic else contrast_proto.spectrum_centroid
                            vec_b = contrast_proto.spectrum_centroid if mech_rotor.is_anisotropic else feat_vec
                            sprouted_axes.append(
                                _deterministic_sprout_axis(
                                    axis_id=axis_id,
                                    name="Structural Anisotropy & Cleavage Axis",
                                    vec_pos=vec_a,
                                    vec_neg=vec_b,
                                    provenance_organs=[
                                        "MechanicalContactOrgan.DirectionalRotor",
                                        "AcousticResonanceOrgan.TapFFT",
                                    ],
                                    genesis_step=step_idx,
                                    genesis_reason=(
                                        f"Discovered empirical separation between directional cleavage solids "
                                        f"and isotropic stiff solids at {target_id}."
                                    ),
                                )
                            )

            region_records.append(
                RegionRotorRecord(
                    step=step_idx,
                    region_id=reg.region_id,
                    target_id=target_id,
                    centroid_yx_norm=(yn, xn),
                    resolution_priority=prio,
                    used_rotor_transfer=used_transfer,
                    optical_rotor=opt_rotor,
                    mechanical_rotor=mech_rotor,
                    acoustic_obs=acoust_obs,
                    strong_push_sample=strong_smp,
                    weak_cleavage_sample=weak_smp,
                    cleavage_transition=cleavage_trans,
                    cleavage_flow_mag=cleavage_flow_mag,
                    cleavage_confirmed=cleavage_confirmed,
                    optical_mech_phase_lock=phase_lock,
                    prediction_surprise=surprise_err,
                    matched_prototype_id=assigned_proto_id,
                )
            )

        causal_field = self._project_rotors_and_axes_to_causal_field(
            periph_obs=periph_obs,
            region_records=region_records,
            prototypes=prototypes,
            sprouted_axes=sprouted_axes,
        )
        audit = self.reasoner.audit_organ_causality(periph_obs)

        return RotorExplorationTrace(
            periphery_obs=periph_obs,
            region_records=region_records,
            prototypes=prototypes,
            sprouted_axes=sprouted_axes,
            criteria_revisions=criteria_revisions,
            rotor_transfer_confirmations=transfer_confirmations,
            final_causal_field=causal_field,
            organ_audit=audit,
        )

    def _project_rotors_and_axes_to_causal_field(
        self,
        periph_obs: OpticalSceneObservation,
        region_records: List[RegionRotorRecord],
        prototypes: List[RotorSpectrumPrototype],
        sprouted_axes: List[GroundedDiscriminativeAxis],
    ) -> CausalField:
        field_obj = self.reasoner.project_to_causal_field(periph_obs)

        # Level 8.0: Rotor-aligned Cross-Modal Prototypes linked to member optical regions
        for idx, proto in enumerate(prototypes):
            p_vid = f"proto_{proto.prototype_id}"
            v_proto = InformationVoxel(
                id=p_vid,
                content={
                    "type": "rotor_spectrum_prototype",
                    "prototype_id": proto.prototype_id,
                    "confirmations": proto.confirmation_count,
                    "parent": proto.parent_prototype_id,
                    "phase_lock": proto.optical_mech_phase_lock,
                },
                tensor=proto.spectrum_centroid.astype(np.float32),
                mass=max(0.8, float(proto.confirmation_count)),
                position=np.array([0.2 + 0.22 * idx, 0.85, 0.7], dtype=np.float32),
                scale_level=8.0,
            )
            field_obj.add_voxel(v_proto)

            for r_id in proto.member_regions:
                if r_id in field_obj.voxels:
                    field_obj.link_voxels(r_id, p_vid, strength=2.4)

        # Level 10.0: Deterministically Sprouted Grounded Axes linked back to prototypes
        for idx, axis in enumerate(sprouted_axes):
            a_vid = axis.axis_id
            v_axis = InformationVoxel(
                id=a_vid,
                content={
                    "type": "grounded_discriminative_axis",
                    "name": axis.name,
                    "threshold": axis.separating_threshold,
                    "provenance": axis.provenance_organs,
                },
                tensor=axis.unit_direction.astype(np.float32),
                mass=1.8,
                position=np.array([0.25 + 0.35 * idx, 0.92, 0.9], dtype=np.float32),
                scale_level=10.0,
            )
            field_obj.add_voxel(v_axis)
            for proto in prototypes:
                p_vid = f"proto_{proto.prototype_id}"
                if p_vid in field_obj.voxels:
                    field_obj.link_voxels(p_vid, a_vid, strength=1.8)

        field_obj.step(dt=0.05)
        return field_obj
