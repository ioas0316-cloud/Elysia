"""
Causal Mechanism Palette & Rectification Engine
===============================================

Treats verified scientific laws, existing Elysia code modules, and distilled
experiential memories as executable, composable "Palette Pigments" (물감) over
`SpectralScaleRotorManifold`, eliminating from-scratch formula re-derivation and
1-to-100 point-grid re-simulation.

4 Continuities & Chromatic Interconnectedness (per AGENTS.md):
- Relationship: Incoming observations and transitions are rectified against the
  palette's continuous L^2 resonance Gram matrix rather than discrete thresholds.
- Connectivity: Vertical scale hierarchies and co-activated palette pigments project
  into `CausalField` as `EngramAttractor` + `InformationVoxel` + `ConnectivityBeam`
  networks.
- Mobility: Every palette operator acts directly in O(K) spectral rotor space and
  injects novelty-driven momentum vectors into `CausalField`.
- Informational Continuity: Unexplained causal friction (> novelty_threshold) is
  distilled into reusable memory pigments so subsequent encounters recall and
  transfer the learned causal law in O(K).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.physics.clifford_spacetime_rotor import CliffordSpacetimeRotor
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

OperatorFn = Callable[
    [SpectralScaleRotorManifold, Dict[str, Any]], SpectralScaleRotorManifold
]


@dataclass
class CausalMechanismPigment:
    """
    A single causal law, existing code operator, or distilled memory preserved
    in the palette for immediate O(K) composition and rectification.
    """

    pigment_id: str
    discipline: str
    origin: str
    description: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    operator: Optional[OperatorFn] = None
    archetype_signature: Optional[SpectralScaleRotorManifold] = None
    dim: Optional[int] = None
    activation_count: int = 0
    chromatic_affinity: Tuple[float, float, float] = (0.33, 0.34, 0.33)

    def apply(
        self,
        manifold: SpectralScaleRotorManifold,
        params: Optional[Dict[str, Any]] = None,
    ) -> SpectralScaleRotorManifold:
        """Executes this pigment directly on `manifold` in O(K) rotor space."""
        if self.dim is not None and manifold.dim != self.dim:
            raise ValueError(
                f"Pigment '{self.pigment_id}' requires dim={self.dim}, got dim={manifold.dim}"
            )
        merged: Dict[str, Any] = dict(self.default_params)
        if params:
            merged.update(params)

        self.activation_count += 1
        if self.operator is not None:
            return self.operator(manifold, merged)
        if self.archetype_signature is not None:
            return self.archetype_signature.copy()
        return manifold.copy()


@dataclass
class RectificationResult:
    """Outcome of projecting an observation or causal transition onto the palette."""

    recipe_weights: Dict[str, float]
    explained_ratio: float
    residual_friction: float
    from_memory: bool
    distilled_pigment_id: Optional[str]
    reconstructed_manifold: SpectralScaleRotorManifold
    residual_manifold: SpectralScaleRotorManifold
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Built-in O(K) Disciplinary & Existing-Code Operators (Palette Primaries)
# ---------------------------------------------------------------------------


def op_thermal_diffusion(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Exact Heat PDE evolution (du/dt = D * Laplacian u) in O(K) without time-stepping:
    - Global modes decay exponentially: exp(-D * tau * ||w||^2).
    - Localized Gaussian wave-packets widen exactly: sigma' = sqrt(sigma^2 + 2*D*tau).
    """
    diffusivity = max(0.0, float(params.get("diffusivity", 0.002)))
    tau = max(0.0, float(params.get("tau", 1.0)))
    dt_eff = diffusivity * tau

    new_bands: List[ScaleRotorBand] = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0 and dt_eff > 0.0:
            w_sq = np.sum(bc.wavevectors ** 2, axis=1)
            if bc.is_global:
                bc.amplitudes = bc.amplitudes * np.exp(-dt_eff * w_sq)
            else:
                s2 = bc.sigmas ** 2
                s2_new = s2 + 2.0 * dt_eff
                ratio = s2 / s2_new
                w_new = ratio[:, None] * bc.wavevectors
                phase_preserve = np.sum((bc.wavevectors - w_new) * bc.centers, axis=1)

                bc.sigmas = np.sqrt(s2_new)
                bc.wavevectors = w_new
                bc.amplitudes = (
                    bc.amplitudes
                    * (ratio ** (0.5 * manifold.dim))
                    * np.exp(-dt_eff * ratio * w_sq + 1j * phase_preserve)
                )
        new_bands.append(bc)

    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=manifold.balance,
        current_time=manifold.current_time + tau,
    )


def op_damped_harmonic(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Exact underdamped harmonic oscillator evolution in O(K):
    x'' + 2*gamma*x' + omega_0^2*x = 0.
    """
    gamma = max(0.0, float(params.get("damping", 0.15)))
    tau = float(params.get("tau", 1.0))

    new_bands: List[ScaleRotorBand] = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            omega_eff = np.sqrt(np.maximum(bc.temporal_omegas ** 2 - gamma ** 2, 0.0))
            bc.amplitudes = bc.amplitudes * np.exp((-gamma - 1j * omega_eff) * tau)
        new_bands.append(bc)

    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=manifold.balance,
        current_time=manifold.current_time + tau,
    )


def op_doppler_advection(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Exact linear advection / Doppler shift (du/dt + v . grad u = 0) in O(K).
    """
    tau = float(params.get("tau", 0.1))
    vel_in = params.get("velocity", None)
    if vel_in is None:
        vel = np.full(manifold.dim, 0.25, dtype=np.float64)
    else:
        vel = np.asarray(vel_in, dtype=np.float64).reshape(manifold.dim)
    return manifold.evolve_spacetime(dt=tau, velocity=vel)


def op_lateral_inhibition(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Retinal / cortical center-surround Difference-of-Gaussians (DoG) contrast
    enhancement in O(K) wavevector space, suppressing uniform DC background while
    sharpening structural edges and contours.
    """
    sigma_c = max(1e-6, float(params.get("sigma_center", 0.02)))
    sigma_s = max(sigma_c + 1e-6, float(params.get("sigma_surround", 0.08)))
    alpha = float(params.get("surround_weight", 0.85))
    contrast_boost = float(params.get("contrast_boost", 0.15))

    new_bands: List[ScaleRotorBand] = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            w_sq = np.sum(bc.wavevectors ** 2, axis=1)
            gain = np.exp(-0.5 * (sigma_c ** 2) * w_sq) - alpha * np.exp(
                -0.5 * (sigma_s ** 2) * w_sq
            )
            bc.amplitudes = bc.amplitudes * gain
        new_bands.append(bc)

    new_balance = SpectralBalanceScale(
        scale_tilt=manifold.balance.scale_tilt,
        contrast_tilt=float(np.clip(manifold.balance.contrast_tilt + contrast_boost, -1.0, 1.0)),
        phase_bias=manifold.balance.phase_bias,
        granularity_scale=manifold.balance.granularity_scale,
    )
    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=new_balance,
        current_time=manifold.current_time,
    )


def op_harmonic_resonance(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Acoustic overtone resonance filter in O(K): reinforces modes aligned with
    integer multiples n * fundamental while attenuating off-harmonic dissonance.
    """
    fundamental = max(1e-6, float(params.get("fundamental", 2.0 * math.pi * 3.0)))
    max_overtones = max(1, int(params.get("max_overtones", 6)))
    bandwidth = max(1e-6, float(params.get("bandwidth", 2.0)))
    floor_gain = float(np.clip(params.get("floor_gain", 0.05), 0.0, 1.0))

    harmonics = fundamental * np.arange(1, max_overtones + 1, dtype=np.float64)
    new_bands: List[ScaleRotorBand] = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            w_mag = np.linalg.norm(bc.wavevectors, axis=1)  # (K,)
            dists = w_mag[:, None] - harmonics[None, :]
            res = np.max(np.exp(-0.5 * (dists / bandwidth) ** 2), axis=1)
            gain = floor_gain + (1.0 - floor_gain) * res
            bc.amplitudes = bc.amplitudes * gain
        new_bands.append(bc)

    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=manifold.balance,
        current_time=manifold.current_time,
    )


def op_snell_refraction(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Optical refractive index dispersion (k -> n*k) coupled with Clifford rotor
    deflection in O(K), decreasing phase velocity while bending wavefront direction.
    """
    index_ratio = max(1e-6, float(params.get("index_ratio", 1.25)))
    angle = float(params.get("angle", 0.25))

    low, high = manifold.domain_bounds
    mid = 0.5 * (low + high)
    rotor = (
        CliffordSpacetimeRotor.create_rotor_3d(np.array([0.0, 0.0, 1.0]), angle)
        if manifold.dim >= 2 and abs(angle) > 1e-12
        else None
    )

    new_bands: List[ScaleRotorBand] = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            w_scaled = bc.wavevectors * index_ratio
            if rotor is not None:
                w3 = np.zeros((bc.num_modes, 3), dtype=np.float64)
                w3[:, :2] = w_scaled[:, :2]
                w3_rot = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, w3)
                w_scaled[:, :2] = w3_rot[:, :2]

                if not bc.is_global:
                    c_rel = bc.centers[:, :2] - mid[None, :2]
                    c3 = np.zeros((bc.num_modes, 3), dtype=np.float64)
                    c3[:, :2] = c_rel
                    c3_rot = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, c3)
                    bc.centers[:, :2] = c3_rot[:, :2] + mid[None, :2]

            delta_w = w_scaled - bc.wavevectors
            bc.amplitudes = bc.amplitudes * np.exp(-1j * (delta_w @ mid))
            bc.wavevectors = w_scaled
        new_bands.append(bc)

    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=manifold.balance,
        current_time=manifold.current_time,
    )


def op_clifford_rotor_twist(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Existing Elysia Code Module (`CliffordSpacetimeRotor`) wrapped as an O(K)
    reversible geometric rotor twist pigment:
    - In dim >= 2: rotates wavefronts & localized centers rigidly around domain
      midpoint using Clifford sandwich product v' = R v R~ (exact rewind via R~ v' R).
    - In dim == 1: rotates complex rotor amplitudes in spinor plane (Re, Im, 0).
    """
    angle = float(params.get("angle", 0.35))
    scale = max(1e-9, float(params.get("scale", 1.0)))
    rewind = bool(params.get("rewind", False))
    axis = np.asarray(params.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64).reshape(3)
    rotor = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    low, high = manifold.domain_bounds
    mid = 0.5 * (low + high)
    new_bands: List[ScaleRotorBand] = []

    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            if manifold.dim >= 2:
                w3 = np.zeros((bc.num_modes, 3), dtype=np.float64)
                w3[:, :2] = bc.wavevectors[:, :2]
                if rewind:
                    w3_out = CliffordSpacetimeRotor.rewind_transform_vector_3d(rotor, w3)
                else:
                    w3_out = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, w3)
                w_new = bc.wavevectors.copy()
                w_new[:, :2] = w3_out[:, :2]

                if not bc.is_global:
                    c3 = np.zeros((bc.num_modes, 3), dtype=np.float64)
                    c3[:, :2] = bc.centers[:, :2] - mid[None, :2]
                    if rewind:
                        c3_out = CliffordSpacetimeRotor.rewind_transform_vector_3d(rotor, c3)
                    else:
                        c3_out = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, c3)
                    bc.centers[:, :2] = c3_out[:, :2] + mid[None, :2]

                delta_w = w_new - bc.wavevectors
                amp_scale = (1.0 / scale) if rewind else scale
                bc.amplitudes = (bc.amplitudes * amp_scale) * np.exp(-1j * (delta_w @ mid))
                bc.wavevectors = w_new
            else:
                v3 = np.zeros((bc.num_modes, 3), dtype=np.float64)
                v3[:, 0] = np.real(bc.amplitudes)
                v3[:, 1] = np.imag(bc.amplitudes)
                if rewind:
                    v3_out = CliffordSpacetimeRotor.rewind_transform_vector_3d(rotor, v3) / scale
                else:
                    v3_out = CliffordSpacetimeRotor.scale_rotor_transform(rotor, scale, v3)
                bc.amplitudes = v3_out[:, 0] + 1j * v3_out[:, 1]
        new_bands.append(bc)

    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=manifold.balance,
        current_time=manifold.current_time,
    )


def op_causal_worldline_orbit(
    manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
) -> SpectralScaleRotorManifold:
    """
    Existing Elysia Code Module (`CausalScaleRotorField`) wrapped as an O(K)
    nested worldline orbital pigment across macro and micro bands.
    """
    time_val = float(params.get("time", 0.5))
    macro_rate = float(params.get("macro_rate", 0.5))
    micro_rate = float(params.get("micro_rate", 2.0))
    macro_radius = float(params.get("macro_radius", 0.04))
    micro_radius = float(params.get("micro_radius", 0.02))

    orbit_field = CausalScaleRotorField(
        [
            RotorNode(
                "macro_orbit",
                ScaleRotor(axis=(0.0, 0.0, 1.0), angular_rate=macro_rate, radius=macro_radius),
            ),
            RotorNode(
                "micro_orbit",
                ScaleRotor(axis=(0.0, 0.0, 1.0), angular_rate=micro_rate, radius=micro_radius),
                parent_id="macro_orbit",
            ),
        ]
    )
    slices = orbit_field.slice_at(time_val)
    macro_shift = slices["macro_orbit"][: manifold.dim]
    micro_shift = slices["micro_orbit"][: manifold.dim]

    new_bands: List[ScaleRotorBand] = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            if bc.level == 0:
                shift = macro_shift
                orbital_phase = macro_rate * time_val
            else:
                shift = micro_shift
                orbital_phase = (macro_rate + micro_rate) * time_val
                if not bc.is_global:
                    bc.centers = bc.centers + shift[None, :]
            phase_shift = -(bc.wavevectors @ shift) - orbital_phase
            bc.amplitudes = bc.amplitudes * np.exp(1j * phase_shift)
        new_bands.append(bc)

    return SpectralScaleRotorManifold(
        dim=manifold.dim,
        domain_bounds=manifold.domain_bounds,
        bands=new_bands,
        balance=manifold.balance,
        current_time=manifold.current_time,
    )


def _make_1d_archetype(
    wavevectors: Sequence[float],
    amplitudes: Sequence[complex],
    centers: Optional[Sequence[float]] = None,
    sigmas: Optional[Sequence[float]] = None,
    is_global: bool = True,
) -> SpectralScaleRotorManifold:
    k = len(wavevectors)
    w = np.asarray(wavevectors, dtype=np.float64).reshape(k, 1)
    a = np.asarray(amplitudes, dtype=np.complex128).reshape(k)
    c = (
        np.zeros((k, 1), dtype=np.float64)
        if centers is None
        else np.asarray(centers, dtype=np.float64).reshape(k, 1)
    )
    s = (
        np.full(k, np.inf, dtype=np.float64)
        if sigmas is None
        else np.asarray(sigmas, dtype=np.float64).reshape(k)
    )
    band = ScaleRotorBand(
        level=0 if is_global else 1,
        dim=1,
        wavevectors=w,
        temporal_omegas=np.abs(w[:, 0]),
        amplitudes=a,
        centers=c,
        sigmas=s,
        is_global=is_global,
        parent_indices=-np.ones(k, dtype=np.int32),
        parent_couplings=np.ones(k, dtype=np.float64),
    )
    return SpectralScaleRotorManifold(dim=1, domain_bounds=([0.0], [1.0]), bands=[band])


def _make_2d_archetype(
    wavevectors: Sequence[Tuple[float, float]],
    amplitudes: Sequence[complex],
    centers: Optional[Sequence[Tuple[float, float]]] = None,
    sigmas: Optional[Sequence[float]] = None,
    is_global: bool = True,
) -> SpectralScaleRotorManifold:
    k = len(wavevectors)
    w = np.asarray(wavevectors, dtype=np.float64).reshape(k, 2)
    a = np.asarray(amplitudes, dtype=np.complex128).reshape(k)
    c = (
        np.zeros((k, 2), dtype=np.float64)
        if centers is None
        else np.asarray(centers, dtype=np.float64).reshape(k, 2)
    )
    s = (
        np.full(k, np.inf, dtype=np.float64)
        if sigmas is None
        else np.asarray(sigmas, dtype=np.float64).reshape(k)
    )
    band = ScaleRotorBand(
        level=0 if is_global else 1,
        dim=2,
        wavevectors=w,
        temporal_omegas=np.linalg.norm(w, axis=1),
        amplitudes=a,
        centers=c,
        sigmas=s,
        is_global=is_global,
        parent_indices=-np.ones(k, dtype=np.int32),
        parent_couplings=np.ones(k, dtype=np.float64),
    )
    return SpectralScaleRotorManifold(
        dim=2, domain_bounds=([0.0, 0.0], [1.0, 1.0]), bands=[band]
    )


def _solve_gram_weights(gram: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    p = gram.shape[0]
    if p == 0:
        return np.empty(0, dtype=np.float64)
    scale = max(float(np.trace(gram)) / float(p), 1e-12)
    reg = gram + (float(ridge) * scale) * np.eye(p, dtype=np.float64)
    try:
        return np.linalg.solve(reg, rhs)
    except np.linalg.LinAlgError:
        sol, _, _, _ = np.linalg.lstsq(reg, rhs, rcond=None)
        return sol


def _distill_transfer_operator(
    before: SpectralScaleRotorManifold,
    after: SpectralScaleRotorManifold,
) -> Tuple[OperatorFn, np.ndarray, np.ndarray]:
    """
    Distills a continuous spectral transfer law H(||w||) directly from an observed
    transition (before -> after) in O(K) without fitting a point grid.
    """
    radii_list: List[float] = []
    ratios_list: List[complex] = []

    after_by_level = {b.level: b for b in after.bands}
    all_after_w = (
        np.concatenate([b.wavevectors for b in after.bands if b.num_modes > 0], axis=0)
        if any(b.num_modes > 0 for b in after.bands)
        else np.empty((0, before.dim), dtype=np.float64)
    )
    all_after_a = (
        np.concatenate([after.effective_amplitudes(b) for b in after.bands if b.num_modes > 0], axis=0)
        if any(b.num_modes > 0 for b in after.bands)
        else np.empty(0, dtype=np.complex128)
    )

    max_before = max(
        (float(np.max(np.abs(before.effective_amplitudes(b)))) for b in before.bands if b.num_modes > 0),
        default=0.0,
    )
    min_amp = max(1e-9 * max_before, 1e-15)

    for bb in before.bands:
        if bb.num_modes == 0:
            continue
        ba = after_by_level.get(bb.level)
        if ba is not None and ba.num_modes > 0:
            cand_w = ba.wavevectors
            cand_a = after.effective_amplitudes(ba)
        else:
            cand_w = all_after_w
            cand_a = all_after_a

        if cand_w.shape[0] == 0:
            continue

        eff_b = before.effective_amplitudes(bb)
        for j in range(bb.num_modes):
            if abs(eff_b[j]) <= min_amp:
                continue
            dists = np.linalg.norm(cand_w - bb.wavevectors[j][None, :], axis=1)
            best_idx = int(np.argmin(dists))
            w_mag = float(np.linalg.norm(bb.wavevectors[j]))
            tol = 1e-6 + 0.05 * max(1.0, w_mag)
            if dists[best_idx] <= tol:
                radii_list.append(w_mag)
                ratios_list.append(complex(cand_a[best_idx] / eff_b[j]))

    if not radii_list:
        e_b = before.wave_resonance(before)["energy_self"]
        e_a = after.wave_resonance(after)["energy_self"]
        gain = math.sqrt(e_a / max(e_b, 1e-18))
        radii_arr = np.array([0.0], dtype=np.float64)
        ratios_arr = np.array([complex(gain, 0.0)], dtype=np.complex128)
    else:
        raw_r = np.asarray(radii_list, dtype=np.float64)
        raw_h = np.asarray(ratios_list, dtype=np.complex128)
        order = np.argsort(raw_r)
        raw_r = raw_r[order]
        raw_h = raw_h[order]

        # Merge duplicate radii cleanly
        uniq_r: List[float] = [float(raw_r[0])]
        uniq_h_accum: List[List[complex]] = [[complex(raw_h[0])]]
        for r_val, h_val in zip(raw_r[1:], raw_h[1:]):
            if abs(float(r_val) - uniq_r[-1]) <= 1e-5:
                uniq_h_accum[-1].append(complex(h_val))
            else:
                uniq_r.append(float(r_val))
                uniq_h_accum.append([complex(h_val)])

        radii_arr = np.asarray(uniq_r, dtype=np.float64)
        ratios_arr = np.asarray([np.mean(grp) for grp in uniq_h_accum], dtype=np.complex128)

    if radii_arr.size >= 2:
        diffs = np.diff(radii_arr)
        pos_diffs = diffs[diffs > 1e-6]
        med_diff = float(np.median(pos_diffs)) if pos_diffs.size > 0 else 5.0
        default_bw = float(np.clip(0.35 * med_diff, 1.0, 6.0))
    else:
        default_bw = 5.0

    def _op_distilled(
        manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
    ) -> SpectralScaleRotorManifold:
        bw = max(1e-4, float(params.get("bandwidth", default_bw)))
        new_bands: List[ScaleRotorBand] = []
        for band in manifold.bands:
            bc = band.copy()
            if bc.num_modes > 0:
                if radii_arr.size == 1:
                    gains = np.full(bc.num_modes, ratios_arr[0], dtype=np.complex128)
                else:
                    r_q = np.linalg.norm(bc.wavevectors, axis=1)
                    dist_sq = ((r_q[:, None] - radii_arr[None, :]) / bw) ** 2
                    ker = np.exp(-0.5 * dist_sq) + 1e-12
                    gains = (ker @ ratios_arr) / np.sum(ker, axis=1)
                bc.amplitudes = bc.amplitudes * gains
            new_bands.append(bc)

        return SpectralScaleRotorManifold(
            dim=manifold.dim,
            domain_bounds=manifold.domain_bounds,
            bands=new_bands,
            balance=manifold.balance,
            current_time=manifold.current_time,
        )

    return _op_distilled, radii_arr, ratios_arr


class CausalMechanismPalette:
    """
    Palette of reusable causal mechanisms across scientific disciplines,
    existing Elysia modules, and distilled experiential memories.
    """

    def __init__(self) -> None:
        self.pigments: Dict[str, CausalMechanismPigment] = {}
        self._memory_counter: int = 0
        self.rectification_history: List[Dict[str, Any]] = []

    def register(self, pigment: CausalMechanismPigment) -> None:
        self.pigments[pigment.pigment_id] = pigment

    def get(self, pigment_id: str) -> CausalMechanismPigment:
        if pigment_id not in self.pigments:
            raise KeyError(f"Unknown pigment '{pigment_id}'")
        return self.pigments[pigment_id]

    def list_pigments(
        self,
        discipline: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> List[str]:
        return [
            pid
            for pid, pig in self.pigments.items()
            if (discipline is None or pig.discipline == discipline)
            and (origin is None or pig.origin == origin)
        ]

    @classmethod
    def with_default_principles(cls) -> "CausalMechanismPalette":
        """
        Seeds the palette with canonical disciplinary laws (physics, biology,
        acoustics, optics), verified Elysia code modules (CliffordSpacetimeRotor,
        CausalScaleRotorField), and fundamental 1D/2D structural archetypes.
        """
        pal = cls()

        # 1. Executable Disciplinary & Code-Module Pigments (Dimension-agnostic)
        pal.register(
            CausalMechanismPigment(
                pigment_id="physics.thermal_diffusion",
                discipline="physics",
                origin="disciplinary_law",
                description="Exact O(K) Heat PDE relaxation (du/dt = D * Laplacian u).",
                default_params={"diffusivity": 0.002, "tau": 1.0},
                operator=op_thermal_diffusion,
                chromatic_affinity=(0.15, 0.65, 0.20),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="physics.damped_harmonic",
                discipline="physics",
                origin="disciplinary_law",
                description="Exact underdamped harmonic oscillator evolution.",
                default_params={"damping": 0.15, "tau": 1.0},
                operator=op_damped_harmonic,
                chromatic_affinity=(0.45, 0.35, 0.20),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="physics.doppler_advection",
                discipline="physics",
                origin="disciplinary_law",
                description="Exact spacetime advection & Doppler transport.",
                default_params={"tau": 0.1},
                operator=op_doppler_advection,
                chromatic_affinity=(0.60, 0.20, 0.20),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="biology.lateral_inhibition",
                discipline="biology",
                origin="disciplinary_law",
                description="Retinal DoG center-surround lateral inhibition & edge relief.",
                default_params={"sigma_center": 0.02, "sigma_surround": 0.08, "surround_weight": 0.85},
                operator=op_lateral_inhibition,
                chromatic_affinity=(0.40, 0.45, 0.15),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="acoustics.harmonic_resonance",
                discipline="acoustics",
                origin="disciplinary_law",
                description="Integer overtone ladder harmonic resonance filter.",
                default_params={"fundamental": 2.0 * math.pi * 3.0, "max_overtones": 6, "bandwidth": 2.0},
                operator=op_harmonic_resonance,
                chromatic_affinity=(0.30, 0.55, 0.15),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="optics.snell_refraction",
                discipline="optics",
                origin="disciplinary_law",
                description="Optical refractive wavevector dispersion & wavefront bending.",
                default_params={"index_ratio": 1.25, "angle": 0.25},
                operator=op_snell_refraction,
                chromatic_affinity=(0.45, 0.30, 0.25),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="code.clifford_rotor_twist",
                discipline="code_module",
                origin="code_module",
                description="Existing CliffordSpacetimeRotor geometric sandwich & spinor twist.",
                default_params={"angle": 0.35, "scale": 1.0, "rewind": False, "axis": [0.0, 0.0, 1.0]},
                operator=op_clifford_rotor_twist,
                chromatic_affinity=(0.50, 0.40, 0.10),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="code.causal_worldline_orbit",
                discipline="code_module",
                origin="code_module",
                description="Existing CausalScaleRotorField nested macro/micro orbital slice.",
                default_params={"time": 0.5, "macro_rate": 0.5, "micro_rate": 2.0},
                operator=op_causal_worldline_orbit,
                chromatic_affinity=(0.55, 0.30, 0.15),
            )
        )

        # 2. Canonical Structural Archetypes (1D)
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.dc_equilibrium_1d",
                discipline="physics",
                origin="disciplinary_law",
                description="Uniform equilibrium baseline (1D).",
                archetype_signature=_make_1d_archetype([0.0], [1.0 + 0.0j], is_global=True),
                dim=1,
                chromatic_affinity=(0.10, 0.80, 0.10),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.harmonic_fundamental_1d",
                discipline="acoustics",
                origin="disciplinary_law",
                description="Macro 1D fundamental carrier cos(2*pi*x).",
                archetype_signature=_make_1d_archetype(
                    [2.0 * math.pi], [1.0 + 0.0j], is_global=True
                ),
                dim=1,
                chromatic_affinity=(0.35, 0.55, 0.10),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.gaussian_bell_1d",
                discipline="physics",
                origin="disciplinary_law",
                description="Centered localized heat-kernel Gaussian bell (1D).",
                archetype_signature=_make_1d_archetype(
                    [0.0], [1.0 + 0.0j], centers=[0.5], sigmas=[0.12], is_global=False
                ),
                dim=1,
                chromatic_affinity=(0.25, 0.55, 0.20),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.dog_contrast_1d",
                discipline="biology",
                origin="disciplinary_law",
                description="Centered 1D Difference-of-Gaussians lateral inhibition motif.",
                archetype_signature=_make_1d_archetype(
                    [0.0, 0.0],
                    [1.0 + 0.0j, -0.5 + 0.0j],
                    centers=[0.5, 0.5],
                    sigmas=[0.05, 0.12],
                    is_global=False,
                ),
                dim=1,
                chromatic_affinity=(0.45, 0.40, 0.15),
            )
        )

        # 3. Canonical Structural Archetypes (2D)
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.dc_surface_2d",
                discipline="physics",
                origin="disciplinary_law",
                description="Uniform 2D equilibrium plane.",
                archetype_signature=_make_2d_archetype([(0.0, 0.0)], [1.0 + 0.0j], is_global=True),
                dim=2,
                chromatic_affinity=(0.10, 0.80, 0.10),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.harmonic_x_2d",
                discipline="optics",
                origin="disciplinary_law",
                description="2D horizontal plane-wave carrier cos(2*pi*x).",
                archetype_signature=_make_2d_archetype(
                    [(0.0, 2.0 * math.pi)], [1.0 + 0.0j], is_global=True
                ),
                dim=2,
                chromatic_affinity=(0.35, 0.55, 0.10),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.harmonic_y_2d",
                discipline="optics",
                origin="disciplinary_law",
                description="2D vertical plane-wave carrier cos(2*pi*y).",
                archetype_signature=_make_2d_archetype(
                    [(2.0 * math.pi, 0.0)], [1.0 + 0.0j], is_global=True
                ),
                dim=2,
                chromatic_affinity=(0.35, 0.55, 0.10),
            )
        )
        pal.register(
            CausalMechanismPigment(
                pigment_id="archetype.radial_gaussian_2d",
                discipline="physics",
                origin="disciplinary_law",
                description="2D centered radial Gaussian kernel.",
                archetype_signature=_make_2d_archetype(
                    [(0.0, 0.0)], [1.0 + 0.0j], centers=[(0.5, 0.5)], sigmas=[0.15], is_global=False
                ),
                dim=2,
                chromatic_affinity=(0.25, 0.55, 0.20),
            )
        )

        return pal

    def compose_and_apply(
        self,
        manifold: SpectralScaleRotorManifold,
        recipe: Sequence[Union[Tuple[str, float, Optional[Dict[str, Any]]], Dict[str, Any]]],
        mode: str = "sequential",
        balance: Optional[SpectralBalanceScale] = None,
    ) -> SpectralScaleRotorManifold:
        """
        Composes and executes a palette recipe directly in O(K) without
        dropping down to a spatial point grid.
        """
        if not recipe:
            out = manifold.copy()
            if balance is not None:
                out = out.tune_balance(
                    scale_tilt=balance.scale_tilt,
                    contrast_tilt=balance.contrast_tilt,
                    phase_bias=balance.phase_bias,
                    granularity_scale=balance.granularity_scale,
                )
            return out

        parsed: List[Tuple[str, float, Optional[Dict[str, Any]]]] = []
        for item in recipe:
            if isinstance(item, dict):
                parsed.append(
                    (str(item["pigment_id"]), float(item.get("weight", 1.0)), item.get("params"))
                )
            else:
                pid, w = item[0], float(item[1])
                prm = item[2] if len(item) > 2 else None
                parsed.append((str(pid), w, prm))

        if mode == "sequential":
            curr = manifold.copy()
            for pid, w, prm in parsed:
                pigment = self.get(pid)
                applied = pigment.apply(curr, prm)
                if abs(w - 1.0) <= 1e-12:
                    curr = applied
                else:
                    curr = curr.scale_field(1.0 - w).superpose(applied.scale_field(w)).prune_negligible()
            out = curr
        elif mode == "parallel":
            acc: Optional[SpectralScaleRotorManifold] = None
            for pid, w, prm in parsed:
                pigment = self.get(pid)
                term = pigment.apply(manifold, prm).scale_field(w)
                acc = term if acc is None else acc.superpose(term)
            assert acc is not None
            out = acc.prune_negligible()
        else:
            raise ValueError(f"Unknown composition mode '{mode}', expected 'sequential' or 'parallel'")

        if balance is not None:
            out = out.tune_balance(
                scale_tilt=balance.scale_tilt,
                contrast_tilt=balance.contrast_tilt,
                phase_bias=balance.phase_bias,
                granularity_scale=balance.granularity_scale,
            )
        return out

    def _empty_like(self, manifold: SpectralScaleRotorManifold) -> SpectralScaleRotorManifold:
        b0 = ScaleRotorBand(
            level=0,
            dim=manifold.dim,
            wavevectors=np.empty((0, manifold.dim), dtype=np.float64),
            temporal_omegas=np.empty(0, dtype=np.float64),
            amplitudes=np.empty(0, dtype=np.complex128),
            centers=np.empty((0, manifold.dim), dtype=np.float64),
            sigmas=np.empty(0, dtype=np.float64),
            is_global=True,
            parent_indices=np.empty(0, dtype=np.int32),
            parent_couplings=np.empty(0, dtype=np.float64),
        )
        return SpectralScaleRotorManifold(
            dim=manifold.dim,
            domain_bounds=manifold.domain_bounds,
            bands=[b0],
        )

    def rectify_field(
        self,
        observed: SpectralScaleRotorManifold,
        novelty_threshold: float = 0.15,
        ridge: float = 1e-6,
        distill_novelty: bool = True,
        memory_label: Optional[str] = None,
    ) -> RectificationResult:
        """
        Projects an incoming wave-rotor field onto existing structural palette
        pigments via continuous L^2 resonance (ZERO point grid). If unexplained
        residual friction exceeds `novelty_threshold`, distills the novel residual
        structure into a reusable memory pigment.
        """
        t0 = time.perf_counter()
        bounds = observed.domain_bounds
        e_obs = max(0.0, observed.wave_resonance(observed, bounds=bounds)["energy_self"])

        candidates = [
            pig
            for pig in self.pigments.values()
            if pig.archetype_signature is not None and pig.archetype_signature.dim == observed.dim
        ]

        if not candidates or e_obs <= 1e-24:
            reconstructed = self._empty_like(observed)
            residual = observed.copy()
            e_res = e_obs
            weights_map: Dict[str, float] = {}
        else:
            p = len(candidates)
            sigs = [c.archetype_signature for c in candidates]
            gram = np.zeros((p, p), dtype=np.float64)
            rhs = np.zeros(p, dtype=np.float64)

            for i in range(p):
                si = sigs[i]
                assert si is not None
                rhs[i] = si.wave_resonance(observed, bounds=bounds)["raw_overlap"]
                for j in range(i, p):
                    sj = sigs[j]
                    assert sj is not None
                    ov = si.wave_resonance(sj, bounds=bounds)["raw_overlap"]
                    gram[i, j] = ov
                    gram[j, i] = ov

            w = _solve_gram_weights(gram, rhs, ridge)
            reconstructed = None
            weights_map = {}
            for idx, (cand, w_val) in enumerate(zip(candidates, w)):
                w_f = float(w_val)
                if abs(w_f) > 1e-9:
                    weights_map[cand.pigment_id] = w_f
                    term = sigs[idx].scale_field(w_f)  # type: ignore[union-attr]
                    reconstructed = term if reconstructed is None else reconstructed.superpose(term)

            if reconstructed is None:
                reconstructed = self._empty_like(observed)
            else:
                reconstructed = reconstructed.prune_negligible()

            residual = observed.superpose(reconstructed.scale_field(-1.0)).prune_negligible()
            e_res = max(0.0, float(e_obs - 2.0 * np.dot(w, rhs) + float(w @ gram @ w)))

        residual_friction = math.sqrt(e_res / max(e_obs, 1e-18)) if e_obs > 1e-24 else 0.0
        explained_ratio = float(np.clip(1.0 - (e_res / max(e_obs, 1e-18)), 0.0, 1.0)) if e_obs > 1e-24 else 1.0

        distilled_id: Optional[str] = None
        if residual_friction > novelty_threshold and distill_novelty:
            from_memory = False
            pid = memory_label or f"memory.field_{self._memory_counter}"
            if memory_label is None or pid in self.pigments:
                while pid in self.pigments:
                    self._memory_counter += 1
                    pid = f"memory.field_{self._memory_counter}"
                if memory_label is None:
                    self._memory_counter += 1
            sig_to_store = residual if candidates else observed
            self.register(
                CausalMechanismPigment(
                    pigment_id=pid,
                    discipline="distilled_memory",
                    origin="distilled_memory",
                    description=f"Distilled {observed.dim}D field pigment from novelty rectification.",
                    archetype_signature=sig_to_store.copy(),
                    dim=observed.dim,
                    chromatic_affinity=tuple(observed.spectral_axis_profile()["chromatic_triad"]),  # type: ignore[arg-type]
                )
            )
            distilled_id = pid
        else:
            from_memory = True
            for pid, w_val in weights_map.items():
                if abs(w_val) > 1e-3 and pid in self.pigments:
                    self.pigments[pid].activation_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        res = RectificationResult(
            recipe_weights=weights_map,
            explained_ratio=explained_ratio,
            residual_friction=residual_friction,
            from_memory=from_memory,
            distilled_pigment_id=distilled_id,
            reconstructed_manifold=reconstructed,
            residual_manifold=residual,
            elapsed_ms=elapsed_ms,
        )
        self.rectification_history.append(
            {
                "kind": "field",
                "explained_ratio": explained_ratio,
                "residual_friction": residual_friction,
                "from_memory": from_memory,
                "distilled_pigment_id": distilled_id,
            }
        )
        return res

    def rectify_transition(
        self,
        before: SpectralScaleRotorManifold,
        after: SpectralScaleRotorManifold,
        candidate_ids: Optional[Sequence[str]] = None,
        novelty_threshold: float = 0.15,
        ridge: float = 1e-6,
        distill_novelty: bool = True,
        memory_label: Optional[str] = None,
    ) -> RectificationResult:
        """
        Identifies which executable palette mechanism(s) caused `before -> after`
        purely in O(K) wave-rotor space. If unexplained transition friction exceeds
        `novelty_threshold`, distills a reusable spectral transfer operator from the
        transition so future observations recall and transfer the law zero-shot.
        """
        if before.dim != after.dim:
            raise ValueError(f"Dimension mismatch: {before.dim} vs {after.dim}")

        t0 = time.perf_counter()
        bounds = (
            np.minimum(before.domain_bounds[0], after.domain_bounds[0]),
            np.maximum(before.domain_bounds[1], after.domain_bounds[1]),
        )
        e_after = max(0.0, after.wave_resonance(after, bounds=bounds)["energy_self"])

        if candidate_ids is not None:
            candidates = [self.get(pid) for pid in candidate_ids]
        else:
            candidates = [
                pig
                for pig in self.pigments.values()
                if pig.operator is not None and (pig.dim is None or pig.dim == before.dim)
            ]

        if not candidates or e_after <= 1e-24:
            reconstructed = self._empty_like(after)
            residual = after.copy()
            e_res = e_after
            weights_map: Dict[str, float] = {}
        else:
            # Evaluate candidate operators on `before` without counting search as activation
            saved_counts = [c.activation_count for c in candidates]
            cand_outputs = [c.apply(before) for c in candidates]
            for c, cnt in zip(candidates, saved_counts):
                c.activation_count = cnt

            p = len(candidates)
            gram = np.zeros((p, p), dtype=np.float64)
            rhs = np.zeros(p, dtype=np.float64)
            for i in range(p):
                rhs[i] = cand_outputs[i].wave_resonance(after, bounds=bounds)["raw_overlap"]
                for j in range(i, p):
                    ov = cand_outputs[i].wave_resonance(cand_outputs[j], bounds=bounds)["raw_overlap"]
                    gram[i, j] = ov
                    gram[j, i] = ov

            w = _solve_gram_weights(gram, rhs, ridge)
            reconstructed = None
            weights_map = {}
            for idx, (cand, w_val) in enumerate(zip(candidates, w)):
                w_f = float(w_val)
                if abs(w_f) > 1e-9:
                    weights_map[cand.pigment_id] = w_f
                    term = cand_outputs[idx].scale_field(w_f)
                    reconstructed = term if reconstructed is None else reconstructed.superpose(term)

            if reconstructed is None:
                reconstructed = self._empty_like(after)
            else:
                reconstructed = reconstructed.prune_negligible()

            residual = after.superpose(reconstructed.scale_field(-1.0)).prune_negligible()
            e_res = max(0.0, float(e_after - 2.0 * np.dot(w, rhs) + float(w @ gram @ w)))

        residual_friction = math.sqrt(e_res / max(e_after, 1e-18)) if e_after > 1e-24 else 0.0
        explained_ratio = float(np.clip(1.0 - (e_res / max(e_after, 1e-18)), 0.0, 1.0)) if e_after > 1e-24 else 1.0

        distilled_id: Optional[str] = None
        if residual_friction > novelty_threshold and distill_novelty:
            from_memory = False
            pid = memory_label or f"memory.transition_{self._memory_counter}"
            if memory_label is None or pid in self.pigments:
                while pid in self.pigments:
                    self._memory_counter += 1
                    pid = f"memory.transition_{self._memory_counter}"
                if memory_label is None:
                    self._memory_counter += 1
            op_fn, anchor_r, _ = _distill_transfer_operator(before, after)
            self.register(
                CausalMechanismPigment(
                    pigment_id=pid,
                    discipline="distilled_memory",
                    origin="distilled_memory",
                    description=(
                        f"Distilled spectral transfer law across {len(anchor_r)} wavevector radii."
                    ),
                    operator=op_fn,
                    dim=before.dim,
                )
            )
            distilled_id = pid
        else:
            from_memory = True
            for pid, w_val in weights_map.items():
                if abs(w_val) > 1e-3 and pid in self.pigments:
                    self.pigments[pid].activation_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        res = RectificationResult(
            recipe_weights=weights_map,
            explained_ratio=explained_ratio,
            residual_friction=residual_friction,
            from_memory=from_memory,
            distilled_pigment_id=distilled_id,
            reconstructed_manifold=reconstructed,
            residual_manifold=residual,
            elapsed_ms=elapsed_ms,
        )
        self.rectification_history.append(
            {
                "kind": "transition",
                "explained_ratio": explained_ratio,
                "residual_friction": residual_friction,
                "from_memory": from_memory,
                "distilled_pigment_id": distilled_id,
            }
        )
        return res

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        last_result: Optional[RectificationResult] = None,
        prefix: str = "palette",
    ) -> Dict[str, Any]:
        """
        Projects palette pigments into CausalField as EngramAttractors +
        InformationVoxels, wires co-activated pigments with ConnectivityBeams,
        and injects Mobility momentum proportional to rectification novelty.
        """
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        weights = last_result.recipe_weights if last_result is not None else {}
        novelty = last_result.residual_friction if last_result is not None else 0.0

        active_ids: List[str] = []
        engram_ids: List[str] = []
        total_w = sum(abs(v) for v in weights.values()) + 1e-12

        for idx, (pid, pig) in enumerate(self.pigments.items()):
            w_val = float(weights.get(pid, 0.0))
            pos = np.zeros(field.dimensions, dtype=np.float32)
            pos[0] = float(pig.chromatic_affinity[0])
            if field.dimensions >= 2:
                pos[1] = float(pig.chromatic_affinity[1])
            if field.dimensions >= 3:
                pos[2] = float(idx) * 0.1

            vid = f"{prefix}_{pid}"
            engram_id = f"{prefix}_engram_{pid}"
            engram = EngramAttractor(
                id=engram_id,
                name=pid,
                position=pos,
                intensity=float(1.0 + 5.0 * abs(w_val) / total_w),
                sigma=2.0,
                active=True,
                tier="macro" if pig.origin != "distilled_memory" else "meso",
                symbolic_intent=pig.description,
            )
            field.register_engram(engram)
            engram_ids.append(engram_id)

            sig = np.array(
                [
                    pig.chromatic_affinity[0],
                    pig.chromatic_affinity[1],
                    pig.chromatic_affinity[2],
                    w_val,
                    float(pig.activation_count),
                ],
                dtype=np.float32,
            )
            voxel = InformationVoxel(
                id=vid,
                content={"pigment_id": pid, "discipline": pig.discipline, "origin": pig.origin},
                tensor=sig,
                mass=max(0.2, 1.0 + abs(w_val)),
                position=pos,
                chromatic_vector=np.array(pig.chromatic_affinity, dtype=np.float32),
            )
            field.add_voxel(voxel)
            if abs(w_val) > 1e-4 or pid == (last_result.distilled_pigment_id if last_result else None):
                active_ids.append(vid)

        beams_linked = 0
        for i in range(len(active_ids) - 1):
            w_pair = abs(float(weights.get(active_ids[i + 1].replace(f"{prefix}_", "", 1), 0.5)))
            field.link_voxels(active_ids[i], active_ids[i + 1], strength=max(0.2, w_pair / total_w))
            beams_linked += 1

        impulse = np.zeros(field.dimensions, dtype=np.float32)
        impulse[0] = float(0.1 + novelty)
        for vid in active_ids:
            field.apply_impact(vid, impulse)

        return {
            "causal_field": field,
            "engram_ids": engram_ids,
            "active_voxel_ids": active_ids,
            "beams_linked": beams_linked,
        }
