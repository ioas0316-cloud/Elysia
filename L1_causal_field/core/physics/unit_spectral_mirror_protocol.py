"""
ELYSIA Unit-1 Spectral Mirror & Living Causal-Operator Superposition Protocol (Phase 9)
========================================================================================

Implements the "Mirror Principle" ("거울원리 / 단위체계 1 정규화 / 결과를 결과화하지 않기"):
Never freeze an observation into a dead scalar difference or flat residual array.
Every comparison against a Planetary Reference Order Globe factors reality as a
live, inspectable, composable causal chain:

    Psi_obs = O_observer ( M_world ( G_ref ) )

Where:
1. G_ref (`reference_generator`): The living operator generating the Unit-1
   Reference Order (`PlanetaryOrderGlobe`).
2. O_observer (`observer_operator`): The causal structure of the observing organ
   or lens (`OpticalOcularEyeOperator`, `AcousticWaveOperator`,
   `StructuralSpectrumOperator`), which can be peered through (`O_observer.invert`)
   and counterfactually swapped (`counterfactual_reobserve`).
3. M_world (`extracted_world_operator`): The executable causal transfer operator
   distilled directly from the Unit-1 mirror ratio H = Psi_delensed / Psi_ref,
   capable of lossless reconstruction, zero-shot transfer to unseen signals
   (`transfer_world_cause_to`), and lineage inspection (`trace_causal_lineage`).
4. Unreflected Order (`unreflected_order_operator`): Energy outside the current
   Globe support is spectrally separated by Shannon entropy & concentration into:
   - `UNKNOWN_COHERENT_EXTERNAL_ORDER`: Structured new order -> triggers curiosity
     and folds directly into the Globe (`absorb_discovered_order`), expanding the mirror!
   - `NON_CAUSAL_RUPTURE`: High-entropy uncoupled tamper/noise -> rejected and
     physically tears beams in `CausalField`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.physics.trinitarian_scale_binding_protocol import (
    CoupledQuantity,
    EqualityBond,
    IntentTarget,
    PinMode,
    ScaleTier,
    TrinitarianScaleRotorSystem,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


class MirrorOriginKind(str, Enum):
    """Causal origin classification produced by Unit-1 spectral superposition."""

    UNIT_ONE_REFLECTION = "unit_one_reflection"
    OBSERVER_STRUCTURE_ARTIFACT = "observer_structure_artifact"
    WORLD_LAWFUL_TRANSFORMATION = "world_lawful_transformation"
    UNKNOWN_COHERENT_EXTERNAL_ORDER = "unknown_coherent_external_order"
    NON_CAUSAL_RUPTURE = "non_causal_rupture"


@dataclass
class MultiScaleOnsetProfile:
    """Macro/Meso/Micro deviation breakdown plus localized onset coordinates."""

    macro_deviation: float
    meso_deviation: float
    micro_deviation: float
    dominant_scale: str
    onset_coord: Tuple[float, ...]
    channel_deviations: Dict[str, float]


# ---------------------------------------------------------------------------
# Modality-Preserving Spectral Transform Helpers
# ---------------------------------------------------------------------------


def _to_spectrum(arr: np.ndarray, is_structural_basis: bool = False) -> np.ndarray:
    data = np.asarray(arr)
    if is_structural_basis:
        return data.astype(np.complex128)
    if data.ndim == 1:
        return np.fft.rfft(data.astype(np.float64))
    if data.ndim == 2:
        return np.fft.fft2(data.astype(np.float64))
    if data.ndim == 3:
        return np.fft.fft2(data.astype(np.float64), axes=(0, 1))
    raise ValueError(f"Unsupported array ndim={data.ndim} for spectral projection")


def _from_spectrum(
    spec: np.ndarray,
    target_shape: Tuple[int, ...],
    is_structural_basis: bool = False,
    keep_complex: bool = False,
) -> np.ndarray:
    if is_structural_basis:
        if keep_complex:
            return spec.astype(np.complex128)
        return np.real(spec).astype(np.float64)
    if len(target_shape) == 1:
        return np.fft.irfft(spec, n=target_shape[0]).astype(np.float64)
    if len(target_shape) == 2:
        out = np.fft.ifft2(spec)
        return out if keep_complex else np.real(out).astype(np.float64)
    if len(target_shape) == 3:
        out = np.fft.ifft2(spec, axes=(0, 1))
        return out if keep_complex else np.real(out).astype(np.float64)
    raise ValueError(f"Unsupported target_shape={target_shape}")


def _fourier_shift_filter_2d(
    height: int,
    width: int,
    dy: float,
    dx: float,
    defocus_sigma: float = 0.0,
) -> np.ndarray:
    ky = 2.0 * math.pi * np.fft.fftfreq(height)[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(width)[None, :]
    shift_phase = np.exp(-1j * (ky * float(dy) + kx * float(dx)))
    if defocus_sigma > 0.0:
        mtf = np.exp(-0.5 * (float(defocus_sigma) ** 2) * (ky * ky + kx * kx))
    else:
        mtf = 1.0
    return shift_phase * mtf


def _spectral_entropy_and_concentration(spec: np.ndarray) -> Tuple[float, float]:
    """
    Measures whether unreflected spectral energy is a sparse, structured mode
    (low normalized entropy, high top-mode concentration) vs unstructured noise.
    """
    power = (np.abs(spec).ravel()) ** 2
    total_energy = float(np.sum(power))
    if total_energy <= 1e-30:
        return 0.0, 0.0

    probs = power / total_energy
    nz = probs[probs > 1e-18]
    max_entropy = math.log(max(2, len(power)))
    entropy = float(-np.sum(nz * np.log(nz)) / max_entropy)
    entropy = float(np.clip(entropy, 0.0, 1.0))

    top_k = 6 if spec.ndim >= 2 else 3
    sorted_p = np.sort(power)
    top_conc = float(np.sum(sorted_p[-min(top_k, len(sorted_p)) :]) / total_energy)
    return entropy, float(np.clip(top_conc, 0.0, 1.0))


def _block_rms_grid_2d(err_2d: np.ndarray, grid_n: int) -> np.ndarray:
    h, w = err_2d.shape[:2]
    gh = min(grid_n, h)
    gw = min(grid_n, w)
    grid = np.zeros((gh, gw), dtype=np.float64)
    for iy in range(gh):
        y0 = (iy * h) // gh
        y1 = max(y0 + 1, ((iy + 1) * h) // gh)
        for ix in range(gw):
            x0 = (ix * w) // gw
            x1 = max(x0 + 1, ((ix + 1) * w) // gw)
            patch = err_2d[y0:y1, x0:x1]
            grid[iy, ix] = float(np.sqrt(np.mean(patch * patch)))
    return grid


def _analyze_multiscale_onset(
    ref: np.ndarray,
    comp: np.ndarray,
    is_structural_basis: bool = False,
    band_names: Tuple[str, ...] = (),
) -> MultiScaleOnsetProfile:
    """
    Diagnoses Macro / Meso / Micro deviation structure and locates where in space,
    time, or structural band space a lawful or unreflected divergence originates.
    """
    ref_arr = np.asarray(ref)
    comp_arr = np.asarray(comp)
    diff = np.abs(comp_arr - ref_arr).astype(np.float64)
    norm_ref = max(float(np.sqrt(np.mean(np.abs(ref_arr) ** 2))), 1e-12)

    if is_structural_basis or comp_arr.ndim == 1 and is_structural_basis:
        flat_diff = diff.ravel()
        channel_devs: Dict[str, float] = {}
        for idx, val in enumerate(flat_diff):
            label = band_names[idx] if idx < len(band_names) else f"band_{idx}"
            channel_devs[label] = float(val / norm_ref)
        peak_idx = int(np.argmax(flat_diff)) if len(flat_diff) > 0 else 0
        onset = (float(peak_idx / max(1, len(flat_diff) - 1)),)
        macro = float(np.mean(flat_diff) / norm_ref) if len(flat_diff) else 0.0
        micro = float(np.max(flat_diff) / norm_ref) if len(flat_diff) else 0.0
        meso = 0.5 * (macro + micro)
        dom = "uniform" if micro <= 1.2 * max(macro, 1e-12) else "micro"
        return MultiScaleOnsetProfile(macro, meso, micro, dom, onset, channel_devs)

    if comp_arr.ndim == 1:
        n = len(diff)
        channel_devs = {"wave": float(np.sqrt(np.mean(diff * diff)) / norm_ref)}

        def _win_rms(n_win: int) -> np.ndarray:
            nw = min(n_win, n)
            vals = np.zeros(nw, dtype=np.float64)
            for k in range(nw):
                s0 = (k * n) // nw
                s1 = max(s0 + 1, ((k + 1) * n) // nw)
                vals[k] = float(np.sqrt(np.mean(diff[s0:s1] ** 2)) / norm_ref)
            return vals

        w_macro = _win_rms(4)
        w_meso = _win_rms(16)
        w_micro = _win_rms(64)
        macro = float(np.std(w_macro) + np.mean(w_macro))
        meso = float(np.max(w_meso))
        micro = float(np.max(w_micro))

        # Earliest micro window exceeding 25% of peak excess anomaly
        thresh = 0.25 * float(np.max(w_micro))
        if float(np.max(w_micro)) <= 1e-12:
            onset_t = 0.0
        else:
            hits = np.where(w_micro >= thresh)[0]
            onset_t = float(hits[0] / max(1, len(w_micro) - 1)) if len(hits) else 0.0

        std_micro = float(np.std(w_micro))
        dom = "uniform" if std_micro < 0.1 * max(float(np.mean(w_micro)), 1e-12) else "micro"
        return MultiScaleOnsetProfile(macro, meso, micro, dom, (onset_t,), channel_devs)

    # 2D grayscale or RGB image
    ref_f = ref_arr.astype(np.float64)
    comp_f = comp_arr.astype(np.float64)
    if comp_f.ndim == 3:
        c_names = ["R", "G", "B"] if comp_f.shape[2] == 3 else [f"C{i}" for i in range(comp_f.shape[2])]
        channel_devs = {
            c_names[c]: float(np.sqrt(np.mean((comp_f[..., c] - ref_f[..., c]) ** 2)) / norm_ref)
            for c in range(comp_f.shape[2])
        }
        # Factor out best-fit per-channel global tint/gain so local shadows/occlusions
        # stand out clearly even under global illumination shifts.
        local_resid = np.zeros_like(comp_f)
        for c in range(comp_f.shape[2]):
            rc = ref_f[..., c]
            oc = comp_f[..., c]
            denom = float(np.sum(rc * rc))
            alpha_c = float(np.sum(oc * rc) / denom) if denom > 1e-12 else 1.0
            local_resid[..., c] = oc - alpha_c * rc
        anom_2d = np.sqrt(np.mean(local_resid * local_resid, axis=2))
        raw_2d = np.sqrt(np.mean(diff * diff, axis=2))
    else:
        channel_devs = {"L": float(np.sqrt(np.mean(diff * diff)) / norm_ref)}
        denom = float(np.sum(ref_f * ref_f))
        alpha = float(np.sum(comp_f * ref_f) / denom) if denom > 1e-12 else 1.0
        anom_2d = np.abs(comp_f - alpha * ref_f)
        raw_2d = diff

    g_macro = _block_rms_grid_2d(raw_2d, 2) / norm_ref
    g_meso = _block_rms_grid_2d(anom_2d, 4) / norm_ref
    g_micro = _block_rms_grid_2d(anom_2d, 8) / norm_ref

    macro = float(np.mean(g_macro))
    meso = float(np.max(g_meso))
    micro = float(np.max(g_micro))

    if micro > 1e-4:
        iy, ix = np.unravel_index(int(np.argmax(g_micro)), g_micro.shape)
        gh, gw = g_micro.shape
        onset_coord = (float((iy + 0.5) / gh), float((ix + 0.5) / gw))
        dom = "micro" if micro > 1.5 * float(np.mean(g_micro) + 1e-12) else "macro"
    else:
        onset_coord = (0.5, 0.5)
        dom = "uniform"

    return MultiScaleOnsetProfile(macro, meso, micro, dom, onset_coord, channel_devs)


# ---------------------------------------------------------------------------
# Live, Composable, Invertible Causal Operators
# ---------------------------------------------------------------------------


class CausalOperator(ABC):
    """
    Base class for every live, inspectable, composable transformation or
    observational organ participating in Unit-1 Mirror Superposition.
    """

    def __init__(
        self,
        name: str,
        role: str,
        modality: str,
        origin_axis: str,
        parameters: Optional[Dict[str, Any]] = None,
        parents: Tuple["CausalOperator", ...] = (),
    ) -> None:
        self.name = name
        self.role = role
        self.modality = modality
        self.origin_axis = origin_axis
        self.parameters = dict(parameters or {})
        self.parents = tuple(parents)

    @abstractmethod
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Forward causal action on a signal/field."""

    @abstractmethod
    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        """Regularized inverse restoring the pre-operator causal state."""

    def compose(self, inner: "CausalOperator") -> "CompositeCausalOperator":
        """Returns outer(inner(x)) while preserving both live parent operators."""
        return CompositeCausalOperator(outer=self, inner=inner)

    def describe_lineage(self) -> Dict[str, Any]:
        safe_params: Dict[str, Any] = {}
        for k, v in self.parameters.items():
            if isinstance(v, np.ndarray):
                safe_params[k] = f"ndarray{tuple(v.shape)}"
            else:
                safe_params[k] = v
        return {
            "name": self.name,
            "type": type(self).__name__,
            "role": self.role,
            "modality": self.modality,
            "origin_axis": self.origin_axis,
            "parameters": safe_params,
            "parents": [p.describe_lineage() for p in self.parents],
        }


class IdentityCausalOperator(CausalOperator):
    """Unmodified identity operator (transparent eye or unperturbed reference)."""

    def __init__(
        self,
        name: str = "identity_lens",
        role: str = "observer_eye",
        modality: str = "generic",
        origin_axis: str = "observer_structure",
    ) -> None:
        super().__init__(name=name, role=role, modality=modality, origin_axis=origin_axis)

    def apply(self, field: np.ndarray) -> np.ndarray:
        return np.asarray(field).copy()

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        return np.asarray(field).copy()


class CompositeCausalOperator(CausalOperator):
    """Serial composition `outer(inner(x))` keeping full live operator lineage."""

    def __init__(self, outer: CausalOperator, inner: CausalOperator) -> None:
        super().__init__(
            name=f"({outer.name} o {inner.name})",
            role=outer.role,
            modality=outer.modality,
            origin_axis=outer.origin_axis,
            parameters={"outer": outer.name, "inner": inner.name},
            parents=(inner, outer),
        )
        self.outer = outer
        self.inner = inner

    def apply(self, field: np.ndarray) -> np.ndarray:
        return self.outer.apply(self.inner.apply(field))

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        return self.inner.invert(self.outer.invert(field, reg=reg), reg=reg)


class OpticalOcularEyeOperator(CausalOperator):
    """
    Physical Ocular Eye Operator:
    Models retina L/M/S cone gain imbalance, corneal Gaussian MTF defocus,
    and subpixel ocular shift as a live reversible operator so Elysia explicitly
    separates visual eye artifacts from external reality.
    """

    def __init__(
        self,
        name: str = "ocular_eye",
        cone_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        corneal_defocus_sigma: float = 0.0,
        ocular_shift: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.cone_gain = tuple(float(g) for g in cone_gain)
        self.corneal_defocus_sigma = float(corneal_defocus_sigma)
        self.ocular_shift = (float(ocular_shift[0]), float(ocular_shift[1]))
        super().__init__(
            name=name,
            role="observer_eye",
            modality="optics_2d",
            origin_axis="observer_structure",
            parameters={
                "cone_gain": self.cone_gain,
                "corneal_defocus_sigma": self.corneal_defocus_sigma,
                "ocular_shift": self.ocular_shift,
            },
        )

    def apply(self, field: np.ndarray) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64)
        h, w = arr.shape[:2]
        filt = _fourier_shift_filter_2d(
            h, w, self.ocular_shift[0], self.ocular_shift[1], self.corneal_defocus_sigma
        )
        if arr.ndim == 2:
            return np.real(np.fft.ifft2(np.fft.fft2(arr) * filt)).astype(np.float64)
        if arr.ndim == 3:
            spec = np.fft.fft2(arr, axes=(0, 1)) * filt[:, :, None]
            out = np.real(np.fft.ifft2(spec, axes=(0, 1))).astype(np.float64)
            gain = np.asarray(self.cone_gain[: arr.shape[2]], dtype=np.float64)
            return out * gain[None, None, :]
        raise ValueError(f"OpticalOcularEyeOperator expects 2D or 3D image, got {arr.shape}")

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64)
        h, w = arr.shape[:2]
        filt = _fourier_shift_filter_2d(
            h, w, self.ocular_shift[0], self.ocular_shift[1], self.corneal_defocus_sigma
        )
        inv_filt = np.conj(filt) / (np.abs(filt) ** 2 + max(1e-12, float(reg)))
        if arr.ndim == 2:
            return np.real(np.fft.ifft2(np.fft.fft2(arr) * inv_filt)).astype(np.float64)
        if arr.ndim == 3:
            gain = np.asarray(self.cone_gain[: arr.shape[2]], dtype=np.float64)
            unscaled = arr / np.maximum(gain, 1e-6)[None, None, :]
            spec = np.fft.fft2(unscaled, axes=(0, 1)) * inv_filt[:, :, None]
            return np.real(np.fft.ifft2(spec, axes=(0, 1))).astype(np.float64)
        raise ValueError(f"OpticalOcularEyeOperator expects 2D or 3D image, got {arr.shape}")


def _planck_rgb_gain(kelvin: float) -> np.ndarray:
    """Computes white-balanced CIE RGB gain normalized to 6500K mean luminance."""
    from core.orchestration.sensory_orchestra_os import CIE_CONE_MATRIX, planck_spectrum

    raw = planck_spectrum(kelvin) @ CIE_CONE_MATRIX
    normed = raw / max(float(np.mean(raw)), 1e-9)
    return normed.astype(np.float64)


class OpticalWorldMediumOperator(CausalOperator):
    """
    Physical External Optical Medium:
    Planck blackbody illumination shift, per-wavelength Cauchy chromatic
    dispersion prism shift, and spatial occlusion shadow.
    """

    def __init__(
        self,
        name: str = "optical_world_medium",
        illumination_kelvin: Optional[float] = None,
        chromatic_dispersion_shifts: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        occlusion_box: Optional[Tuple[int, int, int, int]] = None,
        occlusion_factor: float = 0.3,
    ) -> None:
        self.illumination_kelvin = illumination_kelvin
        self.chromatic_dispersion_shifts = tuple(
            (float(s[0]), float(s[1])) for s in chromatic_dispersion_shifts
        )
        self.occlusion_box = occlusion_box
        self.occlusion_factor = float(occlusion_factor)
        super().__init__(
            name=name,
            role="world_law",
            modality="optics_2d",
            origin_axis="external_world",
            parameters={
                "illumination_kelvin": illumination_kelvin,
                "chromatic_dispersion_shifts": self.chromatic_dispersion_shifts,
                "occlusion_box": occlusion_box,
                "occlusion_factor": self.occlusion_factor,
            },
        )

    def _gain_vec(self, n_channels: int) -> np.ndarray:
        if self.illumination_kelvin is None or n_channels != 3:
            return np.ones(n_channels, dtype=np.float64)
        return _planck_rgb_gain(self.illumination_kelvin)

    def apply(self, field: np.ndarray) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64).copy()
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            dy, dx = self.chromatic_dispersion_shifts[0]
            filt = _fourier_shift_filter_2d(h, w, dy, dx, 0.0)
            out = np.real(np.fft.ifft2(np.fft.fft2(arr) * filt)).astype(np.float64)
        elif arr.ndim == 3:
            c_count = arr.shape[2]
            gain = self._gain_vec(c_count)
            scaled = arr * gain[None, None, :]
            out = np.zeros_like(scaled)
            for c in range(c_count):
                dy, dx = self.chromatic_dispersion_shifts[min(c, len(self.chromatic_dispersion_shifts) - 1)]
                filt = _fourier_shift_filter_2d(h, w, dy, dx, 0.0)
                out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(scaled[..., c]) * filt))
        else:
            raise ValueError(f"Unsupported optics shape {arr.shape}")

        if self.occlusion_box is not None:
            y0, y1, x0, x1 = self.occlusion_box
            out[y0:y1, x0:x1, ...] *= self.occlusion_factor
        return out

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64).copy()
        if self.occlusion_box is not None:
            y0, y1, x0, x1 = self.occlusion_box
            arr[y0:y1, x0:x1, ...] /= max(1e-6, self.occlusion_factor)

        h, w = arr.shape[:2]
        if arr.ndim == 2:
            dy, dx = self.chromatic_dispersion_shifts[0]
            filt = _fourier_shift_filter_2d(h, w, -dy, -dx, 0.0)
            return np.real(np.fft.ifft2(np.fft.fft2(arr) * filt)).astype(np.float64)

        c_count = arr.shape[2]
        out = np.zeros_like(arr)
        for c in range(c_count):
            dy, dx = self.chromatic_dispersion_shifts[min(c, len(self.chromatic_dispersion_shifts) - 1)]
            filt = _fourier_shift_filter_2d(h, w, -dy, -dx, 0.0)
            out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(arr[..., c]) * filt))
        gain = self._gain_vec(c_count)
        return out / np.maximum(gain, 1e-6)[None, None, :]


class AcousticWaveOperator(CausalOperator):
    """
    1D Acoustic Wave Medium / Listener Operator:
    Models propagation phase delay, frequency-dependent attenuation, uniform
    harmonic gain, and optional coherent new harmonic mode emission.
    """

    def __init__(
        self,
        name: str = "acoustic_wave_op",
        role: str = "world_law",
        origin_axis: str = "external_world",
        phase_delay_samples: float = 0.0,
        damping_alpha: float = 0.0,
        harmonic_gain: float = 1.0,
        injected_mode: Optional[Tuple[float, float, float]] = None,
        onset_fraction: float = 0.0,
    ) -> None:
        self.phase_delay_samples = float(phase_delay_samples)
        self.damping_alpha = float(damping_alpha)
        self.harmonic_gain = float(harmonic_gain)
        self.injected_mode = injected_mode
        self.onset_fraction = float(onset_fraction)
        super().__init__(
            name=name,
            role=role,
            modality="acoustics_1d",
            origin_axis=origin_axis,
            parameters={
                "phase_delay_samples": self.phase_delay_samples,
                "damping_alpha": self.damping_alpha,
                "harmonic_gain": self.harmonic_gain,
                "injected_mode": injected_mode,
                "onset_fraction": self.onset_fraction,
            },
        )

    def _rfft_filter(self, n: int) -> np.ndarray:
        freqs = np.arange(n // 2 + 1, dtype=np.float64)
        k_max = max(1.0, float(freqs[-1]))
        phase = np.exp(-1j * (2.0 * math.pi * freqs * self.phase_delay_samples / float(n)))
        atten = np.exp(-self.damping_alpha * ((freqs / k_max) ** 2))
        return self.harmonic_gain * phase * atten

    def _mode_wave(self, n: int) -> np.ndarray:
        if self.injected_mode is None:
            return np.zeros(n, dtype=np.float64)
        freq_cycles, amp, ph = self.injected_mode
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        wave = float(amp) * np.cos(2.0 * math.pi * float(freq_cycles) * t + float(ph))
        if self.onset_fraction > 0.0:
            wave = np.where(t >= self.onset_fraction, wave, 0.0)
        return wave

    def apply(self, field: np.ndarray) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64).ravel()
        n = len(arr)
        filt = self._rfft_filter(n)
        out = np.fft.irfft(np.fft.rfft(arr) * filt, n=n).astype(np.float64)
        if self.injected_mode is not None:
            out = out + self._mode_wave(n)
        return out

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64).ravel()
        n = len(arr)
        if self.injected_mode is not None:
            arr = arr - self._mode_wave(n)
        filt = self._rfft_filter(n)
        inv_filt = np.conj(filt) / (np.abs(filt) ** 2 + max(1e-12, float(reg)))
        return np.fft.irfft(np.fft.rfft(arr) * inv_filt, n=n).astype(np.float64)


class StructuralSpectrumOperator(CausalOperator):
    """
    Modality-preserving structural operator for code execution invariants,
    chemistry stoichiometry/valence spectra, or semantic axis vectors without
    flattening native bands into spatial FFT grids.
    """

    def __init__(
        self,
        name: str,
        modality: str,
        band_names: Sequence[str],
        band_gains: Sequence[float],
        band_phase_rad: Optional[Sequence[float]] = None,
        additive_modes: Optional[Sequence[complex]] = None,
        role: str = "world_law",
        origin_axis: str = "external_world",
    ) -> None:
        self.band_names = tuple(band_names)
        self.band_gains = np.asarray(band_gains, dtype=np.float64)
        self.band_phase_rad = (
            np.asarray(band_phase_rad, dtype=np.float64)
            if band_phase_rad is not None
            else np.zeros_like(self.band_gains)
        )
        self.additive_modes = (
            np.asarray(additive_modes, dtype=np.complex128)
            if additive_modes is not None
            else np.zeros_like(self.band_gains, dtype=np.complex128)
        )
        super().__init__(
            name=name,
            role=role,
            modality=modality,
            origin_axis=origin_axis,
            parameters={
                "band_names": self.band_names,
                "band_gains": self.band_gains.tolist(),
                "band_phase_rad": self.band_phase_rad.tolist(),
            },
        )

    def apply(self, field: np.ndarray) -> np.ndarray:
        arr = np.asarray(field)
        mult = self.band_gains * np.exp(1j * self.band_phase_rad)
        out = arr.astype(np.complex128) * mult + self.additive_modes
        if not np.iscomplexobj(arr) and np.allclose(self.band_phase_rad, 0.0) and np.allclose(np.imag(self.additive_modes), 0.0):
            return np.real(out).astype(np.float64)
        return out

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        arr = np.asarray(field).astype(np.complex128) - self.additive_modes
        mult = self.band_gains * np.exp(1j * self.band_phase_rad)
        inv_mult = np.conj(mult) / (np.abs(mult) ** 2 + max(1e-12, float(reg)))
        out = arr * inv_mult
        if not np.iscomplexobj(field) and np.allclose(self.band_phase_rad, 0.0):
            return np.real(out).astype(np.float64)
        return out


class SpectralTransferOperator(CausalOperator):
    """
    Live executable CausalOperator synthesized directly from a Unit-1 mirror
    superposition: preserves extracted `transfer_H` + optional unreflected order
    so the discovered world cause can reconstruct, invert, or transfer zero-shot.
    """

    def __init__(
        self,
        name: str,
        modality: str,
        transfer_H: np.ndarray,
        support_mask: np.ndarray,
        additive_unreflected_spec: np.ndarray,
        is_structural_basis: bool,
        target_shape: Tuple[int, ...],
        role: str = "world_law",
        origin_axis: str = "external_world",
        parents: Tuple[CausalOperator, ...] = (),
    ) -> None:
        self.transfer_H = np.asarray(transfer_H, dtype=np.complex128).copy()
        self.support_mask = np.asarray(support_mask, dtype=bool).copy()
        self.additive_unreflected_spec = np.asarray(additive_unreflected_spec, dtype=np.complex128).copy()
        self.is_structural_basis = bool(is_structural_basis)
        self.target_shape = tuple(target_shape)
        self.ndim_spatial = 0 if is_structural_basis else (1 if len(target_shape) == 1 else 2)
        super().__init__(
            name=name,
            role=role,
            modality=modality,
            origin_axis=origin_axis,
            parameters={
                "spec_shape": tuple(self.transfer_H.shape),
                "support_bins": int(np.sum(self.support_mask)),
                "has_additive_order": bool(np.sum(np.abs(self.additive_unreflected_spec) ** 2) > 1e-24),
            },
            parents=parents,
        )

    def apply(self, field: np.ndarray) -> np.ndarray:
        arr = np.asarray(field)
        spec = _to_spectrum(arr, self.is_structural_basis)
        out_spec = spec * self.transfer_H + self.additive_unreflected_spec
        keep_c = np.iscomplexobj(arr) or (
            self.is_structural_basis and bool(np.any(np.abs(np.imag(out_spec)) > 1e-12))
        )
        return _from_spectrum(
            out_spec,
            target_shape=arr.shape,
            is_structural_basis=self.is_structural_basis,
            keep_complex=keep_c,
        )

    def invert(self, field: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        arr = np.asarray(field)
        spec = _to_spectrum(arr, self.is_structural_basis) - self.additive_unreflected_spec
        inv_H = np.conj(self.transfer_H) / (np.abs(self.transfer_H) ** 2 + max(1e-12, float(reg)))
        inv_spec = spec * inv_H
        keep_c = np.iscomplexobj(arr) or (
            self.is_structural_basis and bool(np.any(np.abs(np.imag(inv_spec)) > 1e-12))
        )
        return _from_spectrum(
            inv_spec,
            target_shape=arr.shape,
            is_structural_basis=self.is_structural_basis,
            keep_complex=keep_c,
        )

    def pure_additive_operator(self, order_name: str = "discovered_coherent_order") -> "SpectralTransferOperator":
        """Extracts ONLY the coherent unreflected mode so it can join the Globe cleanly."""
        return SpectralTransferOperator(
            name=order_name,
            modality=self.modality,
            transfer_H=np.ones_like(self.transfer_H, dtype=np.complex128),
            support_mask=self.support_mask,
            additive_unreflected_spec=self.additive_unreflected_spec.copy(),
            is_structural_basis=self.is_structural_basis,
            target_shape=self.target_shape,
            role="world_law",
            origin_axis="external_discovered_order",
            parents=(self,),
        )


# ---------------------------------------------------------------------------
# Planetary Reference Order Globe & Living Superposition
# ---------------------------------------------------------------------------


@dataclass
class LivingCausalSuperposition:
    """
    The complete living causal relation produced by Unit-1 mirror superposition.
    Nothing is reduced to a dead number: the globe, eye, extracted world law,
    de-lensed reality, counterfactual re-observation, zero-shot transfer, and
    self-expansion all stay alive inside this object.
    """

    globe: "PlanetaryOrderGlobe"
    observer_operator: CausalOperator
    observed_signal: np.ndarray
    delensed_world_signal: np.ndarray
    extracted_world_operator: SpectralTransferOperator
    unreflected_order_operator: Optional[SpectralTransferOperator]
    origin_kind: MirrorOriginKind
    raw_unit_deviation: float
    world_unit_deviation: float
    world_phase_shift_rad: float
    unreflected_energy_ratio: float
    unreflected_entropy: float
    unreflected_top_concentration: float
    world_transfer_H: np.ndarray
    support_mask: np.ndarray
    onset_profile: MultiScaleOnsetProfile
    explanation: str

    def reconstruct(self) -> np.ndarray:
        """Executes O_observer(M_world(G_ref)) live from first principles."""
        world_out = self.extracted_world_operator.apply(self.globe.reference_signal)
        return self.observer_operator.apply(world_out)

    def reconstruction_error(self) -> float:
        rec = self.reconstruct()
        denom = max(float(np.linalg.norm(self.observed_signal)), 1e-12)
        return float(np.linalg.norm(rec - self.observed_signal) / denom)

    def peer_through_observer(self) -> np.ndarray:
        """Returns reality stripped of the observer's own ocular/sensory distortion."""
        return self.delensed_world_signal.copy()

    def counterfactual_reobserve(self, new_observer: CausalOperator) -> "LivingCausalSuperposition":
        """Re-projects the de-lensed world through a counterfactual eye/lens."""
        reobserved = new_observer.apply(self.delensed_world_signal)
        return self.globe.superpose(
            observed_signal=reobserved,
            observer_operator=new_observer,
            world_operator_hint=self.extracted_world_operator,
        )

    def transfer_world_cause_to(self, new_reference_signal: np.ndarray) -> np.ndarray:
        """Transfers the extracted world transformation directly onto an unseen signal."""
        return self.extracted_world_operator.apply(new_reference_signal)

    def absorb_discovered_order(self, order_name: Optional[str] = None) -> "LivingCausalSuperposition":
        """
        Folds a discovered coherent external order into the Planetary Globe so
        that what was once unreflected now reflects smoothly as Unit-1!
        """
        if self.unreflected_order_operator is None:
            raise ValueError(
                f"Cannot absorb order when origin_kind={self.origin_kind.value} "
                "(no coherent unreflected order operator present)"
            )
        expanded = self.globe.expand_order(
            self.unreflected_order_operator,
            new_name=order_name or f"{self.globe.name}+coherent_order",
        )
        return expanded.superpose(
            observed_signal=self.observed_signal,
            observer_operator=self.observer_operator,
        )

    def trace_causal_lineage(self) -> Dict[str, Any]:
        return {
            "origin_kind": self.origin_kind.value,
            "globe_name": self.globe.name,
            "globe_generator": self.globe.reference_generator.describe_lineage(),
            "observer_operator": self.observer_operator.describe_lineage(),
            "extracted_world_operator": self.extracted_world_operator.describe_lineage(),
            "unreflected_order_operator": (
                self.unreflected_order_operator.describe_lineage()
                if self.unreflected_order_operator is not None
                else None
            ),
            "metrics": {
                "raw_unit_deviation": self.raw_unit_deviation,
                "world_unit_deviation": self.world_unit_deviation,
                "world_phase_shift_rad": self.world_phase_shift_rad,
                "unreflected_energy_ratio": self.unreflected_energy_ratio,
                "unreflected_entropy": self.unreflected_entropy,
                "unreflected_top_concentration": self.unreflected_top_concentration,
                "reconstruction_error": self.reconstruction_error(),
            },
            "onset_profile": {
                "dominant_scale": self.onset_profile.dominant_scale,
                "onset_coord": self.onset_profile.onset_coord,
                "macro_deviation": self.onset_profile.macro_deviation,
                "meso_deviation": self.onset_profile.meso_deviation,
                "micro_deviation": self.onset_profile.micro_deviation,
            },
        }

    def to_trinitarian_system(self) -> TrinitarianScaleRotorSystem:
        """
        Projects this Unit-1 mirror superposition into Phase 8's Trinitarian
        Scale-Binding Protocol so scalar equality & upper safety fences can
        govern or regulate the observation cleanly.
        """
        sys = TrinitarianScaleRotorSystem(name=f"mirror_{self.globe.name}")
        obs_contrib = max(0.0, float(self.raw_unit_deviation - self.world_unit_deviation))
        sys.add_quantity(CoupledQuantity("raw_dev", self.raw_unit_deviation, ScaleTier.VALUE, "", "dial", PinMode.DIAL, 0.0, 1e6))
        sys.add_quantity(CoupledQuantity("observer_contrib", obs_contrib, ScaleTier.VALUE, "", "cause", PinMode.FIXED_BOUNDARY, 0.0, 1e6))
        sys.add_quantity(CoupledQuantity("world_dev", self.world_unit_deviation, ScaleTier.FORMULA, "", "process", PinMode.FIXED_BOUNDARY, 0.0, 1e6))
        sys.add_quantity(CoupledQuantity("unreflected_ratio", self.unreflected_energy_ratio, ScaleTier.MECHANISM, "", "outcome", PinMode.FIXED_BOUNDARY, 0.0, 1.0))
        sys.add_quantity(CoupledQuantity("unreflected_max", 0.02, ScaleTier.UPPER_ROTOR, "", "fence_metric", PinMode.FIXED_BOUNDARY, 0.0, 1.0))

        unity_val = 1.0 / (1.0 + self.world_unit_deviation + self.unreflected_energy_ratio)
        sys.add_quantity(CoupledQuantity("unit_unity_score", unity_val, ScaleTier.FORMULA, "", "outcome", PinMode.FREE, 0.0, 1.0))
        sys.add_quantity(CoupledQuantity("explained_raw_dev", self.world_unit_deviation + obs_contrib, ScaleTier.MECHANISM, "", "process", PinMode.FREE, 0.0, 1e6))

        sys.add_bond(
            EqualityBond(
                name="unit_one_coherence",
                tier=ScaleTier.FORMULA,
                variables=("unit_unity_score", "world_dev", "unreflected_ratio"),
                lhs_fn=lambda v: v["unit_unity_score"],
                rhs_fn=lambda v: 1.0 / (1.0 + v["world_dev"] + v["unreflected_ratio"]),
                lhs_expr="unit_unity_score",
                rhs_expr="1 / (1 + world_dev + unreflected_ratio)",
            )
        )
        sys.add_bond(
            EqualityBond(
                name="observer_world_decomposition",
                tier=ScaleTier.MECHANISM,
                variables=("explained_raw_dev", "world_dev", "observer_contrib"),
                lhs_fn=lambda v: v["explained_raw_dev"],
                rhs_fn=lambda v: v["world_dev"] + v["observer_contrib"],
                lhs_expr="explained_raw_dev",
                rhs_expr="world_dev + observer_contrib",
            )
        )
        sys.add_bond(
            EqualityBond(
                name="mirror_reflection_closure_fence",
                tier=ScaleTier.UPPER_ROTOR,
                variables=("unreflected_ratio", "unreflected_max"),
                lhs_fn=lambda v: v["unreflected_ratio"],
                rhs_fn=lambda v: v["unreflected_max"],
                lhs_expr="unreflected_ratio",
                rhs_expr="unreflected_max",
                kind="fence_le",
                description="Energy outside Unit-1 Planetary Order mirror horizon",
            )
        )
        if self.origin_kind == MirrorOriginKind.NON_CAUSAL_RUPTURE:
            sys.add_intent(
                IntentTarget(
                    name="unity_harmony",
                    variable="unit_unity_score",
                    target_value=0.90,
                    tolerance=0.05,
                    relation="at_least",
                )
            )
        sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        return sys

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "mirror",
    ) -> Dict[str, Any]:
        """
        Projects all 5 mirror actors (Reference Globe, De-Lensed World Cause,
        Observer Eye, Reflected Order, Unreflected Residual) into `CausalField`
        with uniform 4D voxel tensors and physically responsive beams.
        """
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)

        is_unknown_order = self.origin_kind == MirrorOriginKind.UNKNOWN_COHERENT_EXTERNAL_ORDER
        is_rupture = self.origin_kind == MirrorOriginKind.NON_CAUSAL_RUPTURE

        base_strength = 1.0 / (1.0 + 10.0 * self.unreflected_energy_ratio + 5.0 * self.world_unit_deviation)
        if is_rupture:
            beam_strength = max(0.02, min(0.18, 0.25 * base_strength))
        else:
            beam_strength = max(0.05, base_strength)

        specs = [
            (
                "ref_globe",
                [1.0, 0.0, float(self.raw_unit_deviation), 1.0],
                [0.0, 0.0, 0.0],
                0.0,
            ),
            (
                "world_cause",
                [1.0 - min(1.0, self.world_unit_deviation), float(self.world_phase_shift_rad), float(self.world_unit_deviation), 0.7],
                [1.0, 0.2, 0.4],
                0.0,
            ),
            (
                "observer_eye",
                [max(0.0, float(self.raw_unit_deviation - self.world_unit_deviation)), float(self.raw_unit_deviation), 0.0, 0.4],
                [-1.0, 0.2, 0.4],
                0.0,
            ),
            (
                "reflected_order",
                [1.0 / (1.0 + self.world_unit_deviation), float(self.onset_profile.macro_deviation), float(self.onset_profile.micro_deviation), 1.0],
                [0.0, 0.8, 0.8],
                0.0,
            ),
            (
                "unreflected_residual",
                [float(self.unreflected_energy_ratio), float(self.unreflected_entropy), float(self.unreflected_top_concentration), 1.0 if is_unknown_order else 0.0],
                [0.0, -0.8, 0.8],
                1.0 if is_unknown_order else 0.0,
            ),
        ]

        vids: Dict[str, str] = {}
        for role_key, vec4, pos3, curiosity in specs:
            vid = f"{prefix}_{role_key}"
            vids[role_key] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"role": role_key, "origin_kind": self.origin_kind.value},
                    tensor=np.asarray(vec4, dtype=np.float32),
                    mass=1.5 if role_key in ("ref_globe", "reflected_order") else 1.0,
                    position=np.asarray(pos3[: field.dimensions], dtype=np.float32),
                    phase_angle=float(self.world_phase_shift_rad) % (2.0 * math.pi),
                    curiosity_charge=float(curiosity),
                )
            )

        pairs = [
            ("ref_globe", "world_cause", 1.0 / (1.0 + 5.0 * self.world_unit_deviation)),
            ("world_cause", "observer_eye", 1.0 / (1.0 + 3.0 * max(0.0, self.raw_unit_deviation - self.world_unit_deviation))),
            ("world_cause", "reflected_order", beam_strength),
            ("observer_eye", "reflected_order", 1.0 / (1.0 + self.raw_unit_deviation)),
            ("reflected_order", "unreflected_residual", beam_strength),
        ]
        strengths: List[float] = []
        for a, b, st in pairs:
            st_clamped = float(max(0.02, min(1.0, st)))
            field.link_voxels(vids[a], vids[b], strength=st_clamped)
            strengths.append(st_clamped)

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_mirror",
                name=f"Mirror_{self.globe.name}",
                position=np.array([0.0, 0.8, 0.8][: field.dimensions], dtype=np.float32),
                intensity=2.0,
                tier="macro",
                symbolic_intent=f"Unit-1 Mirror origin={self.origin_kind.value}",
            )
        )

        return {
            "causal_field": field,
            "voxels": vids,
            "beams_linked": len(pairs),
            "min_beam_strength": float(min(strengths)) if strengths else 1.0,
            "curiosity_charge": 1.0 if is_unknown_order else 0.0,
        }


class PlanetaryOrderGlobe:
    """
    The Unit-1 Reference Order Manifold for a domain/modality.
    Holds both the canonical reference signal AND its generating operator,
    and performs Unit-1 spectral superposition against observed phenomena.
    """

    def __init__(
        self,
        name: str,
        modality: str,
        reference_signal: np.ndarray,
        reference_generator: Optional[CausalOperator] = None,
        is_structural_basis: bool = False,
        band_names: Sequence[str] = (),
        physical_unit: str = "",
        support_rel_tol: float = 1e-3,
    ) -> None:
        self.name = name
        self.modality = modality
        self.reference_signal = np.asarray(reference_signal).copy()
        self.reference_generator = reference_generator or IdentityCausalOperator(
            name=f"{name}_ref_generator",
            role="reference_generator",
            modality=modality,
            origin_axis="reference_globe",
        )
        self.is_structural_basis = bool(is_structural_basis)
        self.band_names = tuple(band_names)
        self.physical_unit = physical_unit
        self.support_rel_tol = float(support_rel_tol)
        self.known_operators: List[CausalOperator] = [self.reference_generator]

    def _spectral_compare(
        self,
        candidate_signal: np.ndarray,
        support_tol: float,
    ) -> Dict[str, Any]:
        ref_spec = _to_spectrum(self.reference_signal, self.is_structural_basis)
        cand_spec = _to_spectrum(candidate_signal, self.is_structural_basis)
        if cand_spec.shape != ref_spec.shape:
            raise ValueError(
                f"Signal spectrum shape {cand_spec.shape} != Globe reference shape {ref_spec.shape}"
            )

        ref_mag = np.abs(ref_spec)
        max_mag = float(np.max(ref_mag)) if ref_mag.size else 0.0
        support_mask = ref_mag > (support_tol * max(max_mag, 1e-12))
        if not np.any(support_mask) and ref_mag.size > 0:
            support_mask = np.ones_like(ref_mag, dtype=bool)

        transfer_H = np.ones_like(ref_spec, dtype=np.complex128)
        transfer_H[support_mask] = cand_spec[support_mask] / ref_spec[support_mask]
        # Suppress catastrophic amplification where high-frequency ref bins are tiny
        excessive = np.abs(transfer_H) > 25.0
        if np.any(excessive):
            transfer_H[excessive] = transfer_H[excessive] * (25.0 / np.abs(transfer_H[excessive]))

        # Exact unreflected residual guarantees reconstructed == cand_spec everywhere
        expected_spec = ref_spec * transfer_H
        unreflected_spec = cand_spec - expected_spec
        unreflected_spec[support_mask] = 0.0 + 0.0j

        weights = (ref_mag[support_mask] ** 2)
        w_sum = max(float(np.sum(weights)), 1e-30)
        dev_sq = (np.abs(transfer_H[support_mask] - 1.0)) ** 2
        unit_deviation = float(math.sqrt(float(np.sum(weights * dev_sq) / w_sum)))

        phase_angles = np.angle(transfer_H[support_mask])
        phase_shift_rad = float(math.sqrt(float(np.sum(weights * (phase_angles ** 2)) / w_sum)))

        total_cand_energy = max(float(np.sum(np.abs(cand_spec) ** 2)), 1e-30)
        unreflected_energy = float(np.sum(np.abs(unreflected_spec) ** 2))
        unreflected_ratio = float(unreflected_energy / total_cand_energy)

        entropy, top_conc = _spectral_entropy_and_concentration(unreflected_spec)

        return {
            "ref_spec": ref_spec,
            "cand_spec": cand_spec,
            "support_mask": support_mask,
            "transfer_H": transfer_H,
            "unreflected_spec": unreflected_spec,
            "unit_deviation": unit_deviation,
            "phase_shift_rad": phase_shift_rad,
            "unreflected_energy_ratio": unreflected_ratio,
            "unreflected_entropy": entropy,
            "unreflected_top_concentration": top_conc,
        }

    def superpose(
        self,
        observed_signal: np.ndarray,
        observer_operator: Optional[CausalOperator] = None,
        world_operator_hint: Optional[CausalOperator] = None,
        support_rel_tol: Optional[float] = None,
    ) -> LivingCausalSuperposition:
        """
        Superposes `observed_signal` onto the Unit-1 Planetary Order Globe,
        peering through `observer_operator` to separate eye artifacts, lawful
        world transformations, unknown coherent orders, and non-causal ruptures.
        """
        obs_arr = np.asarray(observed_signal).copy()
        eye = observer_operator or IdentityCausalOperator(
            name="clear_identity_eye",
            role="observer_eye",
            modality=self.modality,
            origin_axis="observer_structure",
        )
        tol = float(support_rel_tol if support_rel_tol is not None else self.support_rel_tol)

        raw_cmp = self._spectral_compare(obs_arr, tol)
        delensed = eye.invert(obs_arr)
        world_cmp = self._spectral_compare(delensed, tol)

        raw_dev = raw_cmp["unit_deviation"]
        world_dev = world_cmp["unit_deviation"]
        unref_ratio = world_cmp["unreflected_energy_ratio"]
        entropy = world_cmp["unreflected_entropy"]
        top_conc = world_cmp["unreflected_top_concentration"]

        unit_tol = 0.03 if not self.is_structural_basis else 1e-4
        unref_tol = 0.02

        unref_op: Optional[SpectralTransferOperator] = None
        parents = (world_operator_hint,) if world_operator_hint is not None else ()
        extracted_world_op = SpectralTransferOperator(
            name=f"extracted_world_law[{self.name}]",
            modality=self.modality,
            transfer_H=world_cmp["transfer_H"],
            support_mask=world_cmp["support_mask"],
            additive_unreflected_spec=world_cmp["unreflected_spec"],
            is_structural_basis=self.is_structural_basis,
            target_shape=self.reference_signal.shape,
            role="world_law",
            origin_axis="external_world",
            parents=parents,
        )

        if unref_ratio > unref_tol:
            if entropy < 0.45 and top_conc > 0.50:
                origin_kind = MirrorOriginKind.UNKNOWN_COHERENT_EXTERNAL_ORDER
                unref_op = extracted_world_op.pure_additive_operator(
                    order_name=f"discovered_order[{self.name}]"
                )
            else:
                origin_kind = MirrorOriginKind.NON_CAUSAL_RUPTURE
        elif raw_dev <= unit_tol and world_dev <= unit_tol:
            origin_kind = MirrorOriginKind.UNIT_ONE_REFLECTION
        elif raw_dev > unit_tol and world_dev <= max(unit_tol, 0.15 * raw_dev):
            origin_kind = MirrorOriginKind.OBSERVER_STRUCTURE_ARTIFACT
        else:
            origin_kind = MirrorOriginKind.WORLD_LAWFUL_TRANSFORMATION

        onset_ref_target = (
            obs_arr if origin_kind == MirrorOriginKind.OBSERVER_STRUCTURE_ARTIFACT else delensed
        )
        onset = _analyze_multiscale_onset(
            self.reference_signal,
            onset_ref_target,
            is_structural_basis=self.is_structural_basis,
            band_names=self.band_names,
        )

        explanation = (
            f"[Mirror:{self.name} ({self.modality})] origin={origin_kind.value} | "
            f"raw_dev={raw_dev:.4f}, world_dev(de-lensed)={world_dev:.4f}, "
            f"phase_shift={world_cmp['phase_shift_rad']:.4f} rad, "
            f"unreflected={unref_ratio:.4f} (H_spec={entropy:.3f}, top_conc={top_conc:.3f}) | "
            f"scale={onset.dominant_scale}, onset={tuple(round(c, 3) for c in onset.onset_coord)}"
        )

        return LivingCausalSuperposition(
            globe=self,
            observer_operator=eye,
            observed_signal=obs_arr,
            delensed_world_signal=delensed,
            extracted_world_operator=extracted_world_op,
            unreflected_order_operator=unref_op,
            origin_kind=origin_kind,
            raw_unit_deviation=raw_dev,
            world_unit_deviation=world_dev,
            world_phase_shift_rad=world_cmp["phase_shift_rad"],
            unreflected_energy_ratio=unref_ratio,
            unreflected_entropy=entropy,
            unreflected_top_concentration=top_conc,
            world_transfer_H=world_cmp["transfer_H"],
            support_mask=world_cmp["support_mask"],
            onset_profile=onset,
            explanation=explanation,
        )

    def expand_order(
        self,
        new_operator: CausalOperator,
        new_name: Optional[str] = None,
    ) -> "PlanetaryOrderGlobe":
        """
        Expands this Planetary Globe around a newly validated coherent external
        order operator without erasing prior causal lineage.
        """
        expanded_ref = new_operator.apply(self.reference_signal)
        composed_gen = new_operator.compose(self.reference_generator)
        expanded = PlanetaryOrderGlobe(
            name=new_name or f"{self.name}+expanded",
            modality=self.modality,
            reference_signal=expanded_ref,
            reference_generator=composed_gen,
            is_structural_basis=self.is_structural_basis,
            band_names=self.band_names,
            physical_unit=self.physical_unit,
            support_rel_tol=self.support_rel_tol,
        )
        expanded.known_operators = list(self.known_operators) + [new_operator]
        return expanded
