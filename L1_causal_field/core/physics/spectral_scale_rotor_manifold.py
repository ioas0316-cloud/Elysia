"""
Spectral Scale-Rotor Manifold Engine
====================================

Implements a non-point-array data structure and continuous causal operator engine
designed to escape brute-force 1-to-100 sequential point iteration.

4 Continuities & Chromatic Interconnectedness (per AGENTS.md):
- Relationship: Upper macro carriers/surfaces and lower localized line/event-knot
  rotors couple through continuous Gaussian-Fourier overlap potentials rather than
  hard threshold gates.
- Connectivity: Vertical parent-child links (`parent_indices`, `parent_couplings`)
  bind multi-scale rotor tiers and project directly into `CausalField` via
  `InformationVoxel` and `ConnectivityBeam` topologies.
- Mobility: Spacetime evolution (`evolve_spacetime`) operates as an O(K) closed-form
  semigroup rotation and envelope translation across arbitrary dt, conserving phase
  momentum without step-by-step ticking, and injects flux vectors into `CausalField`.
- Informational Continuity: Smooth continuous balance axes (`SpectralBalanceScale`,
  "명암의 저울") modulate macro/micro scale emphasis, constructive/inhibitory contrast,
  phase bias, and discretization granularity across the continuous field without
  re-scanning raw sample arrays.
- Chromatic Triad: Emergent spectral balance maps naturally to Red (Flux / dynamic
  oscillation & localized kinetic energy), Blue (Order / macro structural carrier
  coherence), and Yellow (Entropy / spectral dispersion across modes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField

_ERF_UFUNC = np.frompyfunc(math.erf, 1, 1)


def _vec_erf(arr: np.ndarray) -> np.ndarray:
    """Vectorized error function backed by stdlib math.erf with float64 output."""
    arr_f = np.asarray(arr, dtype=np.float64)
    if arr_f.size == 0:
        return np.empty_like(arr_f, dtype=np.float64)
    return _ERF_UFUNC(arr_f).astype(np.float64)


def _wrap_phase(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wraps phase angle(s) into [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _containment_factor(
    centers: np.ndarray,
    sigmas: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """
    Computes exact Gaussian mass containment within [low, high]^D per mode.
    Returns shape (...,) product across dimensions.
    """
    sqrt2_s = math.sqrt(2.0) * sigmas[..., None]
    upper = _vec_erf((high - centers) / sqrt2_s)
    lower = _vec_erf((low - centers) / sqrt2_s)
    per_dim = np.clip(0.5 * (upper - lower), 0.0, 1.0)
    return np.prod(per_dim, axis=-1)


@dataclass(frozen=True)
class SpectralBalanceScale:
    """
    Continuous spectral balance axes ("명암의 저울") for observing, tuning, and
    controlling a SpectralScaleRotorManifold without point-array recomputation.

    When instantiated with defaults (0.0, 0.0, 0.0, 1.0), all scale and contrast
    multipliers evaluate to exact mathematical identity (1.0).
    """

    scale_tilt: float = 0.0
    contrast_tilt: float = 0.0
    phase_bias: float = 0.0
    granularity_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_tilt", float(np.clip(self.scale_tilt, -1.0, 1.0)))
        object.__setattr__(self, "contrast_tilt", float(np.clip(self.contrast_tilt, -1.0, 1.0)))
        object.__setattr__(self, "phase_bias", float(_wrap_phase(float(self.phase_bias))))
        object.__setattr__(self, "granularity_scale", max(1e-6, float(self.granularity_scale)))

    def band_scale_weight(self, level: int, num_levels: int, sigmas: np.ndarray) -> np.ndarray:
        """
        Smooth scale-tilt and granularity attenuation weight for modes in a band.
        - scale_tilt < 0 amplifies macro upper bands (level 0).
        - scale_tilt > 0 amplifies micro lower bands (level > 0).
        - granularity_scale > 1.0 smoothly coarsens fine localized scales while
          leaving global modes and granularity_scale <= 1.0 at exact 1.0.
        """
        sigmas_arr = np.asarray(sigmas, dtype=np.float64)
        if sigmas_arr.size == 0:
            return np.empty(0, dtype=np.float64)

        if num_levels <= 1:
            norm_level = 0.5
        else:
            norm_level = float(level) / float(num_levels - 1)

        tilt_factor = math.exp(self.scale_tilt * (norm_level - 0.5) * 2.0)

        excess_gran = max(0.0, self.granularity_scale - 1.0)
        if excess_gran == 0.0:
            gran_factor = np.ones_like(sigmas_arr, dtype=np.float64)
        else:
            eff_ratio = norm_level if level > 0 else 1.0
            local_atten = math.exp(-((excess_gran * eff_ratio) ** 2))
            gran_factor = np.where(np.isinf(sigmas_arr), 1.0, local_atten)

        return np.full(sigmas_arr.shape, tilt_factor, dtype=np.float64) * gran_factor

    def contrast_weight(self, phase_alignment: np.ndarray) -> np.ndarray:
        """
        Smooth Light/Dark contrast weight given cosine alignment in [-1, 1].
        - contrast_tilt > 0 amplifies constructive/in-phase ("bright") modes.
        - contrast_tilt < 0 amplifies destructive/anti-phase ("dark") modes.
        """
        align = np.asarray(phase_alignment, dtype=np.float64)
        return np.exp(self.contrast_tilt * align)


@dataclass
class ScaleRotorBand:
    """
    Vectorized rotor band at a single scale tier.
    Point, Line, and Surface are not separate incompatible classes; they emerge
    as scale-relative roles (`macro_surface`/`macro_carrier`, `meso_line_wave`,
    `micro_event_knot`) of the same continuous rotor substrate.
    """

    level: int
    dim: int
    wavevectors: np.ndarray
    temporal_omegas: np.ndarray
    amplitudes: np.ndarray
    centers: np.ndarray
    sigmas: np.ndarray
    is_global: bool
    parent_indices: np.ndarray
    parent_couplings: np.ndarray

    def __post_init__(self) -> None:
        d = int(self.dim)
        self.dim = d
        self.level = int(self.level)
        self.is_global = bool(self.is_global)

        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128).reshape(-1)
        k = int(self.amplitudes.shape[0])

        self.wavevectors = np.asarray(self.wavevectors, dtype=np.float64).reshape(k, d)
        self.temporal_omegas = np.asarray(self.temporal_omegas, dtype=np.float64).reshape(k)
        self.centers = np.asarray(self.centers, dtype=np.float64).reshape(k, d)
        self.sigmas = np.asarray(self.sigmas, dtype=np.float64).reshape(k)
        self.parent_indices = np.asarray(self.parent_indices, dtype=np.int32).reshape(k)
        self.parent_couplings = np.asarray(self.parent_couplings, dtype=np.float64).reshape(k)

    @property
    def num_modes(self) -> int:
        return int(self.amplitudes.shape[0])

    @property
    def role(self) -> str:
        if self.level == 0:
            return "macro_surface" if self.dim >= 2 else "macro_carrier"
        if self.level == 1:
            return "meso_line_wave"
        return "micro_event_knot"

    @property
    def nbytes(self) -> int:
        return int(
            self.wavevectors.nbytes
            + self.temporal_omegas.nbytes
            + self.amplitudes.nbytes
            + self.centers.nbytes
            + self.sigmas.nbytes
            + self.parent_indices.nbytes
            + self.parent_couplings.nbytes
        )

    def copy(self) -> "ScaleRotorBand":
        return ScaleRotorBand(
            level=self.level,
            dim=self.dim,
            wavevectors=self.wavevectors.copy(),
            temporal_omegas=self.temporal_omegas.copy(),
            amplitudes=self.amplitudes.copy(),
            centers=self.centers.copy(),
            sigmas=self.sigmas.copy(),
            is_global=self.is_global,
            parent_indices=self.parent_indices.copy(),
            parent_couplings=self.parent_couplings.copy(),
        )


def _normalize_bounds(
    dim: int,
    domain_bounds: Tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(domain_bounds[0], dtype=np.float64).reshape(dim)
    high = np.asarray(domain_bounds[1], dtype=np.float64).reshape(dim)
    if np.any(high <= low):
        raise ValueError("domain_bounds upper bound must exceed lower bound in every dimension")
    return low, high


def _band_pair_overlap_matrix(
    band_a: ScaleRotorBand,
    amp_a: np.ndarray,
    band_b: ScaleRotorBand,
    amp_b: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """
    Exact continuous L^2 inner-product contribution matrix M[j, k] between two
    real-valued wave-rotor bands over box [low, high]^D, using ZERO grid points.
    """
    ka = band_a.num_modes
    kb = band_b.num_modes
    if ka == 0 or kb == 0:
        return np.zeros((ka, kb), dtype=np.float64)

    dim = band_a.dim
    wa = band_a.wavevectors  # (Ka, D)
    wb = band_b.wavevectors  # (Kb, D)

    dw_minus = wb[None, :, :] - wa[:, None, :]  # (Ka, Kb, D)
    dw_plus = wb[None, :, :] + wa[:, None, :]   # (Ka, Kb, D)

    aa_conj_b = np.conj(amp_a)[:, None] * amp_b[None, :]
    aa_b = amp_a[:, None] * amp_b[None, :]

    if band_a.is_global and band_b.is_global:
        lengths = high - low
        mid = 0.5 * (low + high)

        def _box_fourier(dw: np.ndarray) -> np.ndarray:
            # np.sinc(z) = sin(pi*z)/(pi*z); z = dw * L / (2*pi)
            sinc_arg = (dw * lengths[None, None, :]) / (2.0 * math.pi)
            factors = (
                lengths[None, None, :]
                * np.exp(1j * dw * mid[None, None, :])
                * np.sinc(sinc_arg)
            )
            return np.prod(factors, axis=-1)

        k_minus = _box_fourier(dw_minus)
        k_plus = _box_fourier(dw_plus)
        return 0.5 * np.real(aa_conj_b * k_minus + aa_b * k_plus)

    if (not band_a.is_global) and (not band_b.is_global):
        s2_a = (band_a.sigmas ** 2)[:, None]       # (Ka, 1)
        s2_b = (band_b.sigmas ** 2)[None, :]       # (1, Kb)
        s2_sum = s2_a + s2_b                       # (Ka, Kb)
        s2_prod = s2_a * s2_b                      # (Ka, Kb)
        s2_eff = s2_prod / s2_sum                  # (Ka, Kb)
        sigma_eff = np.sqrt(s2_eff)

        mu_a = band_a.centers[:, None, :]          # (Ka, 1, D)
        mu_b = band_b.centers[None, :, :]          # (1, Kb, D)
        diff_mu_sq = np.sum((mu_a - mu_b) ** 2, axis=-1)  # (Ka, Kb)

        mu_star = (
            s2_b[:, :, None] * mu_a + s2_a[:, :, None] * mu_b
        ) / s2_sum[:, :, None]                     # (Ka, Kb, D)

        prefactor = ((2.0 * math.pi * s2_eff) ** (0.5 * dim)) * np.exp(
            -diff_mu_sq / (2.0 * s2_sum)
        )
        contain = _containment_factor(mu_star, sigma_eff, low, high)

        dw_minus_sq = np.sum(dw_minus ** 2, axis=-1)
        dw_plus_sq = np.sum(dw_plus ** 2, axis=-1)
        k_minus = prefactor * contain * np.exp(
            -0.5 * s2_eff * dw_minus_sq + 1j * np.sum(mu_star * dw_minus, axis=-1)
        )
        k_plus = prefactor * contain * np.exp(
            -0.5 * s2_eff * dw_plus_sq + 1j * np.sum(mu_star * dw_plus, axis=-1)
        )
        return 0.5 * np.real(aa_conj_b * k_minus + aa_b * k_plus)

    # Mixed Global x Local (or Local x Global)
    if not band_a.is_global:
        # Band A is localized, Band B is global
        s_loc = band_a.sigmas[:, None]             # (Ka, 1)
        s2_loc = s_loc ** 2                        # (Ka, 1)
        mu_loc = band_a.centers[:, None, :]        # (Ka, 1, D)
        contain = _containment_factor(
            band_a.centers, band_a.sigmas, low, high
        )[:, None]                                 # (Ka, 1)
    else:
        # Band A is global, Band B is localized
        s_loc = band_b.sigmas[None, :]             # (1, Kb)
        s2_loc = s_loc ** 2                        # (1, Kb)
        mu_loc = band_b.centers[None, :, :]        # (1, Kb, D)
        contain = _containment_factor(
            band_b.centers, band_b.sigmas, low, high
        )[None, :]                                 # (1, Kb)

    prefactor = ((2.0 * math.pi * s2_loc) ** (0.5 * dim)) * contain
    dw_minus_sq = np.sum(dw_minus ** 2, axis=-1)
    dw_plus_sq = np.sum(dw_plus ** 2, axis=-1)

    k_minus = prefactor * np.exp(
        -0.5 * s2_loc * dw_minus_sq + 1j * np.sum(mu_loc * dw_minus, axis=-1)
    )
    k_plus = prefactor * np.exp(
        -0.5 * s2_loc * dw_plus_sq + 1j * np.sum(mu_loc * dw_plus, axis=-1)
    )
    return 0.5 * np.real(aa_conj_b * k_minus + aa_b * k_plus)


class SpectralScaleRotorManifold:
    """
    Continuous hierarchical wave-rotor field manifold.

    All intrinsic operators (spacetime evolution, balance-scale tuning, definite
    integration, differentiation, superposition, wave resonance, and bidirectional
    Emit<->Observe coupling) operate directly on the compact spectral rotor bands
    in O(K) or O(K_A * K_B) time, independent of spatial/temporal sample count N.
    """

    def __init__(
        self,
        dim: int,
        domain_bounds: Tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray],
        bands: List[ScaleRotorBand],
        balance: Optional[SpectralBalanceScale] = None,
        current_time: float = 0.0,
    ) -> None:
        self.dim = int(dim)
        self.domain_bounds = _normalize_bounds(self.dim, domain_bounds)
        self.bands = sorted((b.copy() for b in bands), key=lambda b: b.level)
        for band in self.bands:
            if band.dim != self.dim:
                raise ValueError(f"Band dim {band.dim} does not match manifold dim {self.dim}")
        self.balance = balance if balance is not None else SpectralBalanceScale()
        self.current_time = float(current_time)

    @property
    def total_modes(self) -> int:
        return sum(b.num_modes for b in self.bands)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.bands)

    def copy(self) -> "SpectralScaleRotorManifold":
        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=[b.copy() for b in self.bands],
            balance=self.balance,
            current_time=self.current_time,
        )

    def scale_field(self, factor: float) -> "SpectralScaleRotorManifold":
        """Returns a manifold with all band amplitudes scaled by `factor` in O(K)."""
        f = float(factor)
        scaled_bands: List[ScaleRotorBand] = []
        for band in self.bands:
            bc = band.copy()
            if bc.num_modes > 0:
                bc.amplitudes = bc.amplitudes * f
            scaled_bands.append(bc)
        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=scaled_bands,
            balance=self.balance,
            current_time=self.current_time,
        )

    def prune_negligible(self, rel_tol: float = 1e-12) -> "SpectralScaleRotorManifold":
        """
        Drops modes whose amplitude magnitude is <= rel_tol * max_amp while
        preserving valid level-0 parent references.
        """
        max_amp = max(
            (float(np.max(np.abs(b.amplitudes))) for b in self.bands if b.num_modes > 0),
            default=0.0,
        )
        thresh = max(0.0, float(rel_tol)) * max_amp

        kept_masks: Dict[int, np.ndarray] = {}
        referenced_l0 = set()
        for band in self.bands:
            if band.level > 0 and band.num_modes > 0:
                mask = (np.abs(band.amplitudes) > thresh) if max_amp > 0.0 else np.zeros(band.num_modes, dtype=bool)
                kept_masks[band.level] = mask
                for pidx in band.parent_indices[mask]:
                    if pidx >= 0:
                        referenced_l0.add(int(pidx))

        l0_remap: Optional[np.ndarray] = None
        pruned_bands: List[ScaleRotorBand] = []
        for band in self.bands:
            if band.level == 0:
                if band.num_modes > 0:
                    mask0 = (np.abs(band.amplitudes) > thresh) if max_amp > 0.0 else np.zeros(band.num_modes, dtype=bool)
                    for r_idx in referenced_l0:
                        if 0 <= r_idx < band.num_modes:
                            mask0[r_idx] = True
                    kept_idx = np.where(mask0)[0]
                    l0_remap = -np.ones(band.num_modes, dtype=np.int32)
                    l0_remap[kept_idx] = np.arange(kept_idx.size, dtype=np.int32)
                    pruned_bands.append(
                        ScaleRotorBand(
                            level=0,
                            dim=self.dim,
                            wavevectors=band.wavevectors[kept_idx],
                            temporal_omegas=band.temporal_omegas[kept_idx],
                            amplitudes=band.amplitudes[kept_idx],
                            centers=band.centers[kept_idx],
                            sigmas=band.sigmas[kept_idx],
                            is_global=band.is_global,
                            parent_indices=band.parent_indices[kept_idx],
                            parent_couplings=band.parent_couplings[kept_idx],
                        )
                    )
                else:
                    pruned_bands.append(band.copy())
            else:
                if band.num_modes == 0:
                    continue
                mask = kept_masks[band.level]
                kept_idx = np.where(mask)[0]
                if kept_idx.size == 0:
                    continue
                p_idx = band.parent_indices[kept_idx].copy()
                if l0_remap is not None:
                    valid = (p_idx >= 0) & (p_idx < l0_remap.size)
                    p_idx = np.where(valid, l0_remap[np.clip(p_idx, 0, max(0, l0_remap.size - 1))], -1).astype(np.int32)
                pruned_bands.append(
                    ScaleRotorBand(
                        level=band.level,
                        dim=self.dim,
                        wavevectors=band.wavevectors[kept_idx],
                        temporal_omegas=band.temporal_omegas[kept_idx],
                        amplitudes=band.amplitudes[kept_idx],
                        centers=band.centers[kept_idx],
                        sigmas=band.sigmas[kept_idx],
                        is_global=band.is_global,
                        parent_indices=p_idx,
                        parent_couplings=band.parent_couplings[kept_idx],
                    )
                )

        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=pruned_bands,
            balance=self.balance,
            current_time=self.current_time,
        )

    def macro_reference_phase(self) -> float:
        """Dominant phase angle of the strongest non-DC mode in the macro band."""
        if not self.bands:
            return 0.0
        macro = self.bands[0]
        if macro.num_modes == 0:
            return 0.0
        w_norm = np.linalg.norm(macro.wavevectors, axis=1)
        non_dc = np.where(w_norm > 1e-12)[0]
        if non_dc.size > 0:
            mags = np.abs(macro.amplitudes[non_dc])
            best = non_dc[int(np.argmax(mags))]
            if np.abs(macro.amplitudes[best]) > 1e-15:
                return float(np.angle(macro.amplitudes[best]))
        best_any = int(np.argmax(np.abs(macro.amplitudes)))
        if np.abs(macro.amplitudes[best_any]) <= 1e-15:
            return 0.0
        return float(np.angle(macro.amplitudes[best_any]))

    def effective_amplitudes(self, band: ScaleRotorBand) -> np.ndarray:
        """
        Applies the continuous SpectralBalanceScale axes (scale_tilt, contrast_tilt,
        phase_bias, granularity_scale) to a band's amplitudes without mutating it.
        Identity when SpectralBalanceScale is neutral.
        """
        if band.num_modes == 0:
            return band.amplitudes.copy()

        scale_w = self.balance.band_scale_weight(band.level, len(self.bands), band.sigmas)
        ref_phase = self.macro_reference_phase()
        phase_align = np.cos(np.angle(band.amplitudes) - ref_phase)
        contrast_w = self.balance.contrast_weight(phase_align)
        phase_rot = np.exp(1j * self.balance.phase_bias)
        return band.amplitudes * scale_w * contrast_w * phase_rot

    def tune_balance(
        self,
        scale_tilt: Optional[float] = None,
        contrast_tilt: Optional[float] = None,
        phase_bias: Optional[float] = None,
        granularity_scale: Optional[float] = None,
    ) -> "SpectralScaleRotorManifold":
        """Returns a manifold with adjusted balance axes in O(K) copy time."""
        new_balance = SpectralBalanceScale(
            scale_tilt=self.balance.scale_tilt if scale_tilt is None else scale_tilt,
            contrast_tilt=self.balance.contrast_tilt if contrast_tilt is None else contrast_tilt,
            phase_bias=self.balance.phase_bias if phase_bias is None else phase_bias,
            granularity_scale=(
                self.balance.granularity_scale if granularity_scale is None else granularity_scale
            ),
        )
        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=[b.copy() for b in self.bands],
            balance=new_balance,
            current_time=self.current_time,
        )

    def evolve_spacetime(
        self,
        dt: float,
        velocity: Optional[Sequence[float] | np.ndarray] = None,
    ) -> "SpectralScaleRotorManifold":
        """
        Closed-form semigroup spacetime evolution over arbitrary time jump dt
        (forward or backward) in O(K) operations, with zero step-by-step iteration.
        """
        dt_f = float(dt)
        vel = None
        if velocity is not None:
            vel = np.asarray(velocity, dtype=np.float64).reshape(self.dim)

        new_bands: List[ScaleRotorBand] = []
        for band in self.bands:
            b_copy = band.copy()
            if b_copy.num_modes > 0:
                phase_shift = -(b_copy.temporal_omegas * dt_f)
                if vel is not None:
                    phase_shift = phase_shift - (b_copy.wavevectors @ vel) * dt_f
                    if not b_copy.is_global:
                        b_copy.centers = b_copy.centers + vel[None, :] * dt_f
                b_copy.amplitudes = b_copy.amplitudes * np.exp(1j * phase_shift)
            new_bands.append(b_copy)

        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=new_bands,
            balance=self.balance,
            current_time=self.current_time + dt_f,
        )

    def differentiate(self, axis: int = 0) -> "SpectralScaleRotorManifold":
        """
        Analytic spectral differentiation along spatial `axis`.
        Exact for global rotor bands; for localized Gaussian wave-packets applies
        the carrier spectral derivative 1j * w_axis (valid in carrier-dominated regime).
        """
        ax = int(axis)
        if not (0 <= ax < self.dim):
            raise ValueError(f"axis {ax} out of range for dim {self.dim}")

        new_bands: List[ScaleRotorBand] = []
        for band in self.bands:
            b_copy = band.copy()
            if b_copy.num_modes > 0:
                b_copy.amplitudes = b_copy.amplitudes * (1j * b_copy.wavevectors[:, ax])
            new_bands.append(b_copy)

        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=new_bands,
            balance=self.balance,
            current_time=self.current_time,
        )

    def superpose(self, other: "SpectralScaleRotorManifold") -> "SpectralScaleRotorManifold":
        """
        Linearly superposes two manifolds at matching scale tiers in O(K_A + K_B),
        preserving vertical parent index offsets.
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot superpose dim {self.dim} with dim {other.dim}")

        bands_a = {b.level: b for b in self.bands}
        bands_b = {b.level: b for b in other.bands}
        all_levels = sorted(set(bands_a.keys()) | set(bands_b.keys()))

        level0_offset = bands_a[0].num_modes if 0 in bands_a else 0
        merged_bands: List[ScaleRotorBand] = []

        for lvl in all_levels:
            ba = bands_a.get(lvl)
            bb = bands_b.get(lvl)
            if ba is not None and bb is None:
                bc = ba.copy()
                bc.amplitudes = self.effective_amplitudes(ba)
                merged_bands.append(bc)
            elif bb is not None and ba is None:
                bc = bb.copy()
                bc.amplitudes = other.effective_amplitudes(bb)
                if lvl > 0 and level0_offset > 0:
                    valid_mask = bc.parent_indices >= 0
                    bc.parent_indices = np.where(
                        valid_mask, bc.parent_indices + level0_offset, -1
                    ).astype(np.int32)
                merged_bands.append(bc)
            else:
                assert ba is not None and bb is not None
                if ba.is_global != bb.is_global:
                    raise ValueError(f"Level {lvl} mismatch in is_global during superpose")
                amp_a = self.effective_amplitudes(ba)
                amp_b = other.effective_amplitudes(bb)
                p_idx_b = bb.parent_indices.copy()
                if lvl > 0 and level0_offset > 0:
                    valid_mask = p_idx_b >= 0
                    p_idx_b = np.where(
                        valid_mask, p_idx_b + level0_offset, -1
                    ).astype(np.int32)
                merged_bands.append(
                    ScaleRotorBand(
                        level=lvl,
                        dim=self.dim,
                        wavevectors=np.concatenate([ba.wavevectors, bb.wavevectors], axis=0),
                        temporal_omegas=np.concatenate(
                            [ba.temporal_omegas, bb.temporal_omegas], axis=0
                        ),
                        amplitudes=np.concatenate([amp_a, amp_b], axis=0),
                        centers=np.concatenate([ba.centers, bb.centers], axis=0),
                        sigmas=np.concatenate([ba.sigmas, bb.sigmas], axis=0),
                        is_global=ba.is_global,
                        parent_indices=np.concatenate([ba.parent_indices, p_idx_b], axis=0),
                        parent_couplings=np.concatenate(
                            [ba.parent_couplings, bb.parent_couplings], axis=0
                        ),
                    )
                )

        low = np.minimum(self.domain_bounds[0], other.domain_bounds[0])
        high = np.maximum(self.domain_bounds[1], other.domain_bounds[1])
        return SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=(low, high),
            bands=merged_bands,
            balance=SpectralBalanceScale(),
            current_time=0.5 * (self.current_time + other.current_time),
        )

    def integrate_over_domain(
        self,
        bounds: Optional[Tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] = None,
    ) -> float:
        """
        Exact O(K) analytic definite integral of the continuous field over [low, high]^D.
        Never builds or scans a spatial point grid.
        """
        low, high = (
            self.domain_bounds
            if bounds is None
            else _normalize_bounds(self.dim, bounds)
        )
        lengths = high - low
        mid = 0.5 * (low + high)
        total = 0.0

        for band in self.bands:
            if band.num_modes == 0:
                continue
            eff_amp = self.effective_amplitudes(band)
            w = band.wavevectors  # (K, D)
            if band.is_global:
                sinc_arg = (w * lengths[None, :]) / (2.0 * math.pi)
                per_dim = (
                    lengths[None, :]
                    * np.exp(1j * w * mid[None, :])
                    * np.sinc(sinc_arg)
                )
                integrals = np.prod(per_dim, axis=1)
            else:
                sigma_col = band.sigmas[:, None]
                base_per_dim = (
                    math.sqrt(2.0 * math.pi)
                    * sigma_col
                    * np.exp(-0.5 * ((sigma_col * w) ** 2) + 1j * w * band.centers)
                )
                contain = _containment_factor(band.centers, band.sigmas, low, high)
                integrals = np.prod(base_per_dim, axis=1) * contain

            total += float(np.sum(np.real(eff_amp * integrals)))

        return total

    def _raw_inner_product_breakdown(
        self,
        other: "SpectralScaleRotorManifold",
        low: np.ndarray,
        high: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Returns (raw_overlap, constructive_overlap, destructive_overlap)."""
        eff_self = [self.effective_amplitudes(b) for b in self.bands]
        eff_other = [other.effective_amplitudes(b) for b in other.bands]

        raw_overlap = 0.0
        constructive = 0.0
        destructive = 0.0

        for b_a, a_a in zip(self.bands, eff_self):
            for b_b, a_b in zip(other.bands, eff_other):
                mat = _band_pair_overlap_matrix(b_a, a_a, b_b, a_b, low, high)
                if mat.size == 0:
                    continue
                raw_overlap += float(np.sum(mat))
                constructive += float(np.sum(np.maximum(mat, 0.0)))
                destructive += float(np.sum(np.minimum(mat, 0.0)))

        return raw_overlap, constructive, destructive

    def wave_resonance(
        self,
        other: "SpectralScaleRotorManifold",
        bounds: Optional[Tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Pure O(K_A * K_B) exact continuous L^2 inner-product resonance between two
        manifolds over [low, high]^D. Zero point grid and zero periodic aliasing.
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot compute resonance between dim {self.dim} and {other.dim}")

        if bounds is None:
            low = np.minimum(self.domain_bounds[0], other.domain_bounds[0])
            high = np.maximum(self.domain_bounds[1], other.domain_bounds[1])
        else:
            low, high = _normalize_bounds(self.dim, bounds)

        raw_overlap, constructive, destructive = self._raw_inner_product_breakdown(
            other, low, high
        )
        energy_a, _, _ = self._raw_inner_product_breakdown(self, low, high)
        energy_b, _, _ = other._raw_inner_product_breakdown(other, low, high)

        energy_a = max(0.0, energy_a)
        energy_b = max(0.0, energy_b)
        denom = math.sqrt(max(energy_a * energy_b, 1e-24))
        norm_res = 0.0 if (energy_a <= 1e-24 or energy_b <= 1e-24) else float(
            np.clip(raw_overlap / denom, -1.0, 1.0)
        )

        return {
            "raw_overlap": raw_overlap,
            "constructive_overlap": constructive,
            "destructive_overlap": destructive,
            "energy_self": energy_a,
            "energy_other": energy_b,
            "normalized_resonance": norm_res,
        }

    def bidirectional_couple(
        self,
        observer: "SpectralScaleRotorManifold",
        coupling_gain: float = 0.25,
        inhibition_gain: float = 0.20,
    ) -> Tuple["SpectralScaleRotorManifold", "SpectralScaleRotorManifold", Dict[str, Any]]:
        """
        Double-buffered, order-independent Emit <-> Observe coupling.
        - Constructive overlap pulls phases toward mutual alignment.
        - Destructive overlap applies continuous inhibitory amplitude damping.
        Operates purely on rotor bands in O(K_E * K_O).
        """
        if self.dim != observer.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {observer.dim}")

        low = np.minimum(self.domain_bounds[0], observer.domain_bounds[0])
        high = np.maximum(self.domain_bounds[1], observer.domain_bounds[1])

        eff_emit = [self.effective_amplitudes(b) for b in self.bands]
        eff_obs = [observer.effective_amplitudes(b) for b in observer.bands]

        diag_emit = [
            np.maximum(
                np.diag(_band_pair_overlap_matrix(b, ea, b, ea, low, high)), 1e-18
            )
            if b.num_modes > 0
            else np.empty(0, dtype=np.float64)
            for b, ea in zip(self.bands, eff_emit)
        ]
        diag_obs = [
            np.maximum(
                np.diag(_band_pair_overlap_matrix(b, eb, b, eb, low, high)), 1e-18
            )
            if b.num_modes > 0
            else np.empty(0, dtype=np.float64)
            for b, eb in zip(observer.bands, eff_obs)
        ]

        # Precompute all pairwise band interactions on immutable snapshots first
        emit_phase_pulls = [np.zeros(b.num_modes, dtype=np.float64) for b in self.bands]
        emit_pos_weights = [np.zeros(b.num_modes, dtype=np.float64) for b in self.bands]
        emit_inh_weights = [np.zeros(b.num_modes, dtype=np.float64) for b in self.bands]

        obs_phase_pulls = [np.zeros(b.num_modes, dtype=np.float64) for b in observer.bands]
        obs_pos_weights = [np.zeros(b.num_modes, dtype=np.float64) for b in observer.bands]
        obs_inh_weights = [np.zeros(b.num_modes, dtype=np.float64) for b in observer.bands]

        total_constructive = 0.0
        total_destructive = 0.0

        for ie, (be, ea, de) in enumerate(zip(self.bands, eff_emit, diag_emit)):
            if be.num_modes == 0:
                continue
            phi_e = np.angle(ea)
            for io, (bo, eo, do) in enumerate(zip(observer.bands, eff_obs, diag_obs)):
                if bo.num_modes == 0:
                    continue
                mat = _band_pair_overlap_matrix(be, ea, bo, eo, low, high)
                norm_mat = mat / np.sqrt(de[:, None] * do[None, :])
                norm_mat = np.clip(norm_mat, -1.0, 1.0)

                pos_mat = np.maximum(norm_mat, 0.0)
                neg_mat = np.maximum(-norm_mat, 0.0)

                total_constructive += float(np.sum(np.maximum(mat, 0.0)))
                total_destructive += float(np.sum(np.minimum(mat, 0.0)))

                phi_o = np.angle(eo)
                diff_e_to_o = _wrap_phase(phi_o[None, :] - phi_e[:, None])
                diff_o_to_e = -diff_e_to_o

                emit_phase_pulls[ie] += np.sum(pos_mat * diff_e_to_o, axis=1)
                emit_pos_weights[ie] += np.sum(pos_mat, axis=1)
                emit_inh_weights[ie] += np.sum(neg_mat, axis=1)

                obs_phase_pulls[io] += np.sum(pos_mat * diff_o_to_e, axis=0)
                obs_pos_weights[io] += np.sum(pos_mat, axis=0)
                obs_inh_weights[io] += np.sum(neg_mat, axis=0)

        new_emit_bands: List[ScaleRotorBand] = []
        emit_damping_total = 0.0
        for ie, b in enumerate(self.bands):
            bc = b.copy()
            if bc.num_modes > 0:
                pos_w = emit_pos_weights[ie]
                avg_pull = np.where(
                    pos_w > 1e-12, emit_phase_pulls[ie] / np.maximum(pos_w, 1e-12), 0.0
                )
                eff_pos = np.tanh(pos_w)
                delta_phi = (0.5 * coupling_gain) * eff_pos * avg_pull
                inh_scale = 1.0 / (1.0 + inhibition_gain * emit_inh_weights[ie])
                emit_damping_total += float(np.sum(1.0 - inh_scale))
                bc.amplitudes = bc.amplitudes * inh_scale * np.exp(1j * delta_phi)
            new_emit_bands.append(bc)

        new_obs_bands: List[ScaleRotorBand] = []
        obs_damping_total = 0.0
        for io, b in enumerate(observer.bands):
            bc = b.copy()
            if bc.num_modes > 0:
                pos_w = obs_pos_weights[io]
                avg_pull = np.where(
                    pos_w > 1e-12, obs_phase_pulls[io] / np.maximum(pos_w, 1e-12), 0.0
                )
                eff_pos = np.tanh(pos_w)
                delta_phi = coupling_gain * eff_pos * avg_pull
                inh_scale = 1.0 / (1.0 + inhibition_gain * obs_inh_weights[io])
                obs_damping_total += float(np.sum(1.0 - inh_scale))
                bc.amplitudes = bc.amplitudes * inh_scale * np.exp(1j * delta_phi)
            new_obs_bands.append(bc)

        coupled_emitter = SpectralScaleRotorManifold(
            dim=self.dim,
            domain_bounds=self.domain_bounds,
            bands=new_emit_bands,
            balance=self.balance,
            current_time=self.current_time,
        )
        coupled_observer = SpectralScaleRotorManifold(
            dim=observer.dim,
            domain_bounds=observer.domain_bounds,
            bands=new_obs_bands,
            balance=observer.balance,
            current_time=observer.current_time,
        )
        metrics = {
            "constructive_overlap": total_constructive,
            "destructive_overlap": total_destructive,
            "emitter_damping_sum": emit_damping_total,
            "observer_damping_sum": obs_damping_total,
            "resonance_after": coupled_emitter.wave_resonance(coupled_observer)[
                "normalized_resonance"
            ],
        }
        return coupled_emitter, coupled_observer, metrics

    def slice_manifestation(self, coords: np.ndarray) -> np.ndarray:
        """
        Manifests (evaluates) the continuous wave-rotor field ONLY at the requested
        query coordinates `coords` of shape (..., D). Used at visualization / test
        boundaries, never inside core rotor operations.
        """
        pts = np.asarray(coords, dtype=np.float64)
        if self.dim == 1 and pts.ndim == 1:
            pts = pts[:, None]
        if pts.shape[-1] != self.dim:
            raise ValueError(f"Expected coords last dim {self.dim}, got {pts.shape[-1]}")

        out_shape = pts.shape[:-1]
        flat_pts = pts.reshape(-1, self.dim)  # (M, D)
        accum = np.zeros(flat_pts.shape[0], dtype=np.float64)

        for band in self.bands:
            if band.num_modes == 0:
                continue
            eff_amp = self.effective_amplitudes(band)
            # Phase dot: (M, K)
            phase_arg = flat_pts @ band.wavevectors.T
            if band.is_global:
                contrib = eff_amp[None, :] * np.exp(1j * phase_arg)
            else:
                diff = flat_pts[:, None, :] - band.centers[None, :, :]  # (M, K, D)
                dist_sq = np.sum(diff ** 2, axis=-1)                    # (M, K)
                env = np.exp(-dist_sq / (2.0 * (band.sigmas[None, :] ** 2)))
                contrib = eff_amp[None, :] * env * np.exp(1j * phase_arg)
            accum += np.sum(np.real(contrib), axis=1)

        return accum.reshape(out_shape)

    def spectral_axis_profile(self) -> Dict[str, Any]:
        """
        Summarizes the manifold's scale hierarchy, Light/Dark phase balance, and
        Chromatic Triad [Red (Flux), Blue (Order), Yellow (Entropy)] in O(K).
        """
        low, high = self.domain_bounds
        ref_phase = self.macro_reference_phase()

        per_level_energy: Dict[int, float] = {}
        per_level_modes: Dict[int, int] = {}
        per_level_roles: Dict[int, str] = {}

        macro_energy = 0.0
        micro_energy = 0.0
        bright_energy = 0.0
        dark_energy = 0.0
        weighted_level_sum = 0.0
        total_spectral_power = 0.0
        spectral_powers: List[np.ndarray] = []

        for band in self.bands:
            eff_amp = self.effective_amplitudes(band)
            if band.num_modes == 0:
                lvl_e = 0.0
            else:
                mat = _band_pair_overlap_matrix(band, eff_amp, band, eff_amp, low, high)
                lvl_e = max(0.0, float(np.sum(mat)))

                mode_pow = np.abs(eff_amp) ** 2
                align = np.cos(np.angle(eff_amp) - ref_phase)
                bright_energy += float(np.sum(mode_pow * np.maximum(align, 0.0)))
                dark_energy += float(np.sum(mode_pow * np.maximum(-align, 0.0)))

                total_p = float(np.sum(mode_pow))
                total_spectral_power += total_p
                weighted_level_sum += float(band.level) * total_p
                spectral_powers.append(mode_pow)

            per_level_energy[band.level] = per_level_energy.get(band.level, 0.0) + lvl_e
            per_level_modes[band.level] = per_level_modes.get(band.level, 0) + band.num_modes
            per_level_roles[band.level] = band.role

            if band.level == 0:
                macro_energy += lvl_e
            else:
                micro_energy += lvl_e

        total_l2_energy = macro_energy + micro_energy
        micro_to_macro_ratio = micro_energy / max(macro_energy, 1e-18)
        bright_dark_sum = bright_energy + dark_energy
        light_dark_balance = (
            (bright_energy - dark_energy) / bright_dark_sum
            if bright_dark_sum > 1e-18
            else 0.0
        )
        bright_to_dark_ratio = bright_energy / max(dark_energy, 1e-18)
        scale_centroid = (
            weighted_level_sum / total_spectral_power
            if total_spectral_power > 1e-18
            else 0.0
        )

        # Chromatic Triad [Red=Flux, Blue=Order, Yellow=Entropy]
        if spectral_powers:
            all_p = np.concatenate(spectral_powers)
            p_sum = float(np.sum(all_p))
            if p_sum > 1e-18 and all_p.size > 1:
                probs = all_p / p_sum
                probs = probs[probs > 1e-15]
                entropy_norm = float(
                    -np.sum(probs * np.log(probs)) / math.log(max(2, all_p.size))
                )
            else:
                entropy_norm = 0.0
        else:
            entropy_norm = 0.0

        red_flux = (micro_energy / max(total_l2_energy, 1e-18)) + 0.15
        blue_order = (macro_energy / max(total_l2_energy, 1e-18)) + 0.15
        yellow_entropy = max(0.05, entropy_norm)
        chroma = np.array([red_flux, blue_order, yellow_entropy], dtype=np.float64)
        chroma /= float(np.sum(chroma))

        return {
            "per_level_energy": per_level_energy,
            "per_level_modes": per_level_modes,
            "per_level_roles": per_level_roles,
            "macro_energy": macro_energy,
            "micro_energy": micro_energy,
            "micro_to_macro_ratio": micro_to_macro_ratio,
            "bright_energy": bright_energy,
            "dark_energy": dark_energy,
            "bright_to_dark_ratio": bright_to_dark_ratio,
            "light_dark_balance": light_dark_balance,
            "scale_centroid": scale_centroid,
            "chromatic_triad": chroma.tolist(),
        }

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "rotor",
    ) -> Dict[str, Any]:
        """
        Projects every ScaleRotorBand into an InformationVoxel inside CausalField,
        wires vertical parent-child ConnectivityBeams across scale levels, and
        injects an explicit Mobility impact vector from spectral balance gradient.
        """
        from core.physics.causal_field import CausalField, InformationVoxel

        cf_dim = max(3, self.dim)
        field = causal_field if causal_field is not None else CausalField(dimensions=cf_dim)
        profile = self.spectral_axis_profile()
        chroma_arr = np.array(profile["chromatic_triad"], dtype=np.float32)

        voxel_ids: List[str] = []
        beam_count = 0

        for band in self.bands:
            vid = f"{prefix}_L{band.level}"
            eff_amp = self.effective_amplitudes(band)
            band_energy = profile["per_level_energy"].get(band.level, 0.0)

            if band.num_modes > 0:
                weights = np.abs(eff_amp) ** 2
                w_sum = float(np.sum(weights)) + 1e-18
                mean_w = np.sum(band.wavevectors * weights[:, None], axis=0) / w_sum
                mean_omega = float(np.sum(band.temporal_omegas * weights) / w_sum)
                mean_phase = float(np.angle(np.sum(eff_amp)))
                sig_vec = np.array(
                    [
                        float(mean_w[0]) if self.dim >= 1 else 0.0,
                        float(mean_w[1]) if self.dim >= 2 else mean_omega,
                        float(band_energy),
                        float(self.balance.scale_tilt),
                        float(self.balance.contrast_tilt),
                        float(band.num_modes),
                    ],
                    dtype=np.float32,
                )
            else:
                mean_w = np.zeros(self.dim, dtype=np.float64)
                mean_omega = 0.0
                mean_phase = 0.0
                sig_vec = np.zeros(6, dtype=np.float32)

            pos = np.zeros(max(3, field.dimensions), dtype=np.float32)
            for d_idx in range(min(self.dim, len(pos) - 1)):
                pos[d_idx] = float(mean_w[d_idx])
            pos[-1] = float(band.level)

            voxel = InformationVoxel(
                id=vid,
                content={"role": band.role, "level": band.level, "num_modes": band.num_modes},
                tensor=sig_vec,
                mass=max(0.1, float(band_energy) + 0.1),
                position=pos[: field.dimensions],
                chromatic_vector=chroma_arr.copy(),
                phase_angle=mean_phase % (2.0 * math.pi),
                scale_level=1.0 / (2.0 ** band.level),
            )
            field.add_voxel(voxel)
            voxel_ids.append(vid)

            if band.level > 0 and f"{prefix}_L{band.level - 1}" in field.voxels:
                avg_coupling = (
                    float(np.mean(band.parent_couplings)) if band.num_modes > 0 else 1.0
                )
                field.link_voxels(
                    f"{prefix}_L{band.level - 1}",
                    vid,
                    strength=max(0.1, avg_coupling),
                )
                beam_count += 1

        # Conservation of Mobility: every scale-balance projection carries an explicit impact
        impulse = np.zeros(field.dimensions, dtype=np.float32)
        impulse[0] = float(profile["chromatic_triad"][0] * (1.0 + self.balance.scale_tilt))
        if field.dimensions >= 2:
            impulse[1] = float(profile["light_dark_balance"])
        if field.dimensions >= 3:
            impulse[2] = float(self.balance.contrast_tilt)

        for vid in voxel_ids:
            field.apply_impact(vid, impulse)

        return {
            "causal_field": field,
            "voxel_ids": voxel_ids,
            "beams_linked": beam_count,
            "chromatic_triad": profile["chromatic_triad"],
        }


def _solve_2x2_packet(
    r_patch: np.ndarray,
    g_patch: np.ndarray,
    phase_arg: np.ndarray,
) -> complex:
    """
    Exact 2x2 real least-squares fit of Re(c * g * exp(i * phase_arg)) to r_patch.
    Guaranteed to reduce residual norm cleanly while handling DC (sin == 0).
    """
    phi1 = g_patch * np.cos(phase_arg)
    phi2 = -g_patch * np.sin(phase_arg)
    n11 = float(np.dot(phi1, phi1))
    n22 = float(np.dot(phi2, phi2))
    rhs1 = float(np.dot(r_patch, phi1))

    if n22 <= 1e-18:
        c_r = rhs1 / max(n11, 1e-24)
        return complex(c_r, 0.0)

    n12 = float(np.dot(phi1, phi2))
    rhs2 = float(np.dot(r_patch, phi2))
    gram = np.array([[n11 + 1e-12, n12], [n12, n22 + 1e-12]], dtype=np.float64)
    sol = np.linalg.solve(gram, np.array([rhs1, rhs2], dtype=np.float64))
    return complex(float(sol[0]), float(sol[1]))


def encode_waveform_to_scale_rotors(
    samples: np.ndarray,
    domain: Tuple[float, float] = (0.0, 1.0),
    max_macro_modes: int = 16,
    max_levels: int = 3,
    max_knots_per_level: int = 8,
    residual_tol: float = 0.03,
    base_sigma_ratio: float = 0.06,
    dispersion_speed: float = 1.0,
) -> SpectralScaleRotorManifold:
    """
    Boundary encoder converting a legacy 1D point array into a hierarchical
    SpectralScaleRotorManifold:
    - Level 0: Global macro carrier modes via rfft (stops early if residual_tol met).
    - Levels 1+: Adaptive localized Gaussian wave-rotors spawned ONLY where
      residual tension exceeds tolerance.
    """
    sig = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = sig.size
    if n < 2:
        raise ValueError("samples must contain at least 2 points")

    a, b = float(domain[0]), float(domain[1])
    length = b - a
    if length <= 0.0:
        raise ValueError("domain end must exceed start")

    dx = length / float(n)
    x = a + np.arange(n, dtype=np.float64) * dx

    fft_vals = np.fft.rfft(sig) / float(n)
    num_bins = fft_vals.size
    weights = np.full(num_bins, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0

    energies = weights * (np.abs(fft_vals) ** 2)
    total_energy = float(np.sum(energies))
    sig_norm = float(np.linalg.norm(sig))

    order = np.argsort(energies)[::-1]
    chosen: List[int] = []
    cum_energy = 0.0
    tol_energy = (residual_tol ** 2) * total_energy
    for idx in order[:max_macro_modes]:
        if energies[idx] <= 1e-24 and chosen:
            break
        chosen.append(int(idx))
        cum_energy += float(energies[idx])
        if (total_energy - cum_energy) <= tol_energy:
            break

    if not chosen:
        chosen = [0]

    chosen_arr = np.array(chosen, dtype=np.int64)
    w_macro = (2.0 * math.pi * chosen_arr.astype(np.float64)) / length
    amps_macro = weights[chosen_arr] * fft_vals[chosen_arr] * np.exp(-1j * w_macro * a)

    band0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=w_macro[:, None],
        temporal_omegas=dispersion_speed * np.abs(w_macro),
        amplitudes=amps_macro,
        centers=np.zeros((len(chosen), 1), dtype=np.float64),
        sigmas=np.full(len(chosen), np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=-np.ones(len(chosen), dtype=np.int32),
        parent_couplings=np.ones(len(chosen), dtype=np.float64),
    )

    bands: List[ScaleRotorBand] = [band0]
    if sig_norm <= 1e-12 or max_levels <= 1:
        return SpectralScaleRotorManifold(dim=1, domain_bounds=([a], [b]), bands=bands)

    # Evaluate macro residual once on the boundary grid
    macro_eval = np.sum(
        np.real(amps_macro[None, :] * np.exp(1j * (x[:, None] * w_macro[None, :]))),
        axis=1,
    )
    residual = sig - macro_eval

    for level in range(1, max_levels):
        rel_err = float(np.linalg.norm(residual)) / sig_norm
        if rel_err <= residual_tol:
            break

        sigma_l = length * base_sigma_ratio / (2.0 ** (level - 1))
        half_fit = max(8, int(math.ceil(4.0 * sigma_l / dx)))
        half_sub = max(12, int(math.ceil(6.0 * sigma_l / dx)))

        w_list: List[float] = []
        omega_list: List[float] = []
        amp_list: List[complex] = []
        center_list: List[float] = []
        sigma_list: List[float] = []
        p_idx_list: List[int] = []
        p_coup_list: List[float] = []

        for _ in range(max_knots_per_level):
            if (float(np.linalg.norm(residual)) / sig_norm) <= residual_tol:
                break

            peak_idx = int(np.argmax(np.abs(residual)))
            if abs(residual[peak_idx]) <= 1e-12 * sig_norm:
                break

            mu = float(x[peak_idx])
            lo_fit = max(0, peak_idx - half_fit)
            hi_fit = min(n, peak_idx + half_fit + 1)
            x_fit = x[lo_fit:hi_fit]
            r_fit = residual[lo_fit:hi_fit]
            g_fit = np.exp(-0.5 * (((x_fit - mu) / sigma_l) ** 2))

            patch_len = max(1, len(x_fit)) * dx
            local_fft = np.fft.rfft(r_fit * g_fit)
            m_peak = int(np.argmax(np.abs(local_fft)))
            w_local = (2.0 * math.pi * float(m_peak)) / patch_len

            c_val = _solve_2x2_packet(r_fit, g_fit, w_local * x_fit)
            if abs(c_val) <= 1e-15:
                break

            lo_sub = max(0, peak_idx - half_sub)
            hi_sub = min(n, peak_idx + half_sub + 1)
            x_sub = x[lo_sub:hi_sub]
            g_sub = np.exp(-0.5 * (((x_sub - mu) / sigma_l) ** 2))
            recon_sub = np.real(c_val * g_sub * np.exp(1j * w_local * x_sub))

            old_sub_energy = float(np.sum(residual[lo_sub:hi_sub] ** 2))
            new_sub_res = residual[lo_sub:hi_sub] - recon_sub
            if float(np.sum(new_sub_res ** 2)) >= old_sub_energy:
                break

            residual[lo_sub:hi_sub] = new_sub_res

            prox = np.exp(-0.5 * ((sigma_l * (w_local - w_macro)) ** 2))
            p_idx = int(np.argmax(prox))
            p_coup = float(prox[p_idx])

            w_list.append(w_local)
            omega_list.append(dispersion_speed * abs(w_local))
            amp_list.append(c_val)
            center_list.append(mu)
            sigma_list.append(sigma_l)
            p_idx_list.append(p_idx)
            p_coup_list.append(p_coup)

        if w_list:
            bands.append(
                ScaleRotorBand(
                    level=level,
                    dim=1,
                    wavevectors=np.array(w_list, dtype=np.float64)[:, None],
                    temporal_omegas=np.array(omega_list, dtype=np.float64),
                    amplitudes=np.array(amp_list, dtype=np.complex128),
                    centers=np.array(center_list, dtype=np.float64)[:, None],
                    sigmas=np.array(sigma_list, dtype=np.float64),
                    is_global=False,
                    parent_indices=np.array(p_idx_list, dtype=np.int32),
                    parent_couplings=np.array(p_coup_list, dtype=np.float64),
                )
            )

    return SpectralScaleRotorManifold(dim=1, domain_bounds=([a], [b]), bands=bands)


def encode_surface_to_scale_rotors(
    grid: np.ndarray,
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
    max_macro_modes: int = 32,
    max_levels: int = 3,
    max_knots_per_level: int = 12,
    residual_tol: float = 0.05,
    base_sigma_ratio: float = 0.07,
    dispersion_speed: float = 1.0,
) -> SpectralScaleRotorManifold:
    """
    Boundary encoder converting a legacy 2D grid (Ny, Nx) into a hierarchical
    SpectralScaleRotorManifold:
    - Level 0: Global macro surface modes via unique-mode rfft2.
    - Levels 1+: Adaptive localized 2D Gaussian wave-rotors spawned ONLY where
      edges, ridges, or event knots exceed residual tolerance.
    """
    surf = np.asarray(grid, dtype=np.float64)
    if surf.ndim != 2 or surf.shape[0] < 2 or surf.shape[1] < 2:
        raise ValueError("grid must be a 2D array with shape >= (2, 2)")

    ny, nx = surf.shape
    ay, by = float(domain_bounds[0][0]), float(domain_bounds[0][1])
    ax, bx = float(domain_bounds[1][0]), float(domain_bounds[1][1])
    ly, lx = by - ay, bx - ax
    if ly <= 0.0 or lx <= 0.0:
        raise ValueError("domain_bounds upper bound must exceed lower bound")

    dy = ly / float(ny)
    dx = lx / float(nx)
    y_coords = ay + np.arange(ny, dtype=np.float64) * dy
    x_coords = ax + np.arange(nx, dtype=np.float64) * dx

    fft_vals = np.fft.rfft2(surf) / float(ny * nx)
    half_nx = nx // 2 + 1
    weights = np.full((ny, half_nx), 2.0, dtype=np.float64)

    # Deduplicate self-conjugate columns mx=0 and (if nx even) mx=nx//2
    self_cols = [0] + ([nx // 2] if nx % 2 == 0 else [])
    for col in self_cols:
        weights[:, col] = 2.0
        weights[0, col] = 1.0
        if ny % 2 == 0:
            weights[ny // 2, col] = 1.0
        if ny > 2:
            weights[ny // 2 + 1 :, col] = 0.0

    energies = weights * (np.abs(fft_vals) ** 2)
    total_energy = float(np.sum(energies))
    surf_norm = float(np.linalg.norm(surf))

    flat_e = energies.reshape(-1)
    order = np.argsort(flat_e)[::-1]
    chosen_flat: List[int] = []
    cum_energy = 0.0
    tol_energy = (residual_tol ** 2) * total_energy
    for idx in order[:max_macro_modes]:
        if flat_e[idx] <= 1e-24 and chosen_flat:
            break
        chosen_flat.append(int(idx))
        cum_energy += float(flat_e[idx])
        if (total_energy - cum_energy) <= tol_energy:
            break

    if not chosen_flat:
        chosen_flat = [0]

    iy_sel, ix_sel = np.unravel_index(np.array(chosen_flat, dtype=np.int64), fft_vals.shape)
    my_signed = np.where(iy_sel <= ny // 2, iy_sel, iy_sel - ny).astype(np.float64)
    mx_vals = ix_sel.astype(np.float64)

    wy_macro = (2.0 * math.pi * my_signed) / ly
    wx_macro = (2.0 * math.pi * mx_vals) / lx
    w_macro = np.stack([wy_macro, wx_macro], axis=1)  # (K0, 2)
    amps_macro = (
        weights[iy_sel, ix_sel]
        * fft_vals[iy_sel, ix_sel]
        * np.exp(-1j * (wy_macro * ay + wx_macro * ax))
    )
    omegas_macro = dispersion_speed * np.linalg.norm(w_macro, axis=1)

    band0 = ScaleRotorBand(
        level=0,
        dim=2,
        wavevectors=w_macro,
        temporal_omegas=omegas_macro,
        amplitudes=amps_macro,
        centers=np.zeros((len(chosen_flat), 2), dtype=np.float64),
        sigmas=np.full(len(chosen_flat), np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=-np.ones(len(chosen_flat), dtype=np.int32),
        parent_couplings=np.ones(len(chosen_flat), dtype=np.float64),
    )

    bands: List[ScaleRotorBand] = [band0]
    if surf_norm <= 1e-12 or max_levels <= 1:
        return SpectralScaleRotorManifold(
            dim=2, domain_bounds=([ay, ax], [by, bx]), bands=bands
        )

    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    pts = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    phase_grid = pts @ w_macro.T
    macro_eval = np.sum(np.real(amps_macro[None, :] * np.exp(1j * phase_grid)), axis=1)
    residual = surf - macro_eval.reshape(ny, nx)

    l_min = min(ly, lx)
    for level in range(1, max_levels):
        if (float(np.linalg.norm(residual)) / surf_norm) <= residual_tol:
            break

        sigma_l = l_min * base_sigma_ratio / (2.0 ** (level - 1))
        ry_fit = max(4, int(math.ceil(4.0 * sigma_l / dy)))
        rx_fit = max(4, int(math.ceil(4.0 * sigma_l / dx)))
        ry_sub = max(6, int(math.ceil(6.0 * sigma_l / dy)))
        rx_sub = max(6, int(math.ceil(6.0 * sigma_l / dx)))

        w_list: List[Tuple[float, float]] = []
        omega_list: List[float] = []
        amp_list: List[complex] = []
        center_list: List[Tuple[float, float]] = []
        sigma_list: List[float] = []
        p_idx_list: List[int] = []
        p_coup_list: List[float] = []

        for _ in range(max_knots_per_level):
            if (float(np.linalg.norm(residual)) / surf_norm) <= residual_tol:
                break

            peak_flat = int(np.argmax(np.abs(residual)))
            py, px = np.unravel_index(peak_flat, residual.shape)
            if abs(residual[py, px]) <= 1e-12 * surf_norm:
                break

            mu_y = float(y_coords[py])
            mu_x = float(x_coords[px])

            y0_f, y1_f = max(0, py - ry_fit), min(ny, py + ry_fit + 1)
            x0_f, x1_f = max(0, px - rx_fit), min(nx, px + rx_fit + 1)
            r_fit = residual[y0_f:y1_f, x0_f:x1_f]
            yy_f, xx_f = np.meshgrid(y_coords[y0_f:y1_f], x_coords[x0_f:x1_f], indexing="ij")
            g_fit = np.exp(
                -(((yy_f - mu_y) ** 2 + (xx_f - mu_x) ** 2) / (2.0 * (sigma_l ** 2)))
            )

            py_len, px_len = r_fit.shape
            local_fft = np.fft.rfft2(r_fit * g_fit)
            pk_y, pk_x = np.unravel_index(int(np.argmax(np.abs(local_fft))), local_fft.shape)
            my_loc = float(pk_y if pk_y <= py_len // 2 else pk_y - py_len)
            wy_loc = (2.0 * math.pi * my_loc) / (float(py_len) * dy)
            wx_loc = (2.0 * math.pi * float(pk_x)) / (float(px_len) * dx)

            phase_fit = (wy_loc * yy_f + wx_loc * xx_f).reshape(-1)
            c_val = _solve_2x2_packet(r_fit.reshape(-1), g_fit.reshape(-1), phase_fit)
            if abs(c_val) <= 1e-15:
                break

            y0_s, y1_s = max(0, py - ry_sub), min(ny, py + ry_sub + 1)
            x0_s, x1_s = max(0, px - rx_sub), min(nx, px + rx_sub + 1)
            yy_s, xx_s = np.meshgrid(y_coords[y0_s:y1_s], x_coords[x0_s:x1_s], indexing="ij")
            g_sub = np.exp(
                -(((yy_s - mu_y) ** 2 + (xx_s - mu_x) ** 2) / (2.0 * (sigma_l ** 2)))
            )
            recon_sub = np.real(c_val * g_sub * np.exp(1j * (wy_loc * yy_s + wx_loc * xx_s)))

            old_sub_energy = float(np.sum(residual[y0_s:y1_s, x0_s:x1_s] ** 2))
            new_sub_res = residual[y0_s:y1_s, x0_s:x1_s] - recon_sub
            if float(np.sum(new_sub_res ** 2)) >= old_sub_energy:
                break

            residual[y0_s:y1_s, x0_s:x1_s] = new_sub_res

            dist_w_sq = (wy_loc - w_macro[:, 0]) ** 2 + (wx_loc - w_macro[:, 1]) ** 2
            prox = np.exp(-0.5 * (sigma_l ** 2) * dist_w_sq)
            p_idx = int(np.argmax(prox))
            p_coup = float(prox[p_idx])

            w_list.append((wy_loc, wx_loc))
            omega_list.append(dispersion_speed * math.hypot(wy_loc, wx_loc))
            amp_list.append(c_val)
            center_list.append((mu_y, mu_x))
            sigma_list.append(sigma_l)
            p_idx_list.append(p_idx)
            p_coup_list.append(p_coup)

        if w_list:
            bands.append(
                ScaleRotorBand(
                    level=level,
                    dim=2,
                    wavevectors=np.array(w_list, dtype=np.float64),
                    temporal_omegas=np.array(omega_list, dtype=np.float64),
                    amplitudes=np.array(amp_list, dtype=np.complex128),
                    centers=np.array(center_list, dtype=np.float64),
                    sigmas=np.array(sigma_list, dtype=np.float64),
                    is_global=False,
                    parent_indices=np.array(p_idx_list, dtype=np.int32),
                    parent_couplings=np.array(p_coup_list, dtype=np.float64),
                )
            )

    return SpectralScaleRotorManifold(
        dim=2, domain_bounds=([ay, ax], [by, bx]), bands=bands
    )
