"""
Elysia Sensory Orchestra OS
===========================

Top-level causal orchestrator uniting:
1. A 3-tier Layered Spacetime Globe Rotor Deck (Macro -> Meso -> Micro) using
   `CausalScaleRotorField`, treating spacetime animation like a reversible,
   continuously scrubbable Globe rather than sequential 1-to-N frame stepping.
2. Human 5-Sense Grounded Instruments (`OpticalPrismLightInstrument`,
   `HarmonicAcousticInstrument`, `TactileResistanceInstrument`), each enforcing
   domain-native physical causality (Planck spectrum -> Cauchy prism dispersion
   -> analytic normals & Fresnel; harmonic resonance -> damped vibration;
   spectral roughness -> thermal warmth & coupling tension).
3. `ElysiaOrchestraOS` conductor binding all modalities via `SpectralBalanceScale`
   ("명암의 저울"), cross-modal chromatic triad coherence, human-readable causal
   explanations, and `CausalField` projection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
import wave

import numpy as np

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


# Physical visible spectrum bands (nanometers): violet -> red
WAVELENGTHS_NM = np.array(
    [430.0, 470.0, 510.0, 550.0, 590.0, 630.0, 670.0], dtype=np.float64
)


def planck_spectrum(kelvin: float) -> np.ndarray:
    """
    Computes normalized blackbody spectral radiance across the 7 visible bands
    [430..670nm] using Planck's radiation law.
    """
    temp = max(500.0, float(kelvin))
    h = 6.62607015e-34
    c = 2.99792458e8
    k_b = 1.380649e-23
    lam_m = WAVELENGTHS_NM * 1e-9
    exponent = np.clip((h * c) / (lam_m * k_b * temp), 1e-6, 700.0)
    radiance = 1.0 / ((lam_m ** 5) * np.expm1(exponent))
    total = float(np.sum(radiance))
    return radiance / max(total, 1e-30)


def cauchy_refractive_index(dispersion_strength: float) -> np.ndarray:
    """
    Cauchy dispersion law n(lambda) = A + B / lambda_um^2 across visible bands.
    Shorter blue/violet wavelengths experience strictly higher refractive index.
    """
    lam_um = WAVELENGTHS_NM * 1e-3
    strength = float(np.clip(dispersion_strength, 0.05, 2.5))
    return 1.52 + (0.045 * strength) / (lam_um ** 2)


def _build_white_balanced_cie_matrix() -> np.ndarray:
    """
    7x3 human cone sensitivity matrix (R, G, B) calibrated so that a 6500K
    Planck daylight spectrum maps to neutral white [1.0, 1.0, 1.0].
    """
    raw = np.array(
        [
            [0.28, 0.04, 1.00],  # 430 nm (violet: short-cone + secondary red lobe)
            [0.05, 0.22, 0.85],  # 470 nm (blue)
            [0.04, 0.78, 0.22],  # 510 nm (cyan-green)
            [0.45, 1.00, 0.03],  # 550 nm (green-yellow)
            [0.92, 0.65, 0.00],  # 590 nm (amber)
            [1.00, 0.16, 0.00],  # 630 nm (orange-red)
            [0.72, 0.02, 0.00],  # 670 nm (deep red)
        ],
        dtype=np.float64,
    )
    white_ref = planck_spectrum(6500.0) @ raw
    return raw / np.maximum(white_ref[None, :], 1e-12)


CIE_CONE_MATRIX = _build_white_balanced_cie_matrix()


@dataclass
class SpacetimeGlobeState:
    """Analytic O(1) snapshot of the 3-tier Macro -> Meso -> Micro spacetime globe."""

    time: float
    scale_zoom: float
    macro_angle: float
    macro_pos: np.ndarray
    meso_angle: float
    meso_pos: np.ndarray
    micro_angle: float
    micro_pos: np.ndarray
    solar_elevation_rad: float
    solar_azimuth_rad: float
    solar_kelvin: float
    prism_dispersion_strength: float
    micro_roughness: float


class SpacetimeGlobeRotorDeck:
    """
    Layered spacetime animation deck built on `CausalScaleRotorField`.
    Arbitrary times t (past, present, future) are evaluated analytically in O(1)
    with zero frame-by-frame sequential stepping.
    """

    def __init__(self, day_period: float = 10.0) -> None:
        self.day_period = max(1e-3, float(day_period))
        self.rotor_field = CausalScaleRotorField(
            [
                RotorNode(
                    "macro_world",
                    ScaleRotor(
                        axis=(0.0, 0.25, 1.0),
                        angular_rate=(2.0 * math.pi) / self.day_period,
                        radius=1.0,
                    ),
                ),
                RotorNode(
                    "meso_form",
                    ScaleRotor(
                        axis=(0.2, 0.0, 1.0),
                        angular_rate=(2.0 * math.pi) / 3.5,
                        radius=0.35,
                    ),
                    parent_id="macro_world",
                ),
                RotorNode(
                    "micro_texture",
                    ScaleRotor(
                        axis=(0.0, 0.35, 1.0),
                        angular_rate=(2.0 * math.pi) / 0.9,
                        radius=0.12,
                    ),
                    parent_id="meso_form",
                ),
            ]
        )

    def slice_spacetime(self, t: float, scale_zoom: float = 1.0) -> SpacetimeGlobeState:
        """Evaluates exact Macro/Meso/Micro globe coordinates at arbitrary time `t`."""
        t_f = float(t)
        zoom = max(0.25, float(scale_zoom))
        positions = self.rotor_field.slice_at(t_f)

        macro_angle = self.rotor_field.nodes["macro_world"].rotor.angle_at(t_f)
        meso_angle = self.rotor_field.nodes["meso_form"].rotor.angle_at(t_f)
        micro_angle = self.rotor_field.nodes["micro_texture"].rotor.angle_at(t_f)

        # Dawn (t=0) -> 0.25 rad; Noon (t=2.5) -> 1.40 rad; Dusk (t=5) -> 0.25 rad; Twilight (t=7.5) -> -0.90 rad
        solar_elev = 0.25 + 1.15 * math.sin(macro_angle)
        solar_az = macro_angle
        elev_norm = float(np.clip((math.sin(solar_elev) + 0.2) / 1.2, 0.0, 1.0))
        solar_kelvin = 2800.0 + 3700.0 * elev_norm

        dispersion_strength = 0.55 + 0.45 * (0.5 + 0.5 * math.cos(meso_angle))
        base_rough = 0.18 + 0.14 * (0.5 + 0.5 * math.sin(micro_angle))
        micro_roughness = float(np.clip(base_rough / math.sqrt(zoom), 0.04, 0.75))

        return SpacetimeGlobeState(
            time=t_f,
            scale_zoom=zoom,
            macro_angle=macro_angle,
            macro_pos=positions["macro_world"],
            meso_angle=meso_angle,
            meso_pos=positions["meso_form"],
            micro_angle=micro_angle,
            micro_pos=positions["micro_texture"],
            solar_elevation_rad=solar_elev,
            solar_azimuth_rad=solar_az,
            solar_kelvin=solar_kelvin,
            prism_dispersion_strength=dispersion_strength,
            micro_roughness=micro_roughness,
        )


@dataclass
class InstrumentPerformance:
    """Output of a single sensory instrument for one spacetime slice."""

    instrument_id: str
    modality: str
    causal_state: Dict[str, Any]
    rotor_manifold: Optional[SpectralScaleRotorManifold]
    manifestation: Optional[np.ndarray]
    chromatic_triad: Tuple[float, float, float]
    explanation: str


class SensoryInstrument(ABC):
    """Base contract for instruments grounded in human sensory causal physics."""

    instrument_id: str
    modality: str

    @abstractmethod
    def perform(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
        manifest: bool = False,
        resolution: int = 128,
        context: Optional[Dict[str, InstrumentPerformance]] = None,
    ) -> InstrumentPerformance:
        raise NotImplementedError


class OpticalPrismLightInstrument(SensoryInstrument):
    """
    Visual causal hierarchy:
    Planck Blackbody Light -> Cauchy Prism Snell Dispersion -> 3-Tier Rotor
    Crystal Surface -> Schlick Fresnel & Micro-Roughness -> Human Cone Retina.
    """

    def __init__(self, instrument_id: str = "vision.optical_prism") -> None:
        self.instrument_id = instrument_id
        self.modality = "vision"

    def build_surface_manifold(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
    ) -> SpectralScaleRotorManifold:
        zoom = globe_state.scale_zoom
        zoom_freq = 1.0 + 0.45 * (zoom - 1.0)

        # Level 0: Macro crystal prism dome carriers
        w0 = np.array(
            [
                [2.0 * math.pi * 1.5, 2.0 * math.pi * 0.8],
                [-2.0 * math.pi * 0.8, 2.0 * math.pi * 1.5],
                [2.0 * math.pi * 2.5, -2.0 * math.pi * 1.2],
            ],
            dtype=np.float64,
        )
        a0 = np.array(
            [
                0.55 * np.exp(-1j * 0.5 * globe_state.macro_angle),
                0.45 * np.exp(1j * 0.3 * globe_state.macro_angle),
                0.30 * np.exp(-1j * 0.4 * globe_state.meso_angle),
            ],
            dtype=np.complex128,
        )
        b0 = ScaleRotorBand(
            level=0,
            dim=2,
            wavevectors=w0,
            temporal_omegas=np.linalg.norm(w0, axis=1),
            amplitudes=a0,
            centers=np.zeros((3, 2), dtype=np.float64),
            sigmas=np.full(3, np.inf, dtype=np.float64),
            is_global=True,
            parent_indices=np.array([-1, -1, -1], dtype=np.int32),
            parent_couplings=np.ones(3, dtype=np.float64),
        )

        # Level 1: Meso prism ridge facets orbiting with meso_angle
        meso_angles = globe_state.meso_angle + np.array([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
        c1 = 0.5 + 0.20 * np.stack([np.sin(meso_angles), np.cos(meso_angles)], axis=1)
        w1 = (2.0 * math.pi * 6.0 * zoom_freq) * np.stack(
            [np.cos(meso_angles), np.sin(meso_angles)], axis=1
        )
        a1 = np.full(4, 0.42, dtype=np.complex128) * np.exp(1j * 0.4 * meso_angles)
        b1 = ScaleRotorBand(
            level=1,
            dim=2,
            wavevectors=w1,
            temporal_omegas=np.linalg.norm(w1, axis=1),
            amplitudes=a1,
            centers=c1,
            sigmas=np.full(4, 0.11 / math.sqrt(zoom), dtype=np.float64),
            is_global=False,
            parent_indices=np.array([0, 1, 2, 0], dtype=np.int32),
            parent_couplings=np.full(4, 0.85, dtype=np.float64),
        )

        # Level 2: Micro crystal texture ripples orbiting with micro_angle
        micro_angles = globe_state.micro_angle + np.array([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
        c2 = 0.5 + 0.13 * np.stack([np.cos(micro_angles), np.sin(micro_angles)], axis=1)
        w2 = (2.0 * math.pi * 14.0 * zoom_freq) * np.stack(
            [-np.sin(micro_angles), np.cos(micro_angles)], axis=1
        )
        a2 = np.full(4, 0.24, dtype=np.complex128) * np.exp(-1j * 0.3 * micro_angles)
        b2 = ScaleRotorBand(
            level=2,
            dim=2,
            wavevectors=w2,
            temporal_omegas=np.linalg.norm(w2, axis=1),
            amplitudes=a2,
            centers=c2,
            sigmas=np.full(4, 0.045 / math.sqrt(zoom), dtype=np.float64),
            is_global=False,
            parent_indices=np.array([0, 1, 2, 1], dtype=np.int32),
            parent_couplings=np.full(4, 0.70, dtype=np.float64),
        )

        base_manifold = SpectralScaleRotorManifold(
            dim=2,
            domain_bounds=([0.0, 0.0], [1.0, 1.0]),
            bands=[b0, b1, b2],
            current_time=globe_state.time,
        )
        twist_angle = 0.25 * math.sin(globe_state.macro_angle)
        return palette.compose_and_apply(
            base_manifold,
            [("code.clifford_rotor_twist", 1.0, {"angle": twist_angle, "scale": 1.0})],
            mode="sequential",
            balance=balance,
        )

    def _render_spectral_retina(
        self,
        surface: SpectralScaleRotorManifold,
        globe_state: SpacetimeGlobeState,
        balance: SpectralBalanceScale,
        sun_spec: np.ndarray,
        n_cauchy: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        h = w = max(16, int(resolution))
        ys = np.linspace(0.0, 1.0, h, endpoint=False) + 0.5 / h
        xs = np.linspace(0.0, 1.0, w, endpoint=False) + 0.5 / w
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        pts = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)

        # Evaluate multi-scale crystal wave relief at display boundary only
        h_wave = surface.slice_manifestation(pts).reshape(h, w)
        dy = 1.0 / h
        dx = 1.0 / w
        dh_dy, dh_dx = np.gradient(h_wave, dy, dx)

        # Central spherical prism globe geometry + surface facet relief
        cx, cy = xx - 0.5, yy - 0.5
        r_sq = cx ** 2 + cy ** 2
        sphere_r = 0.40
        inside_mask = r_sq < (sphere_r ** 2)
        z_sphere = np.sqrt(np.maximum((sphere_r ** 2) - r_sq, 0.0))

        # Sphere normal blended with sculpted rotor facet slopes
        scale_slope = 0.024
        nx_raw = np.where(inside_mask, cx / sphere_r - scale_slope * dh_dx, -0.45 * scale_slope * dh_dx)
        ny_raw = np.where(inside_mask, cy / sphere_r - scale_slope * dh_dy, -0.45 * scale_slope * dh_dy)
        nz_raw = np.where(inside_mask, z_sphere / sphere_r, 1.0)
        n_norm = np.sqrt(nx_raw ** 2 + ny_raw ** 2 + nz_raw ** 2) + 1e-12
        nx_u, ny_u, nz_u = nx_raw / n_norm, ny_raw / n_norm, nz_raw / n_norm

        # Light direction from macro globe solar elevation & azimuth
        elev = globe_state.solar_elevation_rad
        az = globe_state.solar_azimuth_rad
        lx = math.cos(elev) * math.cos(az)
        ly = math.cos(elev) * math.sin(az)
        lz = max(0.18, math.sin(elev) * 0.85 + 0.30)
        l_len = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / l_len, ly / l_len, lz / l_len

        # View vector looking down +Z
        vx, vy, vz = 0.0, 0.0, 1.0
        hx, hy, hz = lx + vx, ly + vy, lz + vz
        h_len = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-12
        hx, hy, hz = hx / h_len, hy / h_len, hz / h_len

        n_dot_l = np.clip(nx_u * lx + ny_u * ly + nz_u * lz, 0.0, 1.0)
        n_dot_v = np.clip(nz_u, 0.05, 1.0)
        n_dot_h = np.clip(nx_u * hx + ny_u * hy + nz_u * hz, 0.0, 1.0)

        # Sun intensity & Rayleigh atmospheric sky spectrum across 7 wavelengths
        sun_vis = float(np.clip((math.sin(elev) + 0.35) / 1.35, 0.06, 1.0))
        rayleigh = (550.0 / WAVELENGTHS_NM) ** 4
        rayleigh = rayleigh / np.sum(rayleigh)
        # Twilight moonlight / deep sky scattering vs daytime sky
        effective_source = (0.85 * sun_vis + 0.05) * sun_spec + (0.28 * (1.0 - sun_vis)) * rayleigh

        # Atmospheric background gradient across y (zenith blue, golden horizon at dawn/dusk)
        horizon = np.clip(yy, 0.0, 1.0)[:, :, None]
        warm_horizon = (1.0 - 0.55 * sun_vis) * sun_vis * (horizon ** 1.8)
        bg_spec = (
            (0.08 + 0.42 * sun_vis) * rayleigh[None, None, :] * (1.15 - 0.50 * horizon)
            + (0.55 * warm_horizon + 0.04) * sun_spec[None, None, :]
        )

        # Per-wavelength Schlick Fresnel (7,) -> broadcast (H, W, 7)
        f0 = ((1.0 - n_cauchy) / (1.0 + n_cauchy)) ** 2  # (7,)
        fresnel = f0[None, None, :] + (1.0 - f0[None, None, :]) * ((1.0 - n_dot_v[:, :, None]) ** 5)

        # Per-wavelength Snell refraction T(lambda) -> narrow-lobe rainbow prism caustics
        # Double-refraction angular deflection inside crystal sphere/prism separates 430nm..670nm cleanly
        eta = 1.0 / n_cauchy  # (7,)
        cos_i = n_dot_l[:, :, None]  # (H, W, 1)
        k_sq = 1.0 - (eta[None, None, :] ** 2) * (1.0 - cos_i ** 2)
        cos_t = np.sqrt(np.maximum(k_sq, 0.0))
        tx = eta[None, None, :] * (-lx) + (eta[None, None, :] * cos_i - cos_t) * nx_u[:, :, None]
        ty = eta[None, None, :] * (-ly) + (eta[None, None, :] * cos_i - cos_t) * ny_u[:, :, None]
        tz = eta[None, None, :] * (-lz) + (eta[None, None, :] * cos_i - cos_t) * nz_u[:, :, None]

        caustic_dir = np.array(
            [0.52 * math.cos(globe_state.meso_angle), 0.52 * math.sin(globe_state.meso_angle), -0.68]
        )
        caustic_dir /= np.linalg.norm(caustic_dir)
        t_dot_c = np.clip(tx * caustic_dir[0] + ty * caustic_dir[1] + tz * caustic_dir[2], 0.0, 1.0)
        # Angular half-width ~4.5 deg matches Cauchy refractive angle spread so rainbow bands resolve cleanly
        prism_caustic = np.exp(-((1.0 - t_dot_c) / 0.0032)) * (1.0 - fresnel) * (
            0.9 + 2.8 * globe_state.prism_dispersion_strength
        )

        # Micro-roughness Blinn-Phong specular reflection
        blinn_exp = float(np.clip(2.0 / max(globe_state.micro_roughness ** 2, 0.008), 8.0, 180.0))
        spec_lobe = (n_dot_h[:, :, None] ** blinn_exp) * (0.4 + 2.2 * fresnel)

        # Crystal body spectral radiance (diffuse shading + vivid chromatic dispersion + Fresnel rim)
        crystal_body = (
            effective_source[None, None, :] * (
                0.55 * n_dot_l[:, :, None] * (0.20 + 0.80 * (1.0 - fresnel))
                + 2.60 * prism_caustic
                + 1.35 * spec_lobe
            )
            + 0.35 * rayleigh[None, None, :] * fresnel
        )

        # Smooth anti-aliased blending between crystal globe/prism and background atmosphere
        edge_dist = (sphere_r - np.sqrt(r_sq)) / max(1.5 / h, 1e-6)
        alpha = np.clip(edge_dist, 0.0, 1.0)[:, :, None]
        bg_modulated = bg_spec * (1.0 + 0.16 * np.tanh(h_wave[:, :, None]))
        radiance_7 = alpha * crystal_body + (1.0 - alpha) * bg_modulated

        # Apply "Light/Dark balance scale" contrast_tilt cleanly in spectral luminance
        contrast_gain = math.exp(0.32 * balance.contrast_tilt)
        radiance_7 = np.maximum(0.0, radiance_7 * contrast_gain)

        # Project 7 physical wavelengths onto CIE human cone responses -> RGB
        rgb_lin = np.einsum("hwc,cj->hwj", radiance_7, CIE_CONE_MATRIX) * 1.65
        rgb_lin = np.maximum(rgb_lin, 0.0)

        # Filmic Reinhard tone mapping + sRGB gamma 2.2
        rgb_tone = rgb_lin / (1.0 + rgb_lin)
        rgb_srgb = np.power(np.clip(rgb_tone, 0.0, 1.0), 1.0 / 2.2)
        return (rgb_srgb * 255.0 + 0.5).astype(np.uint8)

    def perform(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
        manifest: bool = False,
        resolution: int = 128,
        context: Optional[Dict[str, InstrumentPerformance]] = None,
    ) -> InstrumentPerformance:
        surface = self.build_surface_manifold(globe_state, palette, balance)
        sun_spec = planck_spectrum(globe_state.solar_kelvin)
        n_cauchy = cauchy_refractive_index(globe_state.prism_dispersion_strength)
        f0 = ((1.0 - n_cauchy) / (1.0 + n_cauchy)) ** 2

        # Pure O(K) analytic spectral gradient energy
        slope_energy = 0.0
        total_energy = 0.0
        for band in surface.bands:
            if band.num_modes == 0:
                continue
            eff = surface.effective_amplitudes(band)
            p = np.abs(eff) ** 2
            w_sq = np.sum(band.wavevectors ** 2, axis=1)
            slope_energy += float(np.sum(p * w_sq))
            total_energy += float(np.sum(p))
        mean_slope = math.sqrt(slope_energy / max(total_energy, 1e-18))

        dom_wl = float(WAVELENGTHS_NM[int(np.argmax(sun_spec))])
        delta_n = float(n_cauchy[0] - n_cauchy[-1])

        image_rgb = (
            self._render_spectral_retina(
                surface, globe_state, balance, sun_spec, n_cauchy, resolution
            )
            if manifest
            else None
        )

        prof = surface.spectral_axis_profile()
        causal_state = {
            "solar_kelvin": round(globe_state.solar_kelvin, 1),
            "solar_elevation_deg": round(math.degrees(globe_state.solar_elevation_rad), 1),
            "dominant_wavelength_nm": dom_wl,
            "cauchy_n_violet": round(float(n_cauchy[0]), 4),
            "cauchy_n_red": round(float(n_cauchy[-1]), 4),
            "chromatic_dispersion_delta_n": round(delta_n, 4),
            "fresnel_f0_violet": round(float(f0[0]), 4),
            "fresnel_f0_red": round(float(f0[-1]), 4),
            "micro_roughness": round(globe_state.micro_roughness, 4),
            "analytic_slope_norm": round(mean_slope, 3),
        }
        explanation = (
            f"Sunlight at {causal_state['solar_kelvin']:.0f}K (elev {causal_state['solar_elevation_deg']:.1f}deg, "
            f"peak {dom_wl:.0f}nm) refracts through prism dispersion delta_n={delta_n:.4f} "
            f"(n_430={n_cauchy[0]:.3f} > n_670={n_cauchy[-1]:.3f}) over roughness {globe_state.micro_roughness:.3f}."
        )
        return InstrumentPerformance(
            instrument_id=self.instrument_id,
            modality=self.modality,
            causal_state=causal_state,
            rotor_manifold=surface,
            manifestation=image_rgb,
            chromatic_triad=tuple(prof["chromatic_triad"]),  # type: ignore[arg-type]
            explanation=explanation,
        )


class HarmonicAcousticInstrument(SensoryInstrument):
    """
    Auditory causal hierarchy:
    Globe Macro Pitch -> Meso Consonant Chord Ratios -> Micro Overtone Shimmer
    -> Palette Harmonic Resonance & Damping -> Ear PCM Waveform at boundary.
    """

    def __init__(
        self,
        instrument_id: str = "hearing.harmonic_synth",
        duration_s: float = 0.65,
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
        # Smooth continuous fundamental around C4 (261.63 Hz) driven by macro globe angle
        f0 = 261.63 * (2.0 ** (0.15 * math.sin(globe_state.macro_angle)))
        dur = self.duration_s

        # Level 0: Consonant just-intonation chord carriers (root, major 3rd, fifth, octave)
        ratios_0 = np.array([1.0, 1.25, 1.5, 2.0], dtype=np.float64)
        w0 = (2.0 * math.pi * f0 * ratios_0)[:, None]
        chord_weights = np.array(
            [
                1.0,
                0.55 * (0.5 + 0.5 * math.cos(globe_state.meso_angle)),
                0.70 * (0.5 + 0.5 * math.sin(globe_state.meso_angle + 0.4)),
                0.45,
            ],
            dtype=np.float64,
        )
        a0 = chord_weights * np.exp(-1j * (0.25 * globe_state.macro_angle * ratios_0))
        b0 = ScaleRotorBand(
            level=0,
            dim=1,
            wavevectors=w0,
            temporal_omegas=w0[:, 0],
            amplitudes=a0.astype(np.complex128),
            centers=np.zeros((4, 1), dtype=np.float64),
            sigmas=np.full(4, np.inf, dtype=np.float64),
            is_global=True,
            parent_indices=-np.ones(4, dtype=np.int32),
            parent_couplings=np.ones(4, dtype=np.float64),
        )

        # Level 1: Micro overtone shimmer bursts synchronized with micro_angle
        ratios_1 = np.array([3.0, 4.0], dtype=np.float64)
        w1 = (2.0 * math.pi * f0 * ratios_1)[:, None]
        a1 = np.array(
            [
                0.35 * (0.5 + 0.5 * math.cos(globe_state.micro_angle)),
                0.22 * (0.5 + 0.5 * math.sin(globe_state.micro_angle)),
            ],
            dtype=np.complex128,
        ) * np.exp(1j * 0.5 * globe_state.micro_angle)
        b1 = ScaleRotorBand(
            level=1,
            dim=1,
            wavevectors=w1,
            temporal_omegas=w1[:, 0],
            amplitudes=a1,
            centers=np.array([[0.32 * dur], [0.68 * dur]], dtype=np.float64),
            sigmas=np.array([0.22 * dur, 0.22 * dur], dtype=np.float64),
            is_global=False,
            parent_indices=np.array([1, 2], dtype=np.int32),
            parent_couplings=np.array([0.85, 0.85], dtype=np.float64),
        )

        raw_acoustic = SpectralScaleRotorManifold(
            dim=1,
            domain_bounds=([0.0], [dur]),
            bands=[b0, b1],
            current_time=globe_state.time,
        )
        # Apply acoustic harmonic resonance + damped harmonic relaxation via Palette
        damping_rate = 0.35 + 0.45 * globe_state.micro_roughness
        acoustic = palette.compose_and_apply(
            raw_acoustic,
            [
                (
                    "acoustics.harmonic_resonance",
                    1.0,
                    {"fundamental": 2.0 * math.pi * f0, "max_overtones": 5, "bandwidth": 2.0 * math.pi * 80.0},
                ),
                (
                    "physics.damped_harmonic",
                    1.0,
                    {"damping": damping_rate, "tau": 0.35},
                ),
            ],
            mode="sequential",
            balance=balance,
        )

        prof = acoustic.spectral_axis_profile()
        brightness = prof["micro_energy"] / max(prof["macro_energy"] + prof["micro_energy"], 1e-18)
        consonance = float(np.clip(prof["macro_energy"] / max(prof["macro_energy"] + prof["micro_energy"], 1e-18), 0.0, 1.0))

        pcm_audio: Optional[np.ndarray] = None
        if manifest:
            n_samp = max(64, int(round(dur * self.sample_rate)))
            t_grid = np.linspace(0.0, dur, n_samp, endpoint=False)[:, None]
            wave_f = acoustic.slice_manifestation(t_grid)
            # Smooth raised-cosine envelope (15 ms fade) for click-free splicing
            fade = min(n_samp // 6, int(0.015 * self.sample_rate))
            if fade > 0:
                win = np.ones(n_samp, dtype=np.float64)
                ramp = 0.5 * (1.0 - np.cos(math.pi * np.arange(fade) / fade))
                win[:fade] = ramp
                win[-fade:] = ramp[::-1]
                wave_f = wave_f * win
            peak = float(np.max(np.abs(wave_f)))
            if peak > 1e-12:
                wave_f = (wave_f / peak) * 0.80
            pcm_audio = wave_f

        causal_state = {
            "fundamental_hz": round(f0, 2),
            "overtone_consonance": round(consonance, 4),
            "spectral_brightness": round(brightness, 4),
            "damping_rate": round(damping_rate, 4),
        }
        explanation = (
            f"Harmonic chord grounded at {f0:.1f}Hz (consonance {consonance:.2f}, "
            f"overtone brightness {brightness:.3f}, damping {damping_rate:.2f})."
        )
        return InstrumentPerformance(
            instrument_id=self.instrument_id,
            modality=self.modality,
            causal_state=causal_state,
            rotor_manifold=acoustic,
            manifestation=pcm_audio,
            chromatic_triad=tuple(prof["chromatic_triad"]),  # type: ignore[arg-type]
            explanation=explanation,
        )


class TactileResistanceInstrument(SensoryInstrument):
    """
    Tactile causal hierarchy operating strictly in O(K) without rasterizing pixels:
    Analytic Surface Spectral Roughness + Solar Thermal Warmth + Structural Tension.
    """

    def __init__(self, instrument_id: str = "touch.tactile_resistance") -> None:
        self.instrument_id = instrument_id
        self.modality = "touch"

    def perform(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
        manifest: bool = False,
        resolution: int = 128,
        context: Optional[Dict[str, InstrumentPerformance]] = None,
    ) -> InstrumentPerformance:
        surface = None
        if context:
            for perf in context.values():
                if perf.modality == "vision" and perf.rotor_manifold is not None:
                    surface = perf.rotor_manifold
                    break
        if surface is None:
            surface = OpticalPrismLightInstrument().build_surface_manifold(
                globe_state, palette, balance
            )

        macro_pow = 0.0
        fine_grad_pow = 0.0
        for band in surface.bands:
            if band.num_modes == 0:
                continue
            eff = surface.effective_amplitudes(band)
            p = float(np.sum(np.abs(eff) ** 2))
            w_sq = float(np.sum((np.abs(eff) ** 2) * np.sum(band.wavevectors ** 2, axis=1)))
            if band.level == 0:
                macro_pow += p
            else:
                fine_grad_pow += w_sq

        spectral_roughness = math.sqrt(fine_grad_pow / max(macro_pow * 100.0, 1e-12))
        roughness_index = float(np.clip(0.5 * globe_state.micro_roughness + 0.5 * math.tanh(spectral_roughness), 0.0, 1.0))

        warmth_norm = float(np.clip((math.sin(globe_state.solar_elevation_rad) + 0.35) / 1.35, 0.0, 1.0))
        thermal_warmth_c = 12.0 + 22.0 * warmth_norm
        structural_tension = float(
            np.clip(
                0.45
                + 0.25 * balance.contrast_tilt
                + 0.20 * balance.scale_tilt
                + 0.15 * globe_state.prism_dispersion_strength,
                0.0,
                1.0,
            )
        )

        # Tactile Chromatic Triad [Red=Tension/Flux, Blue=Smooth Order, Yellow=Thermal Warmth]
        t_red = max(0.05, roughness_index + 0.3 * structural_tension)
        t_blue = max(0.05, 1.0 - roughness_index)
        t_yellow = max(0.05, warmth_norm)
        triad_arr = np.array([t_red, t_blue, t_yellow], dtype=np.float64)
        triad_arr /= float(np.sum(triad_arr))

        temp_word = "sun-warmed" if thermal_warmth_c >= 24.0 else "cool"
        tex_word = "finely faceted" if roughness_index >= 0.35 else "smooth crystal"
        causal_state = {
            "surface_roughness_index": round(roughness_index, 4),
            "thermal_warmth_celsius": round(thermal_warmth_c, 2),
            "structural_tension": round(structural_tension, 4),
            "tactile_feel": f"{temp_word} {tex_word}",
        }
        explanation = (
            f"Tactile sensation feels {temp_word} ({thermal_warmth_c:.1f}C) and {tex_word} "
            f"(roughness {roughness_index:.3f}, tension {structural_tension:.3f})."
        )
        return InstrumentPerformance(
            instrument_id=self.instrument_id,
            modality=self.modality,
            causal_state=causal_state,
            rotor_manifold=None,
            manifestation=None,
            chromatic_triad=(float(triad_arr[0]), float(triad_arr[1]), float(triad_arr[2])),
            explanation=explanation,
        )


@dataclass
class OrchestraSliceResult:
    """Unified multi-sensory outcome of conducting the orchestra at spacetime slice t."""

    globe_state: SpacetimeGlobeState
    balance: SpectralBalanceScale
    performances: Dict[str, InstrumentPerformance]
    cross_modal_coherence: float
    human_explanation: str
    elapsed_ms: float


class ElysiaOrchestraOS:
    """
    Top-level Sensory Orchestra Conductor / OS.
    Coordinates a 3-tier Spacetime Globe, Causal Mechanism Palette, and pluggable
    human-grounded sensory instruments without touching raw point arrays in OS logic.
    """

    def __init__(
        self,
        globe: Optional[SpacetimeGlobeRotorDeck] = None,
        palette: Optional[CausalMechanismPalette] = None,
        instruments: Optional[Sequence[SensoryInstrument]] = None,
    ) -> None:
        self.globe = globe if globe is not None else SpacetimeGlobeRotorDeck()
        self.palette = (
            palette if palette is not None else CausalMechanismPalette.with_default_principles()
        )
        self.instruments: Dict[str, SensoryInstrument] = {}
        if instruments is None:
            self.register_instrument(OpticalPrismLightInstrument())
            self.register_instrument(HarmonicAcousticInstrument())
            self.register_instrument(TactileResistanceInstrument())
        else:
            for inst in instruments:
                self.register_instrument(inst)

    def register_instrument(self, instrument: SensoryInstrument) -> None:
        self.instruments[instrument.instrument_id] = instrument

    @staticmethod
    def _compute_cross_modal_coherence(
        performances: Dict[str, InstrumentPerformance]
    ) -> float:
        triads = [np.asarray(p.chromatic_triad, dtype=np.float64) for p in performances.values()]
        if len(triads) < 2:
            return 1.0
        sims: List[float] = []
        for i in range(len(triads)):
            ni = float(np.linalg.norm(triads[i]))
            for j in range(i + 1, len(triads)):
                nj = float(np.linalg.norm(triads[j]))
                if ni > 1e-12 and nj > 1e-12:
                    sims.append(float(np.dot(triads[i], triads[j]) / (ni * nj)))
        return float(np.mean(sims)) if sims else 1.0

    def conduct_slice(
        self,
        t: float,
        balance: Optional[SpectralBalanceScale] = None,
        scale_zoom: float = 1.0,
        manifest: bool = False,
        resolution: int = 128,
    ) -> OrchestraSliceResult:
        """
        Conducts every registered sensory instrument at continuous spacetime `t`
        and zoom `scale_zoom`, returning unified causal descriptors & optional
        boundary manifestations (images/audio).
        """
        t0 = time.perf_counter()
        bal = balance if balance is not None else SpectralBalanceScale()
        globe_state = self.globe.slice_spacetime(t, scale_zoom=scale_zoom)

        performances: Dict[str, InstrumentPerformance] = {}
        # Run non-touch instruments first so TactileResistanceInstrument reads exact O(K) surface
        ordered = sorted(
            self.instruments.values(),
            key=lambda ins: 1 if ins.modality == "touch" else 0,
        )
        for inst in ordered:
            perf = inst.perform(
                globe_state=globe_state,
                palette=self.palette,
                balance=bal,
                manifest=manifest,
                resolution=resolution,
                context=performances,
            )
            performances[inst.instrument_id] = perf

        coherence = self._compute_cross_modal_coherence(performances)
        explanation_parts = [
            f"[t={globe_state.time:.2f}, zoom={globe_state.scale_zoom:.2f}, coherence={coherence:.3f}]"
        ]
        for perf in performances.values():
            explanation_parts.append(f"({perf.modality}) {perf.explanation}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return OrchestraSliceResult(
            globe_state=globe_state,
            balance=bal,
            performances=performances,
            cross_modal_coherence=coherence,
            human_explanation=" ".join(explanation_parts),
            elapsed_ms=elapsed_ms,
        )

    def scrub_reversibility_check(self, t_a: float, t_b: float) -> Dict[str, float]:
        """
        Proves non-sequential spacetime globe reversibility: jumps t_a -> t_b -> t_a
        and measures exact restoration error across globe coordinates and manifolds.
        """
        s_a1 = self.conduct_slice(t_a, manifest=False)
        _ = self.conduct_slice(t_b, manifest=False)
        s_a2 = self.conduct_slice(t_a, manifest=False)

        globe_pos_err = max(
            float(np.max(np.abs(s_a1.globe_state.macro_pos - s_a2.globe_state.macro_pos))),
            float(np.max(np.abs(s_a1.globe_state.meso_pos - s_a2.globe_state.meso_pos))),
            float(np.max(np.abs(s_a1.globe_state.micro_pos - s_a2.globe_state.micro_pos))),
        )
        max_field_err = 0.0
        for iid, p1 in s_a1.performances.items():
            p2 = s_a2.performances[iid]
            if p1.rotor_manifold is not None and p2.rotor_manifold is not None:
                dim = p1.rotor_manifold.dim
                low, high = p1.rotor_manifold.domain_bounds
                if dim == 1:
                    pts = np.linspace(low[0], high[0], 128, endpoint=False)[:, None]
                else:
                    ys = np.linspace(low[0], high[0], 24, endpoint=False)
                    xs = np.linspace(low[1], high[1], 24, endpoint=False)
                    yy, xx = np.meshgrid(ys, xs, indexing="ij")
                    pts = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
                err = float(
                    np.max(
                        np.abs(
                            p1.rotor_manifold.slice_manifestation(pts)
                            - p2.rotor_manifold.slice_manifestation(pts)
                        )
                    )
                )
                max_field_err = max(max_field_err, err)

        return {
            "globe_position_error": globe_pos_err,
            "manifold_field_error": max_field_err,
        }

    def compose_symphony_wav(
        self,
        times: Sequence[float],
        output_path: Path,
        slice_duration: float = 0.65,
        sample_rate: int = 22050,
        balance: Optional[SpectralBalanceScale] = None,
    ) -> Path:
        """
        Conducts the HarmonicAcousticInstrument across spacetime waypoints `times`
        and writes a playable 16-bit mono PCM WAV file.
        """
        acoustic_inst: Optional[HarmonicAcousticInstrument] = None
        for inst in self.instruments.values():
            if isinstance(inst, HarmonicAcousticInstrument):
                acoustic_inst = inst
                break
        if acoustic_inst is None:
            acoustic_inst = HarmonicAcousticInstrument(
                duration_s=slice_duration, sample_rate=sample_rate
            )
        else:
            acoustic_inst = HarmonicAcousticInstrument(
                instrument_id=acoustic_inst.instrument_id,
                duration_s=slice_duration,
                sample_rate=sample_rate,
            )

        bal = balance if balance is not None else SpectralBalanceScale()
        segments: List[np.ndarray] = []
        for t in times:
            gs = self.globe.slice_spacetime(float(t))
            perf = acoustic_inst.perform(gs, self.palette, bal, manifest=True)
            if perf.manifestation is not None:
                segments.append(perf.manifestation)

        full_wave = np.concatenate(segments) if segments else np.zeros(sample_rate // 4)
        pcm_int16 = np.clip(full_wave * 32767.0, -32767.0, 32767.0).astype(np.int16)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())
        return out

    def sync_to_causal_field(
        self,
        slice_result: OrchestraSliceResult,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "orch",
    ) -> Dict[str, Any]:
        """
        Projects the 3-tier Spacetime Globe and all active Sensory Instruments into
        `CausalField` as EngramAttractors, InformationVoxels, ConnectivityBeams, and
        conserved Mobility vectors.
        """
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        gs = slice_result.globe_state

        tiers = [
            ("macro_world", gs.macro_pos, gs.macro_angle, "macro", 1.0),
            ("meso_form", gs.meso_pos, gs.meso_angle, "meso", 0.5),
            ("micro_texture", gs.micro_pos, gs.micro_angle, "micro", 0.25),
        ]
        tier_vids: List[str] = []
        for tname, pos, ang, tier_lbl, scale_lvl in tiers:
            vid = f"{prefix}_{tname}"
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"tier": tname, "time": gs.time},
                    tensor=np.array([pos[0], pos[1], pos[2], ang], dtype=np.float32),
                    mass=max(0.2, scale_lvl * 2.0),
                    position=pos[: field.dimensions].astype(np.float32),
                    phase_angle=ang % (2.0 * math.pi),
                    scale_level=scale_lvl,
                )
            )
            field.register_engram(
                EngramAttractor(
                    id=f"{prefix}_engram_{tname}",
                    name=tname,
                    position=pos[: field.dimensions].astype(np.float32),
                    intensity=2.0 * scale_lvl,
                    tier=tier_lbl,
                    symbolic_intent=f"Globe {tname} at t={gs.time:.2f}",
                )
            )
            tier_vids.append(vid)

        beams_linked = 0
        # Macro -> Meso -> Micro vertical hierarchy
        for i in range(len(tier_vids) - 1):
            field.link_voxels(tier_vids[i], tier_vids[i + 1], strength=0.9)
            beams_linked += 1

        inst_vids: List[str] = []
        for idx, (iid, perf) in enumerate(slice_result.performances.items()):
            vid = f"{prefix}_{iid}"
            pos = np.array([float(idx) * 0.5, perf.chromatic_triad[0], perf.chromatic_triad[1]], dtype=np.float32)
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"instrument": iid, "modality": perf.modality},
                    tensor=np.array(list(perf.chromatic_triad) + [slice_result.cross_modal_coherence], dtype=np.float32),
                    mass=1.0,
                    position=pos[: field.dimensions],
                    chromatic_vector=np.array(perf.chromatic_triad, dtype=np.float32),
                )
            )
            field.register_engram(
                EngramAttractor(
                    id=f"{prefix}_engram_{iid}",
                    name=iid,
                    position=pos[: field.dimensions],
                    intensity=2.5,
                    tier="meso",
                    symbolic_intent=perf.explanation,
                )
            )
            inst_vids.append(vid)
            # Connect Globe -> Instrument
            field.link_voxels(tier_vids[min(idx, len(tier_vids) - 1)], vid, strength=0.75)
            beams_linked += 1

        # Cross-modal coherence links across instruments
        for i in range(len(inst_vids)):
            for j in range(i + 1, len(inst_vids)):
                field.link_voxels(inst_vids[i], inst_vids[j], strength=max(0.2, slice_result.cross_modal_coherence))
                beams_linked += 1

        impulse = np.array(
            [
                0.2 + 0.1 * slice_result.cross_modal_coherence,
                float(slice_result.balance.contrast_tilt) * 0.1,
                float(slice_result.balance.scale_tilt) * 0.1,
            ],
            dtype=np.float32,
        )[: field.dimensions]
        for vid in tier_vids + inst_vids:
            field.apply_impact(vid, impulse)

        return {
            "causal_field": field,
            "tier_voxels": tier_vids,
            "instrument_voxels": inst_vids,
            "beams_linked": beams_linked,
        }
