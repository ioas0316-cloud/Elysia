"""
mechano_acoustic_organ.py

ELYSIA Physical Contact Mechanics & Acoustic Resonance Sensory Organs
=====================================================================
Strict Domain-Isolated Physical Transducers:
- MechanicalContactOrgan: Infers effective stiffness (N/m), compliance (m/N),
  hysteretic damping, residual plastic deformation, and tangent stiffness collapse
  (yield) strictly from raw 1D (force_t, disp_t) traces.
- AcousticResonanceOrgan: Infers modal peak frequency (Hz), spectral centroid,
  quality factor Q, and temporal decay rate (1/s) strictly from 1D PCM contact
  waveforms.

Neither organ inspects object names, visual arrays, or world internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.sensory.ocular_scale_organ import LayerProvenance


def _integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


@dataclass
class MechanicalObservation:
    """Measurements extracted strictly from a force-displacement contact curve."""

    effective_stiffness_n_m: float
    compliance_m_n: float
    damping_ratio: float
    residual_plastic_ratio: float
    max_force_n: float
    max_disp_m: float
    yield_detected: bool
    yield_force_n: Optional[float]
    mechanical_regime: str
    provenance: LayerProvenance


@dataclass
class AcousticObservation:
    """Measurements extracted strictly from a 1D impact acoustic waveform."""

    peak_frequency_hz: float
    spectral_centroid_hz: float
    spectral_sharpness_q: float
    decay_rate_per_sec: float
    rms_energy: float
    acoustic_regime: str
    provenance: LayerProvenance


class MechanicalContactOrgan:
    """Extracts elastic, viscous, and plastic mechanics from raw (F(t), x(t)) probes."""

    def observe_push(
        self, force_t: np.ndarray, disp_t: np.ndarray, dt: float = 0.002
    ) -> MechanicalObservation:
        f = np.asarray(force_t, dtype=np.float64).ravel()
        x = np.asarray(disp_t, dtype=np.float64).ravel()
        if f.size < 6 or x.size != f.size:
            raise ValueError(f"Expected matching 1D arrays len>=6, got {f.shape} and {x.shape}")

        max_f = float(np.max(f))
        max_x = float(np.max(x))
        peak_idx = int(np.argmax(f))
        peak_idx = max(3, peak_idx)

        # Elastic loading segment: first 35% of upward force ramp where disp > 0
        load_f = f[: peak_idx + 1]
        load_x = x[: peak_idx + 1]
        init_end = max(3, int(round(0.35 * load_f.size)))
        seg_x = load_x[:init_end]
        seg_f = load_f[:init_end]

        var_x = float(np.var(seg_x))
        if var_x > 1e-12:
            cov_fx = float(np.mean((seg_f - np.mean(seg_f)) * (seg_x - np.mean(seg_x))))
            k_eff = max(0.5, cov_fx / var_x)
        else:
            k_eff = max(0.5, max_f / (max_x + 1e-6))

        compliance = float(1.0 / (k_eff + 1e-6))

        # Hysteresis loop area relative to max work envelope
        loop_area = abs(_integrate_trapz(f, x))
        damping_ratio = float(np.clip(loop_area / (max_f * max_x + 1e-12), 0.0, 1.0))

        # Permanent residual dent after unloading force back to ~0
        residual_plastic_ratio = float(np.clip(abs(x[-1]) / (max_x + 1e-9), 0.0, 1.0))

        # Yield detection: sustained drop in tangent dF/dx (< 0.30 * initial stiffness)
        df = np.diff(load_f)
        dx = np.diff(load_x)
        yield_detected = False
        yield_force_n: Optional[float] = None
        consec_soft = 0
        for i in range(max(1, init_end // 2), df.size):
            if dx[i] > 1e-7 and load_f[i] > 0.15 * max_f:
                tangent_k = df[i] / dx[i]
                if tangent_k < 0.30 * k_eff:
                    consec_soft += 1
                    if consec_soft >= 2:
                        yield_detected = True
                        yield_force_n = float(load_f[i - 1])
                        break
                else:
                    consec_soft = 0

        if not yield_detected and residual_plastic_ratio > 0.25:
            yield_detected = True
            yield_force_n = float(0.5 * max_f)

        if yield_detected:
            regime = "PLASTIC_YIELDING"
        elif k_eff >= 4000.0:
            regime = "ELASTIC_RIGID"
        elif k_eff < 1500.0:
            regime = "COMPLIANT_SOFT"
        else:
            regime = "MODERATE_ELASTIC"

        prov = LayerProvenance(
            layer_id="MECH_L1",
            name="Mechanical Force-Displacement Transduction",
            domain="Mechanics",
            causal_mechanism=(
                "Regress initial loading slope dF/dx, integrate hysteresis loop |integral F dx|, "
                "measure final unloading displacement ratio, and detect tangent stiffness collapse."
            ),
            why_necessary=(
                "Photons cannot reveal stiffness, energy dissipation, or permanent plastic yield thresholds."
            ),
            key_metrics={
                "effective_stiffness_n_m": float(k_eff),
                "damping_ratio": damping_ratio,
                "residual_plastic_ratio": residual_plastic_ratio,
                "yield_detected": yield_detected,
                "dt_s": float(dt),
            },
        )
        return MechanicalObservation(
            effective_stiffness_n_m=float(k_eff),
            compliance_m_n=compliance,
            damping_ratio=damping_ratio,
            residual_plastic_ratio=residual_plastic_ratio,
            max_force_n=max_f,
            max_disp_m=max_x,
            yield_detected=yield_detected,
            yield_force_n=yield_force_n,
            mechanical_regime=regime,
            provenance=prov,
        )


class AcousticResonanceOrgan:
    """Extracts modal resonance frequencies and envelope damping from 1D impact audio."""

    def observe_tap(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> AcousticObservation:
        sig = np.asarray(waveform, dtype=np.float64).ravel()
        n = sig.size
        if n < 32:
            raise ValueError(f"Waveform too short for spectral analysis: len={n}")

        rms = float(np.sqrt(np.mean(sig**2)))
        if rms < 5e-4:
            prov = LayerProvenance(
                layer_id="ACOUSTIC_L1",
                name="Impact Acoustic Resonance Transduction",
                domain="Acoustics",
                causal_mechanism="Windowed FFT and log-envelope decay on 1D PCM contact impact.",
                why_necessary="Impact ring vs damped thud disambiguates internal elasticity when optics match.",
                key_metrics={"rms_energy": rms, "silent_response": True},
            )
            return AcousticObservation(
                peak_frequency_hz=50.0,
                spectral_centroid_hz=50.0,
                spectral_sharpness_q=1.0,
                decay_rate_per_sec=200.0,
                rms_energy=rms,
                acoustic_regime="DAMPED_LOW_THUD",
                provenance=prov,
            )

        win = np.hanning(n)
        spec = np.fft.rfft(sig * win)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
        power = np.abs(spec) ** 2

        valid_mask = freqs >= 50.0
        v_freqs = freqs[valid_mask]
        v_power = power[valid_mask]
        total_p = float(np.sum(v_power)) + 1e-12

        peak_idx = int(np.argmax(v_power))
        peak_freq = float(v_freqs[peak_idx])
        centroid = float(np.sum(v_freqs * v_power) / total_p)

        # Full-width half-maximum bandwidth around dominant spectral peak
        half_p = 0.5 * float(v_power[peak_idx])
        above = v_power >= half_p
        freq_res = float(sample_rate) / float(n)
        lo_idx = peak_idx
        while lo_idx > 0 and above[lo_idx - 1]:
            lo_idx -= 1
        hi_idx = peak_idx
        while hi_idx < v_power.size - 1 and above[hi_idx + 1]:
            hi_idx += 1
        fwhm_hz = max(freq_res, float(v_freqs[hi_idx] - v_freqs[lo_idx]))
        sharpness_q = float(peak_freq / fwhm_hz)

        # Temporal exponential decay rate over 8 envelope windows from max-RMS slice onward
        n_slices = 8
        slice_len = max(4, n // n_slices)
        slice_rms = []
        slice_t = []
        for s_i in range(n_slices):
            a = s_i * slice_len
            b = n if s_i == n_slices - 1 else (s_i + 1) * slice_len
            seg = sig[a:b]
            slice_rms.append(float(np.sqrt(np.mean(seg**2))))
            slice_t.append(0.5 * (a + b) / float(sample_rate))

        start_slice = int(np.argmax(slice_rms))
        t_fit = np.asarray(slice_t[start_slice:], dtype=np.float64)
        y_fit = np.log(np.asarray(slice_rms[start_slice:], dtype=np.float64) + 1e-6)
        if t_fit.size >= 3:
            slope, _ = np.polyfit(t_fit, y_fit, 1)
            decay_rate = max(0.0, float(-slope))
        else:
            decay_rate = 20.0

        if peak_freq >= 1200.0 and sharpness_q >= 3.0:
            regime = "SHARP_HIGH_RING"
        elif peak_freq < 700.0 or decay_rate >= 60.0:
            regime = "DAMPED_LOW_THUD"
        else:
            regime = "MODERATE_RESONANCE"

        prov = LayerProvenance(
            layer_id="ACOUSTIC_L1",
            name="Impact Acoustic Resonance Transduction",
            domain="Acoustics",
            causal_mechanism=(
                "Compute Hanning-windowed rFFT power spectrum, FWHM modal quality Q, "
                "and linear slope of log(RMS envelope) across 8 time bins."
            ),
            why_necessary="Acoustic ringing vs viscous damping measures internal energy storage across impact.",
            key_metrics={
                "peak_frequency_hz": peak_freq,
                "spectral_centroid_hz": centroid,
                "spectral_sharpness_q": sharpness_q,
                "decay_rate_per_sec": decay_rate,
            },
        )
        return AcousticObservation(
            peak_frequency_hz=peak_freq,
            spectral_centroid_hz=centroid,
            spectral_sharpness_q=sharpness_q,
            decay_rate_per_sec=decay_rate,
            rms_energy=rms,
            acoustic_regime=regime,
            provenance=prov,
        )


class MechanoAcousticOrgan:
    """Unified wrapper grouping non-visual mechanical and acoustic physical organs."""

    def __init__(self) -> None:
        self.mechanical = MechanicalContactOrgan()
        self.acoustic = AcousticResonanceOrgan()

    def observe_push(
        self, force_t: np.ndarray, disp_t: np.ndarray, dt: float = 0.002
    ) -> MechanicalObservation:
        return self.mechanical.observe_push(force_t, disp_t, dt=dt)

    def observe_tap(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> AcousticObservation:
        return self.acoustic.observe_tap(waveform, sample_rate=sample_rate)
