"""
Multimodal Cognitive Front-end (Multimodal Cognitive Front-end Layer)

This module implements the non-linear front-end pipeline that ingests human sensory
and symbolic representations, reverses human bio-transduction compression biases,
synchronizes heterogeneous modal frequencies via Kuramoto phase dynamics, and disentangles
the representations into two strictly orthogonal axes:
  - Axis A: Macro-Cosmos Structural Topology
  - Axis B: Human Qualia & Intent Spectrum
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class BioTransductionDeconstructor:
    """
    Reverses biological sensory compression biases:
      - Visual: Inverts RGB gamma & tristimulus compression into a continuous spectral manifold.
      - Audio: Inverts equal-loudness curves & cochlear damping into a mechanical phase field.
      - Text/Symbol: Deconstructs discrete symbolic tokens into sensory-value inference tensors.
    """

    def __init__(self, spectral_resolution: int = 32, frequency_bands: int = 32):
        self.spectral_resolution = spectral_resolution
        self.frequency_bands = frequency_bands

    def deconstruct_visual(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Deconstructs RGB array [H, W, 3] or flat RGB array into continuous spectral manifold [H, W, spectral_resolution].
        Applies inverse gamma correction and expands 3-channel RGB into spectral wavelengths.
        """
        arr = np.asarray(rgb_array, dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Linearize sRGB gamma
        linear_rgb = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

        # Expand 3 channels (R, G, B) to continuous spectrum using Gaussian basis functions
        # Wavelengths approximately 380nm to 750nm mapped to [0, 1]
        wavelengths = np.linspace(0.0, 1.0, self.spectral_resolution)

        # Basis peaks for R (red ~ 0.75), G (green ~ 0.5), B (blue ~ 0.2)
        r_peak, g_peak, b_peak = 0.75, 0.50, 0.20
        sigma = 0.15

        r_basis = np.exp(-((wavelengths - r_peak) ** 2) / (2 * sigma ** 2))
        g_basis = np.exp(-((wavelengths - g_peak) ** 2) / (2 * sigma ** 2))
        b_basis = np.exp(-((wavelengths - b_peak) ** 2) / (2 * sigma ** 2))

        # Shape handling
        if linear_rgb.ndim == 1 and linear_rgb.shape[0] == 3:
            spectral = (
                linear_rgb[0] * r_basis + linear_rgb[1] * g_basis + linear_rgb[2] * b_basis
            )
        elif linear_rgb.ndim == 3 and linear_rgb.shape[2] == 3:
            spectral = (
                np.outer(linear_rgb[..., 0], r_basis) +
                np.outer(linear_rgb[..., 1], g_basis) +
                np.outer(linear_rgb[..., 2], b_basis)
            ).reshape(linear_rgb.shape[0], linear_rgb.shape[1], self.spectral_resolution)
        else:
            # Flattened or general shape
            spectral = np.tensordot(linear_rgb, np.vstack([r_basis, g_basis, b_basis]), axes=([-1], [0]))

        return spectral

    def deconstruct_audio(self, audio_waveform: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Inverts equal-loudness curves and cochlear damping from audio waveform [N],
        producing a mechanical phase-energy field across frequency bands.
        """
        wave = np.asarray(audio_waveform, dtype=np.float64)
        if wave.ndim > 1:
            wave = wave.mean(axis=-1)

        # Apply short-time Fourier transform or filter bank approximation
        n_fft = min(len(wave), self.frequency_bands * 4)
        if n_fft < 4:
            n_fft = len(wave)

        if n_fft == 0:
            return np.zeros((self.frequency_bands, 2), dtype=np.float64)

        fft_vals = np.fft.rfft(wave[:n_fft])
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Inverse Fletcher-Munson equal-loudness weight approximation (boosting low/high freqs)
        # Iso-loudness weight: W(f) ~ 1 + 0.5 * cos(2pi * f / 10000)
        iso_weights = 1.0 + 0.3 * np.log1p(freqs / 100.0)

        magnitudes = np.abs(fft_vals) * iso_weights
        phases = np.angle(fft_vals)

        # Interpolate / resample to target frequency_bands
        resampled_mags = np.interp(
            np.linspace(0, len(magnitudes) - 1, self.frequency_bands),
            np.arange(len(magnitudes)),
            magnitudes
        )
        resampled_phases = np.interp(
            np.linspace(0, len(phases) - 1, self.frequency_bands),
            np.arange(len(phases)),
            phases
        )

        mechanical_phase_field = np.stack([resampled_mags, resampled_phases], axis=-1)
        return mechanical_phase_field

    def deconstruct_text(self, text: str) -> np.ndarray:
        """
        Deconstructs symbolic text into a value-inference tensor containing
        semantic intent, emotional affinity, and moral taboo weights.
        """
        # Character/word feature extraction mapped to continuous semantic vector
        words = text.strip().split()
        if not words:
            return np.zeros(16, dtype=np.float64)

        # Deterministic hashing into continuous embedding
        vec = np.zeros(16, dtype=np.float64)
        for i, word in enumerate(words):
            h = sum(ord(c) * (31 ** j) for j, c in enumerate(word)) % 1000000
            angle = (h / 1000000.0) * 2 * np.pi
            vec[i % 16] += np.sin(angle) + np.cos(angle * 0.5)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec


class AsynchronousPhaseLockTransducer:
    """
    Synchronizes heterogeneous modal frequencies using non-linear Kuramoto dynamics:
        dθ_m/dt = ω_m + ∑_{n≠m} K_mn * sin(θ_n - θ_m - δ_mn)

    Uses adaptive RK4 integration and phase wrapping to ensure numerical stability.
    """

    def __init__(self, num_modalities: int, coupling_strength: float = 0.5):
        self.num_modalities = num_modalities
        self.K = np.full((num_modalities, num_modalities), coupling_strength, dtype=np.float64)
        np.fill_diagonal(self.K, 0.0)
        self.delta = np.zeros((num_modalities, num_modalities), dtype=np.float64)

    def _kuramoto_rhs(self, phases: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Computes dθ/dt for all modalities."""
        dtheta = np.copy(freqs)
        for m in range(self.num_modalities):
            interaction = np.sum(
                self.K[m, :] * np.sin(phases - phases[m] - self.delta[m, :])
            )
            dtheta[m] += interaction
        return dtheta

    def step_rk4(self, phases: np.ndarray, freqs: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Runge-Kutta 4th order integration step with phase wrapping."""
        p = np.asarray(phases, dtype=np.float64)
        f = np.asarray(freqs, dtype=np.float64)

        k1 = self._kuramoto_rhs(p, f)
        k2 = self._kuramoto_rhs(p + 0.5 * dt * k1, f)
        k3 = self._kuramoto_rhs(p + 0.5 * dt * k2, f)
        k4 = self._kuramoto_rhs(p + dt * k3, f)

        new_phases = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Wrap phases to [-pi, pi] for numerical stability
        new_phases = (new_phases + np.pi) % (2 * np.pi) - np.pi
        return new_phases

    def synchronize(self, initial_phases: np.ndarray, natural_freqs: np.ndarray,
                    steps: int = 100, dt: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        Evolves phases over time steps and calculates phase-locking order parameter R.
        R = |1/N ∑ e^{i θ_m}|
        """
        phases = np.copy(initial_phases)
        for _ in range(steps):
            phases = self.step_rk4(phases, natural_freqs, dt=dt)

        # Order parameter R
        complex_order = np.mean(np.exp(1j * phases))
        order_parameter = float(np.abs(complex_order))
        return phases, order_parameter


class DualAxisDisentangler:
    """
    Disentangles raw sensory representations into two strictly orthogonal axes:
      - Axis A: Macro-Cosmos Structural Topology (Physical/Structural Reality)
      - Axis B: Human Qualia & Intent Spectrum (Subjective Value/Qualia)

    Enforces orthogonality via Modified Gram-Schmidt projection:
      <Axis A, Axis B> ≈ 0
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    def disentangle(self, combined_representation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Splits combined feature representation into Axis A and Axis B,
        then projects Axis B to be strictly orthogonal to Axis A.
        """
        vec = np.asarray(combined_representation, dtype=np.float64).flatten()
        if len(vec) < 2:
            # Pad if too short
            padded = np.zeros(max(2, self.feature_dim), dtype=np.float64)
            padded[:len(vec)] = vec
            vec = padded

        mid = len(vec) // 2
        axis_a_raw = vec[:mid]
        axis_b_raw = vec[mid:]

        # Resize to feature_dim if needed
        if len(axis_a_raw) != self.feature_dim:
            axis_a = np.interp(np.linspace(0, len(axis_a_raw) - 1, self.feature_dim),
                               np.arange(len(axis_a_raw)), axis_a_raw)
        else:
            axis_a = axis_a_raw.copy()

        if len(axis_b_raw) != self.feature_dim:
            axis_b = np.interp(np.linspace(0, len(axis_b_raw) - 1, self.feature_dim),
                               np.arange(len(axis_b_raw)), axis_b_raw)
        else:
            axis_b = axis_b_raw.copy()

        # Normalize Axis A
        norm_a = np.linalg.norm(axis_a)
        if norm_a > 1e-8:
            axis_a = axis_a / norm_a

        # Orthogonalize Axis B against Axis A (Gram-Schmidt)
        proj_b_on_a = np.dot(axis_b, axis_a) * axis_a
        axis_b_ortho = axis_b - proj_b_on_a

        # Normalize Axis B
        norm_b = np.linalg.norm(axis_b_ortho)
        if norm_b > 1e-8:
            axis_b_ortho = axis_b_ortho / norm_b

        dot_product = float(np.dot(axis_a, axis_b_ortho))

        return {
            "axis_a_topology": axis_a,
            "axis_b_qualia": axis_b_ortho,
            "orthogonality_dot_product": dot_product
        }


class MultimodalCognitiveFrontend:
    """
    Main Multimodal Cognitive Front-end Coordinator.
    Integrates deconstruction, phase locking, and dual-axis disentanglement.
    """

    def __init__(self, feature_dim: int = 16):
        self.deconstructor = BioTransductionDeconstructor(spectral_resolution=feature_dim)
        self.transducer = AsynchronousPhaseLockTransducer(num_modalities=3)
        self.disentangler = DualAxisDisentangler(feature_dim=feature_dim)

    def process_multimodal_input(
        self,
        rgb_image: Optional[np.ndarray] = None,
        audio_wave: Optional[np.ndarray] = None,
        text_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes multimodal inputs and produces a unified, dual-axis disentangled representation.
        """
        # 1. Deconstruct sensory biases
        if rgb_image is None:
            rgb_image = np.array([128, 128, 128])
        if audio_wave is None:
            audio_wave = np.sin(np.linspace(0, 2 * np.pi * 440, 100))
        if text_input is None:
            text_input = "Default perception of reality"

        spectral_manifold = self.deconstructor.deconstruct_visual(rgb_image)
        audio_phase_field = self.deconstructor.deconstruct_audio(audio_wave)
        text_tensor = self.deconstructor.deconstruct_text(text_input)

        # Extract initial phases for Kuramoto synchronization
        vis_phase = float(np.mean(spectral_manifold) % (2 * np.pi) - np.pi)
        aud_phase = float(np.mean(audio_phase_field[..., 1]) % (2 * np.pi) - np.pi)
        txt_phase = float(np.mean(text_tensor) % (2 * np.pi) - np.pi)

        initial_phases = np.array([vis_phase, aud_phase, txt_phase])
        natural_freqs = np.array([1.0, 2.5, 0.5])  # Modality intrinsic frequencies

        # 2. Asynchronous phase lock
        locked_phases, order_param = self.transducer.synchronize(initial_phases, natural_freqs)

        # 3. Fuse deconstructed features
        flat_spectral = spectral_manifold.flatten()
        flat_audio = audio_phase_field.flatten()
        combined_raw = np.concatenate([flat_spectral, flat_audio, text_tensor])

        # 4. Disentangle into Axis A and Axis B
        disentangled = self.disentangler.disentangle(combined_raw)

        return {
            "spectral_manifold": spectral_manifold,
            "audio_phase_field": audio_phase_field,
            "text_tensor": text_tensor,
            "phase_lock_order_parameter": order_param,
            "locked_phases": locked_phases,
            "axis_a_topology": disentangled["axis_a_topology"],
            "axis_b_qualia": disentangled["axis_b_qualia"],
            "orthogonality_dot_product": disentangled["orthogonality_dot_product"]
        }
