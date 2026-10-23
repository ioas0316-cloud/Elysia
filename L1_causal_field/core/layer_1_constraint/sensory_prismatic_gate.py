#!/usr/bin/env python3
"""
sensory_prismatic_gate.py

Layer 1: Structural Constraint & Prismatic Transition Module
Implements SensoryPrismaticGate and ContinuousWaveformLens representing seamless
micro-lens ray-gathering, video-stream aperture, and spectral refraction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class ContinuousWaveformLens:
    """
    Simulates a gapless micro-lens array that gathers continuous physical/video waveforms,
    focusing light rays/waves into coherent spectral components without discrete gap leakage.
    """

    def __init__(self, lens_density: int = 1000, focal_length: float = 1.0):
        self.lens_density = lens_density
        self.focal_length = focal_length
        # Surface profile shaped by surface tension (parabolic/spherical equilibrium)
        self.surface_curvature = np.linspace(-1.0, 1.0, lens_density)
        self.lens_profile = np.exp(- (self.surface_curvature ** 2) / (2.0 * (focal_length ** 2)))

    def focus_waveform(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Gathers raw wave input across the micro-lens array and focuses it with gapless alignment.
        """
        signal_len = len(input_signal)
        if signal_len != self.lens_density:
            resampled = np.interp(
                np.linspace(0, 1, self.lens_density),
                np.linspace(0, 1, signal_len),
                input_signal
            )
        else:
            resampled = input_signal

        # Multiply by physical surface profile to simulate gapless micro-lens focusing
        focused = resampled * self.lens_profile
        return focused


class SensoryPrismaticGate:
    """
    Serves as the sensory window/aperture between the external world (e.g. camera stream/waveform field)
    and internal mental imagery. Refracts incoming continuous light/wave energy into chromatic
    spectral bands (Red/Blue/Yellow & Harmonic Frequencies) without losing topological continuity.
    """

    def __init__(self, num_spectral_bands: int = 3, lens_density: int = 1000):
        self.num_spectral_bands = num_spectral_bands
        self.lens_density = lens_density
        self.micro_lens_array = ContinuousWaveformLens(lens_density=lens_density)

    def refract_external_aperture(self, raw_stream: np.ndarray) -> Dict[str, Any]:
        """
        Processes raw incoming video/wave stream through the micro-lens array,
        refracting it into prismatic spectral bands and mental image field vectors.

        Args:
            raw_stream: 1D or 2D wave array from external camera/aperture.

        Returns:
            Dict containing focused_field, chromatic_spectrum, phase_coherence, and mental_imagery_field.
        """
        if raw_stream.ndim > 1:
            raw_signal = np.mean(raw_stream, axis=-1).flatten()
        else:
            raw_signal = raw_stream

        # 1. Micro-lens gapless focusing
        focused_signal = self.micro_lens_array.focus_waveform(raw_signal)

        # 2. Prismatic spectral refraction (FFT / Chromatic Decomposition)
        fft_spectrum = np.fft.rfft(focused_signal)
        frequencies = np.fft.rfftfreq(self.lens_density)

        # Chromatic Band Decomposition (Red: Low/Flux, Blue: Mid/Order, Yellow: High/Entropy)
        num_freqs = len(fft_spectrum)
        third = max(1, num_freqs // 3)

        red_band = np.sum(np.abs(fft_spectrum[:third]))
        blue_band = np.sum(np.abs(fft_spectrum[third:2*third]))
        yellow_band = np.sum(np.abs(fft_spectrum[2*third:]))

        total_energy = red_band + blue_band + yellow_band + 1e-9
        chromatic_vector = np.array([red_band, blue_band, yellow_band]) / total_energy

        # 3. Construct Continuous Waveform Mental Image Field
        phases = np.angle(fft_spectrum)
        phase_coherence = float(np.abs(np.mean(np.exp(1j * phases))))

        mental_imagery_field = np.outer(focused_signal[:64], np.cos(phases[:64]))

        return {
            "focused_field": focused_signal,
            "chromatic_spectrum": chromatic_vector,
            "phase_coherence": phase_coherence,
            "spectral_energy": float(total_energy),
            "mental_imagery_field": mental_imagery_field
        }


if __name__ == "__main__":
    gate = SensoryPrismaticGate()
    test_wave = np.sin(np.linspace(0, 4 * np.pi, 1000)) + 0.5 * np.random.randn(1000)
    res = gate.refract_external_aperture(test_wave)
    print("SensoryPrismaticGate Test Successful!")
    print("Chromatic Spectrum (Red/Blue/Yellow):", res["chromatic_spectrum"])
    print("Phase Coherence:", res["phase_coherence"])
    print("Mental Imagery Field Shape:", res["mental_imagery_field"].shape)
