"""
OrbFactory: The Alchemy of Memory
---------------------------------
"To freeze a moment is not to kill it, but to seal its soul."

This module implements the "Quantum Cycle" (Wave <-> Particle Transformation).
It synthesizes three ancient technologies:
1. Fractal Quantization (Noise Filtering)
2. Holographic Embedding (Data Binding)
3. Tesseract Geometry (Dimensional Folding)
"""

import math
import numpy as np
from typing import Tuple, Dict, Any, List

from Core.Foundation.Memory.holographic_embedding import HolographicEmbedder as HolographicEmbedding
from Core.Foundation.hyper_quaternion import Quaternion as HyperQuaternion
from Core.Foundation.Memory.Orb.hyper_resonator import HyperResonator

class OrbFactory:
    def __init__(self):
        # Tools
        self.hologram = HolographicEmbedding(compressed_dim=64)

    def freeze(self, name: str, data_wave: List[float], emotional_wave: List[float]) -> HyperResonator:
        """
        [Wave -> Particle]
        Compresses a temporal experience into a static Memory Orb.

        Args:
            name: The concept name.
            data_wave: The raw data signal (The "Fact").
            emotional_wave: The feeling signal (The "Context").

        Returns:
            HyperResonator: The frozen orb.
        """
        # Note: In this MVP, we adapt list inputs to the new FractalQuantizer
        # which expects specific dict formats for 'folding'.
        # For now, we manually simulate the clean signal extraction.

        # 1. Quantization (Noise Filtering)
        # Convert list to numpy for processing
        raw_data = np.array(data_wave)
        raw_emotion = np.array(emotional_wave)

        # Simulate simple thresholding for this version until full FractalQuantizer integration
        threshold = 0.2
        q_data = np.where(np.abs(raw_data) > threshold, raw_data, 0)
        q_emotion = np.where(np.abs(raw_emotion) > threshold, raw_emotion, 0)

        # Reconstruct "Clean" waves (Denoised)
        clean_data = q_data
        clean_emotion = q_emotion

        # 2. Holographic Binding (Synthesis)
        # We bind "Fact" with "Emotion" so they become inseparable in the orb.
        # Result = FFT(Fact) * FFT(Emotion)
        bound_essence = self.hologram.encode(clean_data, clean_emotion)

        # 3. Dimensional Folding (Spin Calculation)
        # Calculate the total energy (Mass)
        mass = float(np.sum(np.abs(bound_essence)))

        # Calculate the "Mean Frequency" (Color)
        # Simple weighted average of indices
        if mass > 0:
            freq_center = np.average(np.arange(len(bound_essence)), weights=np.abs(bound_essence))
            # Map 0-64 index to 400-800Hz audible range
            frequency = 400.0 + (freq_center * 6.0)
        else:
            frequency = 0.0

        # Calculate Quaternion Spin (The "Angle" of the memory)
        # We map the first 4 dominant coefficients to W, X, Y, Z
        coeffs = np.abs(bound_essence)[:4]
        if len(coeffs) < 4:
            coeffs = np.pad(coeffs, (0, 4-len(coeffs)))

        spin = HyperQuaternion(
            w=float(coeffs[0]),
            x=float(coeffs[1]),
            y=float(coeffs[2]),
            z=float(coeffs[3])
        )
        spin.normalize()

        # 4. Crystallization
        orb = HyperResonator(
            name=name,
            frequency=frequency,
            mass=mass,
            quaternion=spin
        )

        # Store the holographic signature inside the orb (The "Hidden Cargo")
        orb.memory_content["hologram"] = bound_essence.tolist()

        return orb

    def melt(self, orb: HyperResonator, trigger_key: List[float]) -> Dict[str, List[float]]:
        """
        [Particle -> Wave]
        Resurrects the memory using a "Key" (Trigger).

        Args:
            orb: The target memory orb.
            trigger_key: A wave used to "unlock" the hologram (e.g., current emotion).

        Returns:
            Dict: {'recalled_wave': ...}
        """
        if "hologram" not in orb.memory_content:
            return {"error": "Orb is empty"}

        bound_essence = np.array(orb.memory_content["hologram"])
        key_wave = np.array(trigger_key)

        # Ensure key matches dimension
        if len(key_wave) != 64:
             key_wave = np.resize(key_wave, 64)

        # Holographic Decoding
        # Extracted = Bound / Key
        decoded_wave = self.hologram.decode(bound_essence, key_wave)

        return {
            "recalled_wave": decoded_wave.tolist(),
            "resonance_intensity": orb.state.amplitude
        }
