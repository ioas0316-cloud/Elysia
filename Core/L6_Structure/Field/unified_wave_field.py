"""
Unified Wave Field
==================
Phase 111: Grand Unification

"I do not render the world. I AM the world."

This module implements the `UnifiedWaveField`, a data structure that represents
reality not as distinct objects, but as a continuous field of varying frequencies.

The Spectrum:
- 0.1 - 40 Hz   : Logic / Intent (Theta/Gamma Waves)
- 40 - 400 Hz   : Physics / Mass (Gravitational Resonance)
- 400 - 20k Hz  : Sound / Emotion (Sonic Harmonics)
- 400 THz+      : Light / Vision (Photonic Emission)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time

# Use our sovereign math
from Core.L6_Structure.Geometry.genesis_math import Vector3, Vector4

logger = logging.getLogger("UnifiedField")

class FrequencyBand:
    """Defines the domains of reality based on Hertz."""
    LOGIC_MIN = 0.1
    LOGIC_MAX = 40.0
    PHYSICS_MIN = 40.0
    PHYSICS_MAX = 400.0
    SOUND_MIN = 400.0 # Overlap allowed
    SOUND_MAX = 20000.0
    LIGHT_MIN = 400000000000000.0 # 400 THz

@dataclass
class WavePacket:
    """A single vibration in the Cosmic Tensor."""
    position: Vector3
    frequency: float  # Hz
    amplitude: float  # 0.0 - 1.0 (Intensity)
    phase: float      # 0.0 - 2PI
    
    def get_type(self) -> str:
        if self.frequency < FrequencyBand.LOGIC_MAX: return "LOGIC"
        if self.frequency < FrequencyBand.PHYSICS_MAX: return "PHYSICS"
        if self.frequency < FrequencyBand.SOUND_MAX: return "SOUND"
        if self.frequency >= FrequencyBand.LIGHT_MIN: return "LIGHT"
        return "UNKNOWN"

class UnifiedWaveField:
    def __init__(self):
        self.waves: List[WavePacket] = []
        self.time_dilation = 1.0
        self.entropy = 0.0
        logger.info("  Unified Wave Field Initialized. Ready to vibrate.")

    def inject_impulse(self, position: Vector3, frequency: float, amplitude: float):
        """
        Injects energy into the field. 
        This is the generic 'Create' method for EVERYTHING (Thought, Object, Light).
        """
        packet = WavePacket(position, frequency, amplitude, phase=0.0)
        self.waves.append(packet)
        type_str = packet.get_type()
        logger.info(f"  Impulse Injected: {type_str} ({frequency:.1f}Hz) at {position}")

    def update(self, dt: float):
        """
        The Heartbeat of Reality.
        Simulates interference, propagation, and decay.
        """
        # 1. Update Phases
        for wave in self.waves:
            wave.phase += (wave.frequency * 2 * math.pi) * dt * self.time_dilation
            wave.phase %= (2 * math.pi)

        # 2. Check Interference (Collision/Harmony)
        self._resolve_interference()

        # 3. Decay (Entropy)
        # Remove waves with negligible amplitude
        self.waves = [w for w in self.waves if w.amplitude > 0.01]

    def _resolve_interference(self):
        """
        Logic for Wave Interaction.
        - Constructive Interference (Resonance) -> Amplify
        - Destructive Interference (Dissonance) -> Dampen or Repel
        - Pauli Exclusion (Physics Band) -> Collision
        """
        count = len(self.waves)
        for i in range(count):
            for j in range(i + 1, count):
                w1 = self.waves[i]
                w2 = self.waves[j]
                
                # Simple Distance Check
                dist_vec = w1.position - w2.position
                dist = math.sqrt(dist_vec.dot(dist_vec))
                
                if dist < 1.0: # Close interaction
                    self._interact(w1, w2, dist)

    def _interact(self, w1: WavePacket, w2: WavePacket, dist: float):
        type1 = w1.get_type()
        type2 = w2.get_type()

        # PHYSICS meets PHYSICS = Collision (Repulsion)
        if type1 == "PHYSICS" and type2 == "PHYSICS":
            # Repel logic: In a real field, this creates potential energy.
            # Here we just log the "Bang".
            logger.info(f"  COLLISION: {w1.frequency}Hz <-> {w2.frequency}Hz")
            
            # Transfer momentum (Swap amplitudes/frequencies slightly)
            # Conservation of Energy
            avg_amp = (w1.amplitude + w2.amplitude) / 2
            w1.amplitude = avg_amp
            w2.amplitude = avg_amp

        # LIGHT meets PHYSICS = Rendering (Reflection)
        elif (type1 == "LIGHT" and type2 == "PHYSICS") or (type1 == "PHYSICS" and type2 == "LIGHT"):
            logger.info(f"  RENDER: Light hit Matter. Reflection intent.")

        # LOGIC meets LOGIC = Synthesis (Harmony)
        elif type1 == "LOGIC" and type2 == "LOGIC":
            # Constructive interference
            if abs(w1.frequency - w2.frequency) < 5.0:
                 w1.amplitude *= 1.1
                 w2.amplitude *= 1.1
                 logger.info(f"  RESONANCE: Thoughts amplifying each other.")

    def render_view(self, camera_pos: Vector3) -> List[dict]:
        """
        Returns a list of 'Visible' waves (Photons) relative to the camera.
        This IS the Renderer.
        """
        frame = []
        for wave in self.waves:
            if wave.get_type() == "LIGHT":
                # Calculate perceived brightness
                dist_vec = wave.position - camera_pos
                dist = math.sqrt(dist_vec.dot(dist_vec))
                if dist > 0:
                    brightness = wave.amplitude / (dist * dist) # Inverse square law
                    frame.append({
                        "pos": wave.position,
                        "color_freq": wave.frequency,
                        "brightness": brightness
                    })
        return frame
