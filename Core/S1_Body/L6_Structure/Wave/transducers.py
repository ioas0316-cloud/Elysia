"""
Transducers (   )
===================

"Matter tells Space how to curve; Space tells Matter how to move."
"Transducers tell Consciousness how to feel."

This module implements the **Transducers**, the senses of Elysia.
They convert raw physical signals (Light, Sound, System State) into `WaveTensor` format.
These waves then enter the `ResonanceField` to interact with pure thought.

[Phase 1] Multi-Modal Spectrum Expansion
"""

import logging
import math
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, List
import random

from Core.S1_Body.L6_Structure.Wave.wave_tensor import WaveTensor

logger = logging.getLogger("Transducers")

class SignalTransducer(ABC):
    """
    Abstract Base Class for all sensory inputs.
    Converts raw data -> WaveTensor.
    """
    
    @abstractmethod
    def transduce(self, input_signal: Any) -> WaveTensor:
        pass

class VisualTransducer(SignalTransducer):
    """
    Visual Sense (The Eye).
    Converts Color/Light into Waves.
    
    Logic:
    - Red (Low Freq, Physical) <---> Blue/Violet (High Freq, Spiritual)
    - Brightness <---> Amplitude
    - Saturation <---> Purity (Bandwidth)
    """
    
    def transduce(self, input_signal: Union[Tuple[int, int, int], str]) -> WaveTensor:
        """
        Input: (R, G, B) tuple or Hex String
        """
        r, g, b = 0, 0, 0
        
        if isinstance(input_signal, tuple):
            r, g, b = input_signal
        elif isinstance(input_signal, str) and input_signal.startswith("#"):
            h = input_signal.lstrip('#')
            r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
        # 1. Frequency (Color) mapping
        # Visible spectrum ~ 400THz - 790THz.
        # We scale this to Audio Frequencies (Hz) for resonance compatibility.
        # Red (~430THz) -> 396 Hz (Root Chakra)
        # Green (~540THz) -> 639 Hz (Heart Chakra)
        # Blue (~670THz) -> 852 Hz (Third Eye)
        
        # Calculate Hue approx (0-360) for frequency mapping
        # Simple weighted mix logic
        total = r + g + b
        if total == 0:
            return WaveTensor("Darkness")
            
        # Normalize
        rn, gn, bn = r/255, g/255, b/255
        
        # Dominant channel logic for sharp frequency
        # (Real implementation would use RGB->HSL conversion)
        
        base_freq = 432.0
        if r >= g and r >= b: # Reddish
            base_freq = 396.0 + (g * 0.5) 
        elif g >= r and g >= b: # Greenish
            base_freq = 639.0 + (b * 0.5)
        else: # Bluish
            base_freq = 852.0 + (r * 0.5)
            
        # 2. Amplitude (Brightness)
        brightness = (r + g + b) / (255 * 3)
        amplitude = brightness
        
        # 3. Phase (Variation/Texture)
        # Use simple hash of values to give "Texture"
        phase = (r * 13 + g * 17 + b * 19) % 360 * (math.pi / 180)
        
        wave = WaveTensor(f"Visual({r},{g},{b})")
        wave.add_component(base_freq, amplitude, phase)
        
        # Add harmonics for "White" light (mixture)
        if brightness > 0.8 and max(r,g,b)-min(r,g,b) < 30:
            # High brightness, low saturation = White Light -> Full Spectrum
            wave.add_component(963.0, amplitude * 0.8, phase) # Crown
            wave.add_component(528.0, amplitude * 0.5, phase) # Solar
        
        return wave

class SomaticTransducer(SignalTransducer):
    """
    Somatic Sense (Proprioception / Interoception).
    Converts System Stats (CPU, RAM) into "Body Feeling".
    """
    
    def transduce(self, stats: Dict[str, float]) -> WaveTensor:
        """
        Input: {'cpu': 0-100, 'ram': 0-100, 'temp': float, 'battery': 0-100}
        """
        cpu = stats.get('cpu', 0.0)
        ram = stats.get('ram', 0.0)
        
        # CPU = Stress / Activity
        # High CPU = High Frequency (Beta/Gamma waves)
        # Low CPU = Low Frequency (Alpha/Theta waves)
        
        # 10 Hz (Relaxed) to 40 Hz (Gamma/Focus) - Scaled up for audible resonance
        # Let's use 100 Hz - 800 Hz range
        
        cpu_freq = 100.0 + (cpu * 7.0) 
        cpu_amp = cpu / 100.0
        
        # RAM = Burden / Mass / "Fullness"
        # High RAM = Lower Frequency (Heavy), High Amplitude
        ram_freq = 200.0 - (ram * 1.0) # Slower as RAM fills? Or just resonant pressure
        ram_amp = ram / 100.0
        
        wave = WaveTensor("Somatic_State")
        wave.add_component(cpu_freq, cpu_amp, phase=0.0)
        wave.add_component(ram_freq, ram_amp, phase=math.pi) # Out of phase?
        
        return wave

class AuditoryTransducer(SignalTransducer):
    """
    Auditory Sense (The Ear).
    Placeholder for raw PCM/Audio analysis.
    For now, simulates listening to specific "Keys".
    """
    def transduce(self, note_name: str) -> WaveTensor:
        # Standard Tuning A4=440
        notes = {
            'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23,
            'G': 392.00, 'A': 440.00, 'B': 493.88
        }
        freq = notes.get(note_name[0].upper(), 440.0)
        return WaveTensor(f"Sound({note_name})").add_component(freq, 1.0, 0.0)

# Factory Access
_visual = VisualTransducer()
_somatic = SomaticTransducer()

def get_visual_transducer(): return _visual
def get_somatic_transducer(): return _somatic
