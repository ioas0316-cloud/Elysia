import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# Configure Logger
logger = logging.getLogger("Prism")

@dataclass
class SevenChannelQualia:
    """
    The 7-Dimensional Wave Signature of any digested concept.
    """
    physical: float = 0.0    # [0] Form/Color/Texture
    functional: float = 0.0  # [1] Mechanism/Utility
    phenomenal: float = 0.0  # [2] Sensation/Feeling
    causal: float = 0.0      # [3] History/Time/Cause
    mental: float = 0.0      # [4] Logic/Abstraction
    structural: float = 0.0  # [5] Law/Pattern
    spiritual: float = 0.0   # [6] Intent/Love/Will
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.physical, self.functional, self.phenomenal, self.causal,
            self.mental, self.structural, self.spiritual
        ], dtype=np.float32)

@dataclass
class DoubleHelixWave:
    """
    The output of the Double Helix Digestion.
    Contains two strands: Pattern (Phenomenon) and Principle (Essence).
    """
    pattern_strand: np.ndarray   # The "Body" (High-dim Vector)
    principle_strand: np.ndarray # The "Soul" (7-Channel Qualia)
    phase: float = 0.0             # Rotational Phase (0 ~ 2pi)

@dataclass
class WaveDynamics:
    """Compatibility structure for older dimensional parsing and digestion."""
    physical: float = 0.0
    functional: float = 0.0
    phenomenal: float = 0.0
    causal: float = 0.0
    mental: float = 0.0
    structural: float = 0.0
    spiritual: float = 0.0
    mass: float = 1.0

class DoubleHelixPrism:
    """
    [The Optical Instrument of the Soul]
    Splits raw data (Weights/Text) into the Double Helix structure.
    Sovereign Edition: Uses Numpy (CPU) instead of Torch (GPU).
    """
    def __init__(self):
        self.fundamental_frequency = 432.0 # Hz (Standard Reference)
    
    def _load_model(self):
        logger.info("Prism model 'loaded' (Internal Logic).")

    def transduce(self, text: str) -> Any:
        wave = self.refract_text(text)
        
        class Profile:
            def __init__(self, wave_ref):
                # principle_strand is a 7D array
                p = wave_ref.principle_strand
                self.dynamics = WaveDynamics(
                    causal=float(p[3]), # Note: Index mapping fixed
                    physical=float(p[0]),
                    functional=float(p[1]),
                    phenomenal=float(p[2]),
                    structural=float(p[5]),
                    mental=float(p[4]),
                    spiritual=float(p[6])
                )
        return Profile(wave)
        
    def refract_weight(self, weight_array: np.ndarray, layer_name: str) -> DoubleHelixWave:
        """
        Refracts a raw neural weight into a Double Helix Wave.
        """
        # 1. Pattern Strand (The Raw Signal)
        raw_signal = weight_array.flatten().astype(np.float32)
        
        if raw_signal.size > 1024:
            # Downsample
            steps = raw_signal.size // 1024
            if steps > 0:
                 raw_signal = raw_signal[::steps][:1024]
        
        norm = np.linalg.norm(raw_signal)
        pattern_strand = raw_signal / (norm + 1e-9)
        
        # 2. Principle Strand (The 7D Spectrum via FFT)
        fft_spectrum = np.abs(np.fft.rfft(raw_signal))
        total_energy = fft_spectrum.sum() + 1e-9
        
        n_bins = fft_spectrum.shape[0]
        band_width = max(1, n_bins // 7)

        bands = []
        for i in range(7):
            start = i * band_width
            end = (i + 1) * band_width if i < 6 else n_bins
            if start >= n_bins: 
                bands.append(0.0)
                continue
            band_energy = fft_spectrum[start:end].sum()
            bands.append(float(band_energy / total_energy))
            
        # Mapping Bands to Qualia 
        # 0 (Low) -> Causal
        qualia = SevenChannelQualia(
            causal=bands[0],
            physical=bands[1],
            functional=bands[2],
            phenomenal=bands[3],
            structural=bands[4],
            mental=bands[5],
            spiritual=bands[6]
        )

        # 3. Calculate Phase
        peak_idx = np.argmax(fft_spectrum)
        phase = (peak_idx / n_bins) * (2 * np.pi)

        return DoubleHelixWave(
            pattern_strand=pattern_strand,
            principle_strand=qualia.to_vector(),
            phase=phase
        )

    def refract_text(self, text: str) -> DoubleHelixWave:
        """
        Refracts a text string into a Double Helix Wave.
        """
        raw_vals = np.array([ord(c) for c in text[:1024]], dtype=np.float32)
        pattern_strand = np.zeros(1024, dtype=np.float32)
        if raw_vals.size > 0:
            pattern_strand[:raw_vals.size] = raw_vals
            
        norm = np.linalg.norm(pattern_strand)
        pattern_strand = pattern_strand / (norm + 1e-9)

        length = len(text)
        complexity = len(set(text)) / length if length > 0 else 0

        qualia = SevenChannelQualia(
            mental=min(1.0, complexity * 2),
            structural=min(1.0, length / 1000.0),
            physical=0.1, 
            functional=0.5
        )

        return DoubleHelixWave(
            pattern_strand=pattern_strand,
            principle_strand=qualia.to_vector(),
            phase=0.0
        )

# Compatibility Aliases
PrismEngine = DoubleHelixPrism