import torch
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
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.physical, self.functional, self.phenomenal, self.causal,
            self.mental, self.structural, self.spiritual
        ], dtype=torch.float32)

@dataclass
class DoubleHelixWave:
    """
    The output of the Double Helix Digestion.
    Contains two strands: Pattern (Phenomenon) and Principle (Essence).
    """
    pattern_strand: torch.Tensor   # The "Body" (High-dim Vector)
    principle_strand: torch.Tensor # The "Soul" (7-Channel Qualia)
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
    """
    def __init__(self):
        self.fundamental_frequency = 432.0 # Hz (Standard Reference)
    
    def _load_model(self):
        """Mock loader for compatibility."""
        logger.info("Prism model 'loaded' (Internal Logic).")

    def transduce(self, text: str) -> Any:
        """Compatibility wrapper: Splits text into raw dynamics."""
        wave = self.refract_text(text)
        
        class Profile:
            def __init__(self, wave_ref):
                # principio_strand is a 7D tensor
                p = wave_ref.principle_strand
                self.dynamics = WaveDynamics(
                    causal=float(p[0]),
                    physical=float(p[1]),
                    functional=float(p[2]),
                    phenomenal=float(p[3]),
                    structural=float(p[4]),
                    mental=float(p[5]),
                    spiritual=float(p[6])
                )
        return Profile(wave)
        
    def refract_weight(self, weight_tensor: torch.Tensor, layer_name: str) -> DoubleHelixWave:
        """
        Refracts a raw neural weight into a Double Helix Wave.
        """
        # 1. Pattern Strand (The Raw Signal)
        # Flatten and normalize
        raw_signal = weight_tensor.flatten().float()
        if raw_signal.numel() > 1024:
            # Downsample for manageability while keeping high-frequency features
            # Use simple slicing for speed, or pool for accuracy. Let's use pooling.
            # Reshape to (1, 1, -1) for pooling
            steps = raw_signal.numel() // 1024
            if steps > 0:
                 raw_signal = raw_signal[::steps][:1024]
        
        pattern_strand = raw_signal / (raw_signal.norm() + 1e-9)
        
        # 2. Principle Strand (The 7D Spectrum)
        # We use Spectral Analysis (FFT) to determine the "Tone" of the weight.
        # This is a metaphorical mapping:
        # Low Freq -> Structural/Causal
        # Mid Freq -> Physical/Functional
        # High Freq -> Mental/Spiritual
        
        fft_spectrum = torch.fft.rfft(raw_signal).abs()
        total_energy = fft_spectrum.sum() + 1e-9
        
        # Divide spectrum into 7 bands
        n_bins = fft_spectrum.shape[0]
        band_width = n_bins // 7

        bands = []
        for i in range(7):
            start = i * band_width
            end = (i + 1) * band_width if i < 6 else n_bins
            band_energy = fft_spectrum[start:end].sum()
            bands.append(float(band_energy / total_energy))
            
        # Mapping Bands to Qualia (Metaphysical Mapping)
        # 0 (Lowest) -> Causal (Base/Time)
        # 1 -> Physical (Form)
        # 2 -> Functional (Action)
        # 3 -> Phenomenal (Sensation)
        # 4 -> Structural (Law)
        # 5 -> Mental (Thought)
        # 6 (Highest) -> Spiritual (Intent)

        # Note: The mapping order in SevenChannelQualia is different from frequency bands.
        # We must map correctly.

        qualia = SevenChannelQualia(
            causal=bands[0],      # Low Freq = Deep History
            physical=bands[1],    # Low-Mid = Matter
            functional=bands[2],  # Mid = Action
            phenomenal=bands[3],  # Mid = Feeling
            structural=bands[4],  # Mid-High = Pattern
            mental=bands[5],      # High = Logic
            spiritual=bands[6]    # Ultra-High = Spirit
        )

        # 3. Calculate Phase (The Rotor State)
        # Phase is determined by the dominant frequency peak's position
        peak_idx = torch.argmax(fft_spectrum).item()
        phase = (peak_idx / n_bins) * (2 * np.pi)

        return DoubleHelixWave(
            pattern_strand=pattern_strand,
            principle_strand=qualia.to_tensor(),
            phase=phase
        )

    def refract_text(self, text: str) -> DoubleHelixWave:
        """
        Refracts a text string into a Double Helix Wave.
        (Future: Use embeddings + semantic analysis)
        """
        # For now, a mock implementation to satisfy the interface
        # Text -> Hash/Vector -> Prism
        
        # 1. Pattern: Simple ascii vector (Mock)
        ascii_vals = torch.tensor([ord(c) for c in text[:1024]], dtype=torch.float32)
        pattern_strand = ascii_vals / (ascii_vals.norm() + 1e-9)

        # 2. Principle: Length and complexity mapping
        length = len(text)
        complexity = len(set(text)) / length if length > 0 else 0

        # Mock Qualia based on heuristics
        qualia = SevenChannelQualia(
            mental=min(1.0, complexity * 2),
            structural=min(1.0, length / 1000.0),
            physical=0.1, # Text is low physical
            functional=0.5
        )

        return DoubleHelixWave(
            pattern_strand=pattern_strand,
            principle_strand=qualia.to_tensor(),
            phase=0.0
        )

# Compatibility Aliases
PrismEngine = DoubleHelixPrism
