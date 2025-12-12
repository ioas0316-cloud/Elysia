
import pytest
import math
import numpy as np
from Core.Foundation.Math.wave_tensor import WaveTensor, create_harmonic_series

class TestWaveTensor:

    def test_superposition_constructive(self):
        """Test that in-phase waves add up (1+1=2)."""
        w1 = WaveTensor("A")
        w1.add_component(440, 1.0, 0.0)
        
        w2 = WaveTensor("B")
        w2.add_component(440, 1.0, 0.0) # Same Phase
        
        w3 = w1 + w2
        spectrum = w3._spectrum[440]
        
        assert abs(spectrum) == 2.0 # Amplitude doubled
    
    def test_superposition_destructive(self):
        """Test that out-of-phase waves cancel out (1-1=0)."""
        w1 = WaveTensor("A")
        w1.add_component(440, 1.0, 0.0)
        
        w2 = WaveTensor("B")
        w2.add_component(440, 1.0, math.pi) # 180 degrees shift
        
        w3 = w1 + w2
        spectrum = w3._spectrum[440]
        
        assert abs(spectrum) < 1e-9 # Effectively zero

    def test_resonance_perfect(self):
        """Test that identical waves have 100% resonance."""
        w1 = create_harmonic_series(440)
        w2 = create_harmonic_series(440)
        
        resonance = w1 @ w2
        assert math.isclose(resonance, 1.0, rel_tol=1e-5)

    def test_resonance_orthogonal(self):
        """Test that different frequencies have 0% resonance."""
        w1 = WaveTensor("C4")
        w1.add_component(261.63, 1.0)
        
        w2 = WaveTensor("F#4") # Tritone, distinct freq
        w2.add_component(370.00, 1.0)
        
        resonance = w1 @ w2
        assert math.isclose(resonance, 0.0, abs_tol=1e-9)

    def test_partial_resonance(self):
        """Test mixing shared and distinctive frequencies."""
        w1 = WaveTensor("Mix A")
        w1.add_component(100, 1.0)
        w1.add_component(200, 1.0)
        
        w2 = WaveTensor("Mix B")
        w2.add_component(200, 1.0) # Shared
        w2.add_component(300, 1.0) # Different
        
        # Resonance should be roughly 0.5 (one shared component out of two equal vectors)
        # Expected calculation: 
        # A = [1, 1, 0], B = [0, 1, 1] (at indices 100, 200, 300)
        # Dot = 1
        # MagA = sqrt(2), MagB = sqrt(2)
        # Res = 1 / 2 = 0.5
        resonance = w1 @ w2
        assert math.isclose(resonance, 0.5, rel_tol=1e-5)

    def test_phase_shift_effect(self):
        """Resonance should decrease as phase shifts away."""
        w1 = WaveTensor("A")
        w1.add_component(100, 1.0, 0.0)
        
        w2 = WaveTensor("B")
        w2.add_component(100, 1.0, math.pi / 2) # 90 degrees shift
        
        # Dot product of 1 and i is |1*-i| = |-i| = 1. Wait.
        # Dot product formula: z1 * conj(z2).
        # 1 * conj(i) = 1 * -i = -i. Magnitude is 1.
        # Resonance ignores global phase difference if it's constant!
        # Because we want "Shape Similarity".
        # However, RELATIVE phase matters if there are multiple components.
        
        # Single component: Phase shift shouldn't change resonance magnitude (it's just a rotation).
        # |<A, B>| checks alignment.
        resonance = w1 @ w2
        assert math.isclose(resonance, 1.0, rel_tol=1e-5)
        
        # But if we superpose, it changes.
        w_sum = w1 + w2
        # Amp should be sqrt(1^2 + 1^2) = sqrt(2) approx 1.414
        assert math.isclose(w_sum.total_energy, 2.0, rel_tol=1e-5)

if __name__ == "__main__":
    # Integration Sandbox
    w1 = create_harmonic_series(100)
    w2 = create_harmonic_series(100)
    w2.phase_shift(math.pi) # Invert
    
    print(f"Resonance(Inverted): {w1 @ w2}")
    print(f"Superposition Energy(Inverted): {(w1 + w2).total_energy}")
