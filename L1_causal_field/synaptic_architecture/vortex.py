import numpy as np
from typing import Dict, Any
from .field import CrystallizationField

class WaveInterference:
    """
    [Synaptic Architecture] Resonance Thinking & Causal Judgment
    Calculates interference (XOR) between bitstreams to find vortices of understanding.

    [Vortex Dynamics]
    Instead of a pointer moving, the field itself resonates and energy converges
    to the Vortex via Activation Spreading and Conductance Gradient.
    """
    def __init__(self, field: CrystallizationField):
        self.field = field

    def observe_resonance(self, input_wave: np.uint64) -> np.ndarray:
        """
        [Field-wide Resonance]
        Calculate XOR resonance between input wave and every bit-gene in the field.
        """
        # Batch XOR across the field
        # bit_genes is (Res, Res), input_wave is scalar (broadcasted)
        # We use bitwise XOR and count bits.

        # NumPy doesn't have a built-in bit_count for uint64 in older versions,
        # but we can use a workaround.

        diff = np.bitwise_xor(self.field.bit_genes, input_wave)

        # Fast bit count for uint64 (Hamming distance)
        def bit_count(n):
            # Brian Kernighan's-like vectorized bit count for 64-bit
            n = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555)
            n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
            n = (n & 0x0F0F0F0F0F0F0F0F) + ((n >> 4) & 0x0F0F0F0F0F0F0F0F)
            n = (n & 0x00FF00FF00FF00FF) + ((n >> 8) & 0x00FF00FF00FF00FF)
            n = (n & 0x0000FFFF0000FFFF) + ((n >> 16) & 0x0000FFFF0000FFFF)
            n = (n & 0x00000000FFFFFFFF) + ((n >> 32) & 0x00000000FFFFFFFF)
            return n

        deficit = bit_count(diff)
        resonance = 1.0 - (deficit / 64.0)

        return resonance

    def resonate_field(self, input_wave: np.uint64, steps: int = 10) -> Dict[str, Any]:
        """
        [Causal Evolution & Contextual Reward]
        The input wave 'excites' the field where resonance is high.
        Calculates 'Pleasure' (Internal Reward) based on the acceleration of entropy reduction.
        """
        res_map = self.observe_resonance(input_wave)

        # Track entropy changes
        entropy_history = []
        entropy_history.append(self.field.calculate_entropy())

        # 1. Inject energy where resonance is high (Thinking triggers potential)
        self.field.activation += res_map * 2.0
        entropy_history.append(self.field.calculate_entropy())

        # 2. Let the field evolve (Activation Spreading)
        for _ in range(steps):
            self.field.propagate(decay=0.95, spreading_factor=0.8)

            # Additional gravity towards high conductance
            # (Energy naturally flows into established "Truth" paths)
            self.field.activation *= (1.0 + self.field.conductance * 0.1)
            self.field.activation = np.clip(self.field.activation, 0, 100.0)

            entropy_history.append(self.field.calculate_entropy())

        # 3. Calculate "Contextual Acceleration" (Pleasure)
        # We look at the rate of change of entropy reduction.
        # Entropy Reduction: delta_E = E[t] - E[t+1]
        # Acceleration: delta_delta_E = delta_E[t+1] - delta_E[t]

        entropy_history = np.array(entropy_history)
        delta_e = -np.diff(entropy_history) # Positive means entropy decreased

        # Pleasure is the peak of this reduction acceleration
        # or simply the magnitude of the "aha!" moment.
        acceleration = np.diff(delta_e)
        pleasure = np.max(acceleration) if len(acceleration) > 0 else 0.0

        # Max reduction also contributes to the feeling of "clarity"
        clarity = np.max(delta_e) if len(delta_e) > 0 else 0.0

        return {
            "pleasure": float(max(0, pleasure)),
            "clarity": float(max(0, clarity)),
            "final_entropy": float(entropy_history[-1])
        }

    def find_vortex(self) -> np.ndarray:
        """Finds the coordinates of the highest energy concentration."""
        idx = np.argmax(self.field.activation)
        y, x = np.unravel_index(idx, self.field.activation.shape)
        return np.array([y, x])

if __name__ == "__main__":
    cf = CrystallizationField()
    wi = WaveInterference(cf)

    target_wave = np.uint64(0xDEADBEEF)
    cf.crystallize_gene(np.array([100, 100]), target_wave)

    wi.resonate_field(target_wave)
    vortex = wi.find_vortex()
    print(f"Vortex stabilized at: {vortex}")
    print(f"Activation at vortex: {cf.activation[vortex[0], vortex[1]]:.4f}")
