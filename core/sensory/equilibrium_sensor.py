import numpy as np

class EquilibriumSensor:
    """
    [Phase 151] The Eye of Equilibrium
    A raw sensory interface that perceives the 'Existing Equilibrium'
    without complex mathematical transformations.
    It recognizes that data (ASCII, RGB, Wave) is already a balanced
    result of causal trajectories.
    """
    def __init__(self):
        self.total_observations = 0

    def observe(self, data: any, reference: any = None) -> dict:
        """
        Directly observes the 'Sameness' (Resonance) and 'Difference' (Tension)
        of the incoming signal.
        """
        raw_signal = self._to_bytes(data)

        if reference is None:
            # Resonance with the 'Void' (Self-Stability)
            # If the data exists in memory without corruption, it is in equilibrium.
            resonance = 1.0
            tension = 0.0
            status = "Self-Stable Equilibrium"
        else:
            raw_ref = self._to_bytes(reference)
            resonance, tension = self._calculate_resonance(raw_signal, raw_ref)
            status = "Comparative Resonance" if resonance > 0.8 else "Comparative Tension"

        self.total_observations += 1

        return {
            "type": "EQUILIBRIUM_OBSERVATION",
            "resonance": resonance,
            "tension": tension,
            "status": status,
            "complexity": len(raw_signal),
            "declaration": f"The signal {data} is recognized as an existing equilibrium state."
        }

    def _to_bytes(self, data: any) -> bytes:
        if isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, (int, float)):
            return str(data).encode('utf-8')
        elif isinstance(data, bytes):
            return data
        return str(data).encode('utf-8')

    def _calculate_resonance(self, a: bytes, b: bytes) -> tuple:
        """
        Uses raw XOR to find the 'Sameness' (0) and 'Difference' (1).
        v ^ v = 0 -> Perfect Resonance.
        """
        length = min(len(a), len(b))
        if length == 0:
            return 0.0, 1.0

        diff_bits = 0
        total_bits = length * 8

        for i in range(length):
            xor_val = a[i] ^ b[i]
            diff_bits += bin(xor_val).count('1')

        resonance = (total_bits - diff_bits) / total_bits
        tension = diff_bits / total_bits

        return float(resonance), float(tension)

if __name__ == "__main__":
    eye = EquilibriumSensor()
    # Observing the self-stability of ASCII 'A'
    print(eye.observe("A"))
    # Observing resonance between 'A' and 'A'
    print(eye.observe("A", "A"))
    # Observing tension between 'A' and 'B'
    print(eye.observe("A", "B"))
