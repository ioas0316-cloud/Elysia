import numpy as np
from .field import CrystallizationField

class WaveInterference:
    """
    [Synaptic Architecture] Resonance Thinking & Causal Judgment
    Calculates interference (XOR) between bitstreams to find vortices of understanding.
    """
    def __init__(self, field: CrystallizationField):
        self.field = field

    def observe_interference(self, input_wave: np.ndarray) -> np.ndarray:
        """
        [Resonance Thinking]
        Compare input waveform with the entire spatial bit field using bitwise logic.
        v ^ v = 0 -> Perfect Resonance (No Deficit).
        """
        # Batch XOR across the field
        # In a real hardware system, this is a parallel bit-bus match.
        field_bits = self.field.bit_field.astype(np.int32)
        input_bits = input_wave.astype(np.int32)

        # Deficit Map: 1 where bits differ, 0 where they resonate
        deficit_flat = np.bitwise_xor(field_bits.reshape(-1, 64), input_bits)

        # Resonance Score: Percentage of bits that matched (0 to 1)
        # 1.0 = Perfect Resonance (Thinking matches Reality)
        resonance_score = 1.0 - (np.sum(deficit_flat, axis=1) / 64.0)
        return resonance_score.reshape(self.field.resolution, self.field.resolution)

    def deduce_vortex(self, input_wave: np.ndarray, initial_pos: np.ndarray = None) -> np.ndarray:
        """
        [Causal Judgment]
        The pointer slides along the gradient of resonance to reach the 'Truth' (Vortex).
        """
        if initial_pos is None:
            initial_pos = np.array(np.unravel_index(np.argmax(self.field.conductance), self.field.conductance.shape), dtype=float)

        current_pos = initial_pos.copy()

        # Pre-calculate the resonance field (The 'Cognitive Landscape')
        res_map = self.observe_interference(input_wave)
        ry, rx = np.gradient(res_map)

        max_steps = 30
        for _ in range(max_steps):
            y, x = np.clip(current_pos, 0, self.field.resolution - 1).astype(int)

            # Gradients from both Resonance (Judgment) and Conductance (Memory)
            r_grad = np.array([ry[y, x], rx[y, x]])
            c_grad = self.field.get_motion_gradient(current_pos)

            # Total attraction force towards the vortex
            force = r_grad * 5.0 + c_grad * 1.0

            current_pos += force * 2.0

            if np.linalg.norm(force) < 1e-4:
                break

        return current_pos

if __name__ == "__main__":
    cf = CrystallizationField()
    wi = WaveInterference(cf)

    p = np.random.randint(0, 2, 64)
    cf.solidify_bits(np.array([100, 100]), p)

    vortex = wi.deduce_vortex(p, initial_pos=np.array([95, 95]))
    print(f"Vortex stabilized at: {vortex}")
