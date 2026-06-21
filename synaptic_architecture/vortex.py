import numpy as np
from .field import MemristiveField

class VortexConvergence:
    """
    [Synaptic Architecture] Vortex-based Addressing (Convergence Engine)
    Abolishes fixed IDs. Uses vectorized resonance mapping and FFT-style
    annihilation to instantly identify the vortex of information.
    """
    def __init__(self, field: MemristiveField):
        self.field = field

    def resonance_map(self, input_waveform: np.ndarray) -> np.ndarray:
        """
        Calculate the resonance field for the entire spatial map.
        (v ^ v = 0) is achieved when resonance is maximized.
        """
        # Vectorized dot product across the entire data field
        # Result is a 2D map of resonance values [0, 1]
        field_flat = self.field.data_field.reshape(-1, 64)
        input_norm = np.linalg.norm(input_waveform)
        if input_norm == 0: return np.zeros((self.field.resolution, self.field.resolution))

        # Batch dot product
        res_flat = np.dot(field_flat, input_waveform)

        # Norms of the field vectors
        field_norms = np.linalg.norm(field_flat, axis=1)
        field_norms[field_norms == 0] = 1.0 # Avoid div by zero

        res_map = (np.abs(res_flat) / (field_norms * input_norm)).reshape(self.field.resolution, self.field.resolution)
        return res_map

    def converge_to_vortex(self, input_waveform: np.ndarray, initial_pos: np.ndarray = None) -> np.ndarray:
        """
        [Implicit Addressing]
        The pointer 'slides' along the combined gradient of:
        1. Local Resonance (Cognitive Gravity)
        2. Memristive Conductance (Physical Potential)
        """
        if initial_pos is None:
            # If no guess, use global maximum of conductance field
            initial_pos = np.array(np.unravel_index(np.argmax(self.field.conductance), self.field.conductance.shape), dtype=float)

        current_pos = initial_pos.copy()

        # Parameters for sliding
        max_steps = 50
        learning_rate = 2.0

        # Pre-calculate global resonance map once (The 'Echo' step)
        res_map = self.resonance_map(input_waveform)
        ry, rx = np.gradient(res_map) # Cognitive Gravity Gradient

        for _ in range(max_steps):
            y, x = np.clip(current_pos, 0, self.field.resolution - 1).astype(int)

            # Combine gradients: Cognitive Gravity (Resonance) + Physical Conductance (Trace)
            c_grad = self.field.get_potential_gradient(current_pos)
            r_grad = np.array([ry[y, x], rx[y, x]])

            # The 'Sliding' Force
            total_force = r_grad * 10.0 + c_grad * 2.0

            # Update position
            current_pos += total_force * learning_rate

            # Convergence check: If the force is negligible, we are in the Vortex
            if np.linalg.norm(total_force) < 1e-4:
                break

        return current_pos

if __name__ == "__main__":
    mf = MemristiveField()
    vc = VortexConvergence(mf)

    pattern = np.random.randn(64)
    mf.deposit_engram(np.array([100, 100]), pattern)

    target = vc.converge_to_vortex(pattern, initial_pos=np.array([90, 90]))
    print(f"Vortex found at: {target}")
