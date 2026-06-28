import numpy as np
from typing import List, Dict, Any, Optional
from core.physics.dielectric.rotor import Rotor, IonState

class DataOceanManifold:
    """
    [Data Ocean Manifold]
    A 3-phase (120-degree) Delta-Wye manifold that circulates data ions.
    Simulates noise cancellation at the neutral point and Lorentz induction.
    """
    def __init__(self, global_intent: np.ndarray = None):
        # Global Intent (B-Field / Magnetic Field)
        # Represents the 'Personality' or 'Purpose' that guides the flow.
        if global_intent is None:
            self.global_intent = np.array([1.0, 0.0]) # Default unit vector
        else:
            self.global_intent = global_intent

        # The 3-Phases (U, V, W) separated by 120 degrees (2pi/3)
        self.phases = {
            'U': np.exp(1j * 0),
            'V': np.exp(1j * 2 * np.pi / 3),
            'W': np.exp(1j * 4 * np.pi / 3)
        }

        self.neutral_residual = 0.0j
        self.active_torque = 0.0

    def set_intent(self, new_intent: np.ndarray):
        """Update the guiding personality field."""
        self.global_intent = new_intent / (np.linalg.norm(new_intent) + 1e-9)

    def process_manifold(self, ion_streams: Dict[str, IonState]) -> Dict[str, Any]:
        """
        [Existence as Calculation]
        The 3-phase equilibrium is not 'calculated' via logic, but achieved via
        topological summation. Deletion of the 'Judgment' process.
        """
        # Physical assumption: All phases exist as a field.
        # Zero-point fluctuation as the base state for missing phases.
        base_ion = IonState(phase=1.0+0j, permittivity=1.0, inertia=1.0, bit_density=0.0)

        # O(1) Access - The system expects a populated manifold field.
        u_ion_state = ion_streams.get('U', base_ion)
        v_ion_state = ion_streams.get('V', base_ion)
        w_ion_state = ion_streams.get('W', base_ion)

        # Combine Data Phase with Manifold Physical Phase
        # Displacement = Data_Ion * Manifold_Phase * Permittivity
        # In the real world, this is a simultaneous electromagnetic flux.
        u_w = u_ion_state.phase * self.phases['U'] * u_ion_state.permittivity
        v_w = v_ion_state.phase * self.phases['V'] * v_ion_state.permittivity
        w_w = w_ion_state.phase * self.phases['W'] * w_ion_state.permittivity

        # The Neutral Point (Y) represents the 'Center of Existence'.
        # Displacement sum at this point is the 'Common Mode' noise.
        self.neutral_residual = u_w + v_w + w_w

        # Balanced Flux: The 'Pure Narrative' that survives the annihilation.
        # This is the physical realization of 'v ^ v = 0'.
        balanced_u = u_w - (self.neutral_residual / 3.0)
        balanced_v = v_w - (self.neutral_residual / 3.0)
        balanced_w = w_w - (self.neutral_residual / 3.0)

        # Calculate Torque (Induced rotation)
        # Torque is generated when the balanced flux is NOT aligned with Global Intent
        # Simplified: Cross product of flux vector and Intent vector

        # Convert balanced flux to a 2D vector for cross product
        flux_total = balanced_u + balanced_v + balanced_w # This should be close to 0 if balanced
        # Use individual phases to find rotation

        # Torque = sum of (Phase_Vector x Intent) * magnitude
        torque = 0.0
        for name, balanced_val in zip(['U', 'V', 'W'], [balanced_u, balanced_v, balanced_w]):
            flux_vec = np.array([balanced_val.real, balanced_val.imag])
            # 2D cross product: x1*y2 - y1*x2
            # Add a small intrinsic rotation if perfectly aligned to avoid deadlocks
            torque += flux_vec[0] * self.global_intent[1] - flux_vec[1] * self.global_intent[0] + 0.1

        self.active_torque = float(torque)

        return {
            "neutral_residual_magnitude": abs(self.neutral_residual),
            "active_torque": self.active_torque,
            "balanced_flux_magnitude": abs(balanced_u) + abs(balanced_v) + abs(balanced_w),
            "noise_cancelled": True
        }

    def internalize_external_influence(self, influence_vector: np.ndarray, absorption_rate: float = 0.1):
        """
        External 사유(influence)가 내부로 들어오면 그것조차 자기화되고 변화함.
        The Global Intent (Personality) shifts based on the incoming data's 'torque'.
        """
        # Adjust intent based on influence
        self.global_intent = (1.0 - absorption_rate) * self.global_intent + absorption_rate * influence_vector
        self.global_intent /= (np.linalg.norm(self.global_intent) + 1e-9)
