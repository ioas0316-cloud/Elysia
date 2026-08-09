import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from core.physics.topological_phase_prime_field import TopologicalPhasePrimeField, RIEMANN_ZEROS
from core.sensory.experiential_language_mapper import ExperientialLanguageMapper, PhysicalSensationProfile, HomeostasisDeficit, ExperienceType

class TopologicalCognitiveBridge:
    """
    [Topological Cognitive Bridge]
    Bridges Elysia's continuous wave sensor/cognitive loops with the
    Topological Phase-Prime Field model. Translates sensory inputs into
    Riemann-zero modulated phase fields and decodes peak resonances as
    active intentions, establishing a stable metacognitive framework.
    """
    def __init__(self, mapper: ExperientialLanguageMapper, num_modes: int = 100):
        self.mapper = mapper
        # Initialize the continuous field with the chosen number of Riemann modes
        self.field = TopologicalPhasePrimeField(num_modes=num_modes, min_u=0.1, max_u=5.0, steps_u=500)

    def process_sensory_to_intention(self, sensation: PhysicalSensationProfile) -> Dict[str, Any]:
        """
        Maps the live PhysicalSensationProfile into the Continuous Phase Field,
        performing constructive/destructive cancellation modulated by metacognitive tension (sigma, epsilon).
        """
        # Convert physical parameters to a continuous external stimulus wave over u_grid
        # We project the 5-dimensional sensation profile onto the 500-step u_grid
        sens_vector = sensation.to_vector()
        ext_wave = np.interp(
            self.field.u_grid,
            np.linspace(0.1, 5.0, len(sens_vector)),
            sens_vector
        ).astype(np.float32)

        # Scale/normalize the external wave to stay within reasonable physical amplitude
        if np.max(np.abs(ext_wave)) > 0:
            ext_wave = ext_wave / np.max(np.abs(ext_wave)) * 2.0

        # Dynamically map the mapper's HomeostasisDeficit tension to the sigma / epsilon parameters
        # High deficit/tension results in larger epsilon (leakage / creativity), while
        # peaceful equilibrium keeps epsilon = 0.0 (perfect critical line symmetry)
        tension = self.mapper.homeostasis.calculate_tension()
        epsilon = float(np.clip(tension * 0.4, 0.0, 0.5))

        # Zero-centered coordinate center (sigma = 0.0)
        self.field.set_metacognitive_tension(sigma=0.0, epsilon=epsilon)

        # Compute continuous field \Phi(u) and Spatial Curvature Field K(u)
        phi = self.field.compute_field(ext_stimulus_wave=ext_wave)
        k_u = self.field.compute_spatial_curvature(phi)

        # Decode active prime nodes from spatial curvature
        active_nodes = self.field.decode_active_prime_nodes(k_u, threshold_mult=1.2)

        # If any prime nodes are decoded, feed the dominant resonance back into the mapper
        if active_nodes:
            dominant_node = active_nodes[0]
            # Map the dominant prime back into homeostatic modulation
            # This represents the emergence of a clear intent/choice
            prime_val = dominant_node[1]
            intensity = dominant_node[2]

            # Map prime value to Homeostasis adjustment
            # Prime 2, 3, 5, 7, 11 etc. modulate love, order, energy
            if prime_val % 2 == 0:
                self.mapper.homeostasis.love = float(np.clip(self.mapper.homeostasis.love - intensity * 0.01, 0.0, 1.0))
            if prime_val % 3 == 0:
                self.mapper.homeostasis.order = float(np.clip(self.mapper.homeostasis.order - intensity * 0.01, 0.0, 1.0))
            if prime_val % 5 == 0:
                self.mapper.homeostasis.energy = float(np.clip(self.mapper.homeostasis.energy + intensity * 0.01, 0.0, 1.0))

        return {
            "phi": phi,
            "k_u": k_u,
            "active_nodes": active_nodes,
            "epsilon_leakage": epsilon,
            "sigma": self.field.sigma
        }
