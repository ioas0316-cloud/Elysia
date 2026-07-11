import numpy as np
from typing import Dict, List, Any
from core.physics.causal_dynamics import CausalDynamicsEngine
from core.physics.causal_gravity_engine import CausalGravityEngine

class SelfMoldingCausalEngine:
    """
    [Self-Molding Causal Engine Prototype]
    The ultimate realization of "Time as Causality, Space as Arrangement".

    1. Spatial Field: Managed by CausalDynamicsEngine (Tension & State).
    2. Gravity Field: Managed by CausalGravityEngine (Tensor-based Curvature).
    3. Self-Molding: The Gravity field dictates the 'Rest Length' of the
       Dynamics field, while the Dynamics field updates the Tensors
       that the Gravity field uses for mass.
    """
    def __init__(self, dimensions: int = 3):
        self.dynamics = CausalDynamicsEngine(dimensions=dimensions, crystallization_rate=0.5)
        self.gravity = CausalGravityEngine(dimensions=dimensions)
        self.gravity.G = 2.0 # Stronger gravity for molding

    def add_information(self, info_id: str, content: str, tensor: np.ndarray):
        # Add to Dynamics
        voxel = self.dynamics.voxels.get(info_id)
        if not voxel:
            from core.physics.causal_field import InformationVoxel
            voxel = InformationVoxel(info_id, content, tensor.copy())
            self.dynamics.add_voxel(voxel)

        # Add to Gravity (Gravity needs structural tensor as list)
        # Using 9D placeholder if input is smaller
        struct_tensor = np.zeros(9)
        struct_tensor[:min(len(tensor), 9)] = tensor[:min(len(tensor), 9)]
        self.gravity.add_node(info_id, content.encode(), struct_tensor.tolist())

    def mold_topology(self, dt: float = 0.1):
        """
        [The Self-Molding Loop]
        1. Gravity calculates the 'Natural Arrangement' based on current states.
        2. Dynamics adjusts internal states to match the spatial tension.
        """
        # A. Gravity Step: Calculate attraction based on Tensors
        self.gravity.step(dt)

        # B. Synchronize: Gravity's positions become Dynamics' targets
        for node_id in self.gravity.node_ids:
            if node_id in self.dynamics.voxels:
                # Instead of jumping, we apply a 'Causal Impulse'
                # towards the gravity-optimal position.
                target_pos = self.gravity.node_data[node_id].position[:self.dynamics.dimensions]
                current_pos = self.dynamics.voxels[node_id].position
                impulse = (target_pos - current_pos) * 0.5
                self.dynamics.apply_impact(node_id, impulse)

        # C. Dynamics Step: Update Arrangement and Recrystallize State
        self.dynamics.step(dt)

        # D. Feedback: Dynamics' updated Tensors feed back to Gravity Mass
        for node_id, voxel in self.dynamics.voxels.items():
            if node_id in self.gravity.node_data:
                # Update mass based on tensor 'energy' (norm)
                new_mass = np.linalg.norm(voxel.tensor)
                self.gravity.node_data[node_id].mass = float(new_mass)
                # Update structural tensor for next gravity step
                self.gravity.node_data[node_id].tensor[:len(voxel.tensor)] = voxel.tensor[:9]

        # Re-sync gravity matrices
        self.gravity._synchronize_field()

    def get_system_state(self):
        return {
            "dynamics": self.dynamics.get_topology(),
            "gravity": self.gravity.get_equilibrium_state()
        }

if __name__ == "__main__":
    sm = SelfMoldingCausalEngine(dimensions=3)
    sm.add_information("A", "Source", np.array([1,0,0], dtype=np.float32))
    sm.add_information("B", "Target", np.array([0.9,0.1,0], dtype=np.float32))

    print("Molding system...")
    for _ in range(5):
        sm.mold_topology(0.2)
        state = sm.get_system_state()
        dist = np.linalg.norm(np.array(state['dynamics']['voxels']['A']['pos']) -
                             np.array(state['dynamics']['voxels']['B']['pos']))
        print(f" Distance A-B: {dist:.4f}, Potential B: {state['dynamics']['voxels']['B']['potential']:.4f}")
