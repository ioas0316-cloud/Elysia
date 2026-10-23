import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from core.sensory.embodied_causal_substrate import EmbodiedCausalSubstrate, EngramSymbol

class EmbodiedSandboxSimulator:
    """
    Digital Twin Embodied Simulator.
    Simulates reciprocal (Inside-Out & Outside-In) sensory interaction, physical potential impacts
    (wall/obstacle boundary collapse), top-down intent filtering, and self-structuring data lens dynamics.
    """

    def __init__(self, grid_size: int = 10, vector_dim: int = 8):
        self.grid_size = grid_size
        self.vector_dim = vector_dim
        self.agent_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([float(grid_size - 1), float(grid_size - 1)])

        # Obstacle obstacle potential field (e.g. wall at x = 5)
        self.obstacle_x = 5.0

        # Embodied Sensory Substrate
        self.substrate = EmbodiedCausalSubstrate(vector_dim=vector_dim, energy_limit=200.0)

    def run_simulation_step(self, intent_direction: np.ndarray) -> Dict[str, Any]:
        """
        Runs one step of reciprocal interaction in the digital twin sandbox.
        - Inside-Out: Agent projects movement intent.
        - Outside-In: Environment exerts physical resistance (obstacle collision potential).
        """
        # 1. Inside-Out: Set top-down intent vector
        # Encode target direction in first 2 dimensions, fill remaining with intent signature
        intent_vector = np.zeros(self.vector_dim)
        intent_vector[:2] = intent_direction[:2]
        intent_vector[2:] = 1.0  # Intent chromatic/potential signature

        self.substrate.set_intent(intent_vector)

        # 2. Simulate prospective agent movement & Environmental Reaction (Outside-In)
        step_vector = intent_direction[:2]
        next_pos = self.agent_pos + step_vector

        # Environment obstacle boundary potential (Outside-In resistance)
        sensory_impact = np.zeros(self.vector_dim)
        sensory_impact[:2] = step_vector

        if abs(next_pos[0] - self.obstacle_x) < 0.5:
            # Hit wall: high repulsive friction force back on agent
            sensory_impact[0] = -step_vector[0] * 2.0  # Counter-force collision
            sensory_impact[3] = 5.0  # Friction potential peak
        else:
            # Normal movement, slight motion background noise
            sensory_impact[2:] = 1.0 + np.random.normal(0, 0.1, self.vector_dim - 2)
            self.agent_pos = np.clip(next_pos, 0, self.grid_size - 1)

        # 3. Process sensory impact through Structural Data Lens & Membrane
        lens_result = self.substrate.process_data_as_lens(
            data_stream=[sensory_impact],
            layer_name="C_meso"
        )

        dist_to_goal = float(np.linalg.norm(self.agent_pos - self.goal_pos))

        return {
            "agent_pos": self.agent_pos.copy(),
            "dist_to_goal": dist_to_goal,
            "friction": lens_result["total_friction"],
            "noise_reduction_ratio": lens_result["noise_reduction_ratio"],
            "invariant_type": lens_result["invariant_type"],
            "converged_engram": lens_result["converged_engram"],
            "remaining_energy": lens_result["remaining_energy"]
        }

    def run_full_trajectory(self, max_steps: int = 20) -> List[Dict[str, Any]]:
        """
        Runs a full trajectory simulation from start to goal.
        """
        history = []
        for _ in range(max_steps):
            # Compute direction towards goal
            dir_vec = self.goal_pos - self.agent_pos
            norm = np.linalg.norm(dir_vec)
            if norm < 0.5:
                break
            dir_vec = dir_vec / norm

            step_res = self.run_simulation_step(dir_vec)
            history.append(step_res)

            # If high friction hit wall, attempt bypass direction next step
            if step_res["friction"] > 2.0:
                bypass_vec = np.array([0.0, 1.0])  # Move vertically to bypass obstacle
                bypass_res = self.run_simulation_step(bypass_vec)
                history.append(bypass_res)

        return history

if __name__ == "__main__":
    sim = EmbodiedSandboxSimulator(grid_size=10)
    traj = sim.run_full_trajectory()
    print(f"Simulation completed with {len(traj)} steps. Final pos: {sim.agent_pos}")
