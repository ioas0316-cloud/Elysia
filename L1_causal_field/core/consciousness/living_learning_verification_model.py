"""
Minimal Verification Model: Living Learning & Structural Self-Transformation System
========================================================================

Implements continuous phase/vector dynamics for living learning.
Instead of discrete string checks, Elysia interacts with a continuous world tensor space:
1. Dynamic Projection Matrix P (Distinction Criteria)
2. Directional Value Vector V_pref (0_value / Judgment Ground)
3. Relational Coupling Matrix J (Relational Topology)
4. Vector Action Generator A_space (Expanded Action & Observation Possibilities)
"""

import numpy as np
from typing import Dict, Any, List


class LivingWorldEnvironment:
    """
    A continuous reactive world field.
    State, actions, and feedback are represented by physical/informational vectors:
    - Action vector A in R^4
    - World resistance W_res in R^4
    - Hidden harmonic phase angle theta_world
    """
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.world_state = np.array([0.5, 0.8, 0.3, 0.1])
        self.harmonic_axis = np.array([0.0, 0.0, 1.0, 0.5])
        self.harmonic_axis = self.harmonic_axis / np.linalg.norm(self.harmonic_axis)
        self.rigidity = 0.8
        self.interaction_count = 0

    def interact(self, action_vec: np.ndarray) -> Dict[str, float]:
        """
        Execute continuous action vector A and return physical feedback metrics:
        - displacement: physical motion achieved
        - friction: energy lost due to structural strain/mismatch
        - phase_alignment: resonance degree with world harmonic axis
        """
        self.interaction_count += 1
        action_norm = np.linalg.norm(action_vec)
        if action_norm < 1e-6:
            action_dir = np.zeros_like(action_vec)
        else:
            action_dir = action_vec / action_norm

        # Calculate alignment with world harmonic axis
        alignment = float(np.dot(action_dir, self.harmonic_axis))

        # Friction is high if force is large but alignment with harmonic axis is low
        friction = float(action_norm * (1.0 - max(0.0, alignment)) * self.rigidity)

        # Displacement depends on alignment and rigidity
        effective_force = action_norm * (0.2 + 0.8 * max(0.0, alignment))
        displacement = float(effective_force / (self.rigidity + 0.1))

        # World rigidity alters dynamically when resonance is achieved (low friction, high alignment)
        if alignment > 0.85 and action_norm > 0.3:
            self.rigidity = max(0.1, self.rigidity - 0.15)

        return {
            "displacement": displacement,
            "friction": friction,
            "alignment": alignment,
            "world_rigidity": self.rigidity
        }


class LivingElysiaAgent:
    """
    Elysia agent whose internal vector/matrix structures transform through interaction.

    Transforms:
    1. Distinction Criteria: Dynamic Projection Matrix P (k x d) expand when high friction gradient is detected.
    2. Judgment Ground (0_value): Continuous preference vector V_pref in R^d shifts toward minimal friction / maximum resonance.
    3. Relational Topology: Coupling matrix J (d x d) rewired via Hebbian/causal phase updates.
    4. Action & Observation Space: Action basis vectors unlocked as new distinction dimensions emerge.
    """
    def __init__(self, dim: int = 4):
        self.dim = dim

        # 1. Distinction Criteria: Projection Matrix P (k x d). Initially perceives 2 dimensions.
        self.projection_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # 2. Judgment Ground (0_value): Vector V_pref in R^d. Initially values raw force along dim 0.
        self.value_pref = np.array([1.0, 0.0, 0.0, 0.0])
        self.value_pref = self.value_pref / np.linalg.norm(self.value_pref)
        self.target_name = "maximize_raw_force"

        # 3. Relational Topology: Coupling Matrix J (d x d). Maps action to expected displacement/friction.
        self.relational_J = np.eye(dim) * 0.5

        # 4. Action Basis Space (List of basis vectors)
        self.action_bases = [
            np.array([0.2, 0.0, 0.0, 0.0]),  # Light push
            np.array([1.2, 0.1, 0.0, 0.0])   # Hard pressure
        ]

        self.history_reflections: List[str] = []

    def receive_external_knowledge(self, assertion: str) -> None:
        self.history_reflections.append(f"Received external assertion: '{assertion}'")

    def test_and_roll_through_practice(self, world: LivingWorldEnvironment, action_idx: int) -> Dict[str, Any]:
        """
        '돌려보는 연습' (Testing & Rolling through practice):
        1. Act using chosen action basis vector.
        2. Receive continuous feedback metrics (displacement, friction, alignment).
        3. Dynamically update Distinction Matrix, Preference Vector (0_value), Coupling Matrix J, and Action Bases.
        """
        action_vec = self.action_bases[action_idx]
        feedback = world.interact(action_vec)

        friction = feedback["friction"]
        displacement = feedback["displacement"]
        alignment = feedback["alignment"]

        difference_detected = False

        # 1. Distinction Criteria Expansion: If friction > 0.3 and projection dimension < 4
        if friction > 0.3 and self.projection_matrix.shape[0] < 4:
            k = self.projection_matrix.shape[0]
            new_row = np.zeros((1, self.dim))
            new_row[0, k] = 1.0
            self.projection_matrix = np.vstack([self.projection_matrix, new_row])
            difference_detected = True
            self.history_reflections.append(f"DISCOVERY: Expanded Distinction Matrix P to rank {self.projection_matrix.shape[0]}!")

        # 2. Shift Judgment Ground (0_value): If high friction is experienced, shift V_pref away from raw force toward harmonic resonance axis
        if friction > 0.4 and self.target_name == "maximize_raw_force":
            self.target_name = "harmonic_resonance_flow"
            self.value_pref = np.array([0.0, 0.0, 1.0, 0.5])
            self.value_pref = self.value_pref / np.linalg.norm(self.value_pref)
            difference_detected = True
            self.history_reflections.append("VALUE SHIFT: Judgment ground (0_value) shifted from 'Maximize Raw Force' to 'Harmonic Resonance Flow'!")

        # 3. Update Relational Coupling Matrix J
        delta_J = np.outer(action_vec, action_vec) * (displacement - friction)
        self.relational_J += 0.1 * delta_J

        # 4. Action & Observation Possibilities Expansion
        if alignment > 0.8 and len(self.action_bases) < 5:
            # Generate new action basis aligned with harmonic preference
            new_action_1 = np.array([0.0, 0.0, 0.8, 0.4])  # Resonant Pulse
            new_action_2 = np.array([0.0, 0.1, 0.0, 0.9])  # Phase Realignment
            self.action_bases.append(new_action_1)
            self.action_bases.append(new_action_2)
            self.history_reflections.append("ACTION EXPANSION: Discovered new Resonant Pulse and Phase Realignment action bases!")

            if feedback["world_rigidity"] < 0.5 and len(self.action_bases) < 5:
                new_action_3 = np.array([0.5, 0.5, 0.5, 0.5]) # Co-created Flow State
                self.action_bases.append(new_action_3)
                self.history_reflections.append("ACTION EXPANSION: Discovered Co-created Flow State action basis!")

        return {
            "action_taken_idx": action_idx,
            "feedback": feedback,
            "difference_detected": difference_detected,
            "distinction_rank": self.projection_matrix.shape[0],
            "value_pref": self.value_pref.tolist(),
            "target_name": self.target_name,
            "num_action_bases": len(self.action_bases)
        }

    def revise_faulty_judgment_if_needed(self, realignment_signal: float) -> bool:
        """
        Reversibility / Self-Correction check:
        If realignment signal indicates a needs for homeostatic balance,
        Elysia adjusts V_pref to a balanced superposition.
        """
        if realignment_signal > 0.5 and self.target_name == "harmonic_resonance_flow":
            self.target_name = "dynamic_homeostatic_coexistence"
            self.value_pref = np.array([0.25, 0.25, 0.25, 0.25])
            self.history_reflections.append("REVERSIBILITY: Revised judgment criteria (0_value) to 'Dynamic Homeostatic Coexistence'!")
            return True
        return False
