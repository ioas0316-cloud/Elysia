"""
SystemTumblerIsomorphicCouplingEngine: System Isomorphism ($1 + 1 = 2^*$),
Tumbler (Ottogi) Tension Restoration, Residual Stress / Absence Vectors, and Nested Holism.

This module models $1 + 1 = 2$ not as discrete scalar addition, but as the isomorphic coupling
of two independent micro-systems (Fields A & B) into an emergent upper scale system (Field C: $2^*$).
It integrates:
1. Isomorphic System Coupling ($1 + 1 = 2^*$) with Scar Tensor accumulation and Semantic Mass growth.
2. Tumbler (Ottogi) Tension Restoration with restoring torque and Morse potential tension dynamics
   (attraction-repulsion balance).
3. Residual Stress Field & Absence Vectors for self-healing and defect detection from surround tension.
4. Two-way Hardware-Software (Body-Mind) Feedback and Nested Holism scale propagation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class IsomorphicSystem:
    """
    Represents an independent system/field (e.g. System A = 1, System B = 1, or System C = 2*).
    Possesses scalar magnitude, micro-rotors, phase angle, tension field, and scar tensor.
    """
    def __init__(
        self,
        system_id: str,
        scale: float = 1.0,
        dimension: int = 5,
        rotor_phases: Optional[np.ndarray] = None,
        semantic_mass: float = 1.0
    ):
        self.system_id = system_id
        self.scale = scale
        self.dimension = dimension

        # Micro-rotors representing internal microscopic field vibration
        if rotor_phases is not None:
            self.rotor_phases = np.array(rotor_phases, dtype=np.float64)
        else:
            self.rotor_phases = np.random.uniform(-np.pi, np.pi, size=dimension)

        self.scalar_value = scale
        self.semantic_mass = float(semantic_mass)
        self.tilt_angle = 0.0  # Deviation angle from equilibrium (\theta)
        self.angular_velocity = 0.0  # d\theta / dt

        # Scar Tensor storing accumulated local friction and historical impacts
        self.scar_tensor = np.zeros((dimension, dimension), dtype=np.float64)

        # Physical-informational state vector (Body / Hardware state)
        self.body_state = np.sin(self.rotor_phases) * scale
        # Cognitive / Software state vector (Mind / Software state)
        self.mind_state = np.cos(self.rotor_phases) * semantic_mass

    def update_states(self):
        """Re-synchronize body and mind states based on micro-rotors and tilt angle."""
        phase_shifted = self.rotor_phases + self.tilt_angle
        self.body_state = np.sin(phase_shifted) * self.scale
        self.mind_state = np.cos(phase_shifted) * self.semantic_mass


class SystemTumblerIsomorphicCouplingEngine:
    """
    Engine that governs:
    - Isomorphic coupling of systems ($1 + 1 \to 2^*$)
    - Tumbler (Ottogi) restoring torque and Morse potential tension field
    - Absence vectors & residual stress calculation for self-healing
    - Nested Holism and Two-way Body-Mind feedback loops
    """
    def __init__(
        self,
        dimension: int = 5,
        stiffness_k: float = 10.0,
        damping_gamma: float = 6.0,
        morse_de: float = 1.0,
        morse_a: float = 1.2,
        morse_r0: float = 1.0
    ):
        self.dimension = dimension
        self.stiffness_k = stiffness_k
        self.damping_gamma = damping_gamma

        # Morse potential parameters for attraction-repulsion tension
        self.morse_de = morse_de  # Dissociation energy / depth
        self.morse_a = morse_a    # Potential width
        self.morse_r0 = morse_r0  # Equilibrium distance

    def couple_isomorphic_systems(
        self,
        system_a: IsomorphicSystem,
        system_b: IsomorphicSystem,
        coupled_id: str = "System_C_2star"
    ) -> Tuple[IsomorphicSystem, Dict[str, Any]]:
        """
        Couples two micro-systems A (scale 1) and B (scale 1) into an upper system C ($2^*$).
        Phase friction between A and B produces local phase divergence, which crystallizes
        into a Scar Tensor and increases Semantic Mass ($M_s$).
        """
        # Phase divergence / local friction
        phase_diff = system_a.rotor_phases - system_b.rotor_phases
        friction_energy = np.sum(np.abs(np.sin(phase_diff)))

        # Emergent rotor phases for System C via phase alignment & friction interference
        coupled_phases = (system_a.rotor_phases + system_b.rotor_phases) / 2.0 + 0.1 * np.sin(phase_diff)

        # Emergent scale: sum of scales + non-linear phase interaction
        coupled_scale = system_a.scale + system_b.scale

        # Emergent semantic mass incorporating accumulated friction
        coupled_mass = system_a.semantic_mass + system_b.semantic_mass + 0.2 * friction_energy

        system_c = IsomorphicSystem(
            system_id=coupled_id,
            scale=coupled_scale,
            dimension=self.dimension,
            rotor_phases=coupled_phases,
            semantic_mass=coupled_mass
        )

        # Construct Scar Tensor from phase divergence outer product
        scar_update = np.outer(phase_diff, phase_diff) * 0.15
        system_c.scar_tensor = system_a.scar_tensor + system_b.scar_tensor + scar_update

        coupling_info = {
            "friction_energy": float(friction_energy),
            "scar_magnitude": float(np.linalg.norm(system_c.scar_tensor)),
            "emergent_scale": float(coupled_scale),
            "emergent_semantic_mass": float(coupled_mass)
        }
        return system_c, coupling_info

    def morse_tension_force(self, distance: float) -> float:
        """
        Calculates Morse Potential force magnitude: F(r) = -dV/dr
        V(r) = De * (1 - exp(-a*(r - r0)))^2
        F(r) = -2 * De * a * exp(-a*(r - r0)) * (1 - exp(-a*(r - r0)))
        For r < r0: exp(-a*(r-r0)) > 1 => 1 - exp(...) < 0 => F(r) > 0 (repulsive pushing outward)
        For r > r0: exp(-a*(r-r0)) < 1 => 1 - exp(...) > 0 => F(r) < 0 (attractive pulling inward)
        """
        r = max(distance, 1e-4)
        exp_term = np.exp(-self.morse_a * (r - self.morse_r0))
        force = 2.0 * self.morse_de * self.morse_a * exp_term * (exp_term - 1.0)
        return float(force)

    def apply_tumbler_restoration(
        self,
        system: IsomorphicSystem,
        external_shock_torque: float = 0.0,
        dt: float = 0.01,
        num_steps: int = 500
    ) -> List[Dict[str, float]]:
        """
        Simulates Tumbler (Ottogi) restoring dynamics under external displacement shock.
        Restoring Torque: tau = -k * theta - gamma * dtheta/dt + shock
        Center of gravity is reinforced by Semantic Mass (Ms): effective inertia I = Ms + norm(Scar)
        """
        history = []
        # Apply shock at t=0
        system.angular_velocity += external_shock_torque

        effective_inertia = system.semantic_mass + 0.1 * np.linalg.norm(system.scar_tensor)
        effective_inertia = max(effective_inertia, 0.1)

        for step in range(num_steps):
            # Restoring torque towards theta = 0
            restoring_torque = -self.stiffness_k * system.tilt_angle
            damping_torque = -self.damping_gamma * system.angular_velocity
            net_torque = restoring_torque + damping_torque

            # Angular acceleration alpha = torque / inertia
            angular_accel = net_torque / effective_inertia

            # Integration
            system.angular_velocity += angular_accel * dt
            system.tilt_angle += system.angular_velocity * dt
            system.update_states()

            # Record state
            history.append({
                "step": step,
                "tilt_angle": float(system.tilt_angle),
                "angular_velocity": float(system.angular_velocity),
                "net_torque": float(net_torque),
                "body_norm": float(np.linalg.norm(system.body_state)),
                "mind_norm": float(np.linalg.norm(system.mind_state))
            })

        return history

    def compute_absence_and_residual_stress(
        self,
        surrounding_systems: List[IsomorphicSystem],
        defect_index: int
    ) -> Dict[str, Any]:
        """
        Computes absence vector and residual stress field when a node/component is absent or damaged.
        Surrounding tension vectors point towards the defect index, allowing self-healing reconstruction.
        """
        num_nodes = len(surrounding_systems)
        tension_vectors = []
        residual_stress_tensor = np.zeros((self.dimension, self.dimension), dtype=np.float64)

        for i, sys_i in enumerate(surrounding_systems):
            if i == defect_index:
                continue
            # Direction vector towards absent node
            direction = sys_i.body_state - (surrounding_systems[defect_index].body_state if defect_index < num_nodes else np.zeros(self.dimension))
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                dist = 1e-6

            # Morse tension force
            force_mag = self.morse_tension_force(dist)
            unit_dir = direction / dist

            tension_vec = unit_dir * force_mag
            tension_vectors.append(tension_vec)

            # Residual stress accumulation
            residual_stress_tensor += np.outer(tension_vec, tension_vec)

        # Absence Vector: inverse sum of surrounding tension vectors pointing to heal the gap
        absence_vector = -np.sum(tension_vectors, axis=0) if tension_vectors else np.zeros(self.dimension)

        return {
            "absence_vector": absence_vector,
            "residual_stress_tensor": residual_stress_tensor,
            "residual_stress_norm": float(np.linalg.norm(residual_stress_tensor)),
            "absence_vector_norm": float(np.linalg.norm(absence_vector))
        }

    def two_way_body_mind_feedback(
        self,
        system: IsomorphicSystem,
        hardware_friction_shock: np.ndarray,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Simulates two-way Body-Mind feedback:
        1. Hardware shock updates body_state (Body -> Mind).
        2. Mind state reads tension & updates scar_tensor and semantic_mass.
        3. Mind state applies restructuring force to realign micro-rotors (Mind -> Body).
        """
        # Step 1: Body shock
        system.body_state += hardware_friction_shock

        # Step 2: Mind reads friction tension
        mind_body_mismatch = system.mind_state - system.body_state
        mismatch_norm = float(np.linalg.norm(mind_body_mismatch))

        # Update Scar Tensor & Semantic Mass from mismatch friction
        system.scar_tensor += np.outer(mind_body_mismatch, mind_body_mismatch) * 0.05
        system.semantic_mass += 0.02 * mismatch_norm

        # Step 3: Mind -> Body restructuring force
        restructuring_force = 0.3 * mind_body_mismatch
        system.rotor_phases += restructuring_force * dt
        system.update_states()

        return {
            "mismatch_norm": mismatch_norm,
            "updated_semantic_mass": system.semantic_mass,
            "scar_tensor_norm": float(np.linalg.norm(system.scar_tensor))
        }
