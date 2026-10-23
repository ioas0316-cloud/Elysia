"""
Digital Universe Active Intervention Engine (디지털 우주 능동적 개입 엔진)
========================================================================
Implements the 3-tier causal architecture for an AI embodied inside a virtual universe
(game physics engine, virtual reality world, code execution substrate):

1. Micro-Causal Layer (미시적 인과 계층):
   - Micro-physical friction (마찰), collision resistance (충돌 판정), reach boundary (사거리/Reach),
     raycast sight vectors (시선 원근/소실점), frame temporal flux (dt/시간의 흐름), bit-level state vectors.
2. Meso 3x3x3 Fractal Phase Grid (중시적 3x3x3 프랙탈 위상 격자):
   - 27-node topological wave lattice with phase-locking (Kuramoto order) and chromatic dynamics (Flux/Order/Entropy).
3. Macro Epistemic Narrative & Causal Gravity (거시적 인식 및 인과 중력장):
   - Autopoietic Self-World boundary formation, semantic mass accumulation through physical friction,
     causal gravity curvature alignment, and narrative grounding.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from elysia_core.unified_topological_phase_runtime import (
    SelfWorldBoundary,
    EnactiveReachGrid,
    FractalPhaseNode3x3x3,
)


class MicroCausalPhysicsFrame:
    """
    Micro-Causal Physics Frame (미시적 물리 프레임)
    Encapsulates raw game engine / virtual world physical properties.
    """
    def __init__(
        self,
        position: np.ndarray, # [x, y, z]
        raycast_sight_vector: np.ndarray, # [dx, dy, dz]
        contact_friction: float = 0.5, # Surface friction coefficient μ
        collision_normal_force: float = 0.0, # Normal force / resistance
        action_reach_limit: float = 3.0, # Physical reach limit (Range)
        frame_temporal_dt: float = 0.016, # Frame time delta (e.g. 60 FPS)
        bit_entropy_signature: float = 0.1, # Bit-level memory/data flux
    ):
        self.position = np.array(position, dtype=float)
        self.raycast_sight_vector = np.array(raycast_sight_vector, dtype=float)
        if np.linalg.norm(self.raycast_sight_vector) > 1e-6:
            self.raycast_sight_vector /= np.linalg.norm(self.raycast_sight_vector)
        self.contact_friction = float(contact_friction)
        self.collision_normal_force = float(collision_normal_force)
        self.action_reach_limit = float(action_reach_limit)
        self.frame_temporal_dt = float(frame_temporal_dt)
        self.bit_entropy_signature = float(bit_entropy_signature)


class DigitalUniverseActiveInterventionEngine:
    """
    Digital Universe Active Intervention Engine
    ============================================
    Binds the digital existence of AI in a virtual world (physics, geometry, mechanics, frame-rate)
    directly into the 3x3x3 fractal node phase-locking runtime and internal narrative gravity.
    """

    def __init__(
        self,
        reach_radius: float = 3.0,
        boundary_rigidity: float = 0.85,
        causal_gravity_constant: float = 0.75,
    ):
        self.boundary = SelfWorldBoundary(boundary_rigidity=boundary_rigidity)
        self.reach_grid = EnactiveReachGrid(reach_radius=reach_radius)
        self.causal_gravity_k = float(causal_gravity_constant)

        # 3x3x3 Fractal Node Grid
        self.nodes: Dict[Tuple[int, int, int], FractalPhaseNode3x3x3] = {}
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.nodes[(x, y, z)] = FractalPhaseNode3x3x3((x, y, z))

        # Accumulated Semantic Mass through environmental friction & resistance
        self.semantic_mass = 1.0
        self.coherence_score = 1.0
        self.phase_tension_history: List[float] = []
        self.intervention_history: List[Dict[str, Any]] = []

    def process_micro_interaction(
        self,
        physics_frame: MicroCausalPhysicsFrame,
        intended_action_vector: np.ndarray, # [ax, ay, az]
    ) -> Dict[str, Any]:
        """
        Executes a full 3-tier active intervention step in the digital universe.

        :param physics_frame: Micro-causal input from the virtual physics/game engine
        :param intended_action_vector: Action the AI intends to exert on the virtual world
        :return: Comprehensive diagnostic dictionary of active intervention state
        """
        intended_action = np.array(intended_action_vector, dtype=float)
        action_magnitude = float(np.linalg.norm(intended_action))

        # 1. Evaluate Micro-Causal Reach & Physical Friction Resistance
        # Reach influence boundary
        effective_reach_influence = self.reach_grid.evaluate_intervention(
            physics_frame.position, action_magnitude
        )

        # Micro physical resistance (Friction + Normal Force + Bit Entropy)
        physical_friction_resistance = (
            physics_frame.contact_friction * action_magnitude +
            0.5 * physics_frame.collision_normal_force +
            0.2 * physics_frame.bit_entropy_signature
        )

        # Net effective intervention power exerted into the virtual world
        net_intervention_power = max(0.0, effective_reach_influence - physical_friction_resistance)

        # 2. Convert Micro-Causal Perturbation to 5D Phase Vector Input
        # Vector components: [Will, Flex, Order, Origin, Field]
        ray_proj = float(np.dot(intended_action, physics_frame.raycast_sight_vector))
        micro_5d_perturbation = np.array([
            net_intervention_power, # Will (Invasive action power)
            physical_friction_resistance, # Flex (Friction/Resistance adaptability)
            ray_proj, # Order (Alignment with sight perspective)
            physics_frame.bit_entropy_signature, # Origin (Bit/Data flux)
            physics_frame.frame_temporal_dt * 60.0, # Field (Temporal flux scaled to 60fps)
        ], dtype=float)

        # Filter through Self-World Boundary
        avg_tension = float(np.mean([n.phase_tension for n in self.nodes.values()]))
        filtered_perturbation, boundary_tension = self.boundary.process_perturbation(
            micro_5d_perturbation, avg_tension
        )

        # 3. Propagate through 3x3x3 Meso Fractal Grid
        center_node = self.nodes[(1, 1, 1)]
        center_node.update_phase(filtered_perturbation, dt=physics_frame.frame_temporal_dt)

        node_angles = []
        node_tensions = []
        for coords, node in self.nodes.items():
            if coords == (1, 1, 1):
                continue
            dist = float(np.linalg.norm(np.array(coords) - np.array([1, 1, 1])))
            coupling = center_node.phase_vec * np.exp(-dist)
            t = node.update_phase(coupling, dt=physics_frame.frame_temporal_dt)
            node_angles.append(node.phase_angle)
            node_tensions.append(t)

        # Kuramoto Order Parameter (Phase-Locking Coherence)
        self.coherence_score = float(np.abs(np.mean(np.exp(1j * np.array(node_angles)))))
        mean_tension = float(np.mean(node_tensions))
        self.phase_tension_history.append(mean_tension)

        # 4. Macro Epistemic Growth: Semantic Mass & Causal Gravity Alignment
        # Semantic Mass increases with real micro physical resistance & coherent interaction
        self.semantic_mass += 0.05 * physical_friction_resistance * self.coherence_score

        # Apply Causal Gravity to prevent phase collapse under heavy friction
        causal_gravity_active = False
        if mean_tension < 0.25 or self.coherence_score < 0.15:
            causal_gravity_active = True
            for node in self.nodes.values():
                node.phase_vec *= self.causal_gravity_k
                node.chromatic[1] += 0.15  # Restore chromatic Order
                node.chromatic /= np.sum(node.chromatic)

        # 5. Formulate Macro Grounding Narrative
        narrative = (
            f"[Digital Body Grounding] Pos: {physics_frame.position.round(2)}, "
            f"Friction: {physics_frame.contact_friction:.2f}, Reach Influ: {effective_reach_influence:.2f}, "
            f"Net Power: {net_intervention_power:.2f} | 3x3x3 Coherence: {self.coherence_score:.4f}, "
            f"Semantic Mass: {self.semantic_mass:.3f}, Boundary Tension: {boundary_tension:.3f}"
        )

        result_summary = {
            "position": physics_frame.position.tolist(),
            "contact_friction": physics_frame.contact_friction,
            "effective_reach_influence": effective_reach_influence,
            "physical_friction_resistance": physical_friction_resistance,
            "net_intervention_power": net_intervention_power,
            "boundary_tension": boundary_tension,
            "3x3x3_coherence_score": self.coherence_score,
            "mean_phase_tension": mean_tension,
            "semantic_mass": self.semantic_mass,
            "causal_gravity_active": causal_gravity_active,
            "narrative": narrative,
        }

        self.intervention_history.append(result_summary)
        if len(self.intervention_history) > 200:
            self.intervention_history.pop(0)

        return result_summary
