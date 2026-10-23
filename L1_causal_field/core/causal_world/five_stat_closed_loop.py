"""
Elysia Core - 5-Stat Environmental Principle & Closed-Loop Engine
========================================================================
Maps unstructured behavioral/text extraction parameters and macro rotor field
dynamics into 5-stat thermodynamic/topological environmental principles,
maintaining a closed-loop causal feedback between Macro Environment and Micro NPCs.

5-Stats as Environmental Principles:
1. Energy Consumption Rate (S_E): Physical work, fatigue generation, and metabolic decay.
2. Information Bandwidth (S_I): Cognitive processing capacity and sensory throughput.
3. Friction Resistance (S_F): Environmental impedance, social viscosity, and obstacle drag.
4. Adaptation Speed (S_A): Rate of state trajectory shift towards environmental geodesics.
5. Equilibrium Stability (S_S): Homeostatic resilience and attractor field depth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class FiveStatVector:
    """
    Representation of 5 environmental principle state variables for an entity or region.
    All stats are normalized in [0.0, 1.0] continuous phase space.
    """
    energy_consumption: float = 0.5  # S_E
    info_bandwidth: float = 0.5      # S_I
    friction_resistance: float = 0.5 # S_F
    adaptation_speed: float = 0.5    # S_A
    equilibrium_stability: float = 0.5 # S_S

    def to_array(self) -> np.ndarray:
        return np.array([
            self.energy_consumption,
            self.info_bandwidth,
            self.friction_resistance,
            self.adaptation_speed,
            self.equilibrium_stability
        ], dtype=np.float64)

    def update_from_array(self, arr: np.ndarray):
        arr_clipped = np.clip(arr, 0.0, 1.0)
        self.energy_consumption = float(arr_clipped[0])
        self.info_bandwidth = float(arr_clipped[1])
        self.friction_resistance = float(arr_clipped[2])
        self.adaptation_speed = float(arr_clipped[3])
        self.equilibrium_stability = float(arr_clipped[4])


class FiveStatClosedLoopEngine:
    """
    Closed-loop engine linking Macro-scale Spacetime Rotors & Riemannian Metrics
    with Micro-scale NPC 5-Stat state trajectories.
    """

    def __init__(self, initial_environment_5stat: Optional[FiveStatVector] = None):
        self.env_stat = initial_environment_5stat or FiveStatVector(
            energy_consumption=0.3,
            info_bandwidth=0.8,
            friction_resistance=0.2,
            adaptation_speed=0.6,
            equilibrium_stability=0.7
        )
        self.macro_rotor_energy = 1.0
        self.macro_metric_curvature = 0.0

    def map_extraction_to_5stat(self, extraction_result: Dict[str, Any]) -> FiveStatVector:
        """
        Maps extracted topological parameters (ricci_scalar, attractor_potential,
        rotor magnitudes) into a 5-stat representation.
        """
        ricci = abs(extraction_result.get("ricci_scalar", 0.0))
        attractor = extraction_result.get("attractor_potential", 0.5)
        rotation_mag = extraction_result.get("mean_spatial_rotation", 0.0)
        boost_mag = extraction_result.get("mean_temporal_boost", 0.0)

        # Compute 5-stats from manifold dynamics:
        s_e = min(1.0, 0.2 + boost_mag * 0.5)               # Higher boost -> higher energy consumption
        s_i = min(1.0, 0.3 + attractor * 0.6)               # Deeper attractor -> higher info bandwidth
        s_f = min(1.0, 0.1 + ricci * 0.4)                   # Higher curvature -> higher friction
        s_a = min(1.0, 0.2 + rotation_mag * 0.5)            # Higher rotor spin -> faster adaptation
        s_s = min(1.0, 0.4 + (1.0 - ricci) * 0.3 + attractor * 0.3) # Stability from low curvature & deep attractor

        return FiveStatVector(
            energy_consumption=s_e,
            info_bandwidth=s_i,
            friction_resistance=s_f,
            adaptation_speed=s_a,
            equilibrium_stability=s_s
        )

    def step_closed_loop(
        self,
        npc_stat: FiveStatVector,
        extraction_data: Optional[Dict[str, Any]] = None,
        action_intent: str = "walk"
    ) -> Tuple[FiveStatVector, Dict[str, Any]]:
        """
        Executes one step in the macro-micro closed-loop cycle.
        1. Macro Environment exerts rotor dynamics onto NPC 5-Stats.
        2. NPC calculates equilibrium seeking trajectory (e.g., rest/restoration if fatigued).
        3. Micro Action generates feedback onto Macro Environment (modifying friction, curvature, rotor energy).
        """
        if extraction_data:
            extracted_env = self.map_extraction_to_5stat(extraction_data)
            # Update Macro environment via exponential smoothing
            env_arr = 0.7 * self.env_stat.to_array() + 0.3 * extracted_env.to_array()
            self.env_stat.update_from_array(env_arr)

        npc_arr = npc_stat.to_array()
        env_arr = self.env_stat.to_array()

        # Environmental influence: NPC stats shift toward environmental baseline
        coupling_strength = 0.15
        npc_arr += coupling_strength * (env_arr - npc_arr)

        # Micro Action Execution & Fatigue Dynamics
        state_delta = np.zeros(5, dtype=np.float64)
        recorded_action = action_intent

        # If fatigue/energy consumption is too high or equilibrium stability is too low,
        # NPC automatically seeks homeostatic recovery ("rest/sleep")
        if npc_stat.energy_consumption > 0.8 or npc_stat.equilibrium_stability < 0.3:
            recorded_action = "rest_and_recover"
            state_delta[0] -= 0.25  # Lower energy consumption / fatigue
            state_delta[4] += 0.20  # Restore equilibrium stability
            state_delta[2] -= 0.10  # Reduce internal friction
        elif action_intent == "work_or_exert":
            recorded_action = "work_or_exert"
            state_delta[0] += 0.20  # Energy consumption increases
            state_delta[1] += 0.10  # Info bandwidth active
            state_delta[2] += 0.15  # Friction accumulated
        else:  # default walk or adapt
            recorded_action = action_intent
            state_delta[0] += 0.05
            state_delta[3] += 0.05

        # Apply action delta
        npc_arr += state_delta
        updated_npc = FiveStatVector()
        updated_npc.update_from_array(npc_arr)

        # Closed-Loop Macro Feedback: NPC state feeds back into Macro Environmental Rotor Field
        # Macro friction adjusts to collective NPC energy consumption & stability
        macro_feedback_delta = np.zeros(5, dtype=np.float64)
        macro_feedback_delta[2] = (updated_npc.energy_consumption - 0.5) * 0.05  # High fatigue increases macro friction
        macro_feedback_delta[4] = (updated_npc.equilibrium_stability - 0.5) * 0.05 # NPC stability reinforces macro equilibrium

        env_arr += macro_feedback_delta
        self.env_stat.update_from_array(env_arr)

        feedback_summary = {
            "executed_action": recorded_action,
            "macro_friction": self.env_stat.friction_resistance,
            "macro_stability": self.env_stat.equilibrium_stability,
            "npc_energy_consumption": updated_npc.energy_consumption,
            "npc_equilibrium_stability": updated_npc.equilibrium_stability,
            "closed_loop_resonance": float(np.dot(updated_npc.to_array(), self.env_stat.to_array()))
        }

        return updated_npc, feedback_summary
