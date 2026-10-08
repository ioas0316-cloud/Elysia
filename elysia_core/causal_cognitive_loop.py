"""
Project Elysia: 4-Stage Causal Cognitive Loop Engine
====================================================
Implements a 4-Stage Cognitive Loop (Perception -> Discrimination/Judgment
-> Cognition/Reflection -> Action/Internalization) integrated with the 5D Phase
Space framework (R^5) and Stone Librande's One-Page Design Philosophy.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from elysia_core.unified_phase_space import (
    clip_vector,
    EnvironmentPotential,
    NPCEntity,
    ResourceEntity,
    CraftedItemEntity,
    MonsterEntity,
    ElysiaUnifiedEngine,
    AXIS_SHORT_NAMES
)

from core.sensory.emergent_sensory_apparatus import EmergentSensoryApparatus
from core.sensory.central_metacognitive_workspace import CentralMetacognitiveWorkspace
from core.causal_world.internal_simulation_engine import InternalSimulationEngine


class CausalCognitiveAgent(NPCEntity):
    """
    Living Cognitive Agent operating within the 4-Stage Cognitive Loop:
    1. Perception: Perceive macro Zeitgeist / environment potential shift V_Env(t).
    2. Discrimination & Judgment: Calculate cognitive dissonance tension T_diss = ||S - V_Env||.
       If T_diss exceeds threshold, trigger Quantum Phase Transition into a new potential well.
    3. Cognition & Reflection: Apply relaxation dynamics dS/dt = -mu * grad(V) with meta-learning.
    4. Action & Internalization: Manifest speech/actions and reverse-project impact onto macro era V_Era.
    """

    def __init__(
        self,
        name: str,
        base_S: np.ndarray,
        dissonance_threshold: float = 0.8,
        relaxation_rate: float = 0.2,
        reverse_projection_weight: float = 0.05
    ):
        super().__init__(name, base_S, relaxation_rate)
        self.dissonance_threshold = dissonance_threshold
        self.reverse_projection_weight = reverse_projection_weight
        self.cognitive_history: List[Dict[str, Any]] = []
        self.phase_transitions_count: int = 0
        self.current_archetype: str = self._determine_archetype()
        self.last_perceived_env: np.ndarray = np.copy(self.S)
        self.last_dissonance: float = 0.0
        self.last_phase_shifted: bool = False

        # Integrated Emergent Sensory Apparatus, CMW, and Internal Simulation Engine
        self.sensory_apparatus = EmergentSensoryApparatus(dim=16)
        self.cmw = CentralMetacognitiveWorkspace(dim=16)
        self.simulation_engine = InternalSimulationEngine(state_dim=2)

    def _determine_archetype(self) -> str:
        """Determines agent's cognitive archetype based on current 5D phase vector position."""
        will, flex, order, origin, field = self.S
        if order > 0.4 and origin > 0.3:
            return "수호자/맹세자 (Guardian/Legate)"
        elif will > 0.4 and flex > 0.3:
            return "혁명가/실리주의자 (Revolutionary/Pragmatist)"
        elif field > 0.5:
            return "선동가/선지자 (Prophet/Resonator)"
        elif flex < -0.4 and order < -0.4:
            return "방랑자/허무주의자 (Wanderer/Nihilist)"
        elif will < -0.4 and origin > 0.4:
            return "은둔자/자연학자 (Hermit/Ecology Monk)"
        else:
            return "조율자/중립자 (Harmonizer/Neutral Agent)"

    # ------------------------------------------------------------------
    # Stage 1: Perception (지각)
    # ------------------------------------------------------------------
    def perceive_environment(self, env_vec: np.ndarray) -> np.ndarray:
        """Stage 1: Receives macro environment potential V_Env(t) into perceptual space."""
        self.last_perceived_env = clip_vector(env_vec)
        return self.last_perceived_env

    # ------------------------------------------------------------------
    # Stage 2: Discrimination & Judgment (분별과 판단)
    # ------------------------------------------------------------------
    def judge_dissonance(self) -> Tuple[float, bool]:
        """
        Stage 2: Calculates cognitive dissonance tension T_diss = ||S - V_Env||.
        If T_diss > threshold, triggers Quantum Phase Transition to a new potential well.
        """
        diff = self.S - self.last_perceived_env
        self.last_dissonance = float(np.linalg.norm(diff))
        self.dissonance = self.last_dissonance

        phase_shifted = False
        if self.last_dissonance > self.dissonance_threshold:
            # Quantum Phase Transition (위상 전이)
            # Pull phase vector abruptly towards environmental vector along steepest gradient
            self.S = clip_vector(self.S + 0.6 * (self.last_perceived_env - self.S))
            self.phase_transitions_count += 1
            phase_shifted = True
            self.current_archetype = self._determine_archetype()

        self.last_phase_shifted = phase_shifted
        return self.last_dissonance, phase_shifted

    # ------------------------------------------------------------------
    # Stage 3: Cognition & Reflection (사고와 반성)
    # ------------------------------------------------------------------
    def reflect_and_adapt(self) -> np.ndarray:
        """
        Stage 3: Meta-cognitive relaxation dynamics dS/dt = -mu * grad(V).
        Relaxes internal belief system S toward perceived environment potential.
        """
        target = self.last_perceived_env
        gradient = self.S - target
        self.S = clip_vector(self.S - self.relaxation_rate * gradient)
        self.current_archetype = self._determine_archetype()
        return self.S

    # ------------------------------------------------------------------
    # Stage 4: Action & Internalizing Causality (행위와 인과 내재화)
    # ------------------------------------------------------------------
    def act_and_reverse_project(self) -> np.ndarray:
        """
        Stage 4: Manifests internal state externally and reverse-projects back onto macro V_Era.
        Returns delta_V_env impact vector generated by agent's agency.
        """
        # Delta shift feedback onto macro environment: delta_V = beta * (S - V_Env)
        delta_V = self.reverse_projection_weight * (self.S - self.last_perceived_env)

        # Log cognitive cycle step
        self.cognitive_history.append({
            "cycle": len(self.cognitive_history) + 1,
            "S": np.round(self.S, 2).tolist(),
            "dissonance": round(self.last_dissonance, 3),
            "phase_shifted": self.last_phase_shifted,
            "archetype": self.current_archetype,
            "delta_V_projected": np.round(delta_V, 3).tolist()
        })

        return delta_V

    def execute_cognitive_loop(self, env_vec: np.ndarray) -> np.ndarray:
        """Executes full 4-Stage Cognitive Loop sequentially for this agent."""
        self.perceive_environment(env_vec)
        self.judge_dissonance()

        # Integrated Sensory Apparatus, CMW, and Simulation Engine cycle
        sensory_out = self.sensory_apparatus.process_raw_stream(env_vec)
        sim_report = self.simulation_engine.tick(dt=0.05, external_wave=sensory_out["wave"])
        cmw_out = self.cmw.process_cognition(
            symbol_vec=self.S,
            sensory_wave=sensory_out["wave"],
            world_response=sensory_out["cpll_state"].q_axis_torque * np.ones(16, dtype=np.float32),
        )

        self.reflect_and_adapt()
        delta_V = self.act_and_reverse_project()
        return delta_V

    def get_cognitive_status(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update({
            "archetype": self.current_archetype,
            "phase_transitions": self.phase_transitions_count,
            "last_phase_shifted": self.last_phase_shifted,
            "history_length": len(self.cognitive_history)
        })
        return status


class CausalCognitiveEngine(ElysiaUnifiedEngine):
    """
    Integrated Engine managing 5D Phase Space and 4-Stage Cognitive Cycles
    for living cognitive agents, driving causal macro shifts and feedback loops.
    """

    def __init__(self, env_name: str = "신정정치 평화기", env_base_vec: Optional[np.ndarray] = None):
        super().__init__(env_name, env_base_vec)
        self.cognitive_agents: List[CausalCognitiveAgent] = []
        self.total_cycles: int = 0

    def add_cognitive_agent(self, agent: CausalCognitiveAgent) -> None:
        self.cognitive_agents.append(agent)
        # Also register as standard NPC for compatibility
        self.add_npc(agent)

    def step_cognitive_cycle(self, macro_event_delta: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Executes one complete 4-Stage Cognitive Loop across all registered agents and subsystems:
        1. Apply macro event shift if any
        2. Agents perceive environment V_Env
        3. Agents judge dissonance & undergo phase transitions if triggered
        4. Agents reflect and adapt
        5. Agents act and reverse-project impact back onto macro V_Env
        6. Subsystems cascade relaxation
        """
        self.total_cycles += 1

        # Direct external macro event shift
        if macro_event_delta is not None:
            self.env.shift_zeitgeist(macro_event_delta)

        # Process cognitive agents loop & collect reverse projections
        total_reverse_delta = np.zeros(5, dtype=float)
        for agent in self.cognitive_agents:
            delta_V = agent.execute_cognitive_loop(self.env.vector)
            total_reverse_delta += delta_V

        # Apply cumulative reverse projection from agents onto macro Zeitgeist V_Env
        if np.any(total_reverse_delta):
            self.env.shift_zeitgeist(total_reverse_delta)

        # Subsystems (Resources, Monsters, Standard NPCs) cascade adaptation
        self.step_cascade()

        return self.get_cognitive_snapshot()

    def get_cognitive_snapshot(self) -> Dict[str, Any]:
        snapshot = self.get_system_snapshot()
        snapshot["cognitive_agents"] = [a.get_cognitive_status() for a in self.cognitive_agents]
        snapshot["total_cycles"] = self.total_cycles
        return snapshot

    def render_cognitive_one_page_dashboard(self, title_suffix: str = "") -> str:
        """
        Renders a terminal One-Page Design Dashboard Report summarizing
        the 4-Stage Cognitive Loop dynamics and subsystem state.
        Inspired by Stone Librande's 'One Page Design Philosophy'.
        """
        snapshot = self.get_cognitive_snapshot()
        env_data = snapshot["environment"]
        env_vec = env_data["vector_list"]

        lines = []
        lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
        lines.append(f"│ ELYSIA :: CAUSAL COGNITIVE LOOP ONE-PAGE DASHBOARD {title_suffix:<20} │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│ [1. PERCEPTION / MACRO ZEITGEIST POTENTIAL FIELD (V_Env)]                   │")
        lines.append(f"│  Era/Region  : {env_data['name']:<59} │")
        lines.append(f"│  5D Vector   : X1:{env_vec[0]:+0.2f} | X2:{env_vec[1]:+0.2f} | X3:{env_vec[2]:+0.2f} | X4:{env_vec[3]:+0.2f} | X5:{env_vec[4]:+0.2f}     │")
        lines.append(f"│  Total Loop Cycles Completed : {snapshot['total_cycles']:<44} │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [2. COGNITIVE AGENTS (Perception ➔ Judgment ➔ Reflection ➔ Reverse Action)]  │")

        if snapshot["cognitive_agents"]:
            for a in snapshot["cognitive_agents"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in a["phase_vector"]])
                shift_str = "⚡ [PHASE TRANSITION]" if a["last_phase_shifted"] else "✓ [STABLE]"
                lines.append(f"│  • {a['name']:<18} | {a['archetype']:<28} | {shift_str:<20} │")
                lines.append(f"│    └ Dissonance: {a['dissonance']:0.3f} | Phase Vector: [{vec_str}] │")
                lines.append(f"│    └ Speech Pattern: {a['speech_pattern']:<55} │")
        else:
            lines.append("│  (No registered cognitive agents)                                            │")

        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [3. ECOLOGY & RESOURCE CASCADING RELAXATION]                                │")
        if snapshot["resources"]:
            for r in snapshot["resources"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in r["phase_vector"]])
                lines.append(f"│  • Resource: {r['description']:<30} | [{vec_str}] │")
        if snapshot["monsters"]:
            for m in snapshot["monsters"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in m["phase_vector"]])
                lines.append(f"│  • Monster : {m['title']:<30} | [{vec_str}] │")

        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [4. ITEM SUPERPOSITION & QUANTUM ENERGY SHELLS]                               │")
        if snapshot["crafted_items"]:
            for item in snapshot["crafted_items"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in item["phase_vector"]])
                lines.append(f"│  • {item['name']} ({item['tier']}) | Leap Cost: {item['quantum_leap_cost']:>6.1f}J | [{vec_str}] │")
        else:
            lines.append("│  (No registered items)                                                        │")

        lines.append("└──────────────────────────────────────────────────────────────────────────────┘")

        return "\n".join(lines)
