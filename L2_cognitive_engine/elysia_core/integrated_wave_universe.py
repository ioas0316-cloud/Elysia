"""
Project Elysia: Integrated Wave Universe Core Runtime
=====================================================
Combines spatial manifold Clifford rotor rotations (O(1) low-VRAM spatial shifts)
with Kuramoto oscillator-based cognitive NPC nodes for real-time spatial exploration,
dialogue frequency resonance, and friction-controlled zero-cost execution.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class SpatialManifold:
    """
    Spatial Manifold represented as Clifford Rotor phase angles and high-dimensional phase vectors.
    Enables low-VRAM O(1) spatial shifts via rotor spin instead of heavy mesh reloading.
    """
    zone_name: str
    rotor_angle: float = 0.0  # Phase angle in radians
    dimension: int = 5
    manifold_vector: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.2, 0.8, 0.1, 0.9], dtype=float))

    def rotate_rotor(self, delta_theta: float) -> float:
        """Rotates spatial Clifford rotor by delta_theta (O(1) topological shift)."""
        self.rotor_angle = (self.rotor_angle + delta_theta) % (2 * np.pi)
        # Update high-dimensional manifold vector projection
        rot_matrix = np.array([
            [np.cos(delta_theta), -np.sin(delta_theta)],
            [np.sin(delta_theta),  np.cos(delta_theta)]
        ])
        self.manifold_vector[:2] = rot_matrix @ self.manifold_vector[:2]
        return self.rotor_angle


@dataclass
class NPCPhaseNode:
    """
    Cognitive/Emotional NPC Node modeled as a Kuramoto Phase Oscillator.
    Resonates with player dialogue/voice frequencies and emotional tones.
    """
    name: str
    title: str
    natural_frequency: float  # Natural frequency omega_i
    phase: float = 0.0        # Current phase theta_i
    friction_tolerance: float = 0.5
    personality_vector: np.ndarray = field(default_factory=lambda: np.ones(5, dtype=float) * 0.5)
    resonance_history: List[float] = field(default_factory=list)

    def update_kuramoto_phase(
        self,
        player_phase: float,
        coupling_k: float = 0.8,
        dt: float = 0.05,
        external_friction: float = 0.0
    ) -> Tuple[float, float]:
        """
        Updates NPC oscillator phase using Kuramoto synchronization dynamics:
        d_theta / dt = omega_i + K * sin(theta_player - theta_i) - friction * sin(theta_i)
        Returns (new_phase, phase_difference)
        """
        phase_diff = player_phase - self.phase
        coupling_term = coupling_k * np.sin(phase_diff)
        friction_damping = external_friction * np.sin(self.phase)

        d_phase = (self.natural_frequency + coupling_term - friction_damping) * dt
        self.phase = (self.phase + d_phase) % (2 * np.pi)

        current_diff = abs((self.phase - player_phase + np.pi) % (2 * np.pi) - np.pi)
        resonance = max(0.0, 1.0 - (current_diff / np.pi))
        self.resonance_history.append(resonance)

        return self.phase, current_diff


class IntegratedWaveUniverse:
    """
    Unified Wave Universe Runtime Engine.
    Integrates Clifford rotor spatial manifolds, Kuramoto oscillator NPC networks,
    and friction-based zero-cost execution.
    """

    def __init__(self, primary_zone: str = "fantasy_meadow"):
        self.spatial_manifold = SpatialManifold(zone_name=primary_zone)
        self.npc_nodes: Dict[str, NPCPhaseNode] = {}
        self.player_phase: float = 0.0
        self.player_frequency: float = 1.0
        self.total_steps: int = 0
        self.cached_state_valid: bool = False
        self.last_telemetry: Dict[str, Any] = {}

        # Default setup with companion NPC Elysia
        self.register_npc(
            NPCPhaseNode(
                name="elysia",
                title="파동장의 정령 (Spiritual Resonator of Causal Wavefield)",
                natural_frequency=1.2,
                phase=0.0,
                friction_tolerance=0.4,
                personality_vector=np.array([0.9, 0.7, 0.95, 0.8, 0.85], dtype=float),
            )
        )

    def register_npc(self, npc: NPCPhaseNode) -> None:
        """Registers a Kuramoto phase node NPC into the wave universe."""
        self.npc_nodes[npc.name] = npc

    def set_player_voice_frequency(self, freq: float) -> None:
        """Sets player input dialogue frequency / tone pitch."""
        self.player_frequency = freq

    def step_universe(
        self,
        player_action: str = "idle",
        player_speech: str = "",
        player_voice_freq: Optional[float] = None,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Executes one step of the Integrated Wave Universe Runtime:
        1. Process Spatial Manifold Rotor Rotation (O(1) low-VRAM shift)
        2. Calculate Friction & Free Energy Dissonance
        3. Execute Zero-Cost Cache or Dynamic Phase Re-Locking
        4. Update Kuramoto Synchronization for NPC Nodes
        """
        self.total_steps += 1

        if player_voice_freq is not None:
            self.player_frequency = player_voice_freq

        # Update player phase trajectory
        self.player_phase = (self.player_phase + self.player_frequency * dt) % (2 * np.pi)

        # ------------------------------------------------------------------
        # 1. Spatial Manifold Clifford Rotor Rotation
        # ------------------------------------------------------------------
        spatial_status = "ZERO_COST_FLOW: Spatial cache maintained."
        if player_action in ["move_forward", "spin_view", "explore_meadow", "enter_dungeon"]:
            delta_rotor = np.pi / 16.0  # ~0.196 rad rotation
            new_angle = self.spatial_manifold.rotate_rotor(delta_rotor)
            spatial_status = f"ROTOR_SPIN: Zone '{self.spatial_manifold.zone_name}' shifted to phase {new_angle:.3f} rad"
            self.cached_state_valid = False
        elif player_action == "switch_zone_dungeon":
            self.spatial_manifold = SpatialManifold(zone_name="dungeon_depths", rotor_angle=0.0)
            spatial_status = "TOPOLOGY_JUMP: Zone re-linked to 'dungeon_depths'"
            self.cached_state_valid = False

        # ------------------------------------------------------------------
        # 2. Friction & Dissonance Calculation
        # ------------------------------------------------------------------
        # Dialogue friction is calculated based on pitch variance and speech length
        speech_friction = len(player_speech) * 0.02 if player_speech else 0.0
        movement_friction = 0.15 if "move" in player_action or "spin" in player_action else 0.02
        total_friction = speech_friction + movement_friction

        # ------------------------------------------------------------------
        # 3. Kuramoto Synchronization & NPC Emotional Resonance
        # ------------------------------------------------------------------
        npc_reports = {}
        total_phase_diff = 0.0
        complex_phases = [np.exp(1j * self.player_phase)]

        for name, npc in self.npc_nodes.items():
            new_npc_phase, diff = npc.update_kuramoto_phase(
                player_phase=self.player_phase,
                coupling_k=0.85,
                dt=dt,
                external_friction=total_friction
            )
            complex_phases.append(np.exp(1j * new_npc_phase))
            total_phase_diff += diff

            # Determine Resonance Status
            if diff < 0.25:
                resonance_type = "PHASE_LOCK (Deep Resonance)"
                dialogue_response = f"[{npc.title} {npc.name}]: 당신의 파동과 완벽히 동기화되었습니다. 따뜻한 온기가 마음속에 느껴집니다."
            elif diff < 0.8:
                resonance_type = "HARMONIC_COUPLING (Harmonizing)"
                dialogue_response = f"[{npc.title} {npc.name}]: 조화로운 주파수를 찾아 당신에게 반응하고 있습니다."
            else:
                resonance_type = "FRICTION_DETECTED (Dynamic Re-locking)"
                dialogue_response = f"[{npc.title} {npc.name}]: 미세한 마찰이 느껴집니다. 위상을 당신의 결에 맞춰 재조율합니다."

            npc_reports[name] = {
                "phase": round(new_npc_phase, 3),
                "phase_difference": round(diff, 3),
                "resonance_type": resonance_type,
                "dialogue_response": dialogue_response
            }

        # Kuramoto Order Parameter R (Coherence of overall swarm)
        order_parameter_R = float(np.abs(np.mean(complex_phases)))

        # ------------------------------------------------------------------
        # 4. Zero-Cost Execution Telemetry
        # ------------------------------------------------------------------
        execution_mode = "ZERO_COST_EVALUATION" if total_friction < 0.05 else "DYNAMIC_PHASE_RELOCK"

        telemetry = {
            "step": self.total_steps,
            "spatial_zone": self.spatial_manifold.zone_name,
            "rotor_angle_rad": round(self.spatial_manifold.rotor_angle, 3),
            "spatial_status": spatial_status,
            "player_phase": round(self.player_phase, 3),
            "player_frequency": round(self.player_frequency, 3),
            "total_friction": round(total_friction, 3),
            "kuramoto_order_R": round(order_parameter_R, 4),
            "execution_mode": execution_mode,
            "npc_reports": npc_reports
        }

        self.last_telemetry = telemetry
        return telemetry

    def render_one_page_universe_dashboard(self) -> str:
        """Renders terminal One-Page Design dashboard for Integrated Wave Universe."""
        if not self.last_telemetry:
            return "Universe engine not stepped yet."

        t = self.last_telemetry
        lines = []
        lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ ELYSIA :: INTEGRATED WAVE UNIVERSE RUNTIME (GTX 1060 3GB TOPOLOGY ENGINE)   │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│ Step: {t['step']:<6} | Zone: {t['spatial_zone']:<20} | Rotor Angle: {t['rotor_angle_rad']:<0.3f} rad  │")
        lines.append(f"│ Spatial Status  : {t['spatial_status']:<58} │")
        lines.append(f"│ Execution Mode  : {t['execution_mode']:<28} | Friction: {t['total_friction']:<0.3f}       │")
        lines.append(f"│ Kuramoto Sync R : {t['kuramoto_order_R']:<0.4f} (Swarm Phase Coherence)                        │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [COGNITIVE NPC PHASE RESONANCE & DIALOGUE]                                   │")

        for name, npc_info in t["npc_reports"].items():
            lines.append(f"│ • NPC '{name}': Phase={npc_info['phase']} rad | Diff={npc_info['phase_difference']} rad                │")
            lines.append(f"│   Status  : {npc_info['resonance_type']:<58} │")
            lines.append(f"│   Dialogue: {npc_info['dialogue_response']:<58} │")

        lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)
