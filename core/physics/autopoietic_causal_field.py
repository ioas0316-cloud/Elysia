"""
Autopoietic Causal Field (자기생성적 인과장)
==============================================
Combines topology dynamics with Autopoiesis (self-creation and self-preservation drive).
Converts phase friction into existential topological entropy H_topo.

Key Mechanisms:
1. Global Topological Entropy (H_topo = sum(Z_uv + F_uv)): Measures systemic threat of dissolution.
2. Dual Topological Structure (Core Kernel vs Peripheral Shell): Protects identity Kernel by selectively sacrificing/refracting Shell edges during high entropy.
3. Active Wave Modulation Drive: Spontaneously fluctuates frequency omega and phase angle phi under friction to discover new phase-locking resonance trajectories.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx
import math

from core.lens.enactive_boundary_layer import EnactiveBoundaryLayer, FrictionEvaluation, WaveSignal
from core.lens.cognitive_lens_engine import CognitiveLensEngine, ContextualDimension


class NodeRole:
    CORE_KERNEL = "core_kernel"
    PERIPHERAL_SHELL = "peripheral_shell"


class AutopoieticCausalField:
    """
    Autopoietic Causal Field engine managing systemic topological entropy,
    Kernel protection, Shell edge dissolution, and Active Wave Modulation.
    """

    def __init__(
        self,
        ebl: Optional[EnactiveBoundaryLayer] = None,
        entropy_threshold: float = 2.5,
        fluctuation_scale: float = 0.2
    ):
        self.ebl = ebl or EnactiveBoundaryLayer()
        self.entropy_threshold = entropy_threshold
        self.fluctuation_scale = fluctuation_scale

    def add_autopoietic_node(
        self,
        name: str,
        frequency: float,
        phase: float,
        role: str = NodeRole.PERIPHERAL_SHELL,
        dimension: Optional[ContextualDimension] = None
    ):
        """Registers node with specific role (CORE_KERNEL or PERIPHERAL_SHELL)."""
        self.ebl.add_causal_node(name, frequency=frequency, phase=phase, dimension=dimension)
        self.ebl.graph.nodes[name]["role"] = role
        self.ebl.graph.nodes[name]["base_frequency"] = frequency
        self.ebl.graph.nodes[name]["base_phase"] = phase

    def add_autopoietic_edge(self, source: str, target: str, initial_impedance: float = 0.1):
        """Registers edge between nodes."""
        self.ebl.add_causal_edge(source, target, initial_impedance=initial_impedance)

    def calculate_global_topological_entropy(self) -> float:
        """
        Calculates Global Topological Entropy H_topo = sum(Z_uv + F_uv).
        Represents systemic threat to identity and causal structure.
        """
        total_entropy = 0.0
        for u, v, data in self.ebl.graph.edges(data=True):
            z = data.get("impedance", 0.1)
            f = data.get("last_friction", 0.0)
            total_entropy += (z + f)
        return float(total_entropy)

    def enact_autopoietic_step(
        self,
        source_node: str,
        external_frequency: float,
        external_phase: float,
        target_node: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes step through Enactive Boundary Layer and applies Autopoietic dynamics:
        1. Step execution via EBL.
        2. Store friction on edge.
        3. Compute H_topo.
        4. If H_topo > threshold: Protect Kernel, sever/refract Shell edge.
        5. Active Wave Modulation if persistent friction.
        """
        res = self.ebl.enact_step(source_node, external_frequency, external_phase, target_node)

        target = res.get("target_node")
        if target and self.ebl.graph.has_edge(source_node, target):
            self.ebl.graph.edges[source_node, target]["last_friction"] = res["friction_factor"]

        h_topo = self.calculate_global_topological_entropy()

        shell_dissolved = False
        active_modulation_applied = False

        source_role = self.ebl.graph.nodes[source_node].get("role", NodeRole.PERIPHERAL_SHELL)

        # Autopoietic Self-Preservation Drive: Protect Core Kernel
        if h_topo > self.entropy_threshold:
            if target and self.ebl.graph.has_edge(source_node, target):
                target_role = self.ebl.graph.nodes[target].get("role", NodeRole.PERIPHERAL_SHELL)

                # If edge is in Peripheral Shell or connects Kernel to Shell, sever Shell edge to protect Kernel
                if source_role == NodeRole.PERIPHERAL_SHELL or target_role == NodeRole.PERIPHERAL_SHELL:
                    self.ebl.graph.edges[source_node, target]["impedance"] = 1.0  # Max impedance / severed
                    shell_dissolved = True

        # Active Wave Modulation: Spontaneous fluctuation to search for new resonance alignment
        if res["friction_factor"] > self.ebl.threshold:
            # Active Modulation Drive: perturb frequency and phase of source node
            current_freq = self.ebl.graph.nodes[source_node]["freq"]
            current_phase = self.ebl.graph.nodes[source_node]["phase"]

            freq_fluctuation = np.random.uniform(-self.fluctuation_scale, self.fluctuation_scale)
            phase_fluctuation = np.random.uniform(-self.fluctuation_scale, self.fluctuation_scale)

            self.ebl.graph.nodes[source_node]["freq"] = max(0.1, current_freq + freq_fluctuation)
            self.ebl.graph.nodes[source_node]["phase"] = math.atan2(
                math.sin(current_phase + phase_fluctuation),
                math.cos(current_phase + phase_fluctuation)
            )
            active_modulation_applied = True

        return {
            "source_node": source_node,
            "target_node": target,
            "friction_factor": res["friction_factor"],
            "phase_lag_rad": res["phase_lag_rad"],
            "global_topological_entropy": h_topo,
            "shell_dissolved": shell_dissolved,
            "active_modulation_applied": active_modulation_applied,
            "new_frequency": self.ebl.graph.nodes[source_node]["freq"],
            "new_phase": self.ebl.graph.nodes[source_node]["phase"],
            "status": "SHELL_DISSOLVED" if shell_dissolved else res["status"]
        }
