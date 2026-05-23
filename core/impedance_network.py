"""
[IMPEDANCE-DRIVEN PROPAGATION NETWORK]
"Where Ohm's Law meets Phase Synchronization."

This network implements an unsupervised learning algorithm using physical analog principles:
1. Information is current (I = V / Z) flowing through complex impedance links (Z = R + jX).
2. SAMENESS (Coherence) decreases resistance (R), creating a permanent highway (Memory).
3. DIFFERNCE (Dissonance) increases resistance (R), blocking noise.
"""

import math
from typing import Dict, List, Tuple

class ImpedanceLink:
    def __init__(self, node_from: str, node_to: str, initial_R: float = 10.0, initial_X: float = 0.0):
        self.node_from = node_from
        self.node_to = node_to
        
        # Physical impedance state: Z = R + jX
        self.R = float(initial_R)       # Resistance (attenuation / weight)
        self.X = float(initial_X)       # Reactance (phase distortion)
        self.I = 0.0                    # Current flowing through this link

        # Physical limits
        self.min_R = 0.5
        self.max_R = 100.0

    @property
    def impedance_magnitude(self) -> float:
        """Calculate |Z| = sqrt(R^2 + X^2)."""
        return math.sqrt(self.R**2 + self.X**2)

    def update_impedance(self, diff_phase: float, lr: float = 0.5):
        """
        Ohmic adaptation rule based on phase coherence:
        - If phases are aligned (cos(diff) > 0), reduce resistance.
        - If phases are misaligned, increase resistance.
        """
        coherence = math.cos(diff_phase)
        
        # Adaptation is proportional to the current flow (I) and coherence
        adaptation = lr * coherence * abs(self.I)
        
        if coherence > 0:
            # Coherence: reduce resistance (carve path)
            self.R = max(self.min_R, self.R - adaptation)
        else:
            # Dissonance: increase resistance (block path)
            self.R = min(self.max_R, self.R - adaptation * 0.2) # slower decay for blocking


class ImpedancePropagationNetwork:
    def __init__(self):
        self.phases: Dict[str, float] = {} # node_id -> phase angle in radians
        self.links: List[ImpedanceLink] = []
        self.node_layers: Dict[str, int] = {} # node_id -> layer index (0: input, 1: hidden, 2: output)

    def add_node(self, node_id: str, layer: int, initial_phase: float = 0.0):
        self.phases[node_id] = initial_phase % (2 * math.pi)
        self.node_layers[node_id] = layer

    def connect_nodes(self, id_from: str, id_to: str, initial_R: float = 10.0) -> ImpedanceLink:
        link = ImpedanceLink(id_from, id_to, initial_R)
        self.links.append(link)
        return link

    def forward_propagate(self, inputs: Dict[str, float]):
        """
        Propagate signal current from input layer to output layer.
        inputs: { input_node_id: current_amplitude }
        """
        # Reset all link currents
        for link in self.links:
            self.link_current = 0.0

        # Calculate current flow layer by layer
        # Layer 0 (Input) -> Layer 1 (Hidden) -> Layer 2 (Output)
        node_currents = {k: 0.0 for k in self.phases.keys()}
        for k, v in inputs.items():
            node_currents[k] = v

        max_layer = max(self.node_layers.values(), default=0)

        for layer in range(max_layer):
            # Find all nodes in the current layer
            current_layer_nodes = [n for n, l in self.node_layers.items() if l == layer]
            
            for node in current_layer_nodes:
                input_current = node_currents[node]
                if input_current <= 0:
                    continue

                # Find all outgoing links from this node
                outgoing = [l for l in self.links if l.node_from == node]
                if not outgoing:
                    continue

                # Distribute current based on admittance (1 / |Z|)
                # Higher admittance -> More current flows
                admittances = [1.0 / max(0.01, l.impedance_magnitude) for l in outgoing]
                total_admittance = sum(admittances)

                if total_admittance > 0:
                    for link, adm in zip(outgoing, admittances):
                        flow = input_current * (adm / total_admittance)
                        link.I = flow
                        node_currents[link.node_to] += flow

    def tune_network(self, dt: float, lr: float = 0.5):
        """
        Tune link impedances and adapt node phases.
        Nodes pull each other's phases based on current flow and connection strength.
        """
        # 1. Update link impedances
        for link in self.links:
            diff = self.phases[link.node_to] - self.phases[link.node_from]
            # Adjust resistance based on coherence and current
            link.update_impedance(diff, lr)

        # 2. Phase coupling (Kuramoto phase-locking driven by current)
        phase_deltas = {k: 0.0 for k in self.phases.keys()}
        for link in self.links:
            diff = self.phases[link.node_to] - self.phases[link.node_from]
            
            # Torque pull is proportional to coupling strength (1/R) and current flow (I)
            coupling = (1.0 / link.R) * abs(link.I)
            torque = coupling * math.sin(diff) * dt

            # Parent and child pull each other
            phase_deltas[link.node_to] -= torque * 0.5
            phase_deltas[link.node_from] += torque * 0.5

        # Apply phase updates
        for node in self.phases:
            self.phases[node] = (self.phases[node] + phase_deltas[node]) % (2 * math.pi)
