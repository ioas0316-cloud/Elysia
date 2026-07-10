import numpy as np
from typing import List, Dict, Any, Optional

class ThoughtTransistor:
    """
    [Core Intelligence Element: The Thought-Transistor]
    A virtual electronic component representing the minimum unit of reasoning.
    Features 3 terminals (Emitter, Base, Collector) analogous to physical transistors.

    - Emitter: Source of incoming thought energy.
    - Base: Control terminal. Threshold determines if the thought 'fires'.
    - Collector: Target for outgoing thought energy.
    """
    def __init__(self, thought_id: str, concept_tensor: np.ndarray):
        self.id = thought_id
        self.concept = concept_tensor # High-dimensional structural signature

        # 3-Terminal Connections (Stored as IDs of other ThoughtTransistors)
        self.emitters: List[str] = []
        self.collectors: List[str] = []

        # Operational States
        self.base_threshold = 0.5   # Activation barrier (Contextual resistance)
        self.conductance = 1.0      # G = 1/R (Ease of flow)
        self.energy = 0.0           # Current accumulated potential

        self.is_active = False
        self.trace_history: List[Dict[str, Any]] = [] # Memory of recent energy flow events
        self.growth_potential = 0.0 # Accumulator for mitotic expansion
        self.remanence_factor = 0.2 # How much energy remains after firing

    def inject_energy(self, amount: float, source_id: Optional[str] = None):
        """Accumulate energy from emitters."""
        if np.isnan(amount) or np.isinf(amount): return
        self.energy += amount
        self.energy = np.clip(self.energy, 0, 100.0) # Cap energy to prevent overflow

        if source_id:
            # [Process Recognition] Record the source of energy
            self.trace_history.append({"source": source_id, "amount": amount})
            if len(self.trace_history) > 10: self.trace_history.pop(0)

    def process(self, context_bias: float = 0.0) -> float:
        """
        Determines if the transistor 'turns on' (conducts).
        Returns the amount of energy to be passed to collectors.
        """
        # [Lens Mechanism] The base threshold is refracted by the context_bias
        effective_threshold = max(0.1, self.base_threshold - context_bias)

        if self.energy >= effective_threshold:
            self.is_active = True
            # Output energy is proportional to internal conductance
            output_energy = np.tanh(self.energy * self.conductance) * 2.0

            # [Organic Growth] Energy flow increases growth potential
            self.growth_potential += output_energy * 0.5 # Faster growth for demo

            # [Cognitive Remanence] Instead of 0, keep a fraction of energy
            self.energy = self.energy * self.remanence_factor
            return output_energy
        else:
            self.is_active = False
            # Energy remains accumulated but might decay
            self.energy *= 0.95
            return 0.0

    def update_conductance(self, flow: float, decay: float = 0.01):
        """
        [Plasticity]
        Higher flow reinforces the path (Memristive property).
        """
        self.conductance += flow * 0.5 # Increased reinforcement speed
        self.conductance = np.clip(self.conductance, 0.1, 10.0)
        # Natural decay of conductance if not used
        self.conductance -= decay
        self.conductance = max(0.1, self.conductance)

    def __repr__(self):
        return f"ThoughtTransistor(id={self.id}, active={self.is_active}, energy={self.energy:.2f}, G={self.conductance:.2f})"
