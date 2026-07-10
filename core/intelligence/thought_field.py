import numpy as np
from typing import Dict, List, Tuple
from core.intelligence.thought_element import ThoughtTransistor

class ThoughtField:
    """
    [Causal Field Layer: Thought-Field]
    Manages a network of ThoughtTransistors and performs global energy flow calculations.
    Uses a Conductance Matrix approach to solve for simultaneous potential distribution.
    """
    def __init__(self):
        self.elements: Dict[str, ThoughtTransistor] = {}
        self.matrix_indices: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}

        # Matrix state
        self.G_matrix: Optional[np.ndarray] = None # Conductance Matrix
        self.needs_rebuild = True

    def add_element(self, element: ThoughtTransistor):
        self.elements[element.id] = element
        self.needs_rebuild = True

    def connect(self, source_id: str, target_id: str):
        """Creates a directional connection (Source Collector -> Target Emitter)."""
        if source_id in self.elements and target_id in self.elements:
            self.elements[source_id].collectors.append(target_id)
            self.elements[target_id].emitters.append(source_id)
            self.needs_rebuild = True

    def _rebuild_matrix(self):
        """
        Constructs the global Conductance Matrix (G).
        Rows represent input potential, Columns represent output flow.
        """
        ids = sorted(list(self.elements.keys()))
        size = len(ids)
        self.matrix_indices = {id: i for i, id in enumerate(ids)}
        self.index_to_id = {i: id for i, id in enumerate(ids)}

        G = np.zeros((size, size), dtype=np.float32)

        for i, source_id in enumerate(ids):
            source = self.elements[source_id]
            # Diagonal: Self-leakage or internal dissipation (Resistance to ground)
            G[i, i] = 1.0 # Base resistance

            # Off-diagonal: Conductance to other elements
            for target_id in source.collectors:
                j = self.matrix_indices[target_id]
                # The flow depends on the source's conductance
                G[i, j] = -source.conductance
                G[j, j] += source.conductance # Kirchhoff's current law balancing

        self.G_matrix = G
        self.needs_rebuild = False

    def pulse(self, external_inputs: Dict[str, float]):
        """
        Simultaneous energy distribution across the field.
        Solves I = GV (roughly: Flow = Conductance * Potential)
        """
        if self.needs_rebuild:
            self._rebuild_matrix()

        size = len(self.elements)
        if size == 0: return

        # Vector of input currents (External Stimuli)
        I = np.zeros(size, dtype=np.float32)
        for eid, energy in external_inputs.items():
            if eid in self.matrix_indices:
                I[self.matrix_indices[eid]] += energy

        # Solve for Voltage (V = G^-1 * I)
        # In a real circuit, G might be singular if poorly connected, so we use pseudo-inverse
        try:
            V = np.linalg.pinv(self.G_matrix) @ I
        except np.linalg.LinAlgError:
            V = I # Fallback to local injection if matrix is unsolvable

        # Distribute energy based on calculated potentials (V)
        for i, potential in enumerate(V):
            eid = self.index_to_id[i]
            if potential > 1e-6:
                # Potential distribution from the field (No single source for V)
                self.elements[eid].inject_energy(potential, source_id="field_gradient")

    def step(self):
        """Individual element processing and plasticity update."""
        results = {}

        # [Information Lens] Pre-calculate contextual biases based on active neighbors
        # One transistor's state refracts the 'Base' threshold of connected neighbors.
        context_biases = {eid: 0.0 for eid in self.elements}
        for eid, element in self.elements.items():
            if element.energy > 0.1: # Significant potential exists
                for target_id in element.collectors:
                    if target_id in context_biases:
                        # Neighbor's energy lowers my threshold (Excitatory lens)
                        resonance = np.dot(element.concept, self.elements[target_id].concept)
                        context_biases[target_id] += element.energy * resonance * 0.2

        for eid, element in list(self.elements.items()):
            bias = context_biases.get(eid, 0.0)
            out_energy = element.process(context_bias=bias)

            if out_energy > 0:
                results[eid] = out_energy
                # Plasticity: Reinforce target conductance based on flow
                for target_id in element.collectors:
                    if target_id in self.elements:
                        # Process Recognition: Pass the source ID during injection
                        self.elements[target_id].inject_energy(out_energy / len(element.collectors), source_id=eid)
                        self.elements[target_id].update_conductance(out_energy)

            # Dynamic Rewiring: Break links if resistance is too high or mismatch occurs
            self._handle_rewiring(element)

        self.needs_rebuild = True
        return results

    def _handle_rewiring(self, element: ThoughtTransistor):
        """
        [Structural Tearing & Dynamic Coupling]
        If a connection is consistently low-energy or high-resistance, tear it.
        Then, attempt to find a new connection based on semantic gravity (tensor resonance).
        """
        # 1. Structural Tearing
        broken_collectors = []
        for target_id in element.collectors:
            target = self.elements.get(target_id)
            if not target: continue

            # Tension = Resistance * Potential (Simplified)
            resistance = 1.0 / (target.conductance + 1e-6)
            if resistance > 5.0: # Arbitrary threshold for 'Tension'
                broken_collectors.append(target_id)

        for tid in broken_collectors:
            element.collectors.remove(tid)
            if element.id in self.elements[tid].emitters:
                self.elements[tid].emitters.remove(element.id)
            print(f"[Rewire] Link broken: {element.id} -X-> {tid} (High Resistance)")

        # 2. Dynamic Coupling (Semantic Gravity)
        if len(element.collectors) < 2: # Looking for new connections
            potential_targets = []
            for other_id, other in self.elements.items():
                if other_id == element.id or other_id in element.collectors: continue

                # Semantic Gravity = Dot product of concept tensors
                resonance = np.dot(element.concept, other.concept)
                if resonance > 0.8: # Strong resonance threshold
                    potential_targets.append(other_id)

            for new_tid in potential_targets[:1]: # Connect to top candidate
                self.connect(element.id, new_tid)
                print(f"[Rewire] New Link formed: {element.id} ---> {new_tid} (Semantic Resonance)")

if __name__ == "__main__":
    # Test Setup
    field = ThoughtField()
    t1 = ThoughtTransistor("Root", np.array([1, 0]))
    t2 = ThoughtTransistor("Node_A", np.array([0, 1]))
    t3 = ThoughtTransistor("Leaf", np.array([1, 1]))

    field.add_element(t1)
    field.add_element(t2)
    field.add_element(t3)

    field.connect("Root", "Node_A")
    field.connect("Node_A", "Leaf")

    print("Initial Pulse...")
    field.pulse({"Root": 1.0})

    for i in range(5):
        res = field.step()
        print(f"Step {i}: Active Elements -> {res}")
        if res:
            # Re-inject output to simulate continuous flow if needed
            field.pulse({eid: energy for eid, energy in res.items()})
