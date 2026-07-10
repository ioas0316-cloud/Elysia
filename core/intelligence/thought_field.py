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
        Includes Homeostatic Regulation to prevent runaway energy.
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

        # [Homeostasis] Regulate input intensity based on current total field energy
        total_field_energy = sum(e.energy for e in self.elements.values())
        regulation_factor = 1.0 / (1.0 + total_field_energy * 0.05)
        regulated_I = I * regulation_factor

        # Solve for Voltage (V = G^-1 * I)
        try:
            V = np.linalg.pinv(self.G_matrix) @ regulated_I
        except np.linalg.LinAlgError:
            V = regulated_I # Fallback

        # Distribute energy based on calculated potentials (V)
        for i, potential in enumerate(V):
            eid = self.index_to_id[i]
            if potential > 1e-6:
                self.elements[eid].inject_energy(potential, source_id="field_gradient")

    def step(self):
        """Individual element processing and plasticity update."""
        results = {}

        # [Information Lens & Radiative Sensation]
        # One transistor's state refracts the 'Base' threshold of others.
        # Now includes "Radiative" influence: even unconnected nodes sense nearby high-energy nodes in semantic space.
        context_biases = {eid: 0.0 for eid in self.elements}
        active_nodes = [(eid, e) for eid, e in self.elements.items() if e.energy > 0.1]

        for eid, target in self.elements.items():
            for source_id, source in active_nodes:
                if source_id == eid: continue

                # Semantic proximity (Resonance)
                resonance = np.dot(source.concept, target.concept)

                # Excitatory bias: closer in semantic space = more influence
                influence = source.energy * resonance * 0.1

                if target.id in source.collectors: # Direct connectivity has higher weight
                    influence *= 2.0

                context_biases[eid] += influence

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

            # [Mitotic Expansion] If growth potential is high, create a "Bud"
            if element.growth_potential > 5.0:
                self._mitotic_growth(element)

            # [Apoptosis] Remove dead cells
            if element.health <= 0.0:
                print(f"[Apoptosis] Cell death: {eid}")
                self._remove_element(eid)
                self.needs_rebuild = True

        self.needs_rebuild = True
        self._detect_organs()
        return results

    def _detect_organs(self):
        """
        [Differentiation] Identifies high-conductance clusters as "Functional Organs".
        Organs represent stable subsystems of intelligence.
        """
        self.organs = []
        visited = set()

        for eid, element in self.elements.items():
            if eid in visited: continue

            # Use simple BFS to find strong-connected components (G > threshold)
            cluster = []
            queue = [eid]
            visited.add(eid)

            while queue:
                curr_id = queue.pop(0)
                cluster.append(curr_id)

                curr = self.elements[curr_id]
                for neighbor_id in curr.collectors + curr.emitters:
                    if neighbor_id not in visited and neighbor_id in self.elements:
                        neighbor = self.elements[neighbor_id]
                        # Only consider strong bonds as part of an organ
                        if neighbor.conductance > 2.0:
                            visited.add(neighbor_id)
                            queue.append(neighbor_id)

            if len(cluster) >= 3: # Minimum size for an organ
                self.organs.append(cluster)

    def _remove_element(self, eid: str):
        """Cleans up all connections and removes the element from the field."""
        if eid not in self.elements: return
        element = self.elements[eid]

        # Remove from collectors of emitters
        for em_id in element.emitters:
            if em_id in self.elements:
                if eid in self.elements[em_id].collectors:
                    self.elements[em_id].collectors.remove(eid)

        # Remove from emitters of collectors
        for col_id in element.collectors:
            if col_id in self.elements:
                if eid in self.elements[col_id].emitters:
                    self.elements[col_id].emitters.remove(eid)

        del self.elements[eid]

    def _mitotic_growth(self, parent: ThoughtTransistor):
        """
        [Mitosis] Creates a new ThoughtTransistor as a functional expansion of the parent.
        The new 'Bud' specializes by slightly mutating the concept tensor.
        """
        child_id = f"{parent.id}_bud_{len(self.elements)}"
        # Mutate the concept tensor slightly (Specialization)
        mutation = (np.random.randn(*parent.concept.shape) * 0.1).astype(np.float32)
        child_concept = parent.concept + mutation

        child = ThoughtTransistor(child_id, child_concept)
        self.add_element(child)

        # Connect Parent to Child (Forward expansion)
        self.connect(parent.id, child_id)
        # Child also connects back to Parent (Feedback loop)
        self.connect(child_id, parent.id)

        parent.growth_potential = 0.0 # Reset potential after growth
        print(f"[Mitosis] Organic Expansion: {parent.id} spawned {child_id}")

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
