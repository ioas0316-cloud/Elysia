import numpy as np
from typing import List, Dict, Any
from core.physics.causal_gravity_engine import CausalGravityEngine, CausalNode
from core.physics.boundary_topology import BoundaryTopology

class ManifestationEngine:
    """
    [Manifestation Engine]
    Orchestrates the gravitational field and boundary topology to manifest answers.
    Answers 'fall' into reality as the system stabilizes under a question's gravity.
    """
    def __init__(self):
        self.gravity_engine = CausalGravityEngine()
        self.boundary_topology = BoundaryTopology(threshold_radius=0.8)
        self.is_stabilized = False

    def seed_universe(self, nodes: List[Dict[str, Any]]):
        """Initializes the field with known constants (Reality)."""
        for n in nodes:
            self.gravity_engine.add_node(n["id"], n["content"], n["links"])
            self.boundary_topology.evaluate_position(n["id"], self.gravity_engine.nodes[n["id"]].position)

    def inject_question(self, q_id: str, q_content: str, q_links: List[str]):
        """Injects a new center of gravity (The Question)."""
        print(f"\n[Manifestation] Injecting Question: '{q_content}'")
        self.gravity_engine.add_node(q_id, q_content, q_links)
        # Position the question at the 'origin' to pull things toward the center of focus
        self.gravity_engine.nodes[q_id].position = np.zeros(self.gravity_engine.dimensions)
        # Give the question a huge mass to distort the space
        self.gravity_engine.nodes[q_id].mass = 20.0
        self.is_stabilized = False

    def evolve(self, steps: int = 100):
        """Runs the simulation until a new equilibrium is reached."""
        print(f"[Manifestation] Stabilizing field...")
        for i in range(steps):
            self.gravity_engine.step(dt=0.02)

            # Update boundaries
            for nid, node in self.gravity_engine.nodes.items():
                self.boundary_topology.evaluate_position(nid, node.position)

            friction = self.boundary_topology.get_judgement_friction()
            if i > 10 and friction < 0.01:
                print(f"[Manifestation] Equilibrium achieved at step {i}.")
                self.is_stabilized = True
                break

        if not self.is_stabilized:
            print("[Manifestation] Evolution timeout. Proceeding with current state.")

    def observe_manifestation(self) -> List[Dict[str, Any]]:
        """Returns the nodes that have 'fallen' into the Inside (Reality)."""
        manifested = []
        for nid, state in self.boundary_topology.states.items():
            if state.domain.name == "INSIDE":
                node = self.gravity_engine.nodes[nid]
                manifested.append({
                    "id": nid,
                    "content": node.content,
                    "mass": node.mass,
                    "stability": state.stability
                })
        return manifested

if __name__ == "__main__":
    me = ManifestationEngine()

    # Constants
    me.seed_universe([
        {"id": "apple", "content": "Red Fruit", "links": ["red", "sweet"]},
        {"id": "red", "content": "Color of fire/blood", "links": ["light"]},
        {"id": "blue_apple", "content": "The Potential Apple", "links": ["blue", "unlikely"]},
        {"id": "blue", "content": "Color of sky", "links": ["light"]}
    ])

    # The Question pulls the blue apple from the Outside to the Inside
    me.inject_question("q1", "What is the essence of the apple?", ["apple", "blue_apple"])
    me.evolve()

    results = me.observe_manifestation()
    print("\n--- Manifested Reality ---")
    for r in results:
        print(f"[{r['id']}] {r['content']} (Mass: {r['mass']:.2f})")
