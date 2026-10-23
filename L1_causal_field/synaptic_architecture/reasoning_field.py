import numpy as np
from typing import Dict, List, Any
from core.physics.causal_field import CausalField, InformationVoxel

class ReasoningField:
    """
    [Logic/Reasoning Layer: The Synaptic Bridge]
    Maps logical concepts into the CausalField to simulate 'Dynamic Rewiring'
    and 'Conceptual Tension'.
    """
    def __init__(self):
        self.field = CausalField(dimensions=3)
        self.concepts: Dict[str, str] = {} # Mapping voxel_id to conceptual name

    def inject_concept(self, name: str, meaning_vector: np.ndarray):
        """
        Injects a concept as an Information Voxel.
        The meaning_vector acts as the structural tensor.
        """
        voxel_id = f"concept_{name}"
        voxel = InformationVoxel(
            id=voxel_id,
            content=name,
            tensor=meaning_vector.astype(np.float32),
            position=np.random.randn(3).astype(np.float32)
        )
        self.field.add_voxel(voxel)
        self.concepts[voxel_id] = name
        return voxel_id

    def assert_relationship(self, concept_a: str, concept_b: str, logic_strength: float = 1.0):
        """
        Creates a connectivity beam between two concepts.
        """
        id_a = f"concept_{concept_a}"
        id_b = f"concept_{concept_b}"
        self.field.link_voxels(id_a, id_b, strength=logic_strength)

    def apply_logical_impact(self, concept_name: str, impact_vector: np.ndarray):
        """
        Simulates a logical contradiction or new evidence hitting a concept.
        """
        voxel_id = f"concept_{concept_name}"
        self.field.apply_impact(voxel_id, impact_vector.astype(np.float32))

    def evolve(self, steps: int = 5, dt: float = 0.1):
        """
        Allows the logical structure to 'settle' or 'tear' based on tension.
        """
        for _ in range(steps):
            self.field.step(dt)

    def get_logical_state(self):
        topology = self.field.get_topology()
        # Enhance topology with concept names
        for vid, data in topology["voxels"].items():
            data["name"] = self.concepts.get(vid, "Unknown")
        return topology

if __name__ == "__main__":
    rf = ReasoningField()

    # Define a simple logical structure
    # Concept: "Bird" and "Can Fly"
    v_bird = rf.inject_concept("Bird", np.array([1, 0, 0]))
    v_fly = rf.inject_concept("Can Fly", np.array([0.9, 0, 0]))
    rf.assert_relationship("Bird", "Can Fly", logic_strength=5.0)

    print("Initial Logical State:", rf.get_logical_state())

    # Impact: "Penguin" is a Bird but "Cannot Fly"
    # This creates tension in the "Can Fly" relationship
    print("\nApplying logical contradiction: 'Penguin' (A Bird that cannot fly)")
    rf.apply_logical_impact("Can Fly", np.array([0, 10.0, 0]))

    rf.evolve(steps=20)

    state = rf.get_logical_state()
    print("Final Logical State:", state)

    for beam in state["beams"]:
        if beam["broken"]:
            print(f"!!! Logical link broken between {beam['s']} and {beam['t']} due to high tension !!!")
