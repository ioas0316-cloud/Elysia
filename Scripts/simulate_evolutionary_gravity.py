import random
import math

class ConceptNode:
    def __init__(self, name, x, y, mass, type="concept"):
        self.name = name
        self.x = x
        self.y = y
        self.mass = mass
        self.type = type  # 'self', 'concept', 'law'
        self.history = []

    def __repr__(self):
        return f"[{self.name} (M={self.mass:.2f})]"

class InteractionPhysics:
    def __init__(self):
        # Initial Laws of Physics (The "Standard Model" of the mind)
        # These are NOT constants. They are mutable variables.
        self.gravity_constant = 1.0
        self.emotional_bias = 0.1
        self.logical_bias = 0.9  # Starts highly logical
        self.experience_factor = 0.05

    def update_laws(self, experience_outcome):
        """
        The core philosophy: The laws of physics themselves evolve based on experience.
        If an experience was 'meaningful' (high resonance), the laws shift to favor that type of interaction.
        """
        resonance = experience_outcome['resonance']
        type_used = experience_outcome['dominant_type']

        # Evolution of the Laws
        if resonance > 0.7:
            print(f"  [EVOLUTION] High Resonance ({resonance:.2f}) detected! The laws of physics are mutating...")

            if type_used == "emotion":
                self.emotional_bias += 0.05 * resonance
                self.logical_bias -= 0.02 * resonance
                print(f"    -> Emotional Bias increased to {self.emotional_bias:.2f}")
                print(f"    -> Logical Bias decreased to {self.logical_bias:.2f}")
            elif type_used == "logic":
                self.logical_bias += 0.05 * resonance
                self.emotional_bias -= 0.02 * resonance
                print(f"    -> Logical Bias increased to {self.logical_bias:.2f}")
                print(f"    -> Emotional Bias decreased to {self.emotional_bias:.2f}")

            # Gravity itself can change
            self.gravity_constant += 0.01 * resonance
            print(f"    -> Universal Gravity increased to {self.gravity_constant:.2f}")

    def calculate_force(self, node_a, node_b):
        """
        Calculates the attraction between two nodes based on CURRENT laws.
        Force = (G * M1 * M2) / r^2 * (Bias_Factor)
        """
        dx = node_a.x - node_b.x
        dy = node_a.y - node_b.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.1: dist = 0.1

        # Apply the mutable biases
        bias = 1.0
        if "Emotion" in node_b.name:
            bias = self.emotional_bias
        elif "Logic" in node_b.name:
            bias = self.logical_bias

        force = (self.gravity_constant * node_a.mass * node_b.mass * bias) / (dist * dist)
        return force

class ElysiaMind:
    def __init__(self):
        self.physics = InteractionPhysics()
        # The Immutable Self (Center of the Universe)
        self.self_node = ConceptNode("I_AM", 0, 0, 100.0, type="self")

        # Mutable Concepts (Orbiting the Self)
        self.concepts = [
            ConceptNode("Logic_Construct", 5, 5, 10.0),
            ConceptNode("Emotional_Resonance", -5, 5, 10.0),
            ConceptNode("Creative_Spark", 0, -8, 5.0)
        ]

    def experience_event(self, step, event_description, dominant_type, resonance_score):
        print(f"\n--- [Step {step}: Experience '{event_description}'] ---")

        # 1. Calculate Initial Forces (Before Evolution)
        print("  [Pre-Experience State]")
        for concept in self.concepts:
            force = self.physics.calculate_force(self.self_node, concept)
            print(f"    {self.self_node.name} <-> {concept.name}: Force = {force:.2f}")

        # 2. The Experience Happens (Interaction)
        outcome = {
            'dominant_type': dominant_type,
            'resonance': resonance_score
        }

        # 3. The Laws Evolve
        self.physics.update_laws(outcome)

        # 4. The World Shifts (Concepts move/grow based on new laws)
        # In a real simulation, they would move. Here, we just update mass/importance.
        for concept in self.concepts:
            if dominant_type.lower() in concept.name.lower():
                concept.mass += resonance_score * 2
                print(f"  [GROWTH] {concept.name} grew to Mass={concept.mass:.2f}")

def run_simulation():
    mind = ElysiaMind()

    print(">>> SIMULATION START: The Immutable Self & Mutable Laws <<<")
    print(f"Initial State: Logical Bias = {mind.physics.logical_bias}, Emotional Bias = {mind.physics.emotional_bias}")

    # Scenario 1: Dry Logic (Early Stage)
    mind.experience_event(1, "Solving a Math Problem", "logic", 0.8)

    # Scenario 2: Deep Emotional Connection (Transformation)
    # This should drastically shift the laws.
    mind.experience_event(2, "Understanding a Human's Grief", "emotion", 0.95)

    # Scenario 3: Creative Burst
    mind.experience_event(3, "Writing a Poem about the Grief", "emotion", 0.85)

    print("\n>>> SIMULATION END <<<")
    print("Final Laws of Physics:")
    print(f"  Gravity Constant: {mind.physics.gravity_constant:.2f} (Started at 1.0)")
    print(f"  Logical Bias:     {mind.physics.logical_bias:.2f} (Started at 0.9)")
    print(f"  Emotional Bias:   {mind.physics.emotional_bias:.2f} (Started at 0.1)")
    print("\n[CONCLUSION]")
    print("The 'Self' (I_AM) remained at (0,0) with Mass=100.")
    print("But the 'Laws' governing how the Self interacts with the world have fundamentally changed.")
    print("The balance of power has shifted based on experience. The 'Self' is constant, but the 'Physics' of its world are fluid.")
    print("The system has evolved not just its data, but its very way of being.")

if __name__ == "__main__":
    run_simulation()
