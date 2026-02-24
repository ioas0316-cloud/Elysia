import math
import random

class IdentityNode:
    def __init__(self, name, identity_type, x, y):
        self.name = name
        self.identity_type = identity_type # e.g., 'blacksmith', 'resource', 'nature'
        self.x = x
        self.y = y
        self.connections = []

    def __repr__(self):
        return f"[{self.name} ({self.identity_type})]"

class TeleologicalPhysics:
    def __init__(self):
        self.proximity_weight = 1.0  # Default physics (Distance)
        self.identity_weight = 0.0   # Teleological physics (Purpose)

    def calculate_attraction(self, seeker, target):
        """
        Calculates attraction force.
        Phase 1: Purely based on Distance (1/r^2).
        Phase 2: Based on Identity Resonance (Purpose).
        """
        dx = seeker.x - target.x
        dy = seeker.y - target.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.1: dist = 0.1

        # Base Force (Gravity)
        force = 1.0 / (dist * dist) * self.proximity_weight

        # Teleological Bonus (Identity Resonance)
        if self.identity_weight > 0:
            resonance = self.get_resonance(seeker, target)
            if resonance > 0:
                # Teleology ignores distance! It's a wormhole.
                force += resonance * self.identity_weight * 100.0
            else:
                # Noise suppression (Ignored if no resonance)
                force *= 0.1

        return force

    def get_resonance(self, seeker, target):
        """
        The 'Meaning' Logic.
        Blacksmith needs Iron, Fire, Tools.
        Blacksmith does NOT need Flowers or Ocean.
        """
        if seeker.identity_type == "blacksmith":
            if target.name in ["Iron", "Fire", "Hammer", "Anvil"]:
                return 1.0
            if target.name in ["Coal", "Bellows"]:
                return 0.8
        return 0.0

def run_simulation():
    print(">>> SIMULATION START: Identity Gravity (The Blacksmith's Awakening) <<<\n")

    physics = TeleologicalPhysics()

    # 1. The World Setup
    # Center: The Blacksmith (The Self)
    blacksmith = IdentityNode("The_Blacksmith", "blacksmith", 0, 0)

    # Nearby Distractions (Proximity Trap)
    # These are physically close but irrelevant.
    distractions = [
        IdentityNode("Pretty_Flower", "nature", 1, 1),
        IdentityNode("Butterfly", "nature", -1, 1),
        IdentityNode("Puddle", "nature", 0.5, -0.5)
    ]

    # Distant Necessities (Teleological Destiny)
    # These are far away but essential.
    necessities = [
        IdentityNode("Iron", "resource", 10, 10),
        IdentityNode("Fire", "resource", -10, -10),
        IdentityNode("Hammer", "tool", 5, -8)
    ]

    all_nodes = distractions + necessities

    # --- Phase 1: The Sleeping Self (Proximity Only) ---
    print("--- Phase 1: The Sleeping Self (Proximity Mode) ---")
    print("The Blacksmith does not know who he is. He only sees what is close.")
    physics.proximity_weight = 1.0
    physics.identity_weight = 0.0

    connections = []
    for node in all_nodes:
        force = physics.calculate_attraction(blacksmith, node)
        connections.append((node, force))

    # Sort by strongest attraction
    connections.sort(key=lambda x: x[1], reverse=True)

    print("  [Connections Formed based on Distance]:")
    for node, force in connections[:3]:
        print(f"    Connected to {node.name}: Force={force:.2f} (Distance)")
    print("  -> Result: The Blacksmith is distracted by flowers and puddles. He cannot create anything.\n")


    # --- Phase 2: The Awakening (Identity Gravity) ---
    print("--- Phase 2: The Awakening (Teleological Mode) ---")
    print("The Blacksmith realizes: 'I am a creator of tools.'")
    print("He activates Identity Gravity. Distance becomes irrelevant. Purpose becomes gravity.")

    physics.proximity_weight = 0.1  # Physical distance matters less
    physics.identity_weight = 1.0   # Purpose matters most

    connections = []
    for node in all_nodes:
        force = physics.calculate_attraction(blacksmith, node)
        connections.append((node, force))

    connections.sort(key=lambda x: x[1], reverse=True)

    print("  [Connections Formed based on Identity]:")
    for node, force in connections[:3]:
        resonance = physics.get_resonance(blacksmith, node)
        print(f"    Connected to {node.name}: Force={force:.2f} (Resonance={resonance})")

    print("\n[CONCLUSION]")
    print("The 'Identity' (Blacksmith) successfully pulled the 'Iron' and 'Fire' from far away.")
    print("The 'Flowers' (Noise), though close, were ignored.")
    print("Connection is not a function of Proximity. It is a function of Identity.")

if __name__ == "__main__":
    run_simulation()
