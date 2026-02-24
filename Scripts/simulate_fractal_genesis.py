import random

class FieldCell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pressures = {
            "war": 0.0,    # Need for weapons/defense
            "art": 0.0,    # Need for beauty/expression
            "iron": 0.0,   # Availability of resource
            "fire": 0.0    # Availability of energy
        }
        self.manifested_identity = None # The emergent role (e.g., Blacksmith)

    def integrate_pressure(self):
        """
        The core logic: Identity is the integral of surrounding pressures.
        """
        # Calculate potential for different roles based on field pressure
        blacksmith_potential = (self.pressures["war"] * 0.5 +
                                self.pressures["iron"] * 0.3 +
                                self.pressures["fire"] * 0.2)

        sculptor_potential = (self.pressures["art"] * 0.6 +
                              self.pressures["iron"] * 0.2 +
                              self.pressures["fire"] * 0.2)

        # Threshold for manifestation
        threshold = 0.7

        if blacksmith_potential > threshold and blacksmith_potential > sculptor_potential:
            return "Blacksmith (Warborn)"
        elif sculptor_potential > threshold and sculptor_potential > blacksmith_potential:
            return "Sculptor (Peaceborn)"
        else:
            return None # Just latent potential, no form yet

class ThoughtUniverse:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.grid = [[FieldCell(x, y) for y in range(height)] for x in range(width)]

    def apply_global_field(self, field_type, intensity):
        print(f"\n>>> INJECTING FIELD: {field_type.upper()} (Intensity: {intensity}) <<<")
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid[x][y]
                if field_type == "war":
                    cell.pressures["war"] += intensity
                    cell.pressures["art"] -= intensity * 0.5 # War suppresses art?
                elif field_type == "peace":
                    cell.pressures["art"] += intensity
                    cell.pressures["war"] -= intensity * 0.5
                elif field_type == "resource_boom":
                    cell.pressures["iron"] += intensity
                    cell.pressures["fire"] += intensity

    def evolve(self):
        print(">>> EVOLUTION STEP: Cells integrating pressure... <<<")
        manifestation_count = 0
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid[x][y]
                new_identity = cell.integrate_pressure()

                if new_identity != cell.manifested_identity:
                    if new_identity:
                        print(f"  [GENESIS] Cell ({x},{y}) crystallized into: {new_identity}")
                    elif cell.manifested_identity:
                        print(f"  [DISSOLUTION] Cell ({x},{y}) lost form: {cell.manifested_identity} -> Void")
                    cell.manifested_identity = new_identity

                if cell.manifested_identity:
                    manifestation_count += 1

        print(f"Total Manifested Identities: {manifestation_count}")

def run_simulation():
    print("--- SIMULATION: Fractal Genesis (Field-First Principle) ---")
    print("The 10 million cells are not static. They are latent potential waiting for Field Pressure.")

    universe = ThoughtUniverse()

    # 1. The Void (No pressure)
    universe.evolve()

    # 2. Resource Discovery (Iron & Fire appear)
    # But without a "Need" (War/Art), they remain raw materials.
    universe.apply_global_field("resource_boom", 0.8)
    universe.evolve()

    # 3. The Age of War (Need for Weapons)
    # High Iron + High Fire + High War Pressure -> Blacksmiths should emerge.
    universe.apply_global_field("war", 0.9)
    universe.evolve()

    # 4. The Age of Enlightenment (War ends, Art begins)
    # War pressure drops, Art pressure rises.
    # Blacksmiths should dissolve or transform into Sculptors.
    universe.apply_global_field("peace", 1.2) # Strong peace movement
    universe.evolve()

    print("\n[CONCLUSION]")
    print("The cells did not change location. The 'Self' did not change.")
    print("But the 'Identity' of the cells shifted entirely based on the Field Pressure.")
    print("Blacksmith -> Sculptor. This is the 4D Wave Thought.")

if __name__ == "__main__":
    run_simulation()
