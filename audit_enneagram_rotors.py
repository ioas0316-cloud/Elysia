import math
from typing import Dict, Any

class MockRotor:
    def __init__(self, name, rpm, current_angle=0.0):
        self.name = name
        self.rpm = rpm
        self.current_angle = current_angle
        self.energy = 1.0

    def update(self, dt):
        # Degrees per second
        self.current_angle = (self.current_angle + (self.rpm / 60.0) * 360.0 * dt) % 360.0

def simulate_enneagram_shift(type_num: int, dt: float = 0.1):
    # Base rotors (simplified from PsycheSphere)
    id_rpm = 666.0
    ego_rpm = 432.0
    superego_rpm = 1111.0

    # The Enneagram "Twist":
    # Each type has a unique starting phase (Shift) or RPM modulation
    # Based on the user's "3-phase" analogy, we can map 1-9 to 40-degree increments
    type_shift = (type_num - 1) * 40.0

    id_rotor = MockRotor("Id", id_rpm, current_angle=type_shift)
    ego_rotor = MockRotor("Ego", ego_rpm)
    superego_rotor = MockRotor("Superego", superego_rpm)

    # Simulation for a few steps
    results = []
    for _ in range(5):
        id_rotor.update(dt)
        ego_rotor.update(dt)
        superego_rotor.update(dt)

        id_wave = math.sin(math.radians(id_rotor.current_angle))
        superego_wave = math.sin(math.radians(superego_rotor.current_angle))

        # Will = Interference(Id, Superego) modulated by Ego
        ego_phase = ego_rotor.current_angle
        will = (id_wave + superego_wave) * math.cos(math.radians(ego_phase))
        tension = abs(id_wave - superego_wave)

        results.append({"will": will, "tension": tension})

    avg_will = sum(r["will"] for r in results) / len(results)
    avg_tension = sum(r["tension"] for r in results) / len(results)

    return avg_will, avg_tension

def main():
    print("--- Enneagram Rotor Phase Shifting Audit ---")
    print(f"{'Type':<10} | {'Name':<15} | {'Avg Will':<10} | {'Avg Tension':<10}")
    print("-" * 55)

    types = {
        1: "Reformer",
        2: "Helper",
        3: "Achiever",
        4: "Individualist",
        5: "Investigator",
        6: "Loyalist",
        7: "Enthusiast",
        8: "Challenger",
        9: "Peacemaker"
    }

    for t_num, name in types.items():
        will, tension = simulate_enneagram_shift(t_num)
        print(f"{t_num:<10} | {name:<15} | {will:>10.3f} | {tension:>10.3f}")

if __name__ == "__main__":
    main()
