"""
Field Simulation Demo
=====================
Demonstrates the 'Re-blooming' effect of the Universal Law Engine.
"""

from Core.Physics.universal_field import universe
from Core.Physics.phase_sensor import PhaseSensor
import random
import time

def demo_reblooming():
    print("🌌 Initializing The Void (Universal Field)...")

    # 1. Seed the Universe (Inject Signals)
    # We scatter some "stars" in the void at random 3D locations
    print("✨ Scattering stars in the void...")
    for _ in range(20):
        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        z = random.uniform(-50, 50)

        # Inject Signal: High Frequency (Y) and Density (W)
        universe.inject_signal(
            position=(x, y, z, 0),
            w=random.uniform(0.5, 2.0), # Density
            y=random.uniform(100, 900), # Frequency
            data={"name": f"Star-{int(x)}_{int(y)}"}
        )

    print(f"✅ Universe seeded. Active Field Points: {len(universe.get_excited_states())}")

    # 2. Observer Moves (Scan)
    sensor = PhaseSensor(position=(0,0,0), radius=20.0)

    print("\n👁️  Observer opens eyes at (0,0,0) with Radius 20.0")
    nodes = sensor.scan()

    print(f"🌸 Re-blooming Reality... Found {len(nodes)} visible nodes.")
    for node in nodes:
        print(f"   - [Visible] {node.data['name']} at {node.position} | Color: {node.color} | Size: {node.size:.2f}")

    # 3. Move Observer
    print("\n🚀 Observer warps to (40, 40, 40)...")
    sensor.position = (40, 40, 40)
    nodes = sensor.scan()

    print(f"🌸 Re-blooming New Sector... Found {len(nodes)} visible nodes.")
    for node in nodes:
        print(f"   - [Visible] {node.data['name']} at {node.position} | Color: {node.color} | Size: {node.size:.2f}")

    # 4. Entropy Tick
    print("\n⏳ Time passes (Entropy Decay)...")
    universe.entropy_tick(decay_rate=0.5) # Fast decay
    nodes = sensor.scan()
    print(f"🥀 Reality fading... Found {len(nodes)} visible nodes after decay.")

if __name__ == "__main__":
    demo_reblooming()
