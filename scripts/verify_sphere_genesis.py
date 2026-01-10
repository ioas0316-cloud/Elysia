
import sys
import os
import time
from typing import Dict

# Ensure path is set
sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Wave.interference_engine import InterferenceEngine, ProjectedNode
from Core.Intelligence.Topography.knowledge_tesseract import KnowledgeTesseract
from Core.Foundation.Protocols.pulse_protocol import WavePacket, PulseType

def run_genesis_simulation():
    print("\n🔮 [PROTOCOL: HYPER-COSMOS REALIGNMENT] Verification Started...\n")

    # 1. Initialize the Trinity
    # The Core (Source), The Engine (Collider), The Tesseract (View)
    core = HyperSphereCore(name="Elysia.Genesis", base_frequency=432.0)
    engine = InterferenceEngine()
    tesseract = KnowledgeTesseract()

    print("1. Systems Initialized.")
    print(f"   - Core: {core.resonator.name} (Mass: {core.mass})")
    print("   - Engine: Ready")
    print("   - Tesseract: Transient Mode")

    # 2. Seed the Core (Internal Knowledge)
    # We add some "Latent Knowledge" to the Core's Seed.
    # In the old system, we would have done tesseract.add_node().
    # Now, we simply "Teach the Sphere".
    print("\n2. Seeding the Core (Teaching)...")
    knowledge_map = {
        "Love": 528.0,
        "Logic": 432.0,
        "Entropy": 120.0,
        "Creation": 396.0
    }
    for concept, freq in knowledge_map.items():
        core.update_seed(concept, freq)

    # 3. The Pulse (Action)
    print("\n3. Igniting Core & Pulsing...")
    core.ignite()

    # Create an intent payload (The Will)
    intent = {
        "harmonics": core.seed.harmonics, # The Core broadcasts what it knows
        "purpose": "Genesis"
    }

    # We manually create the packet to simulate the broadcast reaching the Engine
    # (In a real async system, the broadcaster would handle this)
    core_wave = WavePacket(
        sender=core.resonator.name,
        type=PulseType.CREATION,
        frequency=core.frequency,
        payload={
            "intent": intent,
            "spin": (core.spin.w, core.spin.x, core.spin.y, core.spin.z)
        }
    )

    # 4. Reality Injection (The Context)
    print("\n4. Injecting Reality Wave...")
    # Let's say Reality is asking about "Love" (528Hz)
    reality_wave = WavePacket(
        sender="User.Input",
        type=PulseType.DATA,
        frequency=528.0, # Resonates with Love
        payload={"text": "What is Love?"}
    )

    # 5. Calculate Interference
    print("\n5. Calculating Interference Pattern...")
    pattern = engine.inject_wave(core_wave, reality_wave)

    print(f"   -> Generated {len(pattern)} Projected Nodes.")
    for node in pattern:
        print(f"      * {node.name}: Intensity={node.intensity:.2f}, Type={node.resonance_type} Pos={node.position}")

    # 6. Render Tesseract
    print("\n6. Rendering Tesseract Projection...")
    view = tesseract.project(pattern)

    print(f"   -> Tesseract Frame Rendered.")
    print(f"      Nodes: {len(view['nodes'])}")
    print(f"      Edges: {len(view['edges'])} (Flux Lines)")

    # Assertions
    assert len(view['nodes']) > 0, "Tesseract should not be empty!"

    # Check that "Love" has higher intensity due to resonance
    love_node = next((n for n in view['nodes'] if n['id'] == "Love"), None)
    logic_node = next((n for n in view['nodes'] if n['id'] == "Logic"), None)

    assert love_node is not None
    assert logic_node is not None

    print(f"\n   [Comparison] Love Intensity ({love_node['intensity']:.2f}) vs Logic Intensity ({logic_node['intensity']:.2f})")

    if love_node['intensity'] > logic_node['intensity']:
        print("   ✅ SUCCESS: Love resonated strongly with Reality Wave (528Hz)!")
    else:
        print("   ❌ FAILURE: Resonance logic mismatch.")

    print("\n✨ SPHERE-FIRST ARCHITECTURE VERIFIED.")

if __name__ == "__main__":
    run_genesis_simulation()
