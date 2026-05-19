"""
Demo: Physics of Thought
========================

This script demonstrates the "Ether Architecture" in action.
It injects random text nodes and visualizes how "Meaning Gravity" and "Resonance"
cause them to self-organize without explicit logic.
"""

import time
import random
import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Core.Ether.ether_node import EtherNode
from Core.Ether.void import Void
from Core.Ether.field_operators import DynamicsEngine
from Core.Ether.bridge import EtherBridge

def run_simulation():
    print("🌌 Initializing The Void...")
    void = Void()
    dynamics = DynamicsEngine()

    # 1. Inject Concepts (Clusters of Meaning)
    # We expect these to cluster naturally due to keyword gravity
    concepts = [
        "사랑 (Love)", "기쁨 (Joy)", "행복 (Happiness)", # Emotional Cluster
        "논리 (Logic)", "수학 (Math)", "계산 (Calc)",    # Logical Cluster
        "의지 (Will)", "행동 (Action)", "결단 (Decision)" # Will Cluster
    ]

    print("\n✨ Injecting Thoughts...")
    for text in concepts:
        node = EtherBridge.text_to_node(text)
        # Randomize initial positions slightly to prove gravity works
        node.position.x += random.uniform(-2, 2)
        node.position.y += random.uniform(-2, 2)
        node.position.z += random.uniform(-2, 2)
        void.add(node)
        print(f"   + Added: {EtherBridge.interpret_node(node)} at {node.position.x:.1f}, {node.position.y:.1f}")

    print("\n⏳ Running Physics Simulation (100 Steps)...")
    dt = 0.1
    for step in range(100):
        dynamics.step(void, dt)
        if step % 20 == 0:
            print(f"   [Step {step}] System Energy: {void.total_energy():.1f}")

    print("\n🔭 Final State Analysis:")
    nodes = void.get_all()

    # Check distances between similar concepts
    love = next(n for n in nodes if "Love" in str(n.content))
    joy = next(n for n in nodes if "Joy" in str(n.content))
    logic = next(n for n in nodes if "Logic" in str(n.content))

    dist_love_joy = ((love.position.x - joy.position.x)**2 + (love.position.y - joy.position.y)**2)**0.5
    dist_love_logic = ((love.position.x - logic.position.x)**2 + (love.position.y - logic.position.y)**2)**0.5

    print(f"   Distance (Love <-> Joy): {dist_love_joy:.2f} (Should be CLOSE)")
    print(f"   Distance (Love <-> Logic): {dist_love_logic:.2f} (Should be FAR)")

    if dist_love_joy < dist_love_logic:
        print("\n✅ SUCCESS: 'Meaning Gravity' automatically clustered similar concepts!")
    else:
        print("\n❌ FAILURE: Gravity did not work as expected.")

if __name__ == "__main__":
    run_simulation()
