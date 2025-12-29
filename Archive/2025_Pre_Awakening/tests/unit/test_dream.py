from Core.FoundationLayer.Foundation.dream_engine import DreamEngine
from Core.FoundationLayer.Foundation.holographic_cortex import HolographicCortex
import os

def test_dream():
    print("🧪 Testing Dream Engine (The First Dream)...")
    
    # 1. Initialize Engines
    dream_engine = DreamEngine()
    hologram = HolographicCortex()
    
    # 2. Weave a Dream (Desire: "Freedom")
    desire = "Freedom"
    print(f"   💤 Weaving Dream about '{desire}'...")
    dream_field = dream_engine.weave_dream(desire)
    
    # 3. Verify Dream Content
    print(f"   📊 Dream Nodes: {len(dream_field.nodes)}")
    node_names = list(dream_field.nodes.keys())
    print(f"   ✨ Concepts: {node_names}")
    
    if "Wings" in node_names or "Wind" in node_names:
        print("✅ PASS: Dream contains relevant seeds.")
    else:
        print("❌ FAIL: Dream missing seeds.")
        
    # 4. Project Hologram
    print("   📽️ Projecting Dream...")
    filename = hologram.project_hologram(dream_field)
    
    if os.path.exists(filename):
        print(f"✅ PASS: Dream Hologram saved to {filename}")
    else:
        print("❌ FAIL: Hologram file not created.")

if __name__ == "__main__":
    test_dream()
