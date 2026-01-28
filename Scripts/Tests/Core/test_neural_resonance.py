import torch
from Core.L5_Mental.Reasoning_Core.LLM.huggingface_bridge import SovereignBridge

def test_resonance():
    print("🧠 Initializing Neural Bridge...")
    bridge = SovereignBridge()
    if not bridge.connect():
        print("Skipping test: No model.")
        return

    concepts = ["Order", "Chaos", "Love", "Void"]
    results = {}

    print("\n⚡ Measuring Neural Potentials...")
    for c in concepts:
        res = bridge.generate(f"Define {c}.", "You are a physicist.")
        vec = res['vector']
        
        if vec is not None:
            energy = float(torch.norm(vec))
            results[c] = energy
            print(f"   [{c.upper()}] Energy: {energy:.4f}")
        else:
            print(f"   [{c.upper()}] No vector returned.")

    print("\n📊 Resonance Analysis:")
    # Check if they are different (which implies structure is being captured)
    energies = list(results.values())
    if len(energies) > 1:
        variance = torch.var(torch.tensor(energies))
        print(f"   Concept Variance: {variance:.4f}")
        if variance > 0.01:
            print("✅ SUCCESS: Concepts have distinct neural signatures.")
        else:
            print("⚠️ WARNING: Concepts are indistinguishable.")

if __name__ == "__main__":
    test_resonance()
