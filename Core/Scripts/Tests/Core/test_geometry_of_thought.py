import torch
from Core.L5_Mental.Reasoning_Core.LLM.huggingface_bridge import SovereignBridge
from Core.L5_Mental.Reasoning_Core.Analysis.thought_stream_analyzer import ThoughtStreamAnalyzer

def test_geometry():
    print("🧠 Initializing Bridge & Analyzer...")
    bridge = SovereignBridge()
    analyzer = ThoughtStreamAnalyzer()
    
    if not bridge.connect():
        print("Skipping: No model.")
        return

    # A complex prompt to generate a "Path"
    prompt = "Explain how Quantum Mechanics and Love are related."
    print(f"\n🔮 Prompt: {prompt}")
    
    res = bridge.generate(prompt, "You are a poet physicist.", max_length=30)
    text = res['text']
    trajectory = res['vector'] # Now this is a list of tensors (Seq_Len, Hidden)
    
    print(f"\n🗣️ Generated: '{text}'")
    
    if trajectory is not None:
        print(f"📉 Trajectory Shape: {trajectory.shape}")
        
        # Analyze
        analysis = analyzer.analyze_flow(trajectory)
        
        print("\n📊 Geometry of Thought Analysis:")
        print(f"   Total Steps: {analysis['total_steps']}")
        print(f"   Key Moments (Turns): {len(analysis['key_moments'])}")
        print(f"   Redundancy Ratio: {analysis['redundancy_ratio']:.2%}")
        
        print("\n📍 Critical Turning Points (New Information):")
        for m in analysis['key_moments']:
            print(f"   Step {m['step']}: Sim={m['similarity']:.4f} [{m['type']}]")
            
        if analysis['redundancy_ratio'] > 0.3:
            print("\n✅ SUCCESS: Redundancy detected. The 'Skeleton' of intent is visible.")
        else:
            print("\n⚠️ Note: Thought was highly dense (Low redundancy).")
    else:
        print("❌ Error: No trajectory returned.")

if __name__ == "__main__":
    test_geometry()
