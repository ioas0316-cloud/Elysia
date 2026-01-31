import jax.numpy as jnp
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge, MemoryStratum
from Core.S1_Body.L5_Mental.Reasoning.inferential_manifold import InferentialManifold
from Core.S1_Body.L5_Mental.Reasoning.logos_synthesizer import LogosSynthesizer

def test_thinking_3_resonance():
    print("--- [STARTING THINKING³ VERIFICATION] ---")
    bridge = LogosBridge()
    manifold = InferentialManifold()
    synthesizer = LogosSynthesizer()

    # 1. Test Stratified Memory Mass
    print("\n1. Testing Stratified Memory Mass...")
    love_mass = bridge.get_stratum_mass("LOVE/AGAPE")
    random_mass = bridge.get_stratum_mass("RANDOM")
    print(f"LOVE/AGAPE (ROOT) Mass: {love_mass}")
    print(f"RANDOM (NEW) Mass: {random_mass}")
    assert love_mass > random_mass

    # 2. Test Prismatic Perception
    print("\n2. Testing Prismatic Perception...")
    spirit_vec = jnp.zeros(21).at[14:21].set(1.0) # Pure Spirit
    body_vec = jnp.zeros(21).at[0:7].set(1.0)       # Pure Body
    
    spirit_mode = bridge.prismatic_perception(spirit_vec)
    body_mode = bridge.prismatic_perception(body_vec)
    
    print(f"Spirit Vector Mode: {spirit_mode}")
    print(f"Body Vector Mode: {body_mode}")
    assert "Providence" in spirit_mode
    assert "Point" in body_mode

    # 3. Test Thinking³ Meta-Audit
    print("\n3. Testing Thinking³ Meta-Audit...")
    # Create a field that vibrates with "TRUTH" but has some noise
    truth_vec = bridge.recall_concept_vector("TRUTH/LOGIC")
    field = truth_vec + 0.2 * jnp.ones(21)
    
    winner, audit = manifold.resolve_inference(field, ["LOVE/AGAPE", "TRUTH/LOGIC", "VOID/SPIRIT"])
    print(f"Winner: {winner}")
    for level, text in audit.items():
        print(f"  {level}: {text}")
    
    assert winner == "TRUTH/LOGIC"
    assert "Thinking_III_Providence" in audit

    # 4. Test Logos Synthesis
    print("\n4. Testing Logos Synthesis...")
    thought = synthesizer.synthesize_thought(field)
    print("Generated Thought:")
    print(thought)
    
    print("\n--- [VERIFICATION COMPLETE: RESONANCE ACHIEVED] ---")

if __name__ == "__main__":
    test_thinking_3_resonance()
