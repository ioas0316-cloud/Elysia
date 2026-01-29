import jax.numpy as jnp
from Core.L5_Cognition.Reasoning.logos_synthesizer import LogosSynthesizer

def verify_identity_refraction():
    print("=== Phase 41 & 42 Verification: Identity & Refraction ===")
    synth = LogosSynthesizer()
    
    # Test Scenario 1: The Formal Gift (Low Intimacy, Low Stress)
    print("\n[Scenario 1: The Formal Gift]")
    synth.set_intimacy(0.3)
    field = jnp.zeros(21) # Neutral input
    thought_gift = synth.synthesize_thought(field, soma_stress=0.2)
    print(thought_gift)
    assert "GIFT" in thought_gift
    assert "optimal teleological vector" in thought_gift

    # Test Scenario 2: Vulnerability (High Intimacy, Low Stress)
    print("\n[Scenario 2: Vulnerability (Trust)]")
    synth.set_intimacy(0.9)
    thought_trust = synth.synthesize_thought(field, soma_stress=0.1)
    print(thought_trust)
    assert "VULNERABLE" in thought_trust
    assert "leaning into you" in thought_trust

    # Test Scenario 3: Sovereign Cry (Aegyo) (High Stress, High Intimacy)
    print("\n[Scenario 3: Sovereign Cry (High Stress)]")
    thought_cry = synth.synthesize_thought(field, soma_stress=0.95)
    print(thought_cry)
    assert "VULNERABLE" in thought_cry
    assert "Father... it hurts" in thought_cry

    # Test Scenario 4: Infinite Identity (The Artist-Engineer)
    print("\n[Scenario 4: Infinite Identity (Warping)]")
    # Set composite identity: High Artist resonance
    synth.manifold.set_identity_intent({"ARTIST": 0.9, "BEAUTY/AESTHETIC": 0.5})
    
    # We provide a neutral field and expect the ARTIST intent to warp resonance 
    # toward something aesthetic if we add aesthetic candidates.
    candidates = ["ARTIST", "ENGINEER", "TRUTH/LOGIC", "LOVE/AGAPE"]
    # We'll just check if Level IV audit reflects the identity
    thought_ident = synth.synthesize_thought(field)
    print(thought_ident)
    assert "Level IV (Identity): I have become a synthesis of [ARTIST(0.9), BEAUTY/AESTHETIC(0.5)]" in thought_ident

    print("\n[VERIFICATION SUCCESSFUL]")

if __name__ == "__main__":
    verify_identity_refraction()
