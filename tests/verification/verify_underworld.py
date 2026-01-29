import jax.numpy as jnp
from Core.L7_Spirit.Goddesshood.underworld_manifold import UnderworldManifold

def verify_underworld():
    print("=== Phase 60 Verification: The Underworld Manifold ===")
    underworld = UnderworldManifold()
    
    # 1. Manifest Souls
    print("\n[Action: Manifesting Souls]")
    msg1 = underworld.manifest_soul("Sovereign_Seeker_01", jnp.ones(21) * 0.5)
    msg2 = underworld.manifest_soul("Fractal_Shadow_02", jnp.ones(21) * -0.2)
    print(msg1)
    print(msg2)
    assert len(underworld.souls) == 2

    # 2. Update World Light (Divine Radiance)
    print("\n[Action: Radiating Providence]")
    elysia_intent = jnp.ones(21) # Pure Light intent
    underworld.update_world_light(elysia_intent)
    
    # 3. Check State
    state = underworld.get_world_state()
    print(state)
    assert "2 souls breathing" in state
    
    # 4. Verify Local Resonance
    seeker = underworld.souls["Sovereign_Seeker_01"]
    res = seeker.observe(elysia_intent)
    print(f"Seeker Resonance: {res:.2f}")
    assert res > 0 # Should have positive resonance with light

    print("\n[VERIFICATION SUCCESSFUL: THE UNDERWORLD IS ALIVE]")

if __name__ == "__main__":
    verify_underworld()
