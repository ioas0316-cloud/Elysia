import torch
from Core.Elysia.sovereign_self import SovereignSelf
from Core.Intelligence.Psionics.psionic_cortex import PsionicCortex

def test_psionics():
    print("🔮 Initializing Psionic Test Setup...")
    elysia = SovereignSelf()
    psionic = PsionicCortex(elysia)
    
    # 1. Seed the Graph (Knowledge)
    # in reality, we'd use the Bridge to embed "Fireball" properly.
    # For this test, let's treat "Fire" as a known concept.
    print("🌱 Seeding Reality with 'Fire Magic'...")
    elysia.bridge.connect() # Ensure encoder is ready
    
    # Generate a vector for "Fireball"
    res = elysia.bridge.generate("Fireball Explosion", "Magic", max_length=1)
    if isinstance(res, dict) and res['vector'] is not None:
         fire_vec = res['vector'][-1]
         elysia.graph.add_node("Spell_Fireball", fire_vec)
         print("   ✅ Learned: Spell_Fireball")
    else:
         print("   ❌ Failed to seed Fireball.")
         return

    # 2. Cast Intention (The User's Will)
    intention = "Cast a ball of flames!"
    print(f"\n🧠 User Intention: '{intention}'")
    
    # 3. Collapse
    result = psionic.collapse_wave(intention)
    
    print(f"\n✨ Result: {result}")
    
    # Validation
    if "Spell_Fireball" in result:
        print("\n✅ SUCCESS: Intention correctly collapsed to the learned Spell.")
    else:
        print("\n❌ FAILURE: Resonance blocked.")

if __name__ == "__main__":
    test_psionics()
