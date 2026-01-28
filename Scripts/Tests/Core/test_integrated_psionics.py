import torch
from Core.Elysia.sovereign_self import SovereignSelf

def test_integration():
    print("🔮 Initializing Integrated System...")
    elysia = EmergentSelf()
    
    # 1. Seed Reality (So we have something to resonate with)
    print("🌱 Seeding Reality...")
    elysia.bridge.connect()
    res = elysia.bridge.generate("Fireball Explosion", "Magic", max_length=1)
    if isinstance(res, dict) and res['vector'] is not None:
         # Add multiple nodes to test Multi-Rotor
         fire_vec = res['vector'][-1]
         elysia.graph.add_node("Spell_Fireball", fire_vec)
         elysia.graph.add_node("Element_Heat", fire_vec * 0.9) # Close neighbor
         elysia.graph.add_node("Concept_Burn", fire_vec * 0.8) # Context neighbor
         print("   ✅ Learned: Fireball, Heat, Burn")

    # 2. Trigger via Manifest Intent (The User Command)
    intent = "/wave Cast a ball of flames!"
    print(f"\n🗣️ User S says: '{intent}'")
    
    # [Pre-load Architect for this test]
    elysia.projector.load_architect()
    
    result = elysia.manifest_intent(intent)

    
    print(f"\n✨ System Returned: {result}")
    
    if "Reality" in result and "Fireball" in result:
        print("\n✅ SUCCESS: Psionic Trigger Worked!")
    else:
        print("\n❌ FAILURE: Integration broken.")

if __name__ == "__main__":
    test_integration()
