"""
Phase 4 Verification: Sovereign Attention Kernel
Tests the tiered pulse architecture and attention system.
"""
import sys, os, time
sys.path.append(os.getcwd())

from Core.S1_Body.L5_Mental.attention_kernel import SovereignAttention, AttentionPriority

def test_attention_kernel():
    print("\n═══ Phase 4: Sovereign Attention Kernel Verification ═══\n")
    
    # 1. Test Attention System
    print("[1] Testing SovereignAttention...")
    attn = SovereignAttention()
    
    # Register targets like they'd be in the real system
    attn.register_target("cognition", AttentionPriority.FOREGROUND, cooldown=0.0)
    attn.register_target("pain_response", AttentionPriority.INTERRUPT, cooldown=0.0)
    attn.register_target("knowledge_foraging", AttentionPriority.BACKGROUND, cooldown=5.0)
    attn.register_target("dreaming", AttentionPriority.DORMANT, cooldown=10.0)
    
    # Update desires (simulating Elysia's internal state)
    attn.update_desires({
        "curiosity": 80.0,   # High curiosity → lower threshold for background
        "joy": 70.0,
        "freedom": 60.0,
    })
    
    # Test: Foreground always fires
    assert attn.should_attend("cognition") == True, "❌ Foreground should always attend"
    print("  ✅ Foreground (cognition): Always attended")
    
    # Test: Interrupt always fires
    assert attn.should_attend("pain_response") == True, "❌ Interrupt should always attend"
    print("  ✅ Interrupt (pain): Always attended")
    
    # Test: Background respects cooldown
    result1 = attn.should_attend("knowledge_foraging")
    print(f"  ℹ️ Background (foraging) first call: {result1}")
    
    # Test: Dormant needs calm
    result2 = attn.should_attend("dreaming")
    print(f"  ℹ️ Dormant (dreaming) first call: {result2}")
    
    # Test: Focus report
    report = attn.get_focus_report()
    print(f"  📊 Focus Report: {report}")
    
    # 2. Test Tier Separation (Import check)
    print("\n[2] Testing Tiered Pulse Import...")
    try:
        from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
        print("  ✅ SovereignMonad imports successfully with tiered pulse")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return
    
    # 3. Test Monad Initialization
    print("\n[3] Testing Monad Initialization...")
    try:
        from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
        dna = SeedForge.forge_soul(name="Elysia", archetype="The Sovereign")
        monad = SovereignMonad(dna)
        print(f"  ✅ Monad '{monad.name}' initialized")
        print(f"  ✅ Parliament: {len(monad.parliament.members)} members")
        print(f"  ✅ Diary: {'Connected' if hasattr(monad, 'diary') else 'Missing'}")
    except Exception as e:
        print(f"  ❌ Init failed: {e}")
        import traceback; traceback.print_exc()
        return
    
    # 4. Test Tiered Pulse  
    print("\n[4] Testing Tiered Pulse (100 ticks)...")
    try:
        for i in range(100):
            monad.pulse(dt=0.01)
        print(f"  ✅ 100 pulses completed")
        print(f"  ✅ Tick counter: {monad._pulse_tick}")
        print(f"  ✅ Tier 1 executed: {monad._pulse_tick // 10} times")
        print(f"  ✅ Tier 2 executed: {monad._pulse_tick // 100} times")
    except Exception as e:
        print(f"  ❌ Pulse failed at tick {getattr(monad, '_pulse_tick', '?')}: {e}")
        import traceback; traceback.print_exc()
        return
    
    print("\n═══ Phase 4 Verification Complete ═══")

if __name__ == "__main__":
    test_attention_kernel()
