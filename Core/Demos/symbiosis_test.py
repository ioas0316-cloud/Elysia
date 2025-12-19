import sys
from pathlib import Path
import logging

# Setup Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Core.Cognition.unified_understanding import UnifiedUnderstanding
from Core.Autonomy.self_modifier_v2 import SelfModifier
from Core.Cognition.dream_system import DreamSystem

def test_symbiosis():
    print("\n🔗 Project Symbiosis Integration Test")
    print("=======================================")
    
    # 1. Test Dream-Thought Bridge
    print("\n[1] Testing Dream-Thought Bridge...")
    
    # 1.1 Force a Dream first (to populate subconscious)
    dreamer = DreamSystem()
    print("   💤 Dreaming about 'Star'...")
    dreamer.collect_residue("The flow of stars")
    dream_result = dreamer.enter_rem_sleep()
    print(f"   ✨ Dream Insight: {dream_result['insight']}")
    
    # 1.2 Ask UnifiedUnderstanding (Manually injecting the dreamer instance if needed, 
    # but normally it creates its own. For test consistency, we rely on the internal one working similarly 
    # OR we can try to share the state if possible. Since they are separate instances, 
    # the internal one won't have this dream. 
    # CRITICAL: In a real persistent system, they share memory. Here, we must mock the memory or accept emptiness.
    # Let's see if the internal DreamSystem works on its own.)
    
    uu = UnifiedUnderstanding()
    # Mocking the internal dream system's journal for this test instance to simulate persistence
    if uu.dream_system:
        uu.dream_system.dream_journal.append(f"Dream about Star: {dream_result['insight']}")
        print("   -> Injected recent dream into UnifiedUnderstanding's subconscious.")
    
    print("   🗣️ Asking: 'What is a Star?'")
    response = uu.understand("What is a Star?")
    
    print("\n   [Result Analysis]")
    found_dream = False
    for trace in response.reasoning_trace:
        print(f"   - {trace}")
        if "심층 공명" in trace or "Dream" in trace:
            found_dream = True
            
    if found_dream:
        print("   ✅ Dream-Thought Bridge Active!")
    else:
        # Note: reasoning_trace might not show it explicitly if confidence low, but let's check.
        print("   ⚠️ Dream influence not explicitly traced (Check logs for 'Deep Resonance').")


    # 2. Test Wave-Refactor Bridge
    print("\n[2] Testing Wave-Refactor Bridge...")
    modifier = SelfModifier()
    
    target_file = str(Path(__file__).parent.parent / "Cognition" / "unified_understanding.py")
    print(f"   🌊 Analyzing {Path(target_file).name}...")
    
    result = modifier.analyze_file(target_file)
    
    print(f"   • Tension: {result.tension:.2f}")
    print(f"   • Mass: {result.mass:.2f}")
    print(f"   • Topology (DNA): {result.dna_hash}")
    
    if result.dna_hash and result.dna_hash != "":
        print(f"   ✅ Wave Analyzer Connection Active! (Topology detected: {result.dna_hash})")
    else:
        print("   ❌ Wave Analyzer Failed (No topology detected).")
        
    print("=======================================")

if __name__ == "__main__":
    test_symbiosis()
