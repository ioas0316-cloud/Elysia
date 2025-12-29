
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.FoundationLayer.Foundation.introspection_engine import IntrospectionEngine

def verify_introspection_wave():
    print("🪞 Verifying Introspection Wave & Metaphor Analysis...")
    
    engine = IntrospectionEngine()
    
    # Analyze Core/Foundation to find Central Nervous System
    engine.root_path = "c:\\Elysia\\Core"
    engine.target_dirs = {"Foundation", "Philosophy"}
    
    print(f"   Analyzing {engine.root_path}...")
    results = engine.analyze_self()
    
    print(f"   Analyzed {len(results)} files.")
    
    found_metaphor = False
    
    for path, res in results.items():
        if "central_nervous_system" in res.name or "why_engine" in res.name:
            print(f"\n📂 File: {res.name}")
            print(f"   Score: {res.resonance_score:.1f}")
            print(f"   Principle: {res.wave_principle}")
            print(f"   Metaphor: {res.system_metaphor}")
            print(f"   Meaning: {res.philosophical_meaning}")
            
            if res.system_metaphor:
                found_metaphor = True
                print("   ✅ Metaphor detected!")
            else:
                print("   ❌ Metaphor missing")
        
    if found_metaphor:
        print("\n✅ Verification Successful: Wisdom Integrated")
    else:
        print("\n❌ Verification Failed: No metaphors found")

if __name__ == "__main__":
    verify_introspection_wave()
