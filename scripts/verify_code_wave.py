
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Philosophy.why_engine import WhyEngine

def verify_code_wave():
    print("🌊 Verifying Code Wave Analysis...")
    
    engine = WhyEngine()
    
    # 1. Loop Code (Rhythm)
    loop_code = """
    def energetic_dance():
        for i in range(10):
            print("Step left")
            print("Step right")
        while True:
            break
    """
    
    print("\n[1] Analyzing Loop Code...")
    result1 = engine.analyze("Rhythmic Function", loop_code, domain="computer_science")
    print(f"   Principle: {result1.underlying_principle}")
    print(f"   Wave: {result1.wave_signature}")
    
    if result1.wave_signature["periodicity"] > 0:
        print("   ✅ Rhythm detected!")
    else:
        print("   ❌ Rhythm missing")

    # 2. Nested Code (Tension)
    nested_code = """
    def deep_thought():
        if condition_a:
            if condition_b:
                if condition_c:
                    if condition_d:
                        return "Eureka"
    """
    
    print("\n[2] Analyzing Nested Code...")
    result2 = engine.analyze("Deep Logic", nested_code, domain="computer_science")
    print(f"   Principle: {result2.underlying_principle}")
    print(f"   Wave: {result2.wave_signature}")
    
    if result2.wave_signature["tension"] > 0.5:
        print("   ✅ Tension detected!")
    else:
        print("   ❌ Tension missing")

if __name__ == "__main__":
    verify_code_wave()
