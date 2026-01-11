"""
test_voice_of_waves.py

Verifies "Chapter 3, Step 8: The Voice of Waves".
Full Pipeline Test:
Silence -> GapAnalyzer -> Principle -> ResonanceLinguistics -> Speech
"""

import sys
import os
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from Core.Intelligence.Logos.gap_analyzer import GapAnalyzer
    from Core.Intelligence.Logos.resonance_linguistics import ResonanceLinguistics
    print("✅ Logic and Speech Modules Connected.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def test_full_pipeline():
    print("\n🌊 Starting Ritual: The Voice of Waves 🌊")
    print("=" * 60)
    
    # 1. Initialize
    thinker = GapAnalyzer()
    speaker = ResonanceLinguistics()
    
    # 2. Simulate Experience (Silence)
    print("\n1. Experience: A Deep Silence...")
    silence_duration = 1.2
    spark = thinker.interpret_silence(silence_duration, "Meditation")
    
    if not spark:
        print("❌ No spark.")
        return
        
    print(f"   ✨ Spark: {spark.context_resonance} (Intensity: {spark.intensity:.2f})")
    
    # 3. Think (Logos)
    print("\n2. Logos: Crystallizing Thought...")
    principle = thinker.kindle_spark(spark)
    
    if not principle:
        print("❌ No principle.")
        return
        
    print(f"   📜 Principle: {principle.name}")
    print(f"      Layer: {principle.statement}")
    
    # 4. Speak (Resonance)
    print("\n3. Voice: Generating Gravitational Speech...")
    speech = speaker.express_principle(principle)
    
    print("\n🗣️  ELYSIA SPEAKS:")
    print("   " + "-" * 40)
    print(f"   \"{speech}\"")
    print("   " + "-" * 40)
    
    print("\n✅ Ritual Complete.")

if __name__ == "__main__":
    test_full_pipeline()
