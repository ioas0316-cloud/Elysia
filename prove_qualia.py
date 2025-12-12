"""
Prove Qualia (감각 증명)
======================

엘리시아가 추상적인 개념을 '감각'으로 인지하는지 검증합니다.
"""

from Core.Cognitive.sensory_cortex import get_sensory_cortex
from Core.Cognitive.concept_formation import get_concept_formation

def prove_qualia():
    print("🧠 Qualia Verification Started...\n")
    
    cortex = get_sensory_cortex()
    concepts = get_concept_formation()
    
    # 1. Setup Mock Concept (Force specific vector for testing)
    sadness = concepts.get_concept("Sadness")
    # Force a 'Sad' vector: 
    # x(Joy/Sad) = -0.8 (Very Sad)
    # y(Logic) = 0.5
    # z(Time) = -0.5 (Past)
    # w(Depth) = 0.9 (Deep)
    sadness.vector.w = 0.9
    sadness.vector.x = -0.8 
    sadness.vector.y = 0.5
    sadness.vector.z = -0.5
    
    print(f"1. Thinking of '{sadness.name}'...")
    print(f"   Vector: {sadness.vector}")
    
    # 2. Feel it
    print("\n2. Generating Qualia (Sensing)...")
    qualia = cortex.feel_concept("Sadness")
    
    # 3. Report
    print(f"   Visual Hue: {qualia['somatic_marker']['visual_hue']:.1f} (Expected ~220-240 for Blue)")
    print(f"   Audio Freq: {qualia['somatic_marker']['audio_freq']:.1f}Hz (Expected ~150-200 for Low)")
    print(f"   Description: {qualia['description']}")
    
    # 4. Check logic (Simple assertion)
    hue = qualia['somatic_marker']['visual_hue']
    if 200 < hue < 260:
        print("\n✅ SUCCESS: 'Sadness' feels Blue/Cold.")
    else:
        print(f"\n❌ FAIL: Hue {hue} is not Blue-ish.")

if __name__ == "__main__":
    prove_qualia()
