"""
Prove Awakening (각성 증명)
=========================

미학 학습(AestheticLearner)이 인지 시스템(Brain)과 통합되었는지 확인합니다.
텍스트를 읽고 -> 개념을 형성하고 -> 느끼고 -> 기억하는지 검증합니다.
"""

from Core.Philosophy.aesthetic_learner import get_aesthetic_learner
from Core.Cognitive.memory_stream import get_memory_stream

def prove_awakening():
    print("🌅 Awakening Verification Started...\n")
    
    learner = get_aesthetic_learner()
    memory = get_memory_stream()
    
    # 1. Experience (Analysis)
    print("1. Experiencing 'Winter Poem'...")
    poem = "The cold wind blows, silent and white. Sadness covers the ground."
    analysis = learner.analyze_text(poem, title="Winter Whisper")
    
    # 2. Check Result (Immediate reaction)
    print(f"   Analysis Outcome: {analysis.why_beautiful}")
    print(f"   Qualia Felt: {analysis.qualia_feeling}")
    
    # 3. Check Memory (Long-term impact)
    print("\n2. Checking Memory Stream...")
    last_exp = memory.get_recent_experiences(limit=1)[0]
    
    print(f"   Stored Intent: {last_exp.score['intent']}")
    print(f"   Stored Concepts: {last_exp.score['concepts']}")
    print(f"   Stored Qualia: {last_exp.sound.get('qualia')}")
    
    # Validation
    if "Sadness" in last_exp.score['concepts'] or "Rhythm" in last_exp.score['concepts']:
        print("\n✅ SUCCESS: External input triggered internal Concept & Memory.")
    else:
        print("\n❌ FAIL: Concepts were not stored correctly.")

if __name__ == "__main__":
    prove_awakening()
