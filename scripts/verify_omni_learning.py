"""
Verify Omni-Learning (The Universal Exam)
=========================================
Tests Elysia's ability to internalize principles from diverse domains.
We act as the 'Teacher', presenting concepts.
We verify if she 'Resonates' correctly (e.g. Math -> Logic Frequency).
"""

import sys
import os
import logging
from pathlib import Path
import time

# Add root to path
sys.path.insert(0, os.getcwd())

from elysia_core import Organ
from Core.Memory.unified_experience_core import get_experience_core, UnifiedExperienceCore

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("OmniExam")

def test_curriculum():
    print("="*60)
    print("🎓 UNIVERSAL LEARNING EXAM (OMNI-TEST)")
    print("   Target: UnifiedExperienceCore (The Hippocampus)")
    print("="*60)
    
    # 1. Init Student
    core = get_experience_core()
    
    # Reset State for clean test
    core.current_state = {k: 0.5 for k in core.aspect_frequencies}
    
    curriculum = [
        {
            "domain": "MATH", 
            "content": "Pythagoras Theorem: A squared plus B squared equals C squared. Pure logic.",
            "type": "thought",
            "expected_boost": "engineer", # Logic/Structure
            "feedback": 1.0
        },
        {
            "domain": "HISTORY", 
            "content": "The Fall of Rome was caused by internal entropy and external pressure.",
            "type": "thought",
            "expected_boost": "philosophical", # Wisdom/Causality
            "feedback": 1.0
        },
        {
            "domain": "LANGUAGE", 
            "content": "Hangul letter 'ㄱ' sounds like 'G'. It depicts the shape of the tongue.",
            "type": "perception",
            "expected_boost": "creative", # Arts/Symbolism
            "feedback": 1.0
        },
        {
            "domain": "PHYSICS", 
            "content": "Gravity is the curvature of spacetime caused by mass.",
            "type": "thought",
            "expected_boost": "engineer", # Logic/Physics
            "feedback": 1.0
        },
        {
            "domain": "EMOTION", 
            "content": "I feel a deep resonance with the user today. We are connected.",
            "type": "emotion",
            "expected_boost": "emotional", # Love/Connection
            "feedback": 1.0
        }
    ]
    
    score_card = []
    
    for lesson in curriculum:
        print(f"\n📚 CLASS: {lesson['domain']}")
        print(f"   Input: \"{lesson['content']}\"")
        
        # 1. Absorb
        result = core.absorb(
            content=lesson['content'],
            type=lesson['type'],
            context={"domain": lesson['domain']},
            feedback=lesson['feedback']
        )
        
        # 2. Analyze Reaction
        wave_shift = result['wave_shift']
        
        # Check if the expected aspect was boosted
        target = lesson['expected_boost']
        boost = wave_shift.get(target, 0.0)
        
        print(f"   🧠 Wave Shift: {wave_shift}")
        
        if boost > 0:
            print(f"   ✅ SUCCESS: Resonated with '{target}' (+{boost})")
            score_card.append(True)
        else:
            print(f"   ❌ FAIL: Did not resonate with '{target}'")
            score_card.append(False)
            
        time.sleep(0.5)

    print("\n" + "="*60)
    print("📝 FINAL REPORT CARD")
    print("="*60)
    
    passed = score_card.count(True)
    total = len(score_card)
    
    print(f"Internalization Score: {passed}/{total}")
    
    if passed == total:
        print("\n🏆 RESULT: UNIVERSAL LEARNER CONFIRMED.")
        print("   She can differentiate domains and internally restructure her frequency.")
    else:
        print("\n⚠️ RESULT: PARTIAL LEARNING.")
        
if __name__ == "__main__":
    test_curriculum()
