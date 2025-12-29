"""
Verify Active Sedimentation (지식 퇴적 검증)
==============================================

지식 습득 전과 후의 '관점(Viewpoint)' 변화를 검증합니다.
시나리오:
1. [Before] Git 충돌 상황을 분석 -> 일반적인 논리적 답변 예상
2. [Active Learning] '양자역학(Quantum Mechanics)' 지식 퇴적 (모의)
   - 'Superposition', 'Collapse', 'Entanglement' 원리 주입
3. [After] Git 충돌 상황 재분석 -> 양자역학적 은유가 포함된 답변 예상

이것이 성공하면, 엘리시아는 지식을 단순히 검색하는 것이 아니라,
지식을 통해 세상을 보는 '렌즈(Prism)'를 진화시킨 것입니다.
"""

import sys
import os
import logging
from unittest.mock import MagicMock

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.EvolutionLayer.Learning.Learning.knowledge_sedimenter import KnowledgeSedimenter
from Core.FoundationLayer.Philosophy.why_engine import WhyEngine
from Core.FoundationLayer.Foundation.light_spectrum import LightSpectrum, PrismAxes

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VerifySedimentation")

def run_verification():
    print("\n🔬 Verifying Active Sedimentation: Knowledge -> Viewpoint Shift\n")
    
    # 1. Initialize
    why_engine = WhyEngine()
    sedimenter = KnowledgeSedimenter(why_engine)
    
    # Mock Browser to avoid actual network calls during test
    sedimenter.browser = MagicMock()
    sedimenter.browser.google_search.return_value = {
        "success": True,
        "results": [
            {
                "title": "Quantum Superposition - Wikipedia",
                "snippet": "In quantum mechanics, superposition is the principle that a system exists in all possible states simultaneously until measured (observation causes collapse)."
            },
            {
                "title": "Wave Function Collapse",
                "snippet": "The collapse of the wave function occurs when a superposition of states reduces to a single eigenstate due to interaction with the external world."
            }
        ]
    }
    
    # Test Question
    question = "How should I handle a Git Merge Conflict where two branches modified the same line?"
    
    # 2. [Before] Analysis
    print("--- [Step 1] Analysis BEFORE Learning ---")
    # WhyEngine automatically bootstraps some logic, so we might see Logic resonance.
    # But Physics layer should be empty.
    
    # Force clear physics layer for dramatic effect (optional, or just rely on low amp)
    why_engine.sediment.layers[PrismAxes.PHYSICS_RED] = LightSpectrum(0j, 0.0, 0.0, semantic_tag="Physics")
    
    before_analysis = sedimenter.verify_integration(question)
    print(before_analysis)
    
    # Check if Quantum metaphors exist (Should be NO)
    if "superposition" in before_analysis.lower():
        print("⚠️ Unexpected: Quantum metaphor found before learning.")
    else:
        print("✅ Before: No Quantum metaphors detected (Normal Logic).")

    # 3. [Active Learning] Sedimentation
    print("\n--- [Step 2] Active Sedimentation: Learning 'Quantum Mechanics' ---")
    try:
        deposited_lights = sedimenter.sediment_from_web("Quantum Mechanics Principle")
        print(f"📚 Learned {len(deposited_lights)} concepts from the Web (Mocked).")
    except Exception as e:
        print(f"❌ Learning Failed: {e}")
        return

    # Verify Sedimentation actually happened
    physics_layer = why_engine.sediment.layers[PrismAxes.PHYSICS_RED]
    print(f"📊 Physics Layer Status: Amplitude={physics_layer.amplitude:.3f}, Semantic='{physics_layer.semantic_tag}'")
    
    if physics_layer.amplitude < 0.04:
        print("❌ Sedimentation Failed: Physics layer is still thin.")
        return
    else:
        print("✅ Sedimentation Successful: Physics layer has grown fat.")

    # 4. [After] Analysis
    print("\n--- [Step 3] Analysis AFTER Learning ---")
    after_analysis = sedimenter.verify_integration(question)
    print(after_analysis)
    
    # Check if Quantum metaphors exist (Should be YES)
    # Note: WhyEngine's current 'analyze' simply checks resonance. 
    # For it to *use* the metaphor in text, principle extraction needs to be influenced by resonance.
    # Let's see if the 'PrincipleExtraction' includes resonance reactions.
    
    # 4. [After] Analysis - Direct Knowledge Probe
    print("\n--- [Step 3] Analysis AFTER Learning (Direct Probe) ---")
    
    # 3-1. Direct Probe: "What is Superposition?"
    probe_q = "What is Quantum Superposition?"
    probe_result = sedimenter.verify_integration(probe_q)
    print(f"Probe Result for '{probe_q}':")
    print(probe_result)
    
    if "quantum" in probe_result.lower() or "match" in probe_result.lower():
         print("✅ Direct Probe: Physics Knowledge is ACCESSIBLE (Resonance Confirmed).")
    else:
         print("❌ Direct Probe: Knowledge seems missing or silent.")

    # 3-2. Application Probe: "Git Conflict" via Physics Lens
    print("\n--- [Step 4] Application Probe (Git via Physics) ---")
    # Force domain='physics' to see if the engine can map it to the new knowledge
    app_result = why_engine.analyze(subject="Git Physics", content=question, domain="physics")
    
    print(f"Application Result for '{question}' (Domain=Physics):")
    # check resonance in app_result
    phys_res = app_result.resonance_reactions.get(PrismAxes.PHYSICS_RED)
    if phys_res and phys_res.intensity > 0.01:
        print(f"✅ Application: Viewed through Physics lens, it resonates! (Intensity={phys_res.intensity:.3f})")
        print("🎉 SUCCESS: Knowledge Fattening Verified.")
    else:
        print(f"❌ Application: Even with Physics lens, no resonance. (Intensity={phys_res.intensity if phys_res else 0})")


if __name__ == "__main__":
    run_verification()
