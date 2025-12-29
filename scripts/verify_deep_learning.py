
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.EvolutionLayer.Learning.Learning.autonomous_learner import AutonomousLearner
from Core.EvolutionLayer.Learning.Learning.hierarchical_learning import HierarchicalKnowledgeGraph, Domain

def verify_deep_learning():
    print("🔬 Verifying Deep Knowledge Resonance...")
    
    learner = AutonomousLearner()
    kg = HierarchicalKnowledgeGraph("data/test_deep_learning.json")
    
    # 1. Experience something mathematical
    print("\n[1] Experiencing Mathematics (Equation)...")
    content = "오일러의 등식 e^(i*pi) + 1 = 0 은 수학에서 가장 아름다운 등식으로 불린다. 5가지 상수가 완벽한 균형을 이룬다."
    
    # We cheat a bit and inject the kg into learner for this test if needed, 
    # but autonomous_learner instantiates its own. 
    # Let's just run experience and check the file later.
    
    result = learner.experience(
        content=content,
        subject="오일러 등식",
        domain="mathematics"
    )
    
    print(f"   Result: {result['knowledge_state']}")
    print(f"   Principle: {result.get('learned_concept')}")
    
    # 2. Check storage
    # Note: AutonomousLearner uses default path, so we might need to check the default file or modify AutonomousLearner to accept path.
    # For now, let's just check the analysis result first.
    
    # Let's test WhyEngine directly for the Physics case to see the principle extraction
    print("\n[2] Testing WhyEngine Physics Analysis...")
    why_engine = learner.why_engine
    p_content = "에너지 보존 법칙에 따르면 에너지는 생성되거나 소멸되지 않고 형태만 바뀐다. 닫힌 계의 에너지는 일정하다."
    analysis = why_engine.analyze("에너지 보존", p_content, domain="physics")
    
    print(f"   Principle: {analysis.underlying_principle}")
    print(f"   Wave Signature: {analysis.wave_signature}")
    
    if "불변의 원리" in analysis.underlying_principle:
        print("   ✅ Physics Principle extracted correctly")
    else:
        print("   ❌ Physics Principle extraction failed")
        
    if analysis.wave_signature:
        print("   ✅ Wave Signature generated")
    else:
        print("   ❌ Wave Signature missing")

if __name__ == "__main__":
    verify_deep_learning()
