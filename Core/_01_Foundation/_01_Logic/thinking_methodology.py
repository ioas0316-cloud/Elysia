"""
Thinking Methodology System
===========================

어휘 전에 사고 방법론 먼저!

- 연역법 (Deduction)
- 귀납법 (Induction)
- 변증법 (Dialectic)
- 귀추법 (Abduction)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ThinkingMethod:
    """사고 방법론"""
    name: str
    description: str
    pattern: str
    orientation: Quaternion  # 사고 방향성


class ThinkingMethodology:
    """
    사고 방법론 체계
    
    어휘 배우기 전에 '어떻게 생각하는가'를 배움!
    """
    
    def __init__(self):
        print("🧠 Initializing Thinking Methodology System...")
        
        # 기본 사고 방법론들
        self.methods = {
            # 연역법 (Deduction): 일반 → 특수
            "deduction": ThinkingMethod(
                name="연역법 (Deduction)",
                description="일반적 원리에서 특수한 결론 도출",
                pattern="All A are B. X is A. Therefore X is B.",
                orientation=Quaternion(1.0, 0.1, 0.9, 0.1)  # 논리적 (y축)
            ),
            
            # 귀납법 (Induction): 특수 → 일반
            "induction": ThinkingMethod(
                name="귀납법 (Induction)",
                description="특수한 사례들로부터 일반 원리 도출",
                pattern="X1, X2, X3 are B. Therefore all X are B.",
                orientation=Quaternion(1.0, 0.3, 0.8, 0.2)  # 논리적 + 직관적
            ),
            
            # 변증법 (Dialectic): 정 → 반 → 합
            "dialectic": ThinkingMethod(
                name="변증법 (Dialectic)",
                description="대립되는 개념의 충돌과 종합",
                pattern="Thesis + Antithesis → Synthesis",
                orientation=Quaternion(1.0, 0.5, 0.5, 0.7)  # 균형 + 윤리
            ),
            
            # 귀추법 (Abduction): 최선의 설명
            "abduction": ThinkingMethod(
                name="귀추법 (Abduction)",
                description="관찰로부터 최선의 설명 추론",
                pattern="X is observed. Y explains X best. Therefore Y.",
                orientation=Quaternion(1.0, 0.6, 0.7, 0.3)  # 직관 + 논리
            ),
            
            # 유추 (Analogy): 유사성 기반
            "analogy": ThinkingMethod(
                name="유추 (Analogy)",
                description="유사한 것으로부터 추론",
                pattern="A is like B. B has X. Therefore A might have X.",
                orientation=Quaternion(1.0, 0.7, 0.6, 0.2)  # 창의적
            ),
        }
        
        print(f"   ✓ Loaded {len(self.methods)} thinking methods")
        print()
        
        # 논리 패턴들
        self.logical_patterns = {
            "modus_ponens": "If P then Q. P. Therefore Q.",
            "modus_tollens": "If P then Q. Not Q. Therefore not P.",
            "syllogism": "All A are B. All B are C. Therefore all A are C.",
            "reductio": "Assume P. P leads to contradiction. Therefore not P.",
        }
        
        print(f"   ✓ Loaded {len(self.logical_patterns)} logical patterns")
        print()
    
    def learn_method(self, method_name: str):
        """사고 방법론 학습"""
        if method_name not in self.methods:
            print(f"⚠️ Unknown method: {method_name}")
            return
        
        method = self.methods[method_name]
        
        print(f"📚 Learning: {method.name}")
        print(f"   설명: {method.description}")
        print(f"   패턴: {method.pattern}")
        print(f"   사고 방향: {method.orientation}")
        print()
    
    def apply_deduction(self, premise1: str, premise2: str) -> str:
        """연역법 적용"""
        print("🔬 Applying Deduction:")
        print(f"   Premise 1: {premise1}")
        print(f"   Premise 2: {premise2}")
        
        # 간단한 연역 시뮬레이션
        conclusion = f"Therefore conclusion follows logically"
        print(f"   ✓ Conclusion: {conclusion}")
        print()
        
        return conclusion
    
    def apply_induction(self, observations: List[str]) -> str:
        """귀납법 적용"""
        print("🔍 Applying Induction:")
        for i, obs in enumerate(observations, 1):
            print(f"   Observation {i}: {obs}")
        
        # 패턴 찾기
        generalization = f"General pattern identified from {len(observations)} cases"
        print(f"   ✓ Generalization: {generalization}")
        print()
        
        return generalization
    
    def apply_dialectic(self, thesis: str, antithesis: str) -> str:
        """변증법 적용"""
        print("⚖️ Applying Dialectic:")
        print(f"   Thesis: {thesis}")
        print(f"   Antithesis: {antithesis}")
        
        # 종합
        synthesis = f"Synthesis: Integration of both perspectives"
        print(f"   ✓ Synthesis: {synthesis}")
        print()
        
        return synthesis
    
    def get_method_for_concept(self, concept: str) -> str:
        """개념에 적합한 사고 방법 추천"""
        
        # 간단한 휴리스틱
        if any(word in concept.lower() for word in ["all", "every", "must"]):
            return "deduction"
        elif any(word in concept.lower() for word in ["some", "many", "often"]):
            return "induction"
        elif any(word in concept.lower() for word in ["vs", "versus", "conflict"]):
            return "dialectic"
        else:
            return "abduction"
    
    def demonstrate_all_methods(self):
        """모든 사고 방법론 시연"""
        print("="*70)
        print("THINKING METHODOLOGY DEMONSTRATION")
        print("="*70)
        print()
        
        # 연역법
        print("1️⃣ DEDUCTION (연역법)")
        print("-" * 70)
        self.learn_method("deduction")
        self.apply_deduction(
            "All humans are mortal",
            "Socrates is human"
        )
        
        # 귀납법
        print("2️⃣ INDUCTION (귀납법)")
        print("-" * 70)
        self.learn_method("induction")
        self.apply_induction([
            "The sun rose today",
            "The sun rose yesterday",
            "The sun has risen every day in history"
        ])
        
        # 변증법
        print("3️⃣ DIALECTIC (변증법)")
        print("-" * 70)
        self.learn_method("dialectic")
        self.apply_dialectic(
            "Individual freedom is paramount",
            "Social responsibility is essential"
        )
        
        print("="*70)
        print("✅ THINKING METHODOLOGY SYSTEM OPERATIONAL")
        print("   사고 방법론 먼저, 그 다음 어휘!")
        print("="*70)


# 데모
if __name__ == "__main__":
    print("="*70)
    print("🧠 THINKING METHODOLOGY SYSTEM")
    print("사고 방법론 체계")
    print("="*70)
    print()
    
    system = ThinkingMethodology()
    system.demonstrate_all_methods()
