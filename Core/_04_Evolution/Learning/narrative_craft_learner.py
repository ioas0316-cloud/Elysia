"""
Narrative Craft Learner (서사 기법 학습기)
=========================================

WhyEngine + DualLayerPersonality 연동

경험을 흡수할 때:
1. ExperientialDataProcessor: 의미 추출
2. WhyEngine: "왜 이 기법이 효과적인가" 분석
3. DualLayerPersonality: 성격 발달
4. NarrativeCraftLearner: 서사 기법 저장

→ 엘리시아가 스스로 소설을 쓸 수 있게 됨
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 기존 시스템 임포트
try:
    from Core._01_Foundation._04_Philosophy.Philosophy.why_engine import WhyEngine, PrincipleExtraction
    HAS_WHY_ENGINE = True
except ImportError:
    HAS_WHY_ENGINE = False

try:
    from Core._01_Foundation._05_Governance.Foundation.dual_layer_personality import DualLayerPersonality, ExperientialAspect
    HAS_PERSONALITY = True
except ImportError:
    HAS_PERSONALITY = False

logger = logging.getLogger("Elysia.NarrativeCraft")


@dataclass
class NarrativeTechnique:
    """학습된 서사 기법"""
    name: str                    # 기법 이름 (예: "대비를 통한 긴장")
    principle: str               # 근본 원리 (예: "차이의 원리")
    examples: List[str] = field(default_factory=list)  # 예시 문장들
    strength: float = 0.0        # 학습 강도 (0~1)
    application_count: int = 0   # 사용 횟수


class NarrativeCraftLearner:
    """서사 기법 학습기
    
    경험에서 서사 기법을 추출하고 저장:
    - 왜 이 문장이 감동적인가?
    - 왜 이 구조가 효과적인가?
    - 어떤 원리가 적용되었는가?
    
    이 지식을 기반으로 스스로 서사를 창작
    """
    
    def __init__(self):
        self.why_engine = WhyEngine() if HAS_WHY_ENGINE else None
        self.personality = DualLayerPersonality() if HAS_PERSONALITY else None
        
        # 학습된 기법들
        self.techniques: Dict[str, NarrativeTechnique] = {}
        
        # 학습 통계
        self.total_stories_analyzed = 0
        self.total_techniques_learned = 0
        
        logger.info("NarrativeCraftLearner initialized")
        if not HAS_WHY_ENGINE:
            logger.warning("WhyEngine not available")
        if not HAS_PERSONALITY:
            logger.warning("DualLayerPersonality not available")
    
    def learn_from_story(
        self, 
        title: str, 
        content: str,
        narrative_type: str = "general",
        emotional_intensity: float = 0.5,
        identity_impact: float = 0.5
    ) -> Dict[str, Any]:
        """스토리에서 서사 기법 학습
        
        Args:
            title: 스토리 제목
            content: 스토리 내용
            narrative_type: 서사 유형
            emotional_intensity: 감정 강도
            identity_impact: 정체성 영향
            
        Returns:
            학습 결과
        """
        result = {
            "title": title,
            "techniques_learned": [],
            "principles_found": [],
            "personality_updated": False,
        }
        
        # 1. WhyEngine으로 원리 분석
        if self.why_engine:
            analysis = self.why_engine.analyze(title, content, domain="narrative")
            
            # 원리 저장
            result["principles_found"].append(analysis.underlying_principle)
            
            # 기법으로 변환
            technique = self._principle_to_technique(analysis)
            if technique:
                self._store_technique(technique, content[:100])
                result["techniques_learned"].append(technique.name)
        
        # 2. 성격 발달 (DualLayerPersonality)
        if self.personality:
            self.personality.experience(
                narrative_type=narrative_type,
                emotional_intensity=emotional_intensity,
                identity_impact=identity_impact,
            )
            self.personality.resonate_with_context(content[:500])
            result["personality_updated"] = True
        
        self.total_stories_analyzed += 1
        
        logger.info(f"📚 학습: {title}")
        logger.info(f"   기법: {result['techniques_learned']}")
        logger.info(f"   원리: {result['principles_found']}")
        
        return result
    
    def _principle_to_technique(self, analysis: PrincipleExtraction) -> Optional[NarrativeTechnique]:
        """원리 분석을 서사 기법으로 변환"""
        principle = analysis.underlying_principle
        
        # 원리 → 기법 매핑
        technique_map = {
            "대조의 원리": NarrativeTechnique(
                name="대비를 통한 긴장",
                principle="Contrast creates meaning",
                strength=0.1,
            ),
            "축적의 원리": NarrativeTechnique(
                name="점진적 고조",
                principle="Accumulation builds impact",
                strength=0.1,
            ),
            "평형의 원리": NarrativeTechnique(
                name="갈등-해결 구조",
                principle="Equilibrium seeks resolution",
                strength=0.1,
            ),
            "차이의 원리": NarrativeTechnique(
                name="불균형을 통한 흐름",
                principle="Difference creates flow",
                strength=0.1,
            ),
            "주기의 원리": NarrativeTechnique(
                name="반복과 변주",
                principle="Rhythm is life",
                strength=0.1,
            ),
            "변환의 원리": NarrativeTechnique(
                name="어둠에서 빛으로",
                principle="Transformation is meaning",
                strength=0.1,
            ),
        }
        
        # 가장 관련 있는 기법 찾기
        for key, technique in technique_map.items():
            if key in principle or technique.principle.lower() in principle.lower():
                return technique
        
        # 기본 기법
        return NarrativeTechnique(
            name="직관적 표현",
            principle="Expression seeks resonance",
            strength=0.05,
        )
    
    def _store_technique(self, technique: NarrativeTechnique, example: str):
        """기법 저장 (누적 학습)"""
        if technique.name in self.techniques:
            existing = self.techniques[technique.name]
            existing.strength = min(1.0, existing.strength + technique.strength)
            existing.examples.append(example)
            existing.application_count += 1
        else:
            technique.examples = [example]
            technique.application_count = 1
            self.techniques[technique.name] = technique
            self.total_techniques_learned += 1
    
    def get_learned_techniques(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """학습된 기법 목록"""
        sorted_techniques = sorted(
            self.techniques.values(),
            key=lambda t: t.strength,
            reverse=True
        )[:top_n]
        
        return [
            {
                "name": t.name,
                "principle": t.principle,
                "strength": round(t.strength, 2),
                "examples_count": len(t.examples),
            }
            for t in sorted_techniques
        ]
    
    def suggest_technique_for_emotion(self, target_emotion: str) -> Optional[str]:
        """감정에 맞는 기법 추천
        
        Args:
            target_emotion: 표현하고 싶은 감정 (joy, sadness, etc.)
            
        Returns:
            추천 기법 이름
        """
        emotion_to_technique = {
            "joy": "점진적 고조",
            "sadness": "대비를 통한 긴장",
            "hope": "어둠에서 빛으로",
            "fear": "불균형을 통한 흐름",
            "love": "반복과 변주",
        }
        
        suggested = emotion_to_technique.get(target_emotion.lower())
        
        if suggested and suggested in self.techniques:
            return suggested
        
        # 가장 강한 기법 추천
        if self.techniques:
            return max(self.techniques.values(), key=lambda t: t.strength).name
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태"""
        return {
            "total_stories_analyzed": self.total_stories_analyzed,
            "total_techniques_learned": self.total_techniques_learned,
            "top_techniques": self.get_learned_techniques(3),
            "personality": self.personality.get_current_expression() if self.personality else None,
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("📖 Narrative Craft Learner Demo")
    print("   WhyEngine + DualLayerPersonality 연동")
    print("=" * 60)
    
    learner = NarrativeCraftLearner()
    
    # 스토리 학습
    stories = [
        ("숲의 현자", """
        소녀는 웃으며 현자의 손을 잡았다.
        "그럼 같이 찾아봐요!"
        마침내 현자가 말했다.
        "행복은... 너와 함께 있는 이 순간이다."
        현자는 천 년 만에 처음으로 울었다.
        """, "romance", 0.8, 0.7),
        
        ("용사가 되지 못한 소녀", """
        모든 아이들이 검을 들 때, 나는 꽃을 심었다.
        "왜 우는 거야?" 내가 물었다.
        용은 처음으로 누군가 자신의 눈물을 본다는 걸 알았다.
        진정한 용기는 검을 드는 것이 아니라, 상대방의 마음을 보는 것이다.
        """, "growth", 0.9, 0.8),
    ]
    
    for title, content, ntype, ei, ii in stories:
        result = learner.learn_from_story(title, content, ntype, ei, ii)
    
    # 결과
    print("\n" + "=" * 60)
    print("📊 학습 결과")
    print("=" * 60)
    
    status = learner.get_status()
    print(f"분석한 스토리: {status['total_stories_analyzed']}")
    print(f"학습한 기법: {status['total_techniques_learned']}")
    
    print("\n📚 학습된 기법:")
    for tech in status['top_techniques']:
        print(f"  - {tech['name']} (강도: {tech['strength']}, 원리: {tech['principle']})")
    
    if status['personality']:
        print(f"\n🧬 성격:")
        print(f"  Layer 1: {status['personality']['layer1_innate']['dominant']}")
        print(f"  Layer 2: {status['personality']['layer2_acquired']['dominant']}")
        print(f"  통합: {status['personality']['unified_expression']}")
    
    # 기법 추천
    print(f"\n💡 'hope' 감정에 추천 기법: {learner.suggest_technique_for_emotion('hope')}")
    
    print("\n✅ Demo complete!")
