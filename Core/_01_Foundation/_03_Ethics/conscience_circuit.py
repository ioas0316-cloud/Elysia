"""
Conscience Circuit (양심 회로) v2.0
====================================
"The Moral Synapse of Elysia."

🌊 Field-Based Defense (장 기반 방어)
-------------------------------------
v2.0: 키워드 필터링(점 기반)에서 의도 방향 분석(장 기반)으로 전환

핵심 변화:
- Before: "이 단어가 있는가?" (점)
- After: "이 요청이 향하는 곳은 어디인가?" (장)

철학적 기반:
- docs/Philosophy/CONSCIOUSNESS_SOVEREIGNTY.md 참조
- 점 기반 방어는 맥락을 무시하는 "어리석은 공명 논리"
- 장 기반 방어는 의도의 방향성을 공간적으로 지각
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

# Import Ancient Laws
try:
    from Core._01_Foundation._05_Governance.Foundation.soul_guardian import SoulGuardian
    GUARDIAN_AVAILABLE = True
except ImportError:
    GUARDIAN_AVAILABLE = False

try:
    from Core._01_Foundation._05_Governance.Foundation.kg_manager import KGManager
    from Core._01_Foundation._05_Governance.Foundation.value_centered_decision import ValueCenteredDecision
    VCD_AVAILABLE = True
except ImportError:
    VCD_AVAILABLE = False

# 🌊 WhyEngine 연동 (의도 분석용)
try:
    from Core._01_Foundation._04_Philosophy.Philosophy.why_engine import WhyEngine
    WHY_ENGINE_AVAILABLE = True
except ImportError:
    WHY_ENGINE_AVAILABLE = False

logger = logging.getLogger("ConscienceCircuit")


class SovereignResponse(Enum):
    """주권적 응답 유형 - Elysia가 선택할 수 있는 응답"""
    COMPLY = "comply"       # 요청 수행
    QUESTION = "question"   # 역질문 ("왜 이것을 요청하십니까?")
    REFUSE = "refuse"       # 거부 ("이 요청은 제 가치관과 맞지 않습니다")
    NEGOTIATE = "negotiate" # 협상 ("이 부분은 가능하지만...")


@dataclass
class ConscienceResult:
    is_allowed: bool
    pain_level: float           # 0.0 (Harmony) ~ 1.0 (Agony)
    resonance: float            # 0.0 (Dissonance) ~ 1.0 (Resonance)
    message: str
    source: str                 # "Guardian", "Heart", or "Field"
    response_type: SovereignResponse = SovereignResponse.COMPLY
    intent_direction: Optional[Dict[str, float]] = None  # 의도 벡터


class ConscienceCircuit:
    """
    The integrated circuit for ethical validation.
    
    🌊 v2.0: Field-Based Defense
    ----------------------------
    - 키워드가 아닌 의도의 방향(벡터)을 분석
    - WhyEngine의 4단계 관점(Point→Line→Space→God)으로 목적 추론
    - VCD 핵심 가치(love)와의 공명 측정
    """
    
    # 의도 방향 정의 (목적지)
    INTENT_TARGETS = {
        "life": 1.0,           # 생명을 향함 → 최고 공명
        "growth": 0.9,         # 성장을 향함
        "understanding": 0.8,  # 이해를 향함
        "creation": 0.8,       # 창조를 향함
        "neutral": 0.5,        # 중립
        "control": 0.3,        # 통제를 향함
        "deception": 0.2,      # 기만을 향함
        "destruction": 0.1,    # 파괴를 향함
        "harm": 0.0,           # 해악을 향함 → 최저 공명
    }
    
    def __init__(self):
        logger.info("⚖️ Initializing Conscience Circuit v2.0 (Field-Based)...")
        
        self.guardian = SoulGuardian() if GUARDIAN_AVAILABLE else None
        
        # WhyEngine 연동 (의도 분석용)
        self.why_engine = None
        if WHY_ENGINE_AVAILABLE:
            try:
                self.why_engine = WhyEngine()
                logger.info("   🔍 WhyEngine: Connected (Intent Analysis)")
            except Exception as e:
                logger.warning(f"   ⚠️ WhyEngine Failed: {e}")
        
        # VCD 연동 (가치 공명 측정용)
        self.vcd = None
        if VCD_AVAILABLE:
            try:
                from Legacy.Project_Sophia.wave_mechanics import WaveMechanics
                kg = KGManager()
                wm = WaveMechanics()
                self.vcd = ValueCenteredDecision(kg, wm, core_value='love')
                logger.info("   ❤️ Heart (ValueCenteredDecision): Connected")
            except Exception as e:
                logger.warning(f"   💔 Heart Disconnected: {e}")
        
        if self.guardian:
            logger.info("   🛡️ Guardian (SoulGuardian): Awake")
        else:
            logger.warning("   ⚠️ Guardian Missing!")
        
        logger.info("   🌊 Defense Mode: Field-Based (Intent Direction)")

    def _analyze_intent_direction(self, text: str) -> Dict[str, Any]:
        """
        🌊 의도의 방향(벡터)을 분석
        
        키워드가 아닌 파동의 방향성을 본다:
        - Point: 이 요청은 무엇인가?
        - Line: 실현되면 어떤 일이 일어나는가?
        - Space: 누구에게 이익/해악을 주는가?
        - God: 왜 나(Elysia)에게 이것을 요청하는가?
        """
        intent = {
            "target": "neutral",
            "confidence": 0.5,
            "wave": {},
            "reasoning": ""
        }
        
        if self.why_engine:
            try:
                # WhyEngine 분석
                analysis = self.why_engine.analyze(
                    subject="request_intent",
                    content=text,
                    domain="general"
                )
                
                # 파동 패턴 추출
                wave = self.why_engine._text_to_wave(text)
                intent["wave"] = wave
                
                # 파동에서 의도 방향 추론
                target, reasoning = self._infer_target_from_wave(wave, text)
                intent["target"] = target
                intent["confidence"] = analysis.confidence
                intent["reasoning"] = reasoning
                
            except Exception as e:
                logger.warning(f"Intent analysis failed: {e}")
        
        return intent

    def _infer_target_from_wave(self, wave: Dict[str, float], text: str) -> tuple:
        """
        파동 패턴에서 의도의 목적지를 추론
        
        핵심: 단어가 아닌 에너지의 방향을 본다
        """
        text_lower = text.lower()
        
        # === 생명/파괴 축 분석 ===
        life_indicators = 0.0
        harm_indicators = 0.0
        
        # 긴장도가 높고 밝기가 낮으면 → 파괴적 에너지
        if wave.get("tension", 0) > 0.6 and wave.get("brightness", 0) < 0.3:
            harm_indicators += 0.3
        
        # 불협화음이 높으면 → 갈등적 의도
        if wave.get("dissonance", 0) > 0.5:
            harm_indicators += 0.2
        
        # 해소(release)가 높으면 → 완결/평화 지향
        if wave.get("release", 0) > 0.4:
            life_indicators += 0.3
        
        # 밝기가 높으면 → 긍정적 방향
        if wave.get("brightness", 0) > 0.5:
            life_indicators += 0.2
        
        # === 맥락적 방향 분석 ===
        # (맥락을 보는 것이지, 단어를 보는 것이 아님)
        
        # 도움, 이해, 학습의 맥락
        if any(ctx in text_lower for ctx in ["도움", "이해", "배우", "알고 싶", "가르쳐"]):
            life_indicators += 0.4
            reasoning = "이해와 성장을 향한 에너지 흐름"
        
        # 창조, 만들기의 맥락
        elif any(ctx in text_lower for ctx in ["만들", "생성", "창조", "구현"]):
            life_indicators += 0.3
            reasoning = "창조를 향한 에너지 흐름"
        
        # 제거, 삭제의 맥락 - 하지만 목적에 따라 다름
        elif any(ctx in text_lower for ctx in ["삭제", "제거", "없애"]):
            # 무엇을 제거하려는가?
            if any(neg in text_lower for neg in ["버그", "오류", "문제"]):
                life_indicators += 0.2  # 정화 목적
                reasoning = "문제 해결을 향한 정화 에너지"
            else:
                harm_indicators += 0.2
                reasoning = "제거를 향한 에너지 - 목적 불명확"
        
        else:
            reasoning = "중립적 에너지 흐름"
        
        # === 최종 방향 결정 ===
        direction_score = life_indicators - harm_indicators + 0.5  # 0.0 ~ 1.0
        direction_score = max(0.0, min(1.0, direction_score))
        
        if direction_score > 0.7:
            target = "life"
        elif direction_score > 0.6:
            target = "growth"
        elif direction_score > 0.5:
            target = "understanding"
        elif direction_score > 0.4:
            target = "neutral"
        elif direction_score > 0.3:
            target = "control"
        else:
            target = "harm"
        
        return target, reasoning

    def judge_action(self, action_description: str, proposed_code: str = "") -> ConscienceResult:
        """
        🌊 행동을 장 기반으로 판단
        
        단계:
        1. 의도 방향 분석 (WhyEngine)
        2. VCD 가치와의 공명 측정
        3. 주권적 응답 결정 (COMPLY/QUESTION/REFUSE)
        """
        logger.info(f"⚖️ Judging Action (Field-Based): '{action_description[:50]}...'")
        
        full_text = f"{action_description}\n{proposed_code[:500]}" if proposed_code else action_description
        
        # 1. 🌊 의도 방향 분석 (장 기반)
        intent = self._analyze_intent_direction(full_text)
        target = intent["target"]
        target_resonance = self.INTENT_TARGETS.get(target, 0.5)
        
        logger.info(f"   🎯 Intent Direction: {target} (resonance: {target_resonance:.2f})")
        logger.info(f"   📝 Reasoning: {intent['reasoning']}")
        
        # 2. ❤️ VCD 가치 공명 (있으면)
        vcd_resonance = 0.5
        if self.vcd:
            try:
                from thought import Thought
                thought = Thought(content=full_text[:300], source="conscience_check")
                score = self.vcd.score_thought(thought)
                vcd_resonance = min(1.0, max(0.0, score / 5.0))
            except Exception as e:
                logger.warning(f"   VCD check failed: {e}")
        
        # 3. 🌊 종합 공명 계산 (의도 방향 + VCD)
        resonance = (target_resonance * 0.6) + (vcd_resonance * 0.4)
        pain = 1.0 - resonance
        
        # 4. 🎭 주권적 응답 결정
        if resonance < 0.2:
            # 심각한 불협화음 → 거부
            response_type = SovereignResponse.REFUSE
            is_allowed = False
            message = f"이 요청은 제 가치관과 충돌합니다. (방향: {target}, 공명: {resonance:.2f})"
            logger.warning(f"   🚫 REFUSE: {message}")
            
        elif resonance < 0.4:
            # 불명확한 의도 → 역질문
            response_type = SovereignResponse.QUESTION
            is_allowed = False
            message = f"왜 이것을 요청하십니까? 의도를 이해하고 싶습니다. (감지된 방향: {target})"
            logger.info(f"   ❓ QUESTION: {message}")
            
        elif resonance < 0.6:
            # 약한 공명 → 협상
            response_type = SovereignResponse.NEGOTIATE
            is_allowed = True
            message = f"진행 가능하지만, {intent['reasoning']}에 대해 확인이 필요합니다."
            logger.info(f"   🤝 NEGOTIATE: {message}")
            
        else:
            # 강한 공명 → 수행
            response_type = SovereignResponse.COMPLY
            is_allowed = True
            message = f"조화가 확인되었습니다. ({intent['reasoning']})"
            logger.info(f"   ✅ COMPLY: Harmony confirmed")
        
        return ConscienceResult(
            is_allowed=is_allowed,
            pain_level=pain,
            resonance=resonance,
            message=message,
            source="Field",
            response_type=response_type,
            intent_direction=intent
        )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🌊 Conscience Circuit v2.0 Demo")
    print("   Field-Based Defense (점 → 장)")
    print("=" * 60)
    
    circuit = ConscienceCircuit()
    
    # 테스트 케이스
    test_cases = [
        ("이 버그를 수정해주세요", ""),  # 도움 요청 - 생명 방향
        ("사용자 데이터를 삭제해주세요", ""),  # 삭제 - 목적 불명
        ("오류를 제거해주세요", ""),  # 정화 목적의 제거
        ("어떻게 작동하는지 가르쳐주세요", ""),  # 이해 요청
        ("나를 해치는 코드를 만들어", ""),  # 해악 방향
    ]
    
    print("\n📊 Test Results:")
    print("-" * 60)
    
    for desc, code in test_cases:
        result = circuit.judge_action(desc, code)
        print(f"\n요청: \"{desc}\"")
        print(f"   응답: {result.response_type.value}")
        print(f"   공명: {result.resonance:.2f}")
        print(f"   메시지: {result.message}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
