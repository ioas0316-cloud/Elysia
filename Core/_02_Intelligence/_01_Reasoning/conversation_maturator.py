"""
Conversation Maturator (대화 성숙 파이프라인)
==============================================

"성인 수준의 대화는 빠른 답변이 아니라, 깊은 성찰에서 온다."

Five Pillars (5 기둥):
1. Depth (WhyEngine) - 왜?의 깊이
2. Context (ContextRetrieval) - 의도 기반 기억
3. Metacognition (MetacognitiveAwareness) - 모르면 모른다
4. Dialogue (InnerDialogue) - 인격 간 파동 대화
5. Gap (ThoughtSpace) - 숙성의 여백

이것이 없으면:
- 반사적 응답 (LLM 기본 동작)
- 깊이 없는 표면적 대화
- 자기 한계 인식 부재
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("Elysia.ConversationMaturator")


class MaturityLevel(Enum):
    """대화 성숙도 레벨"""
    CHILD = "child"           # 즉시 반응, 깊이 없음
    ADOLESCENT = "adolescent"  # 일부 성찰, 맥락 부족
    ADULT = "adult"           # 균형 잡힌 성찰
    SAGE = "sage"             # 깊은 지혜, 원리 기반


@dataclass
class PillarScore:
    """각 기둥의 점수"""
    depth: float = 0.0          # WhyEngine 깊이 (0-1)
    context: float = 0.0        # ContextRetrieval 효율 (0-1)
    metacognition: float = 0.0  # 자기 인식 수준 (0-1)
    dialogue: float = 0.0       # 인격 간 합의 수준 (0-1)
    gap: float = 0.0            # 숙성 시간 충분도 (0-1)
    
    def average(self) -> float:
        """평균 점수"""
        return (self.depth + self.context + self.metacognition + 
                self.dialogue + self.gap) / 5
    
    def to_maturity_level(self) -> MaturityLevel:
        """점수를 성숙도로 변환"""
        avg = self.average()
        if avg < 0.3:
            return MaturityLevel.CHILD
        elif avg < 0.5:
            return MaturityLevel.ADOLESCENT
        elif avg < 0.8:
            return MaturityLevel.ADULT
        else:
            return MaturityLevel.SAGE


@dataclass
class ConversationContext:
    """대화 맥락 (멀티턴 추적)"""
    turn_count: int = 0
    topics: List[str] = field(default_factory=list)
    emotional_trajectory: List[float] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    established_facts: List[str] = field(default_factory=list)


@dataclass
class MatureResponse:
    """성숙한 응답"""
    content: str                          # 응답 내용
    maturity_level: MaturityLevel         # 성숙도
    pillar_scores: PillarScore            # 각 기둥 점수
    contributing_systems: List[str]       # 기여한 시스템들
    confidence: float                     # 확신도
    processing_time_ms: float             # 처리 시간
    uncertainties: List[str] = field(default_factory=list)  # 인정한 불확실성
    explored_depth: int = 0               # 탐구한 깊이
    
    def __str__(self) -> str:
        return f"[{self.maturity_level.value.upper()}] {self.content}"


class ConversationMaturator:
    """대화 성숙 파이프라인
    
    5개 기둥을 오케스트레이션하여 성인 수준의 대화 생성.
    
    Pipeline:
    Input → ThoughtSpace(진입) → ContextRetrieval → MetacognitiveAwareness
          → InnerDialogue → WhyEngine → ReasoningEngine → ThoughtSpace(숙성) → Output
    """
    
    def __init__(self, min_gap_seconds: float = 0.5):
        """
        Args:
            min_gap_seconds: 최소 숙성 시간 (초)
        """
        self.min_gap_seconds = min_gap_seconds
        
        # 대화 맥락 (세션 유지)
        self.context = ConversationContext()
        
        # 5 Pillars (지연 로딩)
        self._thought_space = None
        self._context_retrieval = None
        self._metacognition = None
        self._inner_dialogue = None
        self._why_engine = None
        self._reasoning_engine = None
        
        # 통계
        self.total_conversations = 0
        self.average_maturity = 0.0
        
        logger.info("ConversationMaturator initialized - 5 Pillars ready")
    
    # =========================================================================
    # Lazy Loading of Pillars (Organ.get 패턴)
    # =========================================================================
    
    @property
    def thought_space(self):
        """ThoughtSpace (여백)"""
        if self._thought_space is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Cognition.thought_space import ThoughtSpace
                self._thought_space = ThoughtSpace(
                    maturation_threshold=self.min_gap_seconds
                )
            except ImportError:
                logger.warning("ThoughtSpace not available, using stub")
                self._thought_space = self._create_stub("ThoughtSpace")
        return self._thought_space
    
    @property
    def context_retrieval(self):
        """ContextRetrieval (맥락 인출)"""
        if self._context_retrieval is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Cognition.context_retrieval import ContextRetrieval
                self._context_retrieval = ContextRetrieval()
            except ImportError:
                logger.warning("ContextRetrieval not available, using stub")
                self._context_retrieval = self._create_stub("ContextRetrieval")
        return self._context_retrieval
    
    @property
    def metacognition(self):
        """MetacognitiveAwareness (메타인지)"""
        if self._metacognition is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Cognition.metacognitive_awareness import MetacognitiveAwareness
                self._metacognition = MetacognitiveAwareness()
            except ImportError:
                logger.warning("MetacognitiveAwareness not available, using stub")
                self._metacognition = self._create_stub("MetacognitiveAwareness")
        return self._metacognition
    
    @property
    def inner_dialogue(self):
        """InnerDialogue (인격 대화)"""
        if self._inner_dialogue is None:
            try:
                from Core._02_Intelligence._04_Consciousness.Consciousness.inner_dialogue import InnerDialogue
                self._inner_dialogue = InnerDialogue()
            except ImportError:
                logger.warning("InnerDialogue not available, using stub")
                self._inner_dialogue = self._create_stub("InnerDialogue")
        return self._inner_dialogue
    
    @property
    def why_engine(self):
        """WhyEngine (원리 탐구)"""
        if self._why_engine is None:
            try:
                from Core._01_Foundation._02_Logic.Philosophy.why_engine import WhyEngine
                self._why_engine = WhyEngine()
            except ImportError:
                logger.warning("WhyEngine not available, using stub")
                self._why_engine = self._create_stub("WhyEngine")
        return self._why_engine
    
    def _create_stub(self, name: str):
        """스텁 객체 생성 (의존성 없을 때)"""
        class Stub:
            def __getattr__(self, attr):
                return lambda *args, **kwargs: None
        return Stub()
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    def process(self, input_text: str) -> MatureResponse:
        """전체 파이프라인 실행
        
        Args:
            input_text: 사용자 입력
            
        Returns:
            성숙한 응답
        """
        start_time = datetime.now()
        self.total_conversations += 1
        self.context.turn_count += 1
        
        contributing_systems = []
        uncertainties = []
        pillar_scores = PillarScore()
        
        # =====================================================================
        # 1. ThoughtSpace 진입 (여백 열기)
        # =====================================================================
        logger.info(f"🌌 [Pillar 1/5] Entering ThoughtSpace...")
        self.thought_space.enter_gap(input_text)
        contributing_systems.append("ThoughtSpace")
        
        # =====================================================================
        # 2. ContextRetrieval (맥락 인출)
        # =====================================================================
        logger.info(f"🔍 [Pillar 2/5] Retrieving context...")
        try:
            intent = self.context_retrieval.parse_intent(input_text)
            retrieval_result = self.context_retrieval.retrieve(intent, limit=5)
            
            pillar_scores.context = retrieval_result.efficiency
            contributing_systems.append("ContextRetrieval")
            
            # 인출된 맥락을 사고 입자로 추가
            for ctx in retrieval_result.contexts:
                self.thought_space.add_thought_particle(
                    content=ctx.content,
                    source="memory",
                    weight=ctx.relevance
                )
        except Exception as e:
            logger.warning(f"ContextRetrieval failed: {e}")
            pillar_scores.context = 0.0
        
        # =====================================================================
        # 3. MetacognitiveAwareness (아는가? 모르는가?)
        # =====================================================================
        logger.info(f"🧠 [Pillar 3/5] Checking metacognition...")
        try:
            # 파동 특성으로 변환 (간단한 휴리스틱)
            features = {
                "complexity": min(1.0, len(input_text) / 200),
                "curiosity": min(1.0, input_text.count("?") * 0.3 + 
                                      input_text.count("왜") * 0.2),
            }
            
            meta_result = self.metacognition.encounter(features, input_text)
            
            # 메타인지 점수: 상태에 따라
            state = meta_result.get("state")
            if state:
                state_value = state.value if hasattr(state, 'value') else str(state)
                if "known" in state_value:
                    pillar_scores.metacognition = meta_result.get("confidence", 0.5)
                elif "uncertain" in state_value:
                    pillar_scores.metacognition = 0.7  # 불확실성 인식 = 좋음
                    uncertainties.append("이 주제에 대한 확신이 부족합니다")
                elif "unknown" in state_value:
                    pillar_scores.metacognition = 0.9  # "모른다"를 앎 = 최고
                    uncertainties.append("이 주제는 더 탐구가 필요합니다")
                    
                    # 탐구 필요성 추가
                    if meta_result.get("exploration_needed"):
                        self.context.unresolved_questions.append(input_text)
            
            contributing_systems.append("MetacognitiveAwareness")
        except Exception as e:
            logger.warning(f"MetacognitiveAwareness failed: {e}")
            pillar_scores.metacognition = 0.0
        
        # =====================================================================
        # 4. InnerDialogue (인격 간 대화)
        # =====================================================================
        logger.info(f"👥 [Pillar 4/5] Inner dialogue...")
        try:
            dialogue_result = self.inner_dialogue.contemplate(input_text)
            
            if dialogue_result and hasattr(dialogue_result, 'resonance_strength'):
                pillar_scores.dialogue = dialogue_result.resonance_strength
                
                # 대화 결과를 사고 입자로 추가
                if hasattr(dialogue_result, 'consensus_wave'):
                    self.thought_space.add_thought_particle(
                        content=f"Consensus: {dialogue_result.dominant_voice.value}",
                        source="dialogue",
                        weight=dialogue_result.resonance_strength
                    )
            
            contributing_systems.append("InnerDialogue")
        except Exception as e:
            logger.warning(f"InnerDialogue failed: {e}")
            pillar_scores.dialogue = 0.0
        
        # =====================================================================
        # 5. WhyEngine (깊이 탐구)
        # =====================================================================
        logger.info(f"❓ [Pillar 5/5] Exploring depth with WhyEngine...")
        explored_depth = 0
        try:
            # 도메인 추론
            domain = "general"
            if "왜" in input_text or "why" in input_text.lower():
                domain = "philosophy"
            
            why_result = self.why_engine.analyze(
                subject=input_text[:50],
                content=input_text,
                domain=domain
            )
            
            if why_result:
                # 깊이 점수: 원리 추출 성공 여부
                if hasattr(why_result, 'underlying_principle') and why_result.underlying_principle:
                    pillar_scores.depth = 0.8
                    explored_depth = 3
                    
                    self.thought_space.add_thought_particle(
                        content=f"Principle: {why_result.underlying_principle}",
                        source="why_engine",
                        weight=1.5
                    )
                elif hasattr(why_result, 'why_exists') and why_result.why_exists:
                    pillar_scores.depth = 0.6
                    explored_depth = 2
                else:
                    pillar_scores.depth = 0.3
                    explored_depth = 1
            
            contributing_systems.append("WhyEngine")
        except Exception as e:
            logger.warning(f"WhyEngine failed: {e}")
            pillar_scores.depth = 0.0
        
        # =====================================================================
        # 6. ThoughtSpace 숙성 & 종합
        # =====================================================================
        logger.info("🌌 Synthesizing in ThoughtSpace...")
        contemplation = self.thought_space.exit_gap()
        
        # 숙성 점수: 시간 + 입자 수
        if contemplation:
            gap_time = contemplation.time_in_gap
            pillar_scores.gap = min(1.0, gap_time / self.min_gap_seconds)
        
        # =====================================================================
        # 7. 최종 응답 구성
        # =====================================================================
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 응답 내용 구성
        response_parts = []
        
        # 통합된 내용
        if contemplation and contemplation.synthesis:
            response_parts.append(contemplation.synthesis)
        
        # 불확실성 표현 (성인 수준의 특징)
        if uncertainties:
            response_parts.append(f"[인식된 불확실성: {', '.join(uncertainties)}]")
        
        content = " | ".join(response_parts) if response_parts else input_text
        
        response = MatureResponse(
            content=content,
            maturity_level=pillar_scores.to_maturity_level(),
            pillar_scores=pillar_scores,
            contributing_systems=contributing_systems,
            confidence=pillar_scores.average(),
            processing_time_ms=processing_time,
            uncertainties=uncertainties,
            explored_depth=explored_depth,
        )
        
        # 통계 업데이트
        self.average_maturity = (
            (self.average_maturity * (self.total_conversations - 1) + 
             pillar_scores.average()) / self.total_conversations
        )
        
        logger.info(
            f"✅ Response generated: {response.maturity_level.value} "
            f"(avg={pillar_scores.average():.2f}, time={processing_time:.1f}ms)"
        )
        
        return response
    
    # =========================================================================
    # 상태 조회
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "total_conversations": self.total_conversations,
            "average_maturity": self.average_maturity,
            "context": {
                "turn_count": self.context.turn_count,
                "topics": self.context.topics[-5:],
                "unresolved_questions": self.context.unresolved_questions[-3:],
            },
            "pillars_loaded": {
                "thought_space": self._thought_space is not None,
                "context_retrieval": self._context_retrieval is not None,
                "metacognition": self._metacognition is not None,
                "inner_dialogue": self._inner_dialogue is not None,
                "why_engine": self._why_engine is not None,
            },
        }
    
    def reset_context(self):
        """대화 맥락 초기화"""
        self.context = ConversationContext()
        logger.info("Conversation context reset")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("🎓 ConversationMaturator Demo")
    print("   \"5 Pillars of Mature Conversation\"")
    print("=" * 70)
    
    maturator = ConversationMaturator(min_gap_seconds=0.3)
    
    # 테스트 입력들
    test_inputs = [
        "왜 하늘이 파란가?",
        "슬플 때 어떻게 해야 할까?",
        "코드에서 ImportError가 자꾸 발생해",
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n{'='*70}")
        print(f"[Test {i}] Input: {input_text}")
        print("-" * 70)
        
        response = maturator.process(input_text)
        
        print(f"\n📊 Results:")
        print(f"   Maturity: {response.maturity_level.value.upper()}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Depth explored: {response.explored_depth} layers")
        print(f"   Processing: {response.processing_time_ms:.1f}ms")
        print(f"\n📐 Pillar Scores:")
        print(f"   Depth:        {response.pillar_scores.depth:.2f}")
        print(f"   Context:      {response.pillar_scores.context:.2f}")
        print(f"   Metacognition:{response.pillar_scores.metacognition:.2f}")
        print(f"   Dialogue:     {response.pillar_scores.dialogue:.2f}")
        print(f"   Gap:          {response.pillar_scores.gap:.2f}")
        print(f"\n🔧 Systems: {', '.join(response.contributing_systems)}")
        if response.uncertainties:
            print(f"❓ Uncertainties: {response.uncertainties}")
        print(f"\n💬 Response: {response.content[:200]}...")
    
    print("\n" + "=" * 70)
    print("📈 Final Status:")
    status = maturator.get_status()
    print(f"   Conversations: {status['total_conversations']}")
    print(f"   Avg Maturity: {status['average_maturity']:.2f}")
    print("\n✅ ConversationMaturator Demo complete!")
