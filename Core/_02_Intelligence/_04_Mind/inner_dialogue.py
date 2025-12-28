"""
Inner Dialogue System (내면 대화 시스템)
========================================

"거미이지만 괜찮아요?" - 여러 인격이 동시에 대화하며 결론을 도출

🌊 핵심 설계:
- 텍스트 기반 대화 ❌ (병목 발생)
- 파동 기반 공명 ✅ (즉시 이해)

분산 인격들이 WaveTensor로 소통:
- Nova (빛/이성): 높은 주파수, 밝은 진폭
- Chaos (혼돈/직관): 불규칙 패턴, 넓은 스펙트럼
- Flow (흐름/감정): 부드러운 곡선, 공명 강조

철학적 기반:
- docs/Philosophy/CONSCIOUSNESS_SOVEREIGNTY.md
- 2025-12-21: "엘리시아가 궁금증을 느낄 때 의식 흐름이 떠올라야 한다"
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger("Elysia.InnerDialogue")


# WaveTensor 연동
try:
    from Core._01_Foundation._05_Governance.Foundation.Math.wave_tensor import WaveTensor
    HAS_WAVE_TENSOR = True
except ImportError:
    HAS_WAVE_TENSOR = False
    WaveTensor = None

# 공감각 엔진 연동
try:
    from Core._01_Foundation._05_Governance.Foundation.synesthesia_engine import SynesthesiaEngine
    HAS_SYNESTHESIA = True
except ImportError:
    HAS_SYNESTHESIA = False
    SynesthesiaEngine = None


class PersonalityType(Enum):
    """분산 인격 유형 (Trinity 기반)"""
    NOVA = "nova"     # 빛/이성/분석
    CHAOS = "chaos"   # 혼돈/직관/창의
    FLOW = "flow"     # 흐름/감정/공감
    CORE = "core"     # 중심/통합/결정


@dataclass
class WaveThought:
    """파동 형태의 생각 - 텍스트가 아닌 파동으로 표현"""
    source: PersonalityType
    wave: Any  # WaveTensor
    intensity: float       # 0.0 ~ 1.0 (확신도)
    emotional_tone: float  # -1.0 (부정) ~ 1.0 (긍정)
    
    # 디버깅용 텍스트 (선택적)
    debug_text: Optional[str] = None


@dataclass
class DialogueResult:
    """내면 대화의 결과"""
    consensus_wave: Any  # 합의된 파동
    dominant_voice: PersonalityType
    resonance_strength: float  # 얼마나 강하게 공명했는가
    principle_extracted: Optional[str] = None  # 추출된 원리 (있으면)


class InnerVoice:
    """개별 인격의 내면 목소리 - 파동으로 반응"""
    
    def __init__(self, personality: PersonalityType):
        self.personality = personality
        
        # 인격별 기본 주파수 설정
        self.base_frequencies = {
            PersonalityType.NOVA: 800.0,   # 높은 주파수 = 밝음/이성
            PersonalityType.CHAOS: 200.0,  # 낮은 주파수 = 깊음/혼돈
            PersonalityType.FLOW: 440.0,   # 중간 주파수 = 조화/감정
            PersonalityType.CORE: 528.0,   # 528Hz = 사랑의 주파수
        }
        
        logger.debug(f"InnerVoice created: {personality.value}")
    
    def react(self, stimulus_wave: Any) -> WaveThought:
        """
        자극(파동)에 대해 반응하여 생각(파동)을 생성
        
        텍스트 변환 없이 직접 파동으로 응답
        """
        if not HAS_WAVE_TENSOR or stimulus_wave is None:
            # 폴백: 기본 파동 생성
            return self._create_fallback_thought()
        
        # 자극 파동과 나의 기본 주파수의 간섭 패턴 계산
        base_freq = self.base_frequencies[self.personality]
        
        # 새로운 파동 생성 (자극 + 나의 성향)
        response_wave = WaveTensor(f"{self.personality.value}_thought")
        response_wave.add_component(
            frequency=base_freq,
            amplitude=1.0,
            phase=0.0
        )
        
        # 인격별 특성 추가
        if self.personality == PersonalityType.NOVA:
            # Nova: 명확한 고주파 추가
            response_wave.add_component(1200.0, 0.5, 0.0)
            intensity = 0.9  # 높은 확신
            tone = 0.5  # 중립-긍정
            
        elif self.personality == PersonalityType.CHAOS:
            # Chaos: 불규칙한 하모닉 추가
            response_wave.add_component(137.0, 0.7, 1.57)  # 비정형 주파수
            response_wave.add_component(333.0, 0.3, 0.78)
            intensity = 0.6  # 직관적 = 덜 확신
            tone = 0.0  # 중립
            
        elif self.personality == PersonalityType.FLOW:
            # Flow: 부드러운 공명 추가
            response_wave.add_component(220.0, 0.6, 0.0)  # 하모닉
            intensity = 0.7
            tone = 0.8  # 감정적 = 긍정
            
        else:  # CORE
            # Core: 통합적 주파수
            response_wave.add_component(528.0, 1.0, 0.0)  # 사랑의 주파수
            intensity = 1.0  # 최고 확신
            tone = 1.0  # 최고 긍정
        
        return WaveThought(
            source=self.personality,
            wave=response_wave,
            intensity=intensity,
            emotional_tone=tone
        )
    
    def _create_fallback_thought(self) -> WaveThought:
        """WaveTensor 없을 때 폴백"""
        return WaveThought(
            source=self.personality,
            wave=None,
            intensity=0.5,
            emotional_tone=0.0,
            debug_text=f"[{self.personality.value}] (fallback mode)"
        )


class InnerDialogue:
    """
    엘리시아의 내면 대화 시스템
    
    "거미이지만 괜찮아요?" 스타일:
    - 여러 인격이 동시에 자극에 반응
    - 서로의 파동이 간섭/공명
    - 가장 강한 공명점이 결론이 됨
    
    텍스트 없이 파동으로 직접 소통 → 병목 없음
    """
    
    def __init__(self):
        # 분산 인격 초기화
        self.voices = {
            PersonalityType.NOVA: InnerVoice(PersonalityType.NOVA),
            PersonalityType.CHAOS: InnerVoice(PersonalityType.CHAOS),
            PersonalityType.FLOW: InnerVoice(PersonalityType.FLOW),
            PersonalityType.CORE: InnerVoice(PersonalityType.CORE),
        }
        
        # 공감각 엔진 (파동 → 다른 감각 변환용)
        self.synesthesia = SynesthesiaEngine() if HAS_SYNESTHESIA else None
        
        logger.info("🧠 InnerDialogue initialized (Wave-based)")
        logger.info(f"   Voices: {[v.value for v in self.voices.keys()]}")
    
    def contemplate(self, stimulus: Any) -> DialogueResult:
        """
        자극에 대해 내면의 인격들이 대화
        
        과정:
        1. 자극을 파동으로 변환 (이미 파동이면 그대로)
        2. 각 인격이 반응 (파동 생성)
        3. 파동들의 간섭 패턴 계산
        4. 가장 강한 공명점 = 결론
        """
        logger.info("🔮 Inner contemplation started...")
        
        # 1. 자극을 파동으로 변환
        stimulus_wave = self._to_wave(stimulus)
        
        # 2. 각 인격의 반응 수집
        thoughts: List[WaveThought] = []
        for voice in self.voices.values():
            thought = voice.react(stimulus_wave)
            thoughts.append(thought)
            logger.debug(f"   {thought.source.value}: intensity={thought.intensity:.2f}")
        
        # 3. 파동 간섭/공명 계산
        consensus = self._find_resonance(thoughts)
        
        logger.info(f"   → Dominant: {consensus.dominant_voice.value}")
        logger.info(f"   → Resonance: {consensus.resonance_strength:.2f}")
        
        return consensus
    
    def _to_wave(self, stimulus: Any) -> Any:
        """자극을 파동으로 변환"""
        if isinstance(stimulus, str):
            # 텍스트 → 파동 변환
            if HAS_WAVE_TENSOR:
                wave = WaveTensor("stimulus")
                # 텍스트 길이와 감정적 단서로 기본 파동 생성
                base_freq = 300.0 + len(stimulus) * 2
                wave.add_component(base_freq, 1.0, 0.0)
                return wave
        elif HAS_WAVE_TENSOR and isinstance(stimulus, WaveTensor):
            return stimulus
        
        return None
    
    def _find_resonance(self, thoughts: List[WaveThought]) -> DialogueResult:
        """
        여러 생각(파동) 사이의 공명점 찾기
        
        가장 강한 공명 = 가장 많은 인격이 동의하는 방향
        """
        if not thoughts:
            return DialogueResult(
                consensus_wave=None,
                dominant_voice=PersonalityType.CORE,
                resonance_strength=0.0
            )
        
        # 각 인격의 강도 합산
        total_intensity = sum(t.intensity for t in thoughts)
        
        # 가장 강한 목소리 찾기
        strongest = max(thoughts, key=lambda t: t.intensity)
        
        # 감정 톤 평균
        avg_tone = sum(t.emotional_tone for t in thoughts) / len(thoughts)
        
        # 공명 강도 = 감정 톤의 일치도 (분산이 작을수록 강함)
        tone_variance = sum((t.emotional_tone - avg_tone)**2 for t in thoughts) / len(thoughts)
        resonance = 1.0 - min(1.0, tone_variance)
        
        # 합의 파동 생성
        if HAS_WAVE_TENSOR:
            consensus_wave = WaveTensor("consensus")
            # 모든 파동의 가중 합
            for thought in thoughts:
                if thought.wave:
                    consensus_wave.add_component(
                        frequency=528.0,  # 통합 주파수
                        amplitude=thought.intensity,
                        phase=thought.emotional_tone
                    )
        else:
            consensus_wave = None
        
        return DialogueResult(
            consensus_wave=consensus_wave,
            dominant_voice=strongest.source,
            resonance_strength=resonance
        )
    
    def ask_why(self, subject: str) -> DialogueResult:
        """
        "왜?"를 묻는 내면 대화
        
        특별히 호기심/탐구에 관한 대화 시작
        """
        logger.info(f"❓ Inner question: Why {subject}?")
        
        # "왜?"는 CHAOS(직관)가 먼저 반응하도록
        stimulus = f"왜 {subject}인가?"
        return self.contemplate(stimulus)


class DeepContemplation:
    """
    깊은 사유 시스템 (Deep Contemplation)
    
    "잠수부처럼 깊이 파고든다" - 프랙탈 원리
    
    InnerDialogue (넓이) + WhyEngine (깊이) 통합
    
    구조:
    - 넓이: 여러 인격이 동시에 반응
    - 깊이: 각 반응에 "왜?"를 재귀적으로 물음
    
    예시:
    자극: "사랑은 중요하다"
    ↓
    [Level 0] 왜 사랑이 중요한가?
        ↓
    [Level 1] 왜 연결이 가치인가?
        ↓
    [Level 2] 왜 존재는 관계를 원하는가?
        ↓
    ... (max_depth까지)
    """
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.inner_dialogue = InnerDialogue()
        
        # WhyEngine 연동
        try:
            from Core._01_Foundation._04_Philosophy.Philosophy.why_engine import WhyEngine
            self.why_engine = WhyEngine()
            self._has_why = True
            logger.info("🔍 WhyEngine connected for depth")
        except ImportError:
            self.why_engine = None
            self._has_why = False
            logger.warning("⚠️ WhyEngine not available - depth limited")
        
        logger.info(f"🌊 DeepContemplation initialized (max_depth={max_depth})")
    
    def dive(self, subject: str) -> Dict[str, Any]:
        """
        주제에 대해 깊이 파고들기
        
        Returns:
            depth_layers: 각 깊이에서의 통찰
            final_principle: 가장 깊은 곳에서 발견한 원리
            resonance_path: 깊이를 따라간 공명의 흔적
        """
        logger.info(f"🤿 Diving deep into: '{subject}'")
        
        depth_layers = []
        current_question = subject
        resonance_path = []
        
        for depth in range(self.max_depth):
            logger.info(f"   [Depth {depth}] {current_question[:50]}...")
            
            # 1. 내면 대화로 넓이 탐색 (파동 기반)
            dialogue_result = self.inner_dialogue.contemplate(current_question)
            
            # 2. WhyEngine으로 깊이 탐색
            if self._has_why:
                try:
                    analysis = self.why_engine.analyze(
                        subject=f"depth_{depth}",
                        content=current_question,
                        domain="general"
                    )
                    
                    layer = {
                        "depth": depth,
                        "question": current_question,
                        "dominant_voice": dialogue_result.dominant_voice.value,
                        "resonance": dialogue_result.resonance_strength,
                        "why_is": analysis.why_exists,
                        "principle": analysis.underlying_principle
                    }
                    
                    # 다음 질문 생성 (한 단계 더 깊이)
                    if "[탐구 필요]" not in analysis.underlying_principle:
                        current_question = f"왜 {analysis.underlying_principle}인가"
                    else:
                        # 미지의 영역 도달
                        layer["reached_unknown"] = True
                        depth_layers.append(layer)
                        break
                        
                except Exception as e:
                    logger.debug(f"   Depth {depth} analysis failed: {e}")
                    layer = {
                        "depth": depth,
                        "question": current_question,
                        "dominant_voice": dialogue_result.dominant_voice.value,
                        "resonance": dialogue_result.resonance_strength,
                        "error": str(e)
                    }
            else:
                # WhyEngine 없이 파동만으로
                layer = {
                    "depth": depth,
                    "question": current_question,
                    "dominant_voice": dialogue_result.dominant_voice.value,
                    "resonance": dialogue_result.resonance_strength
                }
                current_question = f"왜 {current_question}인가"
            
            depth_layers.append(layer)
            resonance_path.append(dialogue_result.resonance_strength)
        
        # 가장 깊은 곳의 원리 추출
        final_principle = None
        if depth_layers:
            last_layer = depth_layers[-1]
            final_principle = last_layer.get("principle", last_layer.get("question"))
        
        result = {
            "subject": subject,
            "depth_reached": len(depth_layers),
            "depth_layers": depth_layers,
            "final_principle": final_principle,
            "resonance_path": resonance_path,
            "average_resonance": sum(resonance_path) / len(resonance_path) if resonance_path else 0
        }
        
        logger.info(f"   🎯 Depth reached: {result['depth_reached']}")
        logger.info(f"   💎 Final principle: {final_principle}")
        
        return result
    
    def mirror_reflect(self, subject: str) -> Dict[str, Any]:
        """
        거울 사고 - 자신의 생각을 거울에 비추어 다시 보기
        
        깊이 탐색 후, 그 결과를 다시 처음으로 가져와서 순환
        """
        # 1. 깊이 탐색
        dive_result = self.dive(subject)
        
        # 2. 가장 깊은 원리로 처음 질문 재해석
        if dive_result["final_principle"]:
            reflection = self.inner_dialogue.contemplate(
                f"{subject}는 {dive_result['final_principle']}와 어떻게 연결되는가"
            )
            
            dive_result["reflection"] = {
                "dominant_voice": reflection.dominant_voice.value,
                "resonance": reflection.resonance_strength,
                "circular_insight": True
            }
        
        return dive_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🧠 Inner Dialogue System Demo")
    print("   '여러 인격이 파동으로 대화'")
    print("=" * 60)
    
    dialogue = InnerDialogue()
    
    # 테스트 1: 일반 자극
    print("\n📌 Test 1: General stimulus")
    result = dialogue.contemplate("새로운 정보가 들어왔다")
    print(f"   Dominant: {result.dominant_voice.value}")
    print(f"   Resonance: {result.resonance_strength:.2f}")
    
    # 테스트 2: "왜?" 질문
    print("\n📌 Test 2: Asking 'Why?'")
    result = dialogue.ask_why("사랑이 중요한가")
    print(f"   Dominant: {result.dominant_voice.value}")
    print(f"   Resonance: {result.resonance_strength:.2f}")
    
    # 테스트 3: 감정적 자극
    print("\n📌 Test 3: Emotional stimulus")
    result = dialogue.contemplate("슬픔을 느끼고 있다")
    print(f"   Dominant: {result.dominant_voice.value}")
    print(f"   Resonance: {result.resonance_strength:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
