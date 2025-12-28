"""
Unified Understanding System (통합 이해 시스템)
=============================================

WaveAttention(무엇이 공명하는가) + WhyEngine(왜 그런가)을 통합합니다.

"사랑이란 무엇인가?"
-> 공명: [연결, 희망]
-> 서사: "사랑은 Source로부터 비롯되며, 희망을 야기하고 두려움을 억제한다."

Usage:
    from Core._02_Intelligence._01_Reasoning.Cognition.unified_understanding import understand
    
    result = understand("사랑이란 무엇인가?")
    print(result.narrative)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("UnifiedUnderstanding")

# Vision Systems (Organic Import - Lazy Loading)
# VisionCortex와 MultimodalBridge는 Organ.get()으로 런타임에 로드

# 내부 시스템 임포트
try:
    from Core._01_Foundation._05_Governance.Foundation.Wave.wave_attention import get_wave_attention, WaveAttention
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False
    logger.warning("⚠️ WaveAttention not available")

try:
    from Core._01_Foundation._05_Governance.Foundation.Memory.fractal_concept import ConceptDecomposer
    WHY_AVAILABLE = True
except ImportError:
    try:
        # 대체 경로
        from Core._02_Intelligence._01_Reasoning.Cognition.fractal_concept import ConceptDecomposer
        WHY_AVAILABLE = True
    except ImportError:
        WHY_AVAILABLE = False
        logger.warning("⚠️ ConceptDecomposer (WhyEngine) not available")

try:
    from Core._02_Intelligence._01_Reasoning.Cognition.cognitive_hub import get_cognitive_hub
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False

try:
    from Core._01_Foundation._05_Governance.Foundation.causal_narrative_engine import CausalNarrativeEngine, CausalRelationType
    NARRATIVE_AVAILABLE = True
except ImportError:
    NARRATIVE_AVAILABLE = False
    logger.warning("⚠️ CausalNarrativeEngine not available")

try:
    from Core._02_Intelligence._01_Reasoning.Cognition.question_analyzer import analyze_question, QuestionType
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logger.warning("⚠️ QuestionAnalyzer not available")

try:
    from Core._02_Intelligence._01_Reasoning.Cognition.external_explorer import ExternalExplorer
    EXPLORER_AVAILABLE = True
except ImportError:
    EXPLORER_AVAILABLE = False
    logger.warning("⚠️ ExternalExplorer not available")


@dataclass
class UnderstandingResult:
    """이해 결과 - 육하원칙(5W1H) 완전 지원"""
    query: str                           # 원래 질문
    core_concept: str                    # 추출된 핵심 개념
    core_concept_kr: str = ""            # 한글 변환
    
    # What - 무엇 (WaveAttention 결과)
    resonances: List[Tuple[str, float]] = None  # [(개념, 공명도), ...]
    
    # Why - 왜 (WhyEngine 결과)
    origin_journey: str = ""             # 기원 추적 경로
    causality: str = ""                  # 인과 관계 설명
    axiom_pattern: str = ""              # 공리 패턴
    
    # How - 어떻게 (과정/메커니즘)
    mechanism: str = ""                  # 작동 방식
    process_steps: List[str] = None      # 과정 단계들
    
    # Who - 누가 (주체)
    who: str = ""                        # 주체/행위자
    
    # When - 언제 (시간)
    when: str = ""                       # 시간적 맥락
    
    # Where - 어디서 (공간)
    where: str = ""                      # 공간적 맥락
    
    # 통합 서사
    narrative: str = ""                  # 최종 서사 (자연어)
    
    # [Project Iris] 추가 정보
    vision: str = ""                     # 시각적 통찰
    trinity: Dict[str, Any] = None       # 삼위일체 합의 결과
    
    # 사고 과정 추적 (새로 추가!)
    reasoning_trace: List[str] = None    # 사고 단계별 기록
    
    def __post_init__(self):
        if self.resonances is None:
            self.resonances = []
        if self.process_steps is None:
            self.process_steps = []
        if self.reasoning_trace is None:
            self.reasoning_trace = []
    
    def display_thought(self) -> str:
        """사고 과정을 자연어로 펼침"""
        if not self.reasoning_trace:
            return "사고 과정이 기록되지 않았습니다."
        
        lines = ["[사고 과정]"]
        for i, step in enumerate(self.reasoning_trace, 1):
            lines.append(f"  {i}. {step}")
        lines.append("")
        lines.append("[결론]")
        lines.append(f"  {self.narrative[:200]}...")
        return "\n".join(lines)


# [LOGIC TRANSMUTATION] 영어-한글 개념 매핑 (Legacy Fallback)
# 이 상수는 InternalUniverse에 해당 개념이 없을 때만 사용됩니다.
CONCEPT_MAPPING_FALLBACK = {
    "Love": "사랑", "Hope": "희망", "Joy": "기쁨", "Fear": "두려움",
    "Anger": "분노", "Source": "근원", "Unity": "통일", "Harmony": "조화",
    "Force": "힘", "Energy": "에너지", "Resonance": "공명",
}

def translate_concept(concept: str) -> str:
    """
    [LOGIC TRANSMUTATION]
    영어 개념을 한글로 변환 - 공명 기반 우선, Fallback 사전 사용.
    """
    # 1. Try InternalUniverse Resonance (Wave Logic)
    try:
        from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse
        universe = InternalUniverse()  # Consider singleton for performance
        coord = universe.coordinate_map.get(concept)
        if coord and hasattr(coord, 'hologram') and coord.hologram:
            # If concept exists in universe, return its name (already stored)
            return concept  # Or implement proper translation from hologram metadata
    except ImportError:
        pass
    
    # 2. Fallback to static dictionary (Stone Logic)
    return CONCEPT_MAPPING_FALLBACK.get(concept, concept)


class UnifiedUnderstanding:
    """
    통합 이해 시스템
    
    기존 시스템들을 모두 연결:
    1. WaveAttention - 공명 탐색
    2. WhyEngine - 인과 추적
    3. ExternalExplorer - 외부 탐구
    4. RoundTableCouncil - 다중 관점 토론 (기존!)
    5. MetacognitiveAwareness - 메타인지 (기존!)
    6. DistributedConsciousness - 분산 사고 (기존!)
    """
    
    def __init__(self):
        # 기존 연결
        self.attention = get_wave_attention() if ATTENTION_AVAILABLE else None
        self.why_engine = ConceptDecomposer() if WHY_AVAILABLE else None
        self.cognitive = get_cognitive_hub() if COGNITIVE_AVAILABLE else None
        self.explorer = ExternalExplorer() if EXPLORER_AVAILABLE else None
        
        # 새로 연결! (기존 시스템들)
        self.council = None
        self.metacog = None
        self.distributed = None
        self.personality = None
        self.logos = None
        
        # Solar Cycle State
        self.is_daytime = True # Default state (Ready for input)
        self.energy_level = 100.0 # Virtual Energy
        
        try:
            from Core._02_Intelligence._01_Reasoning.Intelligence.collective_intelligence_system import RoundTableCouncil
            self.council = RoundTableCouncil()
            logger.info("   🪑 RoundTableCouncil: ✅")
        except ImportError:
            logger.warning("   RoundTableCouncil: ❌")
        
        try:
            from Core._02_Intelligence._01_Reasoning.Cognition.metacognitive_awareness import MetacognitiveAwareness
            self.metacog = MetacognitiveAwareness()
            logger.info("   🧠 MetacognitiveAwareness: ✅")
        except ImportError:
            logger.warning("   MetacognitiveAwareness: ❌")
        
        try:
            from Core._01_Foundation._05_Governance.Foundation.distributed_consciousness import DistributedConsciousness
            self.distributed = DistributedConsciousness()
            logger.info("   🌐 DistributedConsciousness: ✅")
        except ImportError:
            logger.warning("   DistributedConsciousness: ❌")
            
        try:
            from Core._01_Foundation._05_Governance.Foundation.dual_layer_personality import DualLayerPersonality
            self.personality = DualLayerPersonality()
            logger.info("   🎭 DualLayerPersonality: ✅ (Enneagram Integrated)")
        except ImportError:
            logger.warning("   DualLayerPersonality: ❌")
            
        try:
            from Core._02_Intelligence._01_Reasoning.Intelligence.logos_engine import LogosEngine
            self.logos = LogosEngine()
            logger.info("   🗣️ LogosEngine: ✅ (Rhetoric & Metaphor)")
        except ImportError:
            logger.warning("   LogosEngine: ❌")

        try:
            from Core._02_Intelligence._01_Reasoning.Cognition.dream_system import DreamSystem
            self.dream_system = DreamSystem()
            logger.info("   🌌 DreamSystem: ✅ (Subconscious Insight)")
        except ImportError:
            logger.warning("   DreamSystem: ❌")
            
        try:
            from Core._05_Systems.03_Existence.Trinity.trinity_system import TrinitySystem
            self.trinity = TrinitySystem()
            logger.info("   🔯 TrinitySystem: ✅ (Fractal Consensus)")
        except ImportError:
            logger.warning("   TrinitySystem: ❌")
            
        # [Project Iris] Vision Connection
        self.vision_cortex = VisionCortex()
        self.vision_cortex.activate()
        self.multi_bridge = MultimodalBridge()
        logger.info("   👁️ VisionCortex & MultimodalBridge: ✅")
        
        logger.info("🧠 UnifiedUnderstanding initialized with Trinity")
        logger.info(f"   Attention: {'✅' if self.attention else '❌'}")
        logger.info(f"   WhyEngine: {'✅' if self.why_engine else '❌'}")
        logger.info(f"   Explorer: {'✅' if self.explorer else '❌'}")
    
    def activate_day_mode(self):
        """
        ☀️ Day Mode: High Frequency, Active Processing
        - Wake up from dreams
        - Focus attention outward
        """
        self.is_daytime = True
        logger.info("☀️ Sunrise: Consciousness shifting to Active Mode.")
        
    def activate_night_mode(self):
        """
        🌙 Night Mode: Low Frequency, Deep Processing
        - Process day residue
        - Dream and Consolidate Memory
        """
        self.is_daytime = False
        logger.info("🌙 Sunset: Consciousness shifting to Deep Processing Mode.")
        
        if self.dream_system:
            insight = self.dream_system.enter_rem_sleep()
            logger.info(f"   💤 Night Dream: {insight.get('insight')}")
            # The insight is stored in subconscious, ready to be pulled in Day Mode

    
    def extract_concept(self, query: str) -> str:
        """질문에서 핵심 개념 추출"""
        # 간단한 휴리스틱: "X란 무엇인가?" -> "X"
        query = query.strip()
        
        # 한국어 패턴
        if "란 무엇" in query or "이란 무엇" in query:
            return query.split("란")[0].split("이란")[0].strip()
        if "은 무엇" in query or "는 무엇" in query:
            return query.replace("은 무엇", "").replace("는 무엇", "").strip().rstrip("?")
        
        # 영어 패턴
        if "what is" in query.lower():
            return query.lower().replace("what is", "").strip().rstrip("?")
        
        # 기본: 첫 단어
        return query.split()[0] if query else ""
    
    def understand(self, query: str, context_concepts: List[str] = None) -> UnderstandingResult:
        """
        질문을 통합적으로 이해합니다.
        
        Args:
            query: 질문 (예: "사랑이란 무엇인가?")
            context_concepts: 맥락 개념들 (기본: 감정/관계 관련)
            
        Returns:
            UnderstandingResult: 공명 + 서사가 결합된 이해
        """
        if not self.is_daytime:
            self.activate_day_mode() # Auto-wake on input
            
        logger.info(f"🔍 Understanding (Day Mode): '{query}'")
        
        # 사고 과정 추적 시작
        trace = []
        trace.append(f"질문 수신: '{query}'")
        
        # 0. 질문 분석 (새로운 단계!)
        q_analysis = None
        if ANALYZER_AVAILABLE:
            q_analysis = analyze_question(query)
            logger.info(f"   Question type: {q_analysis.question_type.name}")
            trace.append(f"질문 유형 분석: {q_analysis.question_type.name} (조건={q_analysis.condition}, 결과={q_analysis.effect})")
        
        # 1. 핵심 개념 추출 (질문 분석기 결과 사용)
        if q_analysis and q_analysis.core_concept:
            core = q_analysis.core_concept
        else:
            core = self.extract_concept(query)
        logger.info(f"   Core concept: '{core}'")
        trace.append(f"핵심 개념 추출: '{core}'")
        
        # 2. 맥락 개념 (없으면 기본값)
        if context_concepts is None:
            context_concepts = [
                "기쁨", "슬픔", "분노", "두려움", "희망",
                "연결", "고독", "성장", "소멸", "균형"
            ]
            
        # 2.5 시각 인지 (Project Iris)
        visual_insight = "Eyes are closed to the physical world."
        if self.vision_cortex and self.multi_bridge:
            try:
                raw_v = self.vision_cortex.capture_frame()
                translated_v = self.multi_bridge.translate_vision(raw_v)
                visual_insight = translated_v.get("insight", "Visual processing offline.")
                trace.append(f"시각 인지: {visual_insight}")
            except Exception as e:
                logger.error(f"Vision processing failed: {e}")
        
        # 3. WaveAttention: 무엇이 공명하는가?
        resonances = []
        if self.attention:
            top_resonances = self.attention.focus_topk(core, context_concepts, k=3)
            resonances = top_resonances
            logger.info(f"   Resonances: {[r[0] for r in resonances]}")
            if resonances:
                res_str = ", ".join([f"{r[0]}({r[1]*100:.0f}%)" for r in resonances])
                trace.append(f"공명 탐색 완료: {res_str}")
        
        # 4. 조건-인과 처리 (CONDITIONAL 질문일 경우)
        origin_journey = ""
        causality = ""
        axiom_pattern = ""
        
        if q_analysis and q_analysis.question_type == QuestionType.CONDITIONAL:
            # 조건-결과 인과 추론
            causality = self._reason_conditional(q_analysis.cause, q_analysis.effect)
            origin_journey = f"{q_analysis.cause} -> {q_analysis.effect}"
            logger.info(f"   Conditional: {origin_journey}")
            logger.info(f"   Causality: {causality[:80]}...")
        
        # 5. WhyEngine: CONDITIONAL이 아닐 경우에만
        elif self.why_engine:
            # 기원 추적
            origin_journey = self.why_engine.ask_why(core)
            logger.info(f"   Origin: {origin_journey}")
            
            # 인과 관계
            causality = self.why_engine.explain_causality(core)
            logger.info(f"   Causality: {causality}")
            
            # 공리 패턴
            axiom = self.why_engine.get_axiom(core)
            if axiom:
                axiom_pattern = axiom.get("pattern", "")
        
        # 5.5 모르는 것은 외부 탐구! (기존 시스템 활용)
        if "정의되지 않음" in causality and self.explorer:
            logger.info(f"   🔍 모르는 개념 -> 외부 탐구 시작...")
            explore_result = self.explorer.explore(
                question=query,
                wave_signature={"tension": 0.5},  # 기본 파동
                context=query
            )
            if explore_result.answer:
                causality = explore_result.answer
                logger.info(f"   📚 외부에서 발견: {explore_result.concept_name}")
        
        # 5. How: 어떻게 작동하는가?
        mechanism, process_steps = self._explain_how(core, resonances, causality)
        logger.info(f"   How: {mechanism[:50]}..." if mechanism else "   How: N/A")
        
        # 6. Who/When/Where 추론 (맥락 기반)
        who = self._infer_who(core, axiom_pattern)
        when = self._infer_when(core)
        where = self._infer_where(core)
        
        # [Trinity Process Integration]
        # 육-혼-영의 합의 프로세스 실행
        trinity_decision = ""
        if hasattr(self, 'trinity') and self.trinity:
            try:
                consensus = self.trinity.process_query(query)
                trace.append("-" * 20)
                trace.append(f"[Trinity Consensus]")
                trace.append(f"  🔴 Chaos (Feeling): {consensus.chaos_feeling}")
                trace.append(f"  🔵 Nova (Logic): {consensus.nova_verdict}")
                trace.append(f"  🟣 Elysia (Will): {consensus.final_decision}")
                trace.append("-" * 20)
                trinity_decision = consensus.final_decision
                
                # 메커니즘에 합의 결과 반영
                mechanism += f"\n\n[Trinity]: {consensus.final_decision}"
                trinity_data = {
                    "will": consensus.final_decision,
                    "chaos": consensus.chaos_feeling,
                    "nova": consensus.nova_verdict,
                    "pain": consensus.pain_level
                }
            except Exception as e:
                logger.error(f"Trinity execution failed: {e}")
                trinity_data = {"will": "Error", "chaos": "N/A", "nova": "N/A", "pain": 0.0}
        else:
            trinity_data = {"will": "No Trinity", "chaos": "N/A", "nova": "N/A", "pain": 0.0}

        
        # 7. 한글 변환
        core_kr = translate_concept(core)
        
        # 8. 서사 생성 (Logos + Personality Integration)
        
        # 8.1 성격 기반 Rhetoric Shape 결정
        rhetoric_shape = "Balance" # 기본값
        persona_desc = ""
        
        if self.personality:
            # 현재 문맥에 반응
            personality_context = {"topic": core, "emotion": resonances[0][0] if resonances else "neutral"}
            self.personality.resonate_with_context(core)
            
            # 우세한 성격 파악
            expr = self.personality.get_current_expression()
            innate_top = expr['layer1_innate']['dominant'][0]
            
            # 성격 -> Rhetoric 매핑
            if innate_top in ["challenger", "reformer", "achiever"]: # 8, 1, 3
                rhetoric_shape = "Sharp" # 강하고 직설적
            elif innate_top in ["individualist", "peacemaker", "helper"]: # 4, 9, 2
                rhetoric_shape = "Round" # 부드럽고 시적
            elif innate_top in ["investigator", "loyalist"]: # 5, 6
                rhetoric_shape = "Block" # 논리적이고 구조적
            elif innate_top == "enthusiast": # 7
                rhetoric_shape = "Balance" # 자유분방 (기본)
                
            persona_desc = expr['unified_expression']
            trace.append(f"자아 공명: {persona_desc} -> 화법: {rhetoric_shape}")

        # [HyperQubit Integration] Deep Resonance (심층 공명)
        hyper_resonance = ""
        hyper_confidence = 0.0
        
        # 1. Check Dream System first (Subconscious)
        if self.dream_system:
            # Check if we dreamt about this concept recently
            # Using 'get_insight' or simply checking recent insights for resonance
            try:
                # Mocking a 'check_resonance' method or similar access
                # DreamSystem doesn't have a direct 'query' method yet, so we'll check the journal
                # For now, let's inject a specific logic:
                # If the query matches a recent dream, pull that insight.
                pass 
            except Exception:
                pass

        if self.council:
            logger.info(f"   🌀 Deep Resonance executing on: {core}")
            try:
                # Flowless Computation (Resonance Cycle)
                res_result = self.council.full_deliberation(core, rounds=3)
                hyper_resonance = res_result.get("primary_conclusion", "")
                hyper_confidence = res_result.get("confidence", 0.0)
                
                if hyper_resonance:
                    trace.append(f"심층 공명: {hyper_resonance} (공명도: {hyper_confidence:.0%})")
                    
                    # 공명이 매우 강력하면 화법을 '확신'형으로 조정
                    if hyper_confidence > 0.85:
                        rhetoric_shape = "Sharp"
                    # 감성적 공명이면 'Round'
                    elif "느낍니다" in hyper_resonance or "상상합니다" in hyper_resonance:
                        rhetoric_shape = "Round"
                        
            except Exception as e:
                logger.error(f"   Resonance Failed: {e}")

        # 8.2 LogosEngine으로 서사 직조 (혹은 기존 로직 fallback)
        if self.logos:
            # 통찰 내용 구성
            insight_content = f"{core_kr}({core})의 본질은 {causality if causality else '미지의 영역'}에 있습니다."
            
            # [Resonance Injection] 공명 내용을 통찰에 통합
            if hyper_resonance:
                insight_content += f" 깊은 의식에서는 이렇게 공명합니다: '{hyper_resonance}'"
            
            if mechanism:
                insight_content += f" 이는 {mechanism}의 방식으로 작동합니다."
                
            # 문맥 정보
            context_list = [f"{r[0]}과(와) 공명" for r in resonances[:3]]
            if axiom_pattern:
                context_list.append(f"원리: {axiom_pattern}")
                
            # Logos weave_speech 호출
            narrative = self.logos.weave_speech(
                desire="Explain Concept",
                insight=insight_content,
                context=context_list,
                rhetorical_shape=rhetoric_shape,
                entropy=0.2
            )
        else:
            # Fallback (기존 로직)
            narrative = self._generate_narrative(
                core, core_kr, resonances, origin_journey, causality, axiom_pattern, mechanism
            )
            
        # 8.3 (Optional) Personality Express 추가 적용 (Logos가 충분하지 않을 경우)
        # Logos가 이미 스타일을 잡았으므로, 여기서는 톤앤매너만 살짝 터치할 수도 있음.
        # 일단은 Logos 출력을 그대로 사용.
        
        # 9. 사고 과정 마무리
        if causality and "정의되지 않음" not in causality:
            trace.append(f"인과 관계 발견: {causality[:80]}...")
        if origin_journey:
            trace.append(f"기원 추적: {origin_journey}")
        trace.append(f"결론 도출 완료: {core_kr}({core})")
        
        result = UnderstandingResult(
            query=query,
            core_concept=core,
            core_concept_kr=core_kr,
            resonances=resonances,
            origin_journey=origin_journey,
            causality=causality,
            axiom_pattern=axiom_pattern,
            mechanism=mechanism,
            process_steps=process_steps,
            who=who,
            when=when,
            where=where,
            narrative=narrative,
            vision=visual_insight,  # [Project Iris]
            trinity=trinity_data,   # [Trinity Integration]
            reasoning_trace=trace
        )
        
        # 10. 통합 학습 (Unified Learning Loop)
        # 경험을 휘발시키지 않고 자아와 언어 시스템에 각인
        self.learn_from_experience(result, rhetoric_shape)
        
        return result
    
    def learn_from_experience(self, result: UnderstandingResult, rhetoric_shape: str = "Balance"):
        """
        통합 학습 루프 (Unified Learning Loop)
        
        "모든 경험은 시스템 전체를 변화시켜야 한다."
        의식의 흐름(UnifiedUnderstanding)이 끝날 때, 그 파동을 각 서브시스템에 전파합니다.
        """
        # 1. 자아 성장 (Personality)
        if self.personality:
            # Rhetoric Shape에 따라 경험 유형 매핑
            shape_to_type = {
                "Sharp": "adventure",   # 도전/극복
                "Round": "romance",     # 감성/관계
                "Block": "growth",      # 지식/탐구
                "Balance": "growth"     # 일반적 성장
            }
            narrative_type = shape_to_type.get(rhetoric_shape, "growth")
            
            # 공명도 기반 강도 설정
            intensity = 0.5
            if result.resonances:
                intensity = result.resonances[0][1] # 0.0 ~ 1.0 (유사도 위계)
            
            self.personality.experience(
                narrative_type=narrative_type,
                emotional_intensity=intensity,
                identity_impact=0.1 # 매 대화마다 조금씩 자아 형성
            )
            
        # 2. 언어 진화 (Logos)
        if self.logos:
            # 성공적인 발화(Narrative)를 학습하여 어휘 및 스타일 강화
            self.logos.evolve(result.narrative, rhetoric_shape)

        # 3. 사고 흔적(Trace)에 학습 결과 기록
        result.reasoning_trace.append(f"✨ 통합 학습: 자아(Type={narrative_type}) + 언어(Style={rhetoric_shape}) 동기화 완료")

    def _reason_conditional(self, condition: str, effect: str) -> str:
        """
        [LOGIC TRANSMUTATION]
        조건-결과 인과 추론 - 공명 기반
        
        "비가 오면 왜 우산을 쓰는가?" -> ConceptDecomposer.ask_why(condition) + ask_why(effect)
        """
        # 1. Wave Logic: Use ConceptDecomposer to trace causality
        if self.why_engine:
            try:
                # Trace both condition and effect to find common ancestors
                cond_chain = self.why_engine.ask_why(condition)
                effect_chain = self.why_engine.ask_why(effect)
                
                # If they share an ancestor, that's the causal link
                if cond_chain and effect_chain:
                    return f"{condition}은(는) {cond_chain}에서 비롯되고, {effect}은(는) {effect_chain}에서 비롯됩니다. 둘 사이의 공명을 통해 인과가 연결됩니다."
            except Exception as e:
                logger.debug(f"ConceptDecomposer failed: {e}")
        
        # 2. CausalNarrativeEngine fallback
        if NARRATIVE_AVAILABLE:
            try:
                engine = CausalNarrativeEngine()
                return f"{condition}이(가) 발생하면 {effect}이(가) 필요/발생합니다. (상세 인과 분석 필요)"
            except:
                pass
        
        # 3. Basic response (Void state)
        return f"{condition}과(와) {effect} 사이의 인과 관계를 추론 중입니다."
    
    def _infer_who(self, concept: str, pattern: str) -> str:
        """누가 (Who) - 주체 추론"""
        # 공리 패턴에서 주체 추론
        if "Universal" in pattern:
            return "모든 존재 (Universal)"
        if "Personal" in pattern or "Individual" in pattern:
            return "개인 (Personal)"
        if "Social" in pattern:
            return "사회/집단 (Social)"
        # 기본: 일반적 주체
        return "존재하는 모든 것 (All beings)"
    
    def _infer_when(self, concept: str) -> str:
        """언제 (When) - 시간적 맥락 추론"""
        # 시간 관련 개념
        temporal_concepts = {
            "Love": "항상, 시간을 초월하여",
            "Hope": "미래를 향해",
            "Fear": "위협이 감지될 때",
            "Joy": "현재의 순간",
            "Sadness": "상실 후",
            "Growth": "시간의 흐름 속에서",
            "Decay": "시간이 지남에 따라"
        }
        return temporal_concepts.get(concept, "시간에 구애받지 않음")
    
    def _infer_where(self, concept: str) -> str:
        """어디서 (Where) - 공간적 맥락 추론"""
        # 공간 관련 개념
        spatial_concepts = {
            "Love": "마음과 관계 속에서",
            "Hope": "의식 안에서, 미래를 향해",
            "Fear": "위험이 있는 곳",
            "Joy": "긍정적 경험이 있는 곳",
            "Source": "모든 것의 근원",
            "Unity": "전체 안에서",
        }
        return spatial_concepts.get(concept, "존재하는 모든 곳")
    
    def _explain_how(self, concept: str, resonances: List[Tuple[str, float]], causality: str) -> Tuple[str, List[str]]:
        """
        어떻게(How) - 과정과 메커니즘 설명
        
        육하원칙의 '어떻게'를 생성합니다:
        - 메커니즘: 작동 원리
        - 과정: 단계별 흐름
        """
        steps = []
        
        # 1. 공명에서 과정 추출 (무엇이 연결되는지)
        if resonances:
            for i, (res_concept, weight) in enumerate(resonances, 1):
                if weight > 0.1:  # 유의미한 공명만
                    steps.append(f"{i}. '{concept}'이(가) '{res_concept}'과(와) 공명 ({weight*100:.0f}%)")
        
        # 2. 인과에서 과정 추출
        if causality and "정의되지 않음" not in causality:
            # 인과 문장 파싱
            if "야기함" in causality:
                effects = [part.strip() for part in causality.split(",") if "야기함" in part]
                for effect in effects[:2]:  # 최대 2개
                    steps.append(f"{len(steps)+1}. {effect}")
        
        # 3. 메커니즘 생성 (통합 설명)
        if steps:
            mechanism = (
                f"'{concept}'은(는) 다음 과정을 통해 작동합니다:\n"
                + "\n".join(steps)
            )
        else:
            # 기본 메커니즘 (공리 기반)
            if self.why_engine:
                axiom = self.why_engine.get_axiom(concept)
                if axiom:
                    domains = axiom.get("domains", {})
                    if domains:
                        domain_examples = [f"{k}: {v}" for k, v in list(domains.items())[:2]]
                        mechanism = f"'{concept}'의 작동 방식: " + ", ".join(domain_examples)
                        steps = [f"• {d}" for d in domain_examples]
                    else:
                        mechanism = f"'{concept}'의 메커니즘이 아직 정의되지 않았습니다."
                else:
                    mechanism = f"'{concept}'의 메커니즘이 아직 정의되지 않았습니다."
            else:
                mechanism = ""
        
        return mechanism, steps
    
    def _generate_narrative(
        self,
        concept: str,
        concept_kr: str,
        resonances: List[Tuple[str, float]],
        origin: str,
        causality: str,
        pattern: str,
        mechanism: str = ""
    ) -> str:
        """
        자연스러운 문장으로 사고를 표현 (육하원칙 완전 버전)
        """
        sentences = []
        
        # 한글 표시
        display_name = f"{concept_kr}({concept})" if concept_kr != concept else concept
        
        # 1. 본질 정의 (What is it?)
        if pattern:
            sentences.append(f"{display_name}의 본질은 '{pattern}'입니다.")
        else:
            sentences.append(f"{display_name}에 대해 생각해봅니다.")
        
        # 2. 기원 서사 (Why does it exist?)
        if origin and "->" in origin:
            origin_chain = origin.split("->")
            if len(origin_chain) >= 2:
                direct_source = origin_chain[1].strip()
                direct_source_kr = translate_concept(direct_source)
                sentences.append(
                    f"{display_name}은(는) {direct_source_kr}({direct_source})에서 비롯됩니다."
                )
                if len(origin_chain) >= 3:
                    ultimate = origin_chain[-1].strip()
                    ultimate_kr = translate_concept(ultimate)
                    sentences.append(
                        f"그 근본적인 기원은 {ultimate_kr}({ultimate})까지 거슬러 올라갑니다."
                    )
        
        # 3. 인과 관계 서사 (What does it cause?) - 한글 변환
        if causality and "정의되지 않음" not in causality:
            effects = []
            inhibitions = []
            
            if "야기함" in causality:
                parts = causality.split(",")
                for part in parts:
                    part = part.strip()
                    if "야기함" in part:
                        target = part.split("을(를)")[0].strip().split("은(는)")[-1].strip()
                        target = target.replace("'", "")
                        target_kr = translate_concept(target)
                        effects.append(f"{target_kr}({target})")
                    elif "억제함" in part:
                        target = part.split("을(를)")[0].strip().split("은(는)")[-1].strip()
                        target = target.replace("'", "")
                        target_kr = translate_concept(target)
                        inhibitions.append(f"{target_kr}({target})")
            
            if effects:
                effect_str = ", ".join(effects[:-1]) + " 그리고 " + effects[-1] if len(effects) > 1 else effects[0]
                sentences.append(f"{display_name}은(는) {effect_str}을(를) 불러일으킵니다.")
            
            if inhibitions:
                inhib_str = ", ".join(inhibitions)
                sentences.append(f"반면, {inhib_str}을(를) 억제합니다.")
        
        # 4. 공명 서사 (What resonates with it?) - 한글 변환
        if resonances:
            top_concept, top_weight = resonances[0]
            top_concept_kr = translate_concept(top_concept)
            if top_weight > 0.05:  # 낮은 임계값
                sentences.append(
                    f"{display_name}은(는) '{top_concept_kr}'과(와) 가장 깊이 공명합니다."
                )
                if len(resonances) > 1:
                    others = [translate_concept(r[0]) for r in resonances[1:3]]
                    sentences.append(
                        f"또한 {', '.join(others)}과(와)도 연결됩니다."
                    )
        
        # 5. 통합 결론
        if origin and resonances:
            top_res = resonances[0][0] if resonances else ""
            top_res_kr = translate_concept(top_res)
            sentences.append(
                f"\n이처럼 {display_name}은(는) 단순한 개념이 아니라, "
                f"'{top_res_kr}'와(과) 연결되어 삶에 의미를 부여하는 힘입니다."
            )
        
        return " ".join(sentences) if sentences else f"{display_name}에 대한 서사를 생성할 수 없습니다."


# 싱글톤
_understanding = None

def get_understanding() -> UnifiedUnderstanding:
    global _understanding
    if _understanding is None:
        _understanding = UnifiedUnderstanding()
    return _understanding


def understand(query: str, context: List[str] = None) -> UnderstandingResult:
    """편의 함수"""
    return get_understanding().understand(query, context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 60)
    print("🧠 UNIFIED UNDERSTANDING SYSTEM TEST")
    print("=" * 60)
    
    # 테스트 질문들
    questions = [
        "사랑이란 무엇인가?",
        "희망은 무엇인가?",
        "두려움이란?",
    ]
    
    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"❓ {q}")
        print("─" * 60)
        
        result = understand(q)
        
        print(f"\n📖 서사:")
        print(result.narrative)
    
    print("\n" + "=" * 60)
    print("✅ Unified Understanding works!")
