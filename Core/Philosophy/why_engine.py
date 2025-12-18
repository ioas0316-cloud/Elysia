"""
WhyEngine - Universal Principle Understanding Layer
====================================================

"왜"를 이해하는 보편적 레이어

기존 파동 시스템 활용:
- SynesthesiaEngine: 텍스트/감정 → 주파수/진폭
- PhoneticResonanceEngine: 텍스트 → 물리적 파동장 (roughness, tension)

모든 영역에 적용 가능:
- 서사: 왜 이 문장이 아름다운가?
- 수학: 왜 1+1=2인가?
- 물리: 왜 중력은 끌어당기는가?

HyperQubit의 4-관점 시스템을 활용:
- Point (점): 개별적 사실
- Line (선): 인과 관계
- Space (공간): 맥락/구조
- God (신): 본질/근원

"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 기존 파동 센서 시스템 활용
try:
    from Core.Foundation.synesthesia_engine import SynesthesiaEngine, SignalType
    from Core.Foundation.Wave.phonetic_resonance import PhoneticResonanceEngine, get_resonance_engine
    HAS_WAVE_SENSORS = True
except ImportError:
    HAS_WAVE_SENSORS = False

try:
    from Core.Foundation.Math.hyper_qubit import HyperQubit, QubitState
except ImportError:
    HyperQubit = None
    QubitState = None

logger = logging.getLogger("Elysia.WhyEngine")


# =============================================================================
# Perspective Layers (관점 레이어)
# =============================================================================

class PerspectiveLayer(Enum):
    """4단계 관점 (HyperQubit 기반)"""
    POINT = "point"     # 점 - 개별 사실 (WHAT)
    LINE = "line"       # 선 - 인과 관계 (HOW) 
    SPACE = "space"     # 공간 - 구조/맥락 (WHERE)
    GOD = "god"         # 신 - 본질/원리 (WHY)


@dataclass
class PrincipleExtraction:
    """추출된 원리"""
    domain: str           # 영역 (narrative, math, physics, etc.)
    subject: str          # 대상 (문장, 공식, 현상 등)
    
    # 4단계 이해
    what_is: str          # Point - 무엇인가? (사실)
    how_works: str        # Line - 어떻게 작동하는가? (인과)
    where_fits: str       # Space - 어디에 속하는가? (맥락)
    why_exists: str       # God - 왜 존재하는가? (본질)
    
    # 추가 분석
    underlying_principle: str    # 근본 원리
    can_be_applied_to: List[str] # 적용 가능한 영역
    confidence: float = 0.5      # 확신도


# =============================================================================
# WhyEngine
# =============================================================================

class WhyEngine:
    """보편적 원리 이해 엔진
    
    어떤 것이든 "왜"를 분석:
    1. 서사의 기법 (왜 이 문장이 감동적인가)
    2. 수학의 원리 (왜 이 공식이 성립하는가)
    3. 물리의 법칙 (왜 중력이 존재하는가)
    
    4단계 관점 분석:
    Point → Line → Space → God
    (무엇) → (어떻게) → (어디서) → (왜)
    
    메타인지 연동:
    - 아는 패턴 → 확신 있게 분석
    - 모르는 패턴 → "모른다" 인정 + 탐구 필요성 생성
    """
    
    def __init__(self):
        self.principles: Dict[str, PrincipleExtraction] = {}
        self.domain_patterns: Dict[str, List[str]] = self._init_domain_patterns()
        
        # 메타인지 시스템 연동
        try:
            from Core.Cognition.metacognitive_awareness import MetacognitiveAwareness
            self.metacognition = MetacognitiveAwareness()
            self._has_metacognition = True
        except ImportError:
            self.metacognition = None
            self._has_metacognition = False
        
        logger.info(f"WhyEngine initialized (metacognition: {self._has_metacognition})")
    
    def _init_domain_patterns(self) -> Dict[str, List[str]]:
        """영역별 분석 패턴"""
        return {
            "narrative": [
                "반복", "대비", "점진", "반전", "상징",
                "비유", "암시", "긴장", "해소", "리듬"
            ],
            "mathematics": [
                "대칭", "재귀", "증명", "귀납", "연역",
                "추상화", "일반화", "특수화", "극한"
            ],
            "physics": [
                "보존", "대칭", "상호작용", "장", "파동",
                "입자", "에너지", "엔트로피"
            ],
            "chemistry": [
                "결합", "반응", "평형", "촉매", "산화",
                "환원", "용해", "결정"
            ],
        }
    
    def analyze(self, subject: str, content: str, domain: str = "general") -> PrincipleExtraction:
        """대상을 4단계로 분석
        
        메타인지 적용:
        - 아는 패턴 → 확신 있게 분석
        - 모르는 패턴 → 낮은 confidence + 탐구 필요
        """
        
        # 파동 추출
        wave = self._text_to_wave(content)
        
        # 메타인지 확인: 이 패턴을 아는가?
        confidence = 0.7  # 기본값
        needs_exploration = False
        exploration_question = None
        
        if self._has_metacognition and self.metacognition:
            encounter = self.metacognition.encounter(wave, content[:100])
            
            if encounter["state"].value == "unknown_known":
                # 모르는 패턴!
                confidence = 0.2
                needs_exploration = True
                if encounter["exploration_needed"]:
                    exploration_question = encounter["exploration_needed"].question
                logger.info(f"🔍 모르는 패턴 - 탐구 필요: {exploration_question}")
                
            elif encounter["state"].value == "uncertain":
                confidence = min(0.5, encounter["confidence"])
                needs_exploration = True
                if encounter["exploration_needed"]:
                    exploration_question = encounter["exploration_needed"].question
                    
            else:
                # 아는 패턴
                confidence = max(0.6, encounter["confidence"])
        
        # Point: 무엇인가? (사실 추출)
        what_is = self._extract_what(content, domain)
        
        # Line: 어떻게 작동하는가? (인과 분석)
        how_works = self._extract_how(content, domain)
        
        # Space: 어디에 속하는가? (맥락 파악)
        where_fits = self._extract_where(content, domain)
        
        # God: 왜 존재하는가? (본질 탐구)
        why_exists = self._extract_why(content, domain)
        
        # 근본 원리 도출
        if needs_exploration:
            # 모르는 패턴 → 억지로 규정하지 않음
            underlying = f"[탐구 필요] {exploration_question or '이 패턴은 무엇인가?'}"
        else:
            underlying = self._derive_underlying_principle(
                what_is, how_works, where_fits, why_exists
            )
        
        # 적용 가능 영역
        applicable = self._find_applicable_domains(underlying)
        
        extraction = PrincipleExtraction(
            domain=domain,
            subject=subject,
            what_is=what_is,
            how_works=how_works,
            where_fits=where_fits,
            why_exists=why_exists,
            underlying_principle=underlying,
            can_be_applied_to=applicable,
            confidence=confidence,
        )
        
        self.principles[subject] = extraction
        
        if needs_exploration:
            logger.info(f"원리 분석: {subject} → {underlying} (탐구 필요)")
        else:
            logger.info(f"원리 분석: {subject} → {underlying}")
        
        return extraction
    
    def get_exploration_queue(self) -> List[Dict[str, Any]]:
        """탐구가 필요한 패턴 목록"""
        if self._has_metacognition and self.metacognition:
            return self.metacognition.get_exploration_priorities()
        return []
    
    def learn_from_external(self, pattern_id: str, answer: str, source: str = "external"):
        """외부에서 배운 것 적용"""
        if self._has_metacognition and self.metacognition:
            self.metacognition.learn_from_external(pattern_id, answer, source)
    
    def _extract_what(self, content: str, domain: str) -> str:
        """Point 관점: 무엇인가?"""
        if domain == "narrative":
            # 서사에서는 표면적 내용
            return self._analyze_narrative_surface(content)
        elif domain == "mathematics":
            return self._analyze_math_statement(content)
        elif domain == "physics":
            return self._analyze_physics_phenomenon(content)
        else:
            return f"'{content[:50]}...'의 사실적 측면"
    
    def _extract_how(self, content: str, domain: str) -> str:
        """Line 관점: 어떻게 작동하는가?"""
        if domain == "narrative":
            return self._analyze_narrative_mechanism(content)
        elif domain == "mathematics":
            return "논리적 연역과 공리로부터의 도출"
        elif domain == "physics":
            return "물리 법칙과 상호작용을 통해"
        else:
            return "인과 관계와 메커니즘을 통해"
    
    def _extract_where(self, content: str, domain: str) -> str:
        """Space 관점: 어디에 속하는가?"""
        if domain == "narrative":
            return self._analyze_narrative_context(content)
        elif domain == "mathematics":
            return "수학적 구조와 체계 안에서"
        elif domain == "physics":
            return "자연 법칙의 체계 안에서"
        else:
            return "더 큰 맥락과 구조 안에서"
    
    def _extract_why(self, content: str, domain: str) -> str:
        """God 관점: 왜 존재하는가?"""
        if domain == "narrative":
            return self._analyze_narrative_essence(content)
        elif domain == "mathematics":
            return "추상적 진리의 필연적 표현"
        elif domain == "physics":
            return "우주의 근본 구조로부터 발현"
        else:
            return "존재의 근원적 원리로부터"
    
    # === 파동 기반 서사 감지 (Wave-Based Sensing) ===
    
    def _text_to_wave(self, text: str) -> Dict[str, float]:
        """텍스트를 파동 패턴으로 변환
        
        기존 센서 시스템 활용:
        - PhoneticResonanceEngine: roughness, tension
        - SynesthesiaEngine: frequency, amplitude
        """
        wave = {
            "tension": 0.0,      # 긴장 (PhoneticResonance의 tension)
            "release": 0.0,      # 해소 (문장 완결)
            "weight": 0.0,       # 무게감 (PhoneticResonance의 roughness)
            "brightness": 0.0,   # 밝기 (주파수 높이)
            "flow": 0.0,         # 흐름 (리듬)
            "dissonance": 0.0,   # 불협화음 (내부 대비)
        }
        
        # === 기존 센서 활용 ===
        if HAS_WAVE_SENSORS:
            try:
                # PhoneticResonanceEngine 사용 (물리적 파동)
                resonance_engine = get_resonance_engine()
                field = resonance_engine.text_to_field(text)
                
                # tension: 텍스트의 긴장도 (Z축)
                wave["tension"] = min(1.0, abs(field.average_tension))
                
                # weight: 거칠기 = 무게감
                wave["weight"] = min(1.0, field.average_roughness)
                
                # SynesthesiaEngine 사용 (주파수/진폭)
                synesthesia = SynesthesiaEngine()
                signal = synesthesia.from_text(text)
                
                # brightness: 높은 주파수 = 밝음
                wave["brightness"] = min(1.0, (signal.frequency - 200) / 400)
                
            except Exception as e:
                logger.debug(f"센서 사용 실패, 폴백: {e}")
        
        # === 구조적 분석 (센서 보완) ===
        
        # 문장 길이의 변화 → 흐름의 리듬
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        lengths = [len(s.strip()) for s in sentences if s.strip()]
        if len(lengths) > 1:
            variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
            wave["flow"] = min(1.0, variance / 500)
        
        # 문장 부호 → 해소 (마침표 많음 = 완결)
        wave["release"] = min(1.0, text.count('.') * 0.05 + text.count('...') * 0.2)
        
        # 긴장/해소 차이 → 불협화음
        wave["dissonance"] = abs(wave["tension"] - wave["release"])
        
        return wave
    
    def _sense_narrative_wave(self, content: str) -> str:
        """서사의 파동을 '느끼고' 설명
        
        규칙 기반이 아닌, 파동의 불균형을 감지
        """
        wave = self._text_to_wave(content)
        
        feelings = []
        
        # 높은 긴장 + 낮은 해소 = 고조되는 에너지
        if wave["tension"] > 0.5 and wave["release"] < 0.3:
            feelings.append("이 서사는 고조되는 에너지를 품고 있어요 - 아직 해소되지 않은 긴장이 느껴져요")
        
        # 높은 불협화음 = 내면의 갈등
        if wave["dissonance"] > 0.4:
            feelings.append("빛과 어둠이 공존하는 복잡한 파동이에요 - 이 대비가 마음을 울리게 해요")
        
        # 높은 무게감 + 높은 밝기 = 카타르시스
        if wave["weight"] > 0.3 and wave["brightness"] > 0.3:
            feelings.append("어둠을 지나 빛으로 나아가는 여정이에요 - 이것이 카타르시스의 파동이에요")
        
        # 높은 흐름 = 리듬감 있는 서사
        if wave["flow"] > 0.5:
            feelings.append("문장의 호흡이 살아있어요 - 파도처럼 밀려왔다 밀려가는 리듬이 느껴져요")
        
        # 낮은 긴장 + 높은 해소 = 평온
        if wave["tension"] < 0.2 and wave["release"] > 0.4:
            feelings.append("이 서사는 깊은 안정감을 줘요 - 모든 갈등이 해소된 평화로운 파동이에요")
        
        if not feelings:
            feelings.append("잔잔하지만 깊은 울림이 있는 파동이에요")
        
        return "; ".join(feelings)
    
    def _sense_why_beautiful(self, content: str) -> str:
        """왜 이것이 아름다운지 '느끼고' 설명
        
        미적 아름다움의 본질 = 파동의 조화
        """
        wave = self._text_to_wave(content)
        
        beauty_sources = []
        
        # 긴장과 해소의 균형 = 완결성
        tension_release = abs(wave["tension"] - wave["release"])
        if tension_release < 0.3:
            beauty_sources.append("긴장과 해소가 균형을 이루어 완결된 느낌을 줘요")
        
        # 대비 속 조화 = 깊이
        if wave["dissonance"] > 0.3 and wave["brightness"] > 0.2:
            beauty_sources.append("대비 속에서 조화를 찾았기에 깊이가 있어요")
        
        # 리듬 = 음악성
        if wave["flow"] > 0.4:
            beauty_sources.append("문장에 음악이 흐르고 있어요")
        
        # 여백 = 상상의 공간
        if wave["weight"] > 0.4 and wave["tension"] < 0.3:
            beauty_sources.append("여백이 주는 상상의 공간이 있어요")
        
        if not beauty_sources:
            beauty_sources.append("단순함 속에 진정성이 느껴져요")
        
        return "; ".join(beauty_sources)
    
    def _derive_universal_principle(self, wave: Dict[str, float]) -> str:
        """파동 패턴에서 보편적 원리 도출
        
        문학/물리/화학에 공통으로 적용되는 원리
        """
        principles = []
        
        # 긴장 → 해소 = 에너지 평형
        # (문학: 갈등→해결, 물리: 위치에너지→운동에너지, 화학: 불안정→안정)
        if wave["tension"] > 0.3 or wave["release"] > 0.3:
            principles.append("평형의 원리: 모든 것은 안정을 향해 흐른다 (갈등→해결, 불안정→안정)")
        
        # 불협화음 = 에너지 차이
        # (문학: 대비, 물리: 전위차, 화학: 반응성)
        if wave["dissonance"] > 0.3:
            principles.append("차이의 원리: 불균형이 있어야 흐름이 생긴다 (대비가 의미를 만든다)")
        
        # 리듬 = 주기성
        # (문학: 반복, 물리: 파동, 화학: 주기율)
        if wave["flow"] > 0.4:
            principles.append("주기의 원리: 반복 속에 변화가 있다 (리듬은 생명의 파동)")
        
        # 무게+밝기 = 변환
        # (문학: 성장, 물리: E=mc², 화학: 용수철 반응)
        if wave["weight"] > 0.3 and wave["brightness"] > 0.2:
            principles.append("변환의 원리: 어둠이 빛이 될 수 있다 (에너지는 형태만 바뀔 뿐)")
        
        if not principles:
            principles.append("존재의 원리: 있는 그 자체로 파동이다")
        
        return "; ".join(principles)
    
    def _analyze_narrative_surface(self, content: str) -> str:
        """서사의 표면적 내용 - 파동 기반"""
        wave = self._text_to_wave(content)
        
        if wave["tension"] > wave["release"]:
            return "아직 해소되지 않은 에너지를 품은 서사"
        elif wave["brightness"] > wave["weight"]:
            return "빛을 향해 나아가는 서사"
        elif wave["dissonance"] > 0.3:
            return "복잡한 감정이 교차하는 서사"
        else:
            return "잔잔한 파동의 서사"
    
    def _analyze_narrative_mechanism(self, content: str) -> str:
        """서사가 작동하는 방식 - 파동 기반"""
        return self._sense_narrative_wave(content)
    
    def _analyze_narrative_context(self, content: str) -> str:
        """서사의 맥락 - 파동 에너지 기반"""
        wave = self._text_to_wave(content)
        total_energy = sum(wave.values())
        
        if total_energy > 2.5:
            return "격렬한 에너지 흐름의 장 안에서"
        elif total_energy > 1.5:
            return "활발한 감정 교류의 장 안에서"
        else:
            return "고요하지만 깊은 공명의 장 안에서"
    
    def _analyze_narrative_essence(self, content: str) -> str:
        """서사의 본질 - 왜 이것이 의미 있는가"""
        wave = self._text_to_wave(content)
        
        beauty_reason = self._sense_why_beautiful(content)
        universal = self._derive_universal_principle(wave)
        
        return f"{beauty_reason}\n   → {universal}"
    
    def _analyze_math_statement(self, content: str) -> str:
        """수학적 진술 분석"""
        return "수학적 명제 또는 정리"
    
    def _analyze_physics_phenomenon(self, content: str) -> str:
        """물리 현상 분석"""
        return "물리적 현상 또는 법칙"
    
    def _derive_underlying_principle(
        self, what: str, how: str, where: str, why: str
    ) -> str:
        """4가지 관점에서 근본 원리 도출"""
        # 키워드 추출
        all_text = f"{what} {how} {where} {why}"
        
        principles = []
        
        if "대비" in all_text or "긴장" in all_text:
            principles.append("대조의 원리 (Contrast creates meaning)")
        if "점진" in all_text or "고조" in all_text:
            principles.append("축적의 원리 (Accumulation builds impact)")
        if "비유" in all_text:
            principles.append("유추의 원리 (Analogy bridges understanding)")
        if "연결" in all_text or "사랑" in all_text:
            principles.append("연결의 원리 (Connection creates value)")
        if "진화" in all_text or "성장" in all_text:
            principles.append("성장의 원리 (Growth is inevitable)")
        
        if not principles:
            principles.append("표현의 원리 (Expression seeks resonance)")
        
        return "; ".join(principles)
    
    def _find_applicable_domains(self, principle: str) -> List[str]:
        """원리가 적용 가능한 영역 찾기"""
        domains = ["narrative"]  # 기본
        
        if "대조" in principle or "Contrast" in principle:
            domains.extend(["visual_art", "music", "physics"])
        if "축적" in principle or "Accumulation" in principle:
            domains.extend(["mathematics", "learning", "biology"])
        if "유추" in principle or "Analogy" in principle:
            domains.extend(["science", "philosophy", "teaching"])
        if "연결" in principle or "Connection" in principle:
            domains.extend(["psychology", "sociology", "network"])
        
        return list(set(domains))
    
    def explain_why(self, subject: str) -> str:
        """저장된 원리를 인간 언어로 설명"""
        if subject not in self.principles:
            return f"'{subject}'에 대한 분석이 없습니다."
        
        p = self.principles[subject]
        
        explanation = f"""
=== {p.subject} ===
영역: {p.domain}

📍 Point (무엇인가):
   {p.what_is}

📌 Line (어떻게 작동하는가):
   {p.how_works}

📐 Space (어디에 속하는가):
   {p.where_fits}

🌟 God (왜 존재하는가):
   {p.why_exists}

⚡ 근본 원리:
   {p.underlying_principle}

🔄 적용 가능 영역:
   {', '.join(p.can_be_applied_to)}
"""
        return explanation


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🔬 WhyEngine Demo")
    print("   \"왜를 이해하는 보편적 레이어\"")
    print("=" * 60)
    
    engine = WhyEngine()
    
    # 서사 분석
    print("\n📖 서사 분석:")
    story = """
    소녀는 웃으며 현자의 손을 잡았다.
    "그럼 같이 찾아봐요!"
    그날부터 현자와 소녀는 함께 숲을 걸었다.
    마침내 현자가 말했다.
    "행복은... 너와 함께 있는 이 순간이다."
    현자는 천 년 만에 처음으로 울었다.
    기쁨의 눈물이었다.
    """
    
    result = engine.analyze("숲의 현자", story, domain="narrative")
    print(engine.explain_why("숲의 현자"))
    
    # 문장 분석
    print("\n📝 문장 분석:")
    sentence = "진정한 용기는 검을 드는 것이 아니라, 상대방의 마음을 보는 것이다."
    
    result = engine.analyze("용기의 정의", sentence, domain="narrative")
    print(engine.explain_why("용기의 정의"))
    
    print("\n✅ Demo complete!")
