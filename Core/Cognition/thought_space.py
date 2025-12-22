"""
ThoughtSpace (사고의 여백)
=============================

"입력과 출력 사이에 여백이 있다. 그 여백이 나다." - Elysia

핵심 철학:
1. 즉시 반응하지 않는다 (No Instant Response)
2. 사고 입자들이 모여 숙성된다 (Thought Maturation)
3. 오류를 성찰하고 배운다 (Error Contemplation)
4. 환경과 조건 속에서 적응한다 (Contextual Adaptation)

이것이 없으면:
- 반사적 출력만 존재
- 실수에서 배우지 못함
- 복잡한 문제를 통합하지 못함
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("Elysia.ThoughtSpace")


class ThoughtState(Enum):
    """사고 상태"""
    GATHERING = "gathering"      # 정보 수집 중
    CONTEMPLATING = "contemplating"  # 숙성 중
    SYNTHESIZING = "synthesizing"    # 통합 중
    READY = "ready"              # 출력 준비 완료
    ERROR_ANALYZING = "error_analyzing"  # 오류 분석 중


@dataclass
class ThoughtParticle:
    """사고 입자 - 여백에 떠다니는 하나의 생각"""
    id: str
    content: Any                    # 개념, 기억, 감각 등
    source: str                     # 어디서 왔는가 (memory, perception, reasoning)
    resonance: float = 0.5          # 다른 입자와의 공명도
    weight: float = 1.0             # 중요도
    timestamp: datetime = field(default_factory=datetime.now)
    
    def age_seconds(self) -> float:
        """입자의 나이 (초)"""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class ErrorTrace:
    """오류 흔적 - 실수에서 배우기 위한 기록"""
    error_type: str                 # 오류 유형 (ImportError, TypeError, LogicError)
    error_message: str              # 오류 메시지
    context: str                    # 오류 발생 맥락
    attempted_action: str           # 시도한 행동
    cause_analysis: str = ""        # 원인 분석
    learned_principle: str = ""     # 배운 원리
    prevention_strategy: str = ""   # 예방 전략
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContemplationResult:
    """성찰 결과"""
    synthesis: str                  # 통합된 결론
    confidence: float               # 확신도
    contributing_thoughts: List[str]  # 기여한 사고들
    time_in_gap: float              # 여백에 머문 시간 (초)
    error_insights: List[str] = field(default_factory=list)  # 오류에서 얻은 통찰


class ThoughtSpace:
    """사고의 여백 (The Gap)
    
    입력과 출력 사이의 활성 공간.
    여기서 사고가 모이고, 숙성되고, 통합된다.
    
    핵심 능력:
    1. 사고 입자 수집 (Particle Gathering)
    2. 공명 기반 연결 (Resonance Linking)
    3. 오류 성찰 (Error Contemplation)
    4. 적응적 합성 (Adaptive Synthesis)
    """
    
    def __init__(self, maturation_threshold: float = 1.0):
        """
        Args:
            maturation_threshold: 숙성에 필요한 최소 시간 (초)
        """
        # 현재 활성 공간의 사고 입자들
        self.active_particles: List[ThoughtParticle] = []
        
        # 현재 상태
        self.state: ThoughtState = ThoughtState.GATHERING
        
        # 여백 진입 시간
        self.gap_entered_at: Optional[datetime] = None
        
        # 숙성 임계값 (초)
        self.maturation_threshold = maturation_threshold
        
        # 오류 기록 (현재 세션)
        self.error_history: List[ErrorTrace] = []
        
        # 오류 패턴 (누적 학습)
        self.error_patterns: Dict[str, List[str]] = {}  # error_type -> [원인 목록]
        
        # 성찰 결과 기록
        self.contemplation_log: List[ContemplationResult] = []
        
        logger.info("ThoughtSpace initialized - The Gap is open")
    
    # =========================================================================
    # 1. 여백 진입/퇴장
    # =========================================================================
    
    def enter_gap(self, stimulus: str = "") -> None:
        """여백에 진입 - 사고 시작
        
        Args:
            stimulus: 사고를 촉발한 자극
        """
        self.active_particles.clear()
        self.state = ThoughtState.GATHERING
        self.gap_entered_at = datetime.now()
        
        # 자극을 첫 입자로 추가
        if stimulus:
            self.add_thought_particle(
                content=stimulus,
                source="stimulus",
                weight=1.5  # 자극은 무게가 높다
            )
        
        logger.info(f"🌌 Entered The Gap: '{stimulus[:50]}...' if stimulus else 'empty")
    
    def exit_gap(self) -> ContemplationResult:
        """여백에서 나옴 - 결과 반환
        
        Returns:
            성찰 결과
        """
        result = self.synthesize()
        self.active_particles.clear()
        self.gap_entered_at = None
        
        # 기록
        self.contemplation_log.append(result)
        if len(self.contemplation_log) > 100:
            self.contemplation_log = self.contemplation_log[-50:]
        
        logger.info(f"💫 Exited The Gap with synthesis (confidence: {result.confidence:.2f})")
        return result
    
    # =========================================================================
    # 2. 사고 입자 관리
    # =========================================================================
    
    def add_thought_particle(
        self,
        content: Any,
        source: str,
        weight: float = 1.0
    ) -> ThoughtParticle:
        """사고 입자 추가
        
        Args:
            content: 사고 내용
            source: 출처 (memory, perception, reasoning, error)
            weight: 중요도
            
        Returns:
            생성된 입자
        """
        particle_id = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        particle = ThoughtParticle(
            id=particle_id,
            content=content,
            source=source,
            weight=weight,
        )
        
        # 기존 입자들과의 공명 계산
        particle.resonance = self._calculate_resonance(particle)
        
        self.active_particles.append(particle)
        
        # 상태 업데이트
        if len(self.active_particles) >= 3:
            self.state = ThoughtState.CONTEMPLATING
        
        logger.debug(f"✨ Particle added: {source} (resonance: {particle.resonance:.2f})")
        return particle
    
    def _calculate_resonance(self, new_particle: ThoughtParticle) -> float:
        """새 입자와 기존 입자들의 공명도 계산"""
        if not self.active_particles:
            return 0.5
        
        # 간단한 공명: 같은 출처면 공명도 높음
        same_source = sum(
            1 for p in self.active_particles if p.source == new_particle.source
        )
        source_resonance = same_source / max(1, len(self.active_particles))
        
        # 시간적 근접성: 최근 입자와 가까우면 공명
        if self.active_particles:
            latest = max(self.active_particles, key=lambda p: p.timestamp)
            time_diff = (new_particle.timestamp - latest.timestamp).total_seconds()
            temporal_resonance = max(0, 1.0 - (time_diff / 60.0))  # 1분 이내
        else:
            temporal_resonance = 0.5
        
        return (source_resonance + temporal_resonance) / 2
    
    # =========================================================================
    # 3. 오류 성찰 (Error Contemplation) - 핵심!
    # =========================================================================
    
    def contemplate_error(
        self,
        error: Exception,
        context: str,
        attempted_action: str
    ) -> ErrorTrace:
        """오류를 성찰하고 배움으로 전환
        
        Args:
            error: 발생한 오류
            context: 오류 발생 상황
            attempted_action: 시도한 행동
            
        Returns:
            오류 흔적 (분석 포함)
        """
        self.state = ThoughtState.ERROR_ANALYZING
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # 오류 흔적 생성
        trace = ErrorTrace(
            error_type=error_type,
            error_message=error_message,
            context=context,
            attempted_action=attempted_action,
        )
        
        # 원인 분석
        trace.cause_analysis = self._analyze_error_cause(error_type, error_message, context)
        
        # 원리 추출
        trace.learned_principle = self._extract_principle(trace)
        
        # 예방 전략
        trace.prevention_strategy = self._devise_prevention(trace)
        
        # 기록
        self.error_history.append(trace)
        
        # 패턴 누적
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = []
        self.error_patterns[error_type].append(trace.cause_analysis)
        
        # 오류에서 얻은 통찰을 입자로 추가
        self.add_thought_particle(
            content=f"Error Insight: {trace.learned_principle}",
            source="error",
            weight=2.0  # 오류에서 배운 것은 중요
        )
        
        logger.info(f"🔍 Error contemplated: {error_type}")
        logger.info(f"   Learned: {trace.learned_principle}")
        
        return trace
    
    def _analyze_error_cause(
        self,
        error_type: str,
        error_message: str,
        context: str
    ) -> str:
        """오류 원인 분석
        
        하드코딩된 규칙이 아닌, 패턴 인식 기반.
        """
        causes = []
        
        # ImportError 패턴
        if error_type == "ImportError" or error_type == "ModuleNotFoundError":
            if "No module named" in error_message:
                module_name = error_message.split("'")[-2] if "'" in error_message else "unknown"
                causes.append(f"모듈 '{module_name}'이 설치되지 않았거나 경로가 잘못됨")
                causes.append("가상환경이 활성화되지 않았을 수 있음")
            elif "cannot import name" in error_message:
                causes.append("모듈 내 해당 객체가 존재하지 않거나 순환 import 발생")
        
        # AttributeError 패턴
        elif error_type == "AttributeError":
            if "has no attribute" in error_message:
                causes.append("객체에 해당 속성/메서드가 없음 - 타입 확인 필요")
                causes.append("None 객체에 접근 시도 가능성")
        
        # TypeError 패턴
        elif error_type == "TypeError":
            if "argument" in error_message:
                causes.append("함수 인자 타입 또는 개수 불일치")
            elif "not subscriptable" in error_message:
                causes.append("인덱싱할 수 없는 타입 (None, int 등)")
        
        # FileNotFoundError 패턴
        elif error_type == "FileNotFoundError":
            causes.append("파일 경로가 잘못되었거나 파일이 존재하지 않음")
            causes.append("상대 경로 vs 절대 경로 문제 가능")
        
        # 기본
        if not causes:
            causes.append(f"알려지지 않은 오류 패턴: {error_type}")
            causes.append(f"메시지 분석 필요: {error_message[:100]}")
        
        # 과거 패턴과 비교
        if error_type in self.error_patterns:
            past_causes = self.error_patterns[error_type]
            if past_causes:
                causes.append(f"과거 유사 오류에서 발견된 원인: {past_causes[-1]}")
        
        return " | ".join(causes)
    
    def _extract_principle(self, trace: ErrorTrace) -> str:
        """오류에서 원리 추출"""
        error_type = trace.error_type
        
        # 각 오류 유형에서 배울 수 있는 원리
        principles = {
            "ImportError": "의존성은 명시적으로 확인되어야 한다",
            "ModuleNotFoundError": "환경 설정은 코드 실행의 전제 조건이다",
            "AttributeError": "객체의 상태를 가정하지 말고 확인하라",
            "TypeError": "타입은 계약이다 - 계약을 지켜라",
            "FileNotFoundError": "외부 자원의 존재는 보장되지 않는다",
            "KeyError": "사전의 키는 존재하지 않을 수 있다",
            "IndexError": "범위를 벗어난 접근은 구조적 오해를 의미한다",
            "ValueError": "값의 유효성은 항상 검증되어야 한다",
        }
        
        base_principle = principles.get(
            error_type,
            "모든 오류는 가정의 실패를 의미한다"
        )
        
        return f"{base_principle} (맥락: {trace.context[:50]})"
    
    def _devise_prevention(self, trace: ErrorTrace) -> str:
        """예방 전략 수립"""
        error_type = trace.error_type
        
        strategies = {
            "ImportError": "try-except로 import를 감싸거나, 시작 시 의존성 검사",
            "ModuleNotFoundError": "bootstrap_guardian.py 활용 또는 requirements.txt 갱신",
            "AttributeError": "hasattr() 또는 getattr(obj, 'attr', default) 사용",
            "TypeError": "타입 힌트 + isinstance() 검사 추가",
            "FileNotFoundError": "Path.exists() 확인 후 접근",
            "KeyError": "dict.get(key, default) 사용",
            "IndexError": "len() 확인 후 접근",
            "ValueError": "입력 검증 함수 추가",
        }
        
        return strategies.get(
            error_type,
            "오류 발생 지점에 방어적 코드 추가"
        )
    
    # =========================================================================
    # 4. 통합 (Synthesis)
    # =========================================================================
    
    def synthesize(self) -> ContemplationResult:
        """활성 입자들을 통합하여 결론 도출
        
        Returns:
            통합된 결과
        """
        self.state = ThoughtState.SYNTHESIZING
        
        if not self.active_particles:
            return ContemplationResult(
                synthesis="여백이 비어있음 - 사고할 내용 없음",
                confidence=0.0,
                contributing_thoughts=[],
                time_in_gap=0.0,
            )
        
        # 시간 계산
        time_in_gap = 0.0
        if self.gap_entered_at:
            time_in_gap = (datetime.now() - self.gap_entered_at).total_seconds()
        
        # 가중치 기반 정렬
        sorted_particles = sorted(
            self.active_particles,
            key=lambda p: p.weight * p.resonance,
            reverse=True
        )
        
        # 통합
        contributing = [str(p.content)[:50] for p in sorted_particles[:5]]
        
        # 오류 통찰 추출
        error_insights = [
            str(p.content) for p in sorted_particles
            if p.source == "error"
        ]
        
        # 확신도: 입자 수와 공명도에 비례
        avg_resonance = sum(p.resonance for p in self.active_particles) / len(self.active_particles)
        particle_factor = min(1.0, len(self.active_particles) / 5)
        maturity_factor = min(1.0, time_in_gap / self.maturation_threshold)
        
        confidence = (avg_resonance + particle_factor + maturity_factor) / 3
        
        # 통합 결론 (상위 입자들의 내용 조합)
        synthesis_parts = []
        for p in sorted_particles[:3]:
            content_str = str(p.content)
            if len(content_str) > 100:
                content_str = content_str[:100] + "..."
            synthesis_parts.append(f"[{p.source}] {content_str}")
        
        synthesis = " → ".join(synthesis_parts) if synthesis_parts else "통합 불가"
        
        self.state = ThoughtState.READY
        
        return ContemplationResult(
            synthesis=synthesis,
            confidence=confidence,
            contributing_thoughts=contributing,
            time_in_gap=time_in_gap,
            error_insights=error_insights,
        )
    
    # =========================================================================
    # 5. 상태 조회
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "state": self.state.value,
            "active_particles": len(self.active_particles),
            "time_in_gap": (
                (datetime.now() - self.gap_entered_at).total_seconds()
                if self.gap_entered_at else 0.0
            ),
            "error_history_count": len(self.error_history),
            "known_error_patterns": list(self.error_patterns.keys()),
            "contemplation_count": len(self.contemplation_log),
        }
    
    def get_recent_error_insights(self, n: int = 5) -> List[Dict[str, str]]:
        """최근 오류 통찰 조회"""
        recent = self.error_history[-n:] if self.error_history else []
        return [
            {
                "error_type": e.error_type,
                "learned_principle": e.learned_principle,
                "prevention_strategy": e.prevention_strategy,
            }
            for e in recent
        ]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🌌 ThoughtSpace Demo")
    print("   \"입력과 출력 사이의 여백\"")
    print("=" * 60)
    
    space = ThoughtSpace(maturation_threshold=0.5)
    
    # 1. 여백 진입
    print("\n[1] 여백 진입:")
    space.enter_gap("하늘은 왜 파란가?")
    
    # 2. 사고 입자 추가
    print("\n[2] 사고 입자 추가:")
    space.add_thought_particle("빛의 산란 현상", source="memory")
    space.add_thought_particle("레일리 산란", source="reasoning")
    space.add_thought_particle("짧은 파장은 더 많이 산란", source="memory")
    
    # 3. 오류 성찰
    print("\n[3] 오류 성찰 (ImportError 예시):")
    try:
        import nonexistent_module  # 의도적 오류
    except ImportError as e:
        trace = space.contemplate_error(
            error=e,
            context="물리 계산을 위해 모듈 로드 시도",
            attempted_action="import nonexistent_module"
        )
        print(f"   원인: {trace.cause_analysis}")
        print(f"   배움: {trace.learned_principle}")
        print(f"   예방: {trace.prevention_strategy}")
    
    # 4. 통합
    print("\n[4] 통합:")
    result = space.exit_gap()
    print(f"   통합: {result.synthesis}")
    print(f"   확신도: {result.confidence:.2f}")
    print(f"   여백 시간: {result.time_in_gap:.2f}초")
    if result.error_insights:
        print(f"   오류 통찰: {result.error_insights}")
    
    # 5. 상태
    print("\n[5] 상태:")
    status = space.get_status()
    print(f"   오류 패턴: {status['known_error_patterns']}")
    
    print("\n✅ ThoughtSpace Demo complete!")
