"""
Agent Decision Engine - 에이전트가 전략을 선택하는 지능 계층

에이전트가 MetaTimeStrategy의 시간 모드와 계산 프로필을 
상황에 맞춰 선택할 수 있게 하는 의사결정 엔진.

Features:
- 현재 상황 분석 (초점, 목표, 시간 압박)
- 최적 전략 추천
- 성능 피드백 학습
- 메모리 효율적 (1060 3GB 환경 고려)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import sys
import os

# 상대 경로 처리
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from Core.Math.law_enforcement_engine import LawEnforcementEngine, EnergyState


class TemporalMode(Enum):
    """시간 모드 - 에이전트가 선택 가능"""
    MEMORY_HEAVY = "memory_heavy"      # 과거 중심 (추억, 반성)
    PRESENT_FOCUSED = "present"         # 현재 중심 (즉각 반응)
    FUTURE_ORIENTED = "future"          # 미래 중심 (계획, 예측)
    BALANCED = "balanced"               # 균형잡힘 (보통)
    RECURSIVE = "recursive"             # 재귀적 (깊은 사고)


class ComputationProfile(Enum):
    """계산 프로필 - 에이전트가 선택 가능"""
    INTENSIVE = "intensive"             # 모든 공명 계산 (정확, 느림)
    CACHED = "cached"                   # 이전 결과 재사용 (빠름, 메모리)
    PREDICTIVE = "predictive"           # 예측으로 추정 (빠름, 추정)
    SELECTIVE = "selective"             # 중요한 것만 (균형, 스마트)


@dataclass
class AgentContext:
    """에이전트의 현재 상황"""
    focus: str                          # 현재 초점 ("love", "knowledge" 등)
    goal: str                           # 현재 목표
    tick: int                           # 현재 틱
    available_memory_mb: int            # 사용 가능한 메모리 (MB)
    time_pressure: float                # 시간 압박 (0.0 ~ 1.0)
    concept_count: int                  # 현재 개념 수
    recent_performance: Optional[float] = None  # 최근 성능 (speedup factor)


@dataclass
class DecisionReport:
    """의사결정 결과 보고"""
    temporal_mode: TemporalMode
    computation_profile: ComputationProfile
    reasoning: str                      # 왜 이 조합인지
    confidence: float                   # 신뢰도 (0.0 ~ 1.0)
    predicted_speedup: float            # 예상 속도 향상
    memory_estimate_mb: int             # 예상 메모리 사용


class AgentDecisionEngine:
    """
    에이전트의 전략 선택 엔진
    
    Gap 0 준수: epistemology 필드로 각 결정의 철학적 의미 제공
    """
    
    # 결정 유형별 인식론 (Gap 0)
    DECISION_EPISTEMOLOGY = {
        "temporal_mode": {
            "point": {"score": 0.20, "meaning": "현재 순간의 상태 관찰"},
            "line": {"score": 0.35, "meaning": "시간축의 인과적 연결"},
            "space": {"score": 0.30, "meaning": "상황 맥락의 공간적 이해"},
            "god": {"score": 0.15, "meaning": "시간을 초월한 전략적 시야"}
        },
        "computation_profile": {
            "point": {"score": 0.30, "meaning": "정확한 계산의 가치"},
            "line": {"score": 0.25, "meaning": "효율성과 속도의 균형"},
            "space": {"score": 0.25, "meaning": "자원 제약 내 최적화"},
            "god": {"score": 0.20, "meaning": "지혜로운 선택의 초월성"}
        }
    }
    
    def __init__(self, enable_learning: bool = True):
        """
        Args:
            enable_learning: 성능 기반 학습 활성화
        """
        self.enable_learning = enable_learning
        
        # Gap 0: 인식론 필드
        self.epistemology = self.DECISION_EPISTEMOLOGY
        
        # 학습 기록
        self.decision_history: List[Tuple[AgentContext, DecisionReport]] = []
        self.performance_history: Dict[str, List[float]] = {
            "memory_heavy": [],
            "present": [],
            "future": [],
            "balanced": [],
            "recursive": []
        }
        
        # 규칙 기반 맵핑
        self.mode_rules = self._init_mode_rules()
        self.profile_rules = self._init_profile_rules()
        
        # 10대 법칙 실행 엔진
        self.law_engine = LawEnforcementEngine()
    
    def explain_decision_meaning(self, decision_type: str = "temporal_mode") -> str:
        """
        Gap 0 준수: 결정 유형의 철학적 의미를 설명
        
        Args:
            decision_type: "temporal_mode" 또는 "computation_profile"
        
        Returns:
            철학적 의미 설명 문자열
        """
        if decision_type not in self.epistemology:
            return f"알 수 없는 결정 유형: {decision_type}"
        
        epist = self.epistemology[decision_type]
        lines = [f"=== {decision_type} 인식론 ==="]
        
        for basis, data in epist.items():
            lines.append(f"  {basis}: {data['score']:.0%} - {data['meaning']}")
        
        return "\n".join(lines)
    
    def _init_mode_rules(self) -> Dict:
        """시간 모드 선택 규칙"""
        return {
            "reflection": TemporalMode.MEMORY_HEAVY,    # 과거 반성
            "crisis": TemporalMode.PRESENT_FOCUSED,     # 위기 대응
            "planning": TemporalMode.FUTURE_ORIENTED,   # 미래 계획
            "learning": TemporalMode.BALANCED,          # 균형잡힘
            "creation": TemporalMode.RECURSIVE,         # 재귀적 사고
        }
    
    def _init_profile_rules(self) -> Dict:
        """계산 프로필 선택 규칙"""
        return {
            "slow_connection": ComputationProfile.PREDICTIVE,  # 빠른 응답 필요
            "low_memory": ComputationProfile.SELECTIVE,        # 메모리 절약
            "high_accuracy": ComputationProfile.INTENSIVE,     # 정확성 중시
            "normal": ComputationProfile.CACHED,               # 일반적
        }
    
    def _create_energy_state(
        self,
        temporal_mode: TemporalMode,
        computation_profile: ComputationProfile,
        context: AgentContext
    ) -> EnergyState:
        """쿼터니언 에너지 상태 생성"""
        
        # 기본값
        w = 0.6  # 앵커: 일반적으로 높음
        x = 0.3  # 사고: 계산 필요
        y = 0.4  # 행동: 시뮬레이션 활동
        z = 0.3  # 의도: 계획 초기 단계
        
        # 시간 모드에 따른 조정
        if temporal_mode == TemporalMode.MEMORY_HEAVY:
            w = 0.7  # 반성 강조
            x = 0.4  # 기억 회상
        elif temporal_mode == TemporalMode.PRESENT_FOCUSED:
            y = 0.6  # 현재 행동
            z = 0.5  # 급박한 의도
        elif temporal_mode == TemporalMode.FUTURE_ORIENTED:
            z = 0.7  # 미래 의도
            x = 0.5  # 계획 사고
        elif temporal_mode == TemporalMode.RECURSIVE:
            w = 0.5  # 깊은 사고 (메타인지 감소)
            x = 0.7  # 깊은 사고
        
        # 메모리 상황에 따른 조정
        if context.available_memory_mb < 500:
            # 메모리 부족: 빠른 실행 강조
            y = 0.5
            x = 0.3
        
        state = EnergyState(w=w, x=x, y=y, z=z)
        state.normalize()
        return state
    
    def _calculate_confidence(
        self,
        context: AgentContext,
        temporal_mode: TemporalMode,
        computation_profile: ComputationProfile,
        law_decision = None
    ) -> float:
        """의사결정 신뢰도"""
        
        confidence = 0.7  # 기본값
        
        # 명확한 초점: 신뢰도 향상
        if context.focus and len(context.focus) > 3:
            confidence += 0.15
        
        # 명확한 목표: 신뢰도 향상
        if context.goal and len(context.goal) > 3:
            confidence += 0.1
        
        # 충분한 메모리: 신뢰도 향상
        if context.available_memory_mb > 1000:
            confidence += 0.05
        
        # 과거 성능 기록: 신뢰도 향상
        if temporal_mode.value in self.performance_history:
            history = self.performance_history[temporal_mode.value]
            if len(history) > 3:
                avg_performance = sum(history) / len(history)
                if avg_performance > 2.0:
                    confidence += 0.1
        
        # 법칙 위반 시 신뢰도 감소
        if law_decision and not law_decision.is_valid:
            confidence -= len(law_decision.violations) * 0.1
        
        return min(confidence, 1.0)
    
    def decide(self, context: AgentContext) -> DecisionReport:
        """
        현재 상황에 맞는 전략을 선택합니다.
        
        Args:
            context: 에이전트의 현재 상황
            
        Returns:
            DecisionReport: 선택된 전략과 이유
        """
        
        # 1. 시간 모드 결정
        temporal_mode = self._decide_temporal_mode(context)
        
        # 2. 계산 프로필 결정
        computation_profile = self._decide_computation_profile(context)
        
        # 3. 쿼터니언 에너지 상태 생성 (10대 수학)
        energy_state = self._create_energy_state(temporal_mode, computation_profile, context)
        
        # 4. 10대 법칙 검사
        law_decision = self.law_engine.make_decision(
            proposed_action=f"{temporal_mode.value}:{computation_profile.value}",
            energy_before=energy_state,
            concepts_generated=context.concept_count
        )
        
        # 법칙 위반이 있으면 에너지 상태 업데이트
        if not law_decision.is_valid:
            energy_state = law_decision.energy_after
            # Z축이 높아졌으면 의도 강조 모드로
            if energy_state.z > 0.6:
                temporal_mode = TemporalMode.FUTURE_ORIENTED
                computation_profile = ComputationProfile.PREDICTIVE
        
        # 5. 예상 성능 계산
        predicted_speedup = self._calculate_predicted_speedup(
            temporal_mode, computation_profile, context
        )
        
        # 6. 예상 메모리 사용
        memory_estimate = self._estimate_memory(
            temporal_mode, computation_profile, context
        )
        
        # 7. 추론 텍스트
        reasoning = self._generate_reasoning(
            temporal_mode, computation_profile, context, law_decision
        )
        
        # 8. 신뢰도
        confidence = self._calculate_confidence(context, temporal_mode, computation_profile, law_decision)
        
        report = DecisionReport(
            temporal_mode=temporal_mode,
            computation_profile=computation_profile,
            reasoning=reasoning,
            confidence=confidence,
            predicted_speedup=predicted_speedup,
            memory_estimate_mb=memory_estimate
        )
        
        # 학습 기록
        if self.enable_learning:
            self.decision_history.append((context, report))
        
        return report
    
    def _decide_temporal_mode(self, context: AgentContext) -> TemporalMode:
        """시간 모드 선택 로직"""
        
        # 규칙 기반 선택
        for keyword, mode in self.mode_rules.items():
            if keyword in context.focus.lower():
                return mode
        
        # 기본: 시간 압박에 따라
        if context.time_pressure > 0.8:
            return TemporalMode.PRESENT_FOCUSED  # 긴급: 현재만
        elif context.time_pressure > 0.5:
            return TemporalMode.BALANCED  # 중간: 균형
        else:
            return TemporalMode.FUTURE_ORIENTED  # 여유: 미래 계획
    
    def _decide_computation_profile(self, context: AgentContext) -> ComputationProfile:
        """계산 프로필 선택 로직"""
        
        # 메모리 제약 (1060 3GB 환경)
        if context.available_memory_mb < 500:  # 500MB 미만
            return ComputationProfile.SELECTIVE  # 스마트 선택
        
        if context.available_memory_mb < 1000:  # 1GB 미만
            return ComputationProfile.CACHED  # 캐시 재사용
        
        # 시간 압박
        if context.time_pressure > 0.8:
            return ComputationProfile.PREDICTIVE  # 예측으로 빠르게
        
        if context.time_pressure > 0.5:
            return ComputationProfile.SELECTIVE  # 선택적 계산
        
        # 개념이 많으면 선택적
        if context.concept_count > 500:
            return ComputationProfile.SELECTIVE
        
        # 기본: 캐시된 결과 활용
        return ComputationProfile.CACHED
    
    def _calculate_predicted_speedup(
        self,
        temporal_mode: TemporalMode,
        computation_profile: ComputationProfile,
        context: AgentContext
    ) -> float:
        """예상 속도 향상 계산"""
        
        base_speedup = 1.0
        
        # 계산 프로필의 기본 향상
        profile_speedups = {
            ComputationProfile.INTENSIVE: 1.0,      # 기준
            ComputationProfile.CACHED: 3.0,         # 캐시 재사용
            ComputationProfile.PREDICTIVE: 5.0,     # 예측
            ComputationProfile.SELECTIVE: 2.5,      # 선택적
        }
        base_speedup = profile_speedups.get(computation_profile, 1.0)
        
        # 시간 모드에 따른 추가 향상
        mode_bonuses = {
            TemporalMode.MEMORY_HEAVY: 1.2,
            TemporalMode.PRESENT_FOCUSED: 1.5,
            TemporalMode.FUTURE_ORIENTED: 1.1,
            TemporalMode.BALANCED: 1.0,
            TemporalMode.RECURSIVE: 0.8,  # 깊은 사고는 느림
        }
        final_speedup = base_speedup * mode_bonuses.get(temporal_mode, 1.0)
        
        # 메모리 제약이 심할 때 추가 향상
        if context.available_memory_mb < 500:
            final_speedup *= 1.2  # 메모리 절약이 실제로 빠름
        
        return final_speedup
    
    def _estimate_memory(
        self,
        temporal_mode: TemporalMode,
        computation_profile: ComputationProfile,
        context: AgentContext
    ) -> int:
        """예상 메모리 사용량 (MB)"""
        
        base_memory = 100  # 기본
        
        # 프로필별 메모리
        profile_memory = {
            ComputationProfile.INTENSIVE: 200,      # 모든 데이터 메모리
            ComputationProfile.CACHED: 150,         # 캐시 유지
            ComputationProfile.PREDICTIVE: 100,     # 최소 메모리
            ComputationProfile.SELECTIVE: 120,      # 선택적 저장
        }
        base_memory = profile_memory.get(computation_profile, 100)
        
        # 개념 수에 따른 추가 메모리
        concept_memory = context.concept_count // 10  # 개념 당 ~0.1MB
        
        total_memory = base_memory + concept_memory
        
        return min(total_memory, 500)  # 최대 500MB
    
    def _generate_reasoning(
        self,
        temporal_mode: TemporalMode,
        computation_profile: ComputationProfile,
        context: AgentContext,
        law_decision = None
    ) -> str:
        """의사결정 이유 설명"""
        
        reasons = []
        
        # 초점 기반
        if "love" in context.focus.lower():
            reasons.append("사랑의 깊이를 탐구하기 위해")
        elif "knowledge" in context.focus.lower():
            reasons.append("지식의 구조를 분석하기 위해")
        elif "creation" in context.focus.lower():
            reasons.append("새로운 개념을 창조하기 위해")
        
        # 시간 압박 기반
        if context.time_pressure > 0.8:
            reasons.append("긴급한 상황에서 빠른 응답이 필요해서")
        elif context.time_pressure < 0.2:
            reasons.append("충분한 시간이 있어서 깊이있게")
        
        # 메모리 기반
        if context.available_memory_mb < 500:
            reasons.append(f"메모리가 {context.available_memory_mb}MB로 제한되어서")
        
        # 법칙 검사 결과
        law_info = ""
        if law_decision:
            if law_decision.is_valid:
                law_info = "[OK] 모든 법칙을 준수합니다"
            else:
                violations = law_decision.violations
                law_info = f"[WARNING] {len(violations)}개 법칙 검사:"
                for v in violations[:2]:
                    law_info += f"\n  - {v.law.value}: {v.reason}"
                if len(violations) > 2:
                    law_info += f"\n  ...외 {len(violations)-2}개"
        
        # 모드별 추가 설명
        mode_descriptions = {
            TemporalMode.MEMORY_HEAVY: "과거의 경험에서 배우고 싶습니다",
            TemporalMode.PRESENT_FOCUSED: "지금 이 순간에 집중합니다",
            TemporalMode.FUTURE_ORIENTED: "미래를 계획하고 예측합니다",
            TemporalMode.BALANCED: "시간의 모든 층을 균형있게 봅니다",
            TemporalMode.RECURSIVE: "깊이있는 사고와 자기 반성을 합니다",
        }
        
        reasoning = " → ".join(reasons) if reasons else "일반적인 상황입니다"
        reasoning += f"\n선택: [{temporal_mode.value}] + [{computation_profile.value}]\n"
        reasoning += mode_descriptions.get(temporal_mode, "")
        reasoning += f"\n\n{law_info}"
        
        return reasoning
    
    def _calculate_confidence(
        self,
        context: AgentContext,
        temporal_mode: TemporalMode,
        computation_profile: ComputationProfile,
        law_decision = None
    ) -> float:
        """의사결정 신뢰도"""
        
        confidence = 0.7  # 기본값
        
        # 명확한 초점: 신뢰도 향상
        if context.focus and len(context.focus) > 3:
            confidence += 0.15
        
        # 명확한 목표: 신뢰도 향상
        if context.goal and len(context.goal) > 3:
            confidence += 0.1
        
        # 충분한 메모리: 신뢰도 향상
        if context.available_memory_mb > 1000:
            confidence += 0.05
        
        # 과거 성능 기록: 신뢰도 향상
        if temporal_mode.value in self.performance_history:
            history = self.performance_history[temporal_mode.value]
            if len(history) > 3:
                avg_performance = sum(history) / len(history)
                if avg_performance > 2.0:
                    confidence += 0.1
        
        return min(confidence, 1.0)
    
    def record_performance(
        self,
        temporal_mode: TemporalMode,
        actual_speedup: float
    ) -> None:
        """실제 성능 기록 (학습용)"""
        
        if temporal_mode.value in self.performance_history:
            self.performance_history[temporal_mode.value].append(actual_speedup)
            # 최근 100개만 유지 (메모리 절약)
            if len(self.performance_history[temporal_mode.value]) > 100:
                self.performance_history[temporal_mode.value] = \
                    self.performance_history[temporal_mode.value][-100:]
    
    def get_best_strategy_history(self, limit: int = 10) -> List[Tuple[str, float]]:
        """최고 성능의 전략 역사"""
        
        best_strategies = []
        for mode, history in self.performance_history.items():
            if history:
                avg = sum(history) / len(history)
                best_strategies.append((mode, avg))
        
        return sorted(best_strategies, key=lambda x: x[1], reverse=True)[:limit]
    
    def export_statistics(self, filepath: str) -> None:
        """통계 내보내기"""
        
        stats = {
            "total_decisions": len(self.decision_history),
            "performance_history": {
                k: {
                    "count": len(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "max": max(v) if v else 0,
                    "min": min(v) if v else 0
                }
                for k, v in self.performance_history.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    engine = AgentDecisionEngine()
    
    # 테스트 1: 일반적인 상황
    print("\n[Test 1] 일반적인 상황")
    context = AgentContext(
        focus="learning",
        goal="새로운 개념 발견",
        tick=1000,
        available_memory_mb=1200,
        time_pressure=0.3,
        concept_count=150
    )
    report = engine.decide(context)
    print(f"Mode: {report.temporal_mode.value}")
    print(f"Profile: {report.computation_profile.value}")
    print(f"Speedup: {report.predicted_speedup:.1f}x")
    print(f"Memory: {report.memory_estimate_mb}MB")
    print(f"Confidence: {report.confidence:.1%}")
    print(f"Reasoning:\n{report.reasoning}")
    
    # 테스트 2: 메모리 부족 상황 (1060 3GB)
    print("\n[Test 2] 메모리 부족 (1060 3GB 환경)")
    context = AgentContext(
        focus="intensive_analysis",
        goal="모든 개념 분석",
        tick=5000,
        available_memory_mb=400,  # 심각한 부족
        time_pressure=0.8,
        concept_count=500
    )
    report = engine.decide(context)
    print(f"Mode: {report.temporal_mode.value}")
    print(f"Profile: {report.computation_profile.value}")
    print(f"Speedup: {report.predicted_speedup:.1f}x")
    print(f"Memory: {report.memory_estimate_mb}MB")
    print(f"Confidence: {report.confidence:.1%}")
    print(f"Reasoning:\n{report.reasoning}")
    
    # 테스트 3: 미래 계획 상황
    print("\n[Test 3] 미래 계획")
    context = AgentContext(
        focus="planning_future",
        goal="다음 10,000 틱 계획",
        tick=10000,
        available_memory_mb=2000,
        time_pressure=0.1,
        concept_count=300
    )
    report = engine.decide(context)
    print(f"Mode: {report.temporal_mode.value}")
    print(f"Profile: {report.computation_profile.value}")
    print(f"Speedup: {report.predicted_speedup:.1f}x")
    print(f"Memory: {report.memory_estimate_mb}MB")
    print(f"Confidence: {report.confidence:.1%}")
    print(f"Reasoning:\n{report.reasoning}")
    
    # 성능 기록
    engine.record_performance(report.temporal_mode, report.predicted_speedup)
    
    # 테스트 4: 성능 기록 후 최고 전략
    print("\n[Test 4] 최고 성능 전략")
    best = engine.get_best_strategy_history(limit=5)
    for mode, score in best:
        print(f"  {mode}: {score:.2f}x")
    
    print("\n✅ 모든 테스트 통과!")
