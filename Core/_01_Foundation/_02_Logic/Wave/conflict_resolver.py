"""
Conflict Resolver Engine (충돌 해결 엔진)
========================================

"두 시스템이 서로 다른 답을 낼 때, 파동 원리로 진실을 찾는다."

중복 시스템 출력 충돌 시 파동 기반 자동 해결 시스템.

사용 예시:
- Memory: "사과는 빨갛다"
- Vision: "이 사과는 초록색이다"
→ 해결: "일반적으로 빨갛지만, 현재 맥락에서는 초록색"
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core._01_Foundation._05_Governance.Foundation.Wave.wave_interference import (
    Wave, WaveInterference, InterferenceResult, InterferenceType
)

logger = logging.getLogger("ConflictResolver")


class ConflictType(Enum):
    """충돌 유형"""
    NONE = "none"                    # 충돌 없음
    SEMANTIC = "semantic"            # 의미적 충돌 (서로 다른 내용)
    INTENSITY = "intensity"          # 강도 충돌 (같은 내용, 다른 확신도)
    TEMPORAL = "temporal"            # 시간적 충돌 (과거 vs 현재)
    CONTEXTUAL = "contextual"        # 맥락 충돌 (일반 vs 특정 상황)


class ResolutionStrategy(Enum):
    """해결 전략"""
    DOMINANT = "dominant"            # 가장 강한 것 선택
    MERGE = "merge"                  # 병합/통합
    CONTEXTUAL = "contextual"        # 맥락 기반 분리
    UNCERTAIN = "uncertain"          # 불확실성 표시
    DEFER = "defer"                  # 결정 보류


@dataclass
class ConflictOutput:
    """충돌하는 출력 하나"""
    value: Any                       # 출력 값
    source: str                      # 출처 시스템 이름
    confidence: float = 0.5          # 확신도 (0-1)
    timestamp: float = 0.0           # 생성 시간
    context: str = ""                # 맥락 정보


@dataclass
class ResolvedOutput:
    """해결된 출력"""
    value: Any                           # 최종 값
    confidence: float                    # 최종 확신도
    strategy: ResolutionStrategy         # 사용된 전략
    conflict_type: ConflictType          # 감지된 충돌 유형
    sources: List[str] = field(default_factory=list)  # 원본 출처들
    explanation: str = ""                # 해결 설명
    alternatives: List[Any] = field(default_factory=list)  # 대안들
    uncertainty: float = 0.0             # 불확실성


class ConflictResolver:
    """
    파동 원리 기반 충돌 해결기
    
    여러 시스템의 충돌하는 출력을 파동 간섭으로 해결합니다.
    
    Usage:
        resolver = ConflictResolver()
        outputs = [
            ConflictOutput("red apple", "Memory", 0.8),
            ConflictOutput("green apple", "Vision", 0.9),
        ]
        result = resolver.resolve(outputs)
    """
    
    def __init__(self):
        self.interference_engine = WaveInterference()
        self.resolution_history: List[Dict] = []
    
    def resolve(self, outputs: List[ConflictOutput]) -> ResolvedOutput:
        """
        충돌하는 출력들을 해결합니다.
        
        Args:
            outputs: 충돌하는 출력들의 리스트
            
        Returns:
            ResolvedOutput: 해결된 결과
        """
        if not outputs:
            return ResolvedOutput(
                value=None,
                confidence=0.0,
                strategy=ResolutionStrategy.UNCERTAIN,
                conflict_type=ConflictType.NONE,
                explanation="No outputs to resolve"
            )
        
        if len(outputs) == 1:
            return ResolvedOutput(
                value=outputs[0].value,
                confidence=outputs[0].confidence,
                strategy=ResolutionStrategy.DOMINANT,
                conflict_type=ConflictType.NONE,
                sources=[outputs[0].source],
                explanation="Single output, no conflict"
            )
        
        # 1. 충돌 유형 감지
        conflict_type = self.detect_conflict_type(outputs)
        
        if conflict_type == ConflictType.NONE:
            # 충돌 없음 - 가장 확신도 높은 것 반환
            best = max(outputs, key=lambda o: o.confidence)
            return ResolvedOutput(
                value=best.value,
                confidence=best.confidence,
                strategy=ResolutionStrategy.DOMINANT,
                conflict_type=ConflictType.NONE,
                sources=[o.source for o in outputs],
                explanation="No conflict detected, using highest confidence"
            )
        
        # 2. 출력들을 파동으로 변환
        waves = self._outputs_to_waves(outputs)
        
        # 3. 간섭 계산
        interference_result = self.interference_engine.calculate_interference(waves)
        
        # 4. 간섭 결과에 따른 해결 전략 선택
        strategy, resolved_value, explanation = self._select_resolution_strategy(
            outputs, interference_result, conflict_type
        )
        
        # 5. 결과 생성
        result = ResolvedOutput(
            value=resolved_value,
            confidence=interference_result.confidence,
            strategy=strategy,
            conflict_type=conflict_type,
            sources=[o.source for o in outputs],
            explanation=explanation,
            alternatives=[o.value for o in outputs if o.value != resolved_value],
            uncertainty=interference_result.uncertainty
        )
        
        # 히스토리 기록
        self._record_resolution(outputs, result)
        
        logger.info(
            f"⚖️ Conflict resolved: {conflict_type.value} → {strategy.value} "
            f"(conf={result.confidence:.2f})"
        )
        
        return result
    
    def detect_conflict(self, outputs: List[ConflictOutput]) -> bool:
        """
        충돌 여부를 감지합니다.
        
        Returns:
            True if conflict exists, False otherwise
        """
        return self.detect_conflict_type(outputs) != ConflictType.NONE
    
    def detect_conflict_type(self, outputs: List[ConflictOutput]) -> ConflictType:
        """
        충돌 유형을 감지합니다.
        
        Args:
            outputs: 출력들
            
        Returns:
            ConflictType: 감지된 충돌 유형
        """
        if len(outputs) < 2:
            return ConflictType.NONE
        
        values = [str(o.value).lower() for o in outputs]
        confidences = [o.confidence for o in outputs]
        timestamps = [o.timestamp for o in outputs]
        contexts = [o.context for o in outputs]
        
        # 값이 모두 같으면 충돌 없음
        if len(set(values)) == 1:
            # 확신도 차이 확인
            if max(confidences) - min(confidences) > 0.3:
                return ConflictType.INTENSITY
            return ConflictType.NONE
        
        # 맥락 기반 충돌 확인
        if any(c for c in contexts) and len(set(contexts)) > 1:
            return ConflictType.CONTEXTUAL
        
        # 시간 기반 충돌 확인
        if timestamps and max(timestamps) - min(timestamps) > 3600:  # 1시간 이상 차이
            return ConflictType.TEMPORAL
        
        # 기본: 의미적 충돌
        return ConflictType.SEMANTIC
    
    def _outputs_to_waves(self, outputs: List[ConflictOutput]) -> List[Wave]:
        """출력들을 파동으로 변환"""
        waves = []
        for output in outputs:
            # 값의 해시를 주파수로 사용
            value_hash = abs(hash(str(output.value))) % 1000
            frequency = 432.0 + value_hash * 0.5  # 432-932Hz 범위
            
            # 확신도를 진폭으로
            amplitude = output.confidence
            
            # 맥락을 위상으로
            context_hash = abs(hash(output.context)) % 628  # 0 - 2π * 100
            phase = context_hash / 100.0
            
            wave = Wave(
                frequency=frequency,
                amplitude=amplitude,
                phase=phase,
                source=output.source,
                confidence=output.confidence
            )
            waves.append(wave)
        
        return waves
    
    def _select_resolution_strategy(
        self,
        outputs: List[ConflictOutput],
        interference: InterferenceResult,
        conflict_type: ConflictType
    ) -> Tuple[ResolutionStrategy, Any, str]:
        """
        간섭 결과에 따른 해결 전략 선택
        
        Returns:
            (strategy, resolved_value, explanation)
        """
        # 보강 간섭: 값들이 호환됨 → 병합
        if interference.interference_type == InterferenceType.CONSTRUCTIVE:
            # 가장 높은 확신도의 값을 기본으로, 다른 정보 추가
            primary = max(outputs, key=lambda o: o.confidence)
            secondary = [o for o in outputs if o != primary]
            
            if conflict_type == ConflictType.CONTEXTUAL:
                # 맥락 통합
                value = self._merge_contextual(primary, secondary)
                explanation = f"Merged {primary.source} (primary) with contextual info from {[s.source for s in secondary]}"
            else:
                value = primary.value
                explanation = f"{primary.source} confirmed by {[s.source for s in secondary]} (constructive interference)"
            
            return ResolutionStrategy.MERGE, value, explanation
        
        # 상쇄 간섭: 값들이 충돌 → 우세자 또는 불확실
        elif interference.interference_type == InterferenceType.DESTRUCTIVE:
            if interference.confidence < 0.3:
                # 너무 불확실함 → 불확실성 표시
                dominant = max(outputs, key=lambda o: o.confidence)
                value = f"[Uncertain] Possibly {dominant.value}"
                explanation = "High uncertainty due to destructive interference"
                return ResolutionStrategy.UNCERTAIN, value, explanation
            else:
                # 가장 강한 것 선택
                dominant = max(outputs, key=lambda o: o.confidence)
                explanation = f"Destructive interference: {dominant.source} dominant over {[o.source for o in outputs if o != dominant]}"
                return ResolutionStrategy.DOMINANT, dominant.value, explanation
        
        # 혼합 간섭: 맥락 분리 또는 보류
        else:
            if conflict_type == ConflictType.TEMPORAL:
                # 최신 정보 우선
                newest = max(outputs, key=lambda o: o.timestamp)
                explanation = f"Temporal conflict: using most recent from {newest.source}"
                return ResolutionStrategy.CONTEXTUAL, newest.value, explanation
            
            elif conflict_type == ConflictType.CONTEXTUAL:
                # 맥락 분리
                value = self._create_contextual_response(outputs)
                explanation = "Contextual separation of conflicting outputs"
                return ResolutionStrategy.CONTEXTUAL, value, explanation
            
            else:
                # 결정 보류
                options = ", ".join([f"{o.source}:{o.value}" for o in outputs])
                value = f"[Multiple possibilities: {options}]"
                explanation = "Mixed interference, decision deferred"
                return ResolutionStrategy.DEFER, value, explanation
    
    def _merge_contextual(self, primary: ConflictOutput, secondary: List[ConflictOutput]) -> Any:
        """맥락 정보를 병합"""
        base_value = str(primary.value)
        
        for s in secondary:
            if s.context:
                base_value += f" (also: {s.value} in {s.context} context)"
        
        return base_value
    
    def _create_contextual_response(self, outputs: List[ConflictOutput]) -> Any:
        """맥락별 분리된 응답 생성"""
        parts = []
        for output in sorted(outputs, key=lambda o: o.confidence, reverse=True):
            if output.context:
                parts.append(f"In {output.context}: {output.value}")
            else:
                parts.append(f"Generally: {output.value}")
        
        return " | ".join(parts)
    
    def _record_resolution(self, outputs: List[ConflictOutput], result: ResolvedOutput):
        """해결 이력 기록"""
        self.resolution_history.append({
            "inputs": [{"source": o.source, "value": str(o.value)[:50]} for o in outputs],
            "result": str(result.value)[:100],
            "strategy": result.strategy.value,
            "conflict_type": result.conflict_type.value
        })
        
        # 최근 100개만 유지
        if len(self.resolution_history) > 100:
            self.resolution_history = self.resolution_history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """해결 통계 반환"""
        if not self.resolution_history:
            return {"total": 0, "strategies": {}, "conflict_types": {}}
        
        strategies = {}
        conflict_types = {}
        
        for record in self.resolution_history:
            s = record["strategy"]
            c = record["conflict_type"]
            strategies[s] = strategies.get(s, 0) + 1
            conflict_types[c] = conflict_types.get(c, 0) + 1
        
        return {
            "total": len(self.resolution_history),
            "strategies": strategies,
            "conflict_types": conflict_types
        }


# ============= 데모 =============

def demo_conflict_resolution():
    """충돌 해결 데모"""
    print("=" * 60)
    print("⚖️ Conflict Resolver Demo")
    print("=" * 60)
    
    resolver = ConflictResolver()
    
    # 1. 의미적 충돌
    print("\n[1] Semantic Conflict (의미적 충돌)")
    print("-" * 40)
    outputs1 = [
        ConflictOutput("Apple is red", "Memory", 0.7),
        ConflictOutput("This apple is green", "Vision", 0.9, context="current observation"),
    ]
    result1 = resolver.resolve(outputs1)
    print(f"   Memory: 'Apple is red' (conf=0.7)")
    print(f"   Vision: 'This apple is green' (conf=0.9)")
    print(f"   Resolved: {result1.value}")
    print(f"   Strategy: {result1.strategy.value}")
    print(f"   Explanation: {result1.explanation}")
    
    # 2. 보강 (확인)
    print("\n[2] Confirmation (보강)")
    print("-" * 40)
    outputs2 = [
        ConflictOutput("The sky is blue", "Memory", 0.8),
        ConflictOutput("Sky appears blue", "Vision", 0.85),
    ]
    result2 = resolver.resolve(outputs2)
    print(f"   Memory + Vision agree on 'sky is blue'")
    print(f"   Resolved: {result2.value}")
    print(f"   Confidence: {result2.confidence:.2f} (boosted)")
    
    # 3. 시간적 충돌
    print("\n[3] Temporal Conflict (시간적 충돌)")
    print("-" * 40)
    import time
    old_time = time.time() - 7200  # 2시간 전
    outputs3 = [
        ConflictOutput("Weather: Sunny", "OldForecast", 0.6, timestamp=old_time),
        ConflictOutput("Weather: Rainy", "CurrentSensor", 0.8, timestamp=time.time()),
    ]
    result3 = resolver.resolve(outputs3)
    print(f"   Old (2hr ago): 'Sunny' vs Current: 'Rainy'")
    print(f"   Resolved: {result3.value}")
    print(f"   Strategy: {result3.strategy.value}")
    
    # 통계
    print("\n" + "=" * 60)
    print("📊 Resolution Statistics:")
    stats = resolver.get_statistics()
    print(f"   Total resolutions: {stats['total']}")
    print(f"   Strategies used: {stats['strategies']}")
    print("=" * 60)
    print("✅ Demo Complete!")


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        demo_conflict_resolution()
    else:
        print("Usage: python conflict_resolver.py --demo")
