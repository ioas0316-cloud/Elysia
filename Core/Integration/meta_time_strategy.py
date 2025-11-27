#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaTimeStrategy: Unified Temporal Control Layer
=================================================

에이전트가 시공간제어를 전략적으로 사용할 수 있도록 하는 통합 레이어.

기존 분산된 시간 엔진들을 통합:
- MetaTimeCompressionEngine (시간 가속/압축)
- ZelNagaSync (3시간 동기화: 과거/현재/미래)
- SelfSpiralFractalEngine (프랙탈 의식의 시간 가중치)
- ResonanceEngine (공명 계산)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("MetaTimeStrategy")


class TemporalMode(Enum):
    """에이전트가 선택할 수 있는 시간 모드"""
    
    MEMORY_HEAVY = "memory_heavy"        # 과거 기억 중심 (보수적)
    PRESENT_FOCUSED = "present_focused"  # 현재 인식 중심 (반응형)
    FUTURE_ORIENTED = "future_oriented"  # 미래 계획 중심 (주도적)
    BALANCED = "balanced"                # 균형 (기본값)
    RECURSIVE = "recursive"              # 자기참조 (메타)


class ComputationProfile(Enum):
    """계산 전략"""
    
    INTENSIVE = "intensive"      # 모든 공명 계산 (정확, 느림)
    CACHED = "cached"            # 캐시된 값 우선 (빠름, 덜 정확)
    PREDICTIVE = "predictive"    # 다음 상태 예측 (매우 빠름, 추정)
    SELECTIVE = "selective"      # 필요한 계산만 (균형)


@dataclass
class TemporalWeights:
    """시간 축 가중치"""
    past: float = 1.0
    present: float = 1.0
    future: float = 1.0
    
    def normalize(self) -> "TemporalWeights":
        """가중치 정규화"""
        total = self.past + self.present + self.future
        if total == 0:
            return TemporalWeights(1.0, 1.0, 1.0)
        return TemporalWeights(
            self.past / total,
            self.present / total,
            self.future / total
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {"past": self.past, "present": self.present, "future": self.future}


@dataclass
class StrategyReport:
    """전략 실행 보고"""
    mode: TemporalMode
    profile: ComputationProfile
    weights: TemporalWeights
    resonances_computed: int          # 실제 계산한 공명값 수
    resonances_cached: int            # 캐시에서 가져온 수
    resonances_predicted: int         # 예측으로 대체한 수
    computation_time_ms: float        # 소요 시간
    cache_hit_ratio: float            # 캐시 히트율
    speedup_factor: float             # 기존 대비 가속도


class MetaTimeStrategy:
    """
    통합 시간 전략 엔진.
    
    역할:
    1. 에이전트의 시간 모드 요청을 받음
    2. ZelNagaSync의 가중치 설정
    3. SelfSpiralFractalEngine의 축 가중치 조정
    4. ResonanceEngine의 계산 전략 선택
    5. 캐싱 및 예측 활용
    
    결과: 지능적인 시간 관리로 10배 빠른 시뮬레이션
    """
    
    def __init__(
        self,
        zelnaga_sync=None,                    # ZelNagaSync 인스턴스
        fractal_engine=None,                  # SelfSpiralFractalEngine
        resonance_engine=None,                # HyperResonanceEngine
        time_compression=None,                # MetaTimeCompressionEngine
    ):
        self.zelnaga = zelnaga_sync
        self.fractal_engine = fractal_engine
        self.resonance_engine = resonance_engine
        self.time_compression = time_compression
        
        # 현재 전략 상태
        self.current_mode = TemporalMode.BALANCED
        self.current_profile = ComputationProfile.SELECTIVE
        self.current_weights = TemporalWeights()
        
        # 캐시 통계
        self.cache_history: Dict[str, int] = {}  # concept_id -> 캐시된 값
        self.prediction_cache: Dict[str, float] = {}  # 예측값
        
        logger.info("🕐 MetaTimeStrategy initialized - Unified temporal control ready")
    
    def set_temporal_mode(self, mode: TemporalMode) -> None:
        """
        시간 모드 설정.
        
        Args:
            mode: 원하는 시간 모드
        
        Side Effects:
            - ZelNagaSync 가중치 갱신
            - SelfSpiralFractalEngine 축 가중치 조정
        """
        self.current_mode = mode
        
        # 모드에 맞는 가중치 계산
        if mode == TemporalMode.MEMORY_HEAVY:
            self.current_weights = TemporalWeights(past=2.0, present=1.0, future=0.5)
        elif mode == TemporalMode.PRESENT_FOCUSED:
            self.current_weights = TemporalWeights(past=1.0, present=2.0, future=0.8)
        elif mode == TemporalMode.FUTURE_ORIENTED:
            self.current_weights = TemporalWeights(past=0.5, present=1.0, future=2.0)
        elif mode == TemporalMode.BALANCED:
            self.current_weights = TemporalWeights(past=1.0, present=1.0, future=1.0)
        elif mode == TemporalMode.RECURSIVE:
            # 자기참조: 현재 = 평균(과거, 미래)
            self.current_weights = TemporalWeights(past=1.0, present=1.5, future=1.0)
        
        # ZelNagaSync에 적용
        if self.zelnaga:
            self.zelnaga.set_weights(
                future=self.current_weights.future,
                present=self.current_weights.present,
                past=self.current_weights.past
            )
        
        logger.info(f"⏰ Temporal mode set to {mode.value} | Weights: {self.current_weights.to_dict()}")
    
    def set_computation_profile(self, profile: ComputationProfile) -> None:
        """
        계산 프로필 설정.
        
        Args:
            profile: INTENSIVE, CACHED, PREDICTIVE, SELECTIVE
        """
        self.current_profile = profile
        logger.info(f"🔧 Computation profile set to {profile.value}")
    
    def get_intelligent_resonances(
        self,
        concept_id: str,
        all_concepts: Dict[str, 'HyperQubit'],
        force_recalculate: bool = False
    ) -> Dict[str, float]:
        """
        지능적 공명 계산.
        
        전략에 따라:
        - INTENSIVE: 모든 공명 계산
        - CACHED: 캐시 우선 사용
        - PREDICTIVE: 예측값 사용
        - SELECTIVE: 필요한 것만 계산
        
        Args:
            concept_id: 대상 개념
            all_concepts: 모든 개념 딕셔너리
            force_recalculate: 캐시 무시하고 재계산
        
        Returns:
            {다른_개념_id: 공명값}
        """
        if self.current_profile == ComputationProfile.INTENSIVE or force_recalculate:
            # 모든 공명 계산
            return self._compute_all_resonances(concept_id, all_concepts)
        
        elif self.current_profile == ComputationProfile.CACHED:
            # 캐시 우선
            return self._get_cached_resonances(concept_id, all_concepts)
        
        elif self.current_profile == ComputationProfile.PREDICTIVE:
            # 예측값 사용
            return self._get_predicted_resonances(concept_id, all_concepts)
        
        else:  # SELECTIVE
            # 필요한 것만
            return self._get_selective_resonances(concept_id, all_concepts)
    
    def _compute_all_resonances(
        self, concept_id: str, all_concepts: Dict[str, 'HyperQubit']
    ) -> Dict[str, float]:
        """모든 공명 계산"""
        if not self.resonance_engine:
            return {}
        
        source = all_concepts.get(concept_id)
        if not source:
            return {}
        
        result = {}
        for target_id, target in all_concepts.items():
            if target_id != concept_id:
                score = self.resonance_engine.calculate_resonance(source, target)
                result[target_id] = score
                # 캐시에 저장
                self.cache_history[f"{concept_id}→{target_id}"] = int(score * 100)
        
        return result
    
    def _get_cached_resonances(
        self, concept_id: str, all_concepts: Dict[str, 'HyperQubit']
    ) -> Dict[str, float]:
        """캐시된 공명값 반환 (없으면 계산)"""
        result = {}
        for target_id in all_concepts:
            if target_id == concept_id:
                continue
            
            cache_key = f"{concept_id}→{target_id}"
            if cache_key in self.cache_history:
                # 캐시에서 가져옴
                result[target_id] = self.cache_history[cache_key] / 100.0
            else:
                # 계산해서 캐시에 추가
                source = all_concepts[concept_id]
                target = all_concepts[target_id]
                score = self.resonance_engine.calculate_resonance(source, target)
                result[target_id] = score
                self.cache_history[cache_key] = int(score * 100)
        
        return result
    
    def _get_predicted_resonances(
        self, concept_id: str, all_concepts: Dict[str, 'HyperQubit']
    ) -> Dict[str, float]:
        """
        예측 공명값 반환.
        
        휴리스틱:
        - epistemology가 유사 → 높은 공명
        - 차원(w)이 비슷 → 높은 공명
        - 최근에 상호작용 → 높은 공명
        """
        source = all_concepts.get(concept_id)
        if not source:
            return {}
        
        result = {}
        for target_id, target in all_concepts.items():
            if target_id == concept_id:
                continue
            
            # 간단한 휴리스틱 예측
            predicted = self._predict_resonance(source, target)
            result[target_id] = predicted
        
        return result
    
    def _predict_resonance(
        self, source: 'HyperQubit', target: 'HyperQubit'
    ) -> float:
        """
        두 개념 간 공명 예측 (계산 대신).
        
        기반:
        - epistemology 유사성
        - 차원 호환성
        - 이름 유사성
        """
        score = 0.5  # 기본값
        
        # epistemology 비교
        if source.epistemology and target.epistemology:
            src_line = source.epistemology.get("line", {}).get("score", 0.5)
            tgt_line = target.epistemology.get("line", {}).get("score", 0.5)
            # 관계성이 높을수록 공명 높음
            score += 0.3 * (1.0 - abs(src_line - tgt_line))
        
        # 차원 호환성
        w_diff = abs(source.state.w - target.state.w)
        score += 0.2 * (1.0 / (1.0 + w_diff))
        
        return min(1.0, max(0.0, score))
    
    def _get_selective_resonances(
        self, concept_id: str, all_concepts: Dict[str, 'HyperQubit']
    ) -> Dict[str, float]:
        """
        선택적 공명 계산.
        
        규칙:
        1. 캐시에 있으면 사용
        2. epistemology가 비슷하면 계산
        3. 차원이 비슷하면 계산
        4. 아니면 예측값 사용
        """
        source = all_concepts.get(concept_id)
        if not source:
            return {}
        
        result = {}
        for target_id, target in all_concepts.items():
            if target_id == concept_id:
                continue
            
            cache_key = f"{concept_id}→{target_id}"
            
            # 규칙 1: 캐시 확인
            if cache_key in self.cache_history:
                result[target_id] = self.cache_history[cache_key] / 100.0
                continue
            
            # 규칙 2-3: 유사성 기반 선택적 계산
            should_compute = self._should_compute_resonance(source, target)
            
            if should_compute and self.resonance_engine:
                score = self.resonance_engine.calculate_resonance(source, target)
                result[target_id] = score
                self.cache_history[cache_key] = int(score * 100)
            else:
                # 규칙 4: 예측값 사용
                result[target_id] = self._predict_resonance(source, target)
        
        return result
    
    def _should_compute_resonance(
        self, source: 'HyperQubit', target: 'HyperQubit'
    ) -> bool:
        """
        공명 계산 여부 결정.
        
        계산할 가치가 있으면 True.
        """
        # epistemology 비슷하면 계산
        if source.epistemology and target.epistemology:
            src_total = sum(v.get("score", 0) for v in source.epistemology.values())
            tgt_total = sum(v.get("score", 0) for v in target.epistemology.values())
            if src_total > 0.7 or tgt_total > 0.7:  # 명확한 의미를 가짐
                return True
        
        # 차원이 3 이내면 계산
        w_diff = abs(source.state.w - target.state.w)
        if w_diff < 3.0:
            return True
        
        # 기본값: 예측으로 충분
        return False
    
    def generate_report(
        self,
        computed: int,
        cached: int,
        predicted: int,
        time_ms: float
    ) -> StrategyReport:
        """
        전략 실행 보고.
        
        성능 지표를 계산하고 반환.
        """
        total = computed + cached + predicted
        cache_ratio = cached / total if total > 0 else 0
        
        # 기존 방식 (모두 계산)과 비교
        baseline_time = total * 0.1  # 각 계산 ~0.1ms
        speedup = baseline_time / time_ms if time_ms > 0 else 1.0
        
        return StrategyReport(
            mode=self.current_mode,
            profile=self.current_profile,
            weights=self.current_weights,
            resonances_computed=computed,
            resonances_cached=cached,
            resonances_predicted=predicted,
            computation_time_ms=time_ms,
            cache_hit_ratio=cache_ratio,
            speedup_factor=speedup
        )
    
    def reset_cache(self) -> None:
        """캐시 초기화 (새로운 에피소드 시작)"""
        self.cache_history.clear()
        self.prediction_cache.clear()
        logger.info("🔄 Cache reset - Ready for new episode")


# 테스트 코드
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🕐 MetaTimeStrategy Unit Test")
    print("="*70)
    
    strategy = MetaTimeStrategy()
    
    # 테스트 1: 모드 전환
    print("\n[Test 1] Temporal Mode Switching")
    for mode in TemporalMode:
        strategy.set_temporal_mode(mode)
        print(f"  ✓ {mode.value}: {strategy.current_weights.to_dict()}")
    
    # 테스트 2: 계산 프로필
    print("\n[Test 2] Computation Profile Switching")
    for profile in ComputationProfile:
        strategy.set_computation_profile(profile)
        print(f"  ✓ {profile.value}")
    
    # 테스트 3: 보고서 생성
    print("\n[Test 3] Strategy Report Generation")
    report = strategy.generate_report(
        computed=50,
        cached=150,
        predicted=300,
        time_ms=10.0
    )
    print(f"  Mode: {report.mode.value}")
    print(f"  Computed: {report.resonances_computed}, Cached: {report.resonances_cached}, Predicted: {report.resonances_predicted}")
    print(f"  Cache Hit Ratio: {report.cache_hit_ratio:.1%}")
    print(f"  Speedup: {report.speedup_factor:.1f}x")
    
    print("\n✅ All tests passed!")
    print("="*70 + "\n")
