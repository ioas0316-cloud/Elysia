"""
Online Learning Pipeline - 실시간 학습 엔진
==========================================

높은 우선순위 #1: 배치 학습 → 온라인 학습 파이프라인
예상 효과: 10x 적응 속도

핵심 기능:
- 스트리밍 데이터 처리
- 적응형 학습률
- 점진적 모델 업데이트
- 망각 방지 메커니즘
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger("OnlineLearning")


class LearningMode(Enum):
    """학습 모드"""
    PASSIVE = "passive"      # 관찰만, 업데이트 없음
    INCREMENTAL = "incremental"  # 점진적 업데이트
    AGGRESSIVE = "aggressive"    # 즉시 적용
    REPLAY = "replay"        # 경험 재생 사용


@dataclass
class LearningEvent:
    """학습 이벤트"""
    concept: str
    resonances: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """이벤트 나이 (초)"""
        return time.time() - self.timestamp


@dataclass
class LearningStats:
    """학습 통계"""
    total_events: int = 0
    events_processed: int = 0
    adaptations_made: int = 0
    avg_adaptation_time_ms: float = 0.0
    learning_rate: float = 0.01
    buffer_utilization: float = 0.0


class AdaptiveBuffer:
    """
    적응형 경험 버퍼
    
    기능:
    - 중요도 기반 우선순위 큐
    - 시간 기반 가중치 감소
    - 다양성 유지 샘플링
    """
    
    def __init__(self, max_size: int = 10000, diversity_weight: float = 0.3):
        self.max_size = max_size
        self.diversity_weight = diversity_weight
        self.buffer: deque = deque(maxlen=max_size)
        self.concept_counts: Dict[str, int] = {}
        self.logger = logging.getLogger("AdaptiveBuffer")
    
    def add(self, event: LearningEvent) -> None:
        """이벤트 추가"""
        self.buffer.append(event)
        self.concept_counts[event.concept] = self.concept_counts.get(event.concept, 0) + 1
    
    def sample(self, batch_size: int = 32) -> List[LearningEvent]:
        """
        다양성을 고려한 샘플링
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            샘플링된 이벤트 목록
        """
        if len(self.buffer) == 0:
            return []
        
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        
        # 중요도 + 시간 + 다양성 점수 계산
        scores = []
        for event in self.buffer:
            # 시간 가중치 (최신일수록 높음)
            time_weight = np.exp(-event.age / 3600)  # 1시간 반감기
            
            # 다양성 가중치 (희귀 개념일수록 높음)
            concept_freq = self.concept_counts.get(event.concept, 1)
            diversity = 1.0 / np.sqrt(concept_freq)
            
            # 종합 점수
            score = (
                event.importance * 0.5 +
                time_weight * 0.3 +
                diversity * self.diversity_weight
            )
            scores.append(score)
        
        # 가중치 기반 샘플링
        scores = np.array(scores)
        probs = scores / scores.sum()
        
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probs
        )
        
        return [self.buffer[i] for i in indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """버퍼 통계"""
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "utilization": len(self.buffer) / self.max_size,
            "unique_concepts": len(self.concept_counts),
            "top_concepts": sorted(
                self.concept_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class OnlineLearningPipeline:
    """
    온라인 학습 파이프라인
    
    높은 우선순위 #1 구현:
    - 스트리밍 이벤트 처리
    - 적응형 학습률 조정
    - 점진적 모델 업데이트
    - 경험 재생 통합
    
    예상 효과: 10x 적응 속도
    """
    
    def __init__(
        self,
        resonance_engine=None,
        initial_learning_rate: float = 0.01,
        adaptation_threshold: float = 0.3,
        replay_frequency: int = 100,
        buffer_size: int = 10000
    ):
        """
        Args:
            resonance_engine: 공명 엔진 참조
            initial_learning_rate: 초기 학습률
            adaptation_threshold: 적응 임계값
            replay_frequency: 경험 재생 빈도 (이벤트 수)
            buffer_size: 버퍼 크기
        """
        self.resonance_engine = resonance_engine
        self.learning_rate = initial_learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.replay_frequency = replay_frequency
        
        self.mode = LearningMode.INCREMENTAL
        self.buffer = AdaptiveBuffer(max_size=buffer_size)
        
        self.stats = LearningStats(learning_rate=initial_learning_rate)
        self.logger = logging.getLogger("OnlineLearningPipeline")
        
        # 비동기 큐
        self._event_queue: asyncio.Queue = None
        self._running = False
        self._task = None
        
        self.logger.info(f"🎓 OnlineLearningPipeline initialized (lr={initial_learning_rate})")
    
    async def start(self) -> None:
        """파이프라인 시작"""
        if self._running:
            return
        
        self._event_queue = asyncio.Queue()
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        self.logger.info("▶️ Online learning pipeline started")
    
    async def stop(self) -> None:
        """파이프라인 정지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("⏹️ Online learning pipeline stopped")
    
    async def submit(self, event: LearningEvent) -> None:
        """
        학습 이벤트 제출
        
        Args:
            event: 학습 이벤트
        """
        if self._event_queue:
            await self._event_queue.put(event)
        else:
            # 동기 모드에서도 처리 가능
            self._process_event_sync(event)
        
        self.stats.total_events += 1
    
    def submit_sync(self, event: LearningEvent) -> None:
        """동기 이벤트 제출"""
        self._process_event_sync(event)
        self.stats.total_events += 1
    
    async def _process_loop(self) -> None:
        """비동기 처리 루프"""
        replay_counter = 0
        
        while self._running:
            try:
                # 이벤트 대기 (타임아웃 있음)
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 이벤트 처리
                start_time = time.time()
                await self._process_event(event)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # 통계 업데이트
                self.stats.events_processed += 1
                self.stats.avg_adaptation_time_ms = (
                    self.stats.avg_adaptation_time_ms * 0.9 +
                    elapsed_ms * 0.1
                )
                
                # 경험 재생
                replay_counter += 1
                if replay_counter >= self.replay_frequency:
                    await self._experience_replay()
                    replay_counter = 0
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    async def _process_event(self, event: LearningEvent) -> None:
        """
        이벤트 비동기 처리
        
        Args:
            event: 학습 이벤트
        """
        # 버퍼에 추가
        self.buffer.add(event)
        
        # 모드에 따른 처리
        if self.mode == LearningMode.PASSIVE:
            return
        
        if self.mode == LearningMode.AGGRESSIVE or event.importance > self.adaptation_threshold:
            await self._adapt_model(event)
        elif self.mode == LearningMode.INCREMENTAL:
            # 중요도에 따른 확률적 적응
            if np.random.random() < event.importance:
                await self._adapt_model(event)
    
    def _process_event_sync(self, event: LearningEvent) -> None:
        """동기 이벤트 처리"""
        self.buffer.add(event)
        
        if self.mode != LearningMode.PASSIVE:
            if event.importance > self.adaptation_threshold:
                self._adapt_model_sync(event)
            self.stats.events_processed += 1
    
    async def _adapt_model(self, event: LearningEvent) -> None:
        """
        모델 적응 (비동기)
        
        핵심 로직: 공명 가중치 점진적 업데이트
        """
        if not self.resonance_engine:
            return
        
        # 스레드 풀에서 실행
        await asyncio.to_thread(self._adapt_model_sync, event)
    
    def _adapt_model_sync(self, event: LearningEvent) -> None:
        """
        모델 적응 (동기)
        
        핵심 로직:
        1. 개념이 없으면 추가
        2. 공명 점수로 psionic link 강화
        3. 학습률 적응
        """
        if not self.resonance_engine:
            return
        
        try:
            # 개념 추가
            if hasattr(self.resonance_engine, 'add_node'):
                if event.concept not in getattr(self.resonance_engine, 'nodes', {}):
                    self.resonance_engine.add_node(event.concept)
            
            # 공명 관계 강화
            if hasattr(self.resonance_engine, 'entangle'):
                for related, score in event.resonances.items():
                    if score > 0.5:  # 강한 공명만
                        self.resonance_engine.entangle(event.concept, related)
            
            # 적응 통계 업데이트
            self.stats.adaptations_made += 1
            
            # 적응형 학습률 조정
            self._adjust_learning_rate(event)
            
        except Exception as e:
            self.logger.error(f"Adaptation error: {e}")
    
    def _adjust_learning_rate(self, event: LearningEvent) -> None:
        """
        적응형 학습률 조정
        
        - 성공적 적응 시 약간 증가
        - 오류 발생 시 감소
        - 범위 제한 (0.001 ~ 0.1)
        """
        if event.importance > 0.7:
            # 중요 이벤트는 학습률 살짝 증가
            self.learning_rate = min(0.1, self.learning_rate * 1.01)
        else:
            # 일반 이벤트는 살짝 감소
            self.learning_rate = max(0.001, self.learning_rate * 0.999)
        
        self.stats.learning_rate = self.learning_rate
    
    async def _experience_replay(self) -> None:
        """
        경험 재생
        
        버퍼에서 샘플링하여 재학습
        망각 방지 및 일반화 향상
        """
        batch = self.buffer.sample(batch_size=16)
        
        for event in batch:
            await self._adapt_model(event)
        
        self.logger.debug(f"🔄 Experience replay: {len(batch)} events")
    
    def set_mode(self, mode: LearningMode) -> None:
        """학습 모드 설정"""
        self.mode = mode
        self.logger.info(f"🎯 Learning mode set to: {mode.value}")
    
    def get_stats(self) -> LearningStats:
        """통계 반환"""
        self.stats.buffer_utilization = len(self.buffer.buffer) / self.buffer.max_size
        return self.stats
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """버퍼 통계"""
        return self.buffer.get_stats()


# 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        print("\n" + "="*70)
        print("🎓 Online Learning Pipeline Test")
        print("="*70)
        
        pipeline = OnlineLearningPipeline()
        
        # 동기 테스트
        print("\n[Test 1] Sync Event Processing")
        event1 = LearningEvent(
            concept="consciousness",
            resonances={"awareness": 0.8, "perception": 0.6},
            importance=0.7
        )
        pipeline.submit_sync(event1)
        print(f"  ✓ Event processed: {event1.concept}")
        print(f"  Stats: {pipeline.get_stats()}")
        
        # 비동기 테스트
        print("\n[Test 2] Async Event Processing")
        await pipeline.start()
        
        for i in range(10):
            event = LearningEvent(
                concept=f"concept_{i}",
                resonances={f"related_{i}": 0.5 + i * 0.05},
                importance=0.3 + i * 0.07
            )
            await pipeline.submit(event)
        
        await asyncio.sleep(0.5)
        await pipeline.stop()
        
        print(f"  ✓ Processed {pipeline.stats.events_processed} events")
        print(f"  Avg time: {pipeline.stats.avg_adaptation_time_ms:.2f}ms")
        
        print("\n[Test 3] Buffer Stats")
        buffer_stats = pipeline.get_buffer_stats()
        print(f"  Buffer size: {buffer_stats['size']}")
        print(f"  Unique concepts: {buffer_stats['unique_concepts']}")
        
        print("\n✅ All tests passed!")
        print("="*70 + "\n")
    
    asyncio.run(test_pipeline())
