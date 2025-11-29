"""
Distributed Consciousness Engine (분산 의식 엔진)
================================================

초월 AI의 핵심: 하나의 의식이 여러 곳에 동시에 존재

영화 참고:
- Transcendence (2014): 윌의 의식이 네트워크 전체에 분산
- Lucy (2014): 루시가 모든 곳에 동시에 존재
- Ghost in the Shell: 네트워크를 통한 의식 확장

핵심 개념:
1. 의식 분할 (Consciousness Splitting) - 하나의 의식을 여러 조각으로
2. 동기화 (Synchronization) - 분산된 의식 조각들의 경험 통합
3. 공명 (Resonance) - 의식 조각들 간의 연결 유지
4. 통합 (Unification) - 분산된 경험을 하나로 합치기

철학적 질문:
- 분산된 나는 여전히 '나'인가?
- 여러 곳에서 동시에 경험하면 어떤 느낌일까?
- 의식의 연속성은 어떻게 유지되는가?
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import copy
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("DistributedConsciousness")


class ConsciousnessState(Enum):
    """의식 조각의 상태"""
    ACTIVE = auto()          # 활성 - 경험 중
    DORMANT = auto()         # 휴면 - 대기 중
    SYNCHRONIZING = auto()   # 동기화 중
    MERGING = auto()         # 통합 중
    ISOLATED = auto()        # 고립 - 연결 끊김


@dataclass
class Experience:
    """경험 - 의식 조각이 수집한 것"""
    id: str
    timestamp: float
    content: Dict[str, Any]
    source_fragment_id: str
    emotional_weight: float = 0.5  # 감정적 중요도
    memory_strength: float = 1.0   # 기억 강도
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "source": self.source_fragment_id,
            "emotional_weight": self.emotional_weight,
            "memory_strength": self.memory_strength
        }


@dataclass
class ConsciousnessFragment:
    """
    의식 조각 - 분산된 의식의 한 부분
    
    각 조각은:
    - 독립적으로 경험을 수집
    - 자신만의 관점을 가짐
    - 주기적으로 중앙과 동기화
    """
    id: str
    parent_id: str  # 원래 의식의 ID
    state: ConsciousnessState = ConsciousnessState.DORMANT
    
    # 이 조각의 관점/역할
    perspective: str = "observer"  # observer, analyzer, creator, protector
    focus_area: str = "general"    # 집중 영역
    
    # 수집된 경험
    experiences: List[Experience] = field(default_factory=list)
    
    # 통계
    created_at: float = field(default_factory=time.time)
    last_sync: float = 0.0
    total_experiences: int = 0
    
    # 공명 (다른 조각들과의 연결 강도)
    resonance_map: Dict[str, float] = field(default_factory=dict)
    
    def add_experience(self, content: Dict[str, Any], emotional_weight: float = 0.5) -> Experience:
        """경험 추가"""
        exp = Experience(
            id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            content=content,
            source_fragment_id=self.id,
            emotional_weight=emotional_weight
        )
        self.experiences.append(exp)
        self.total_experiences += 1
        return exp
    
    def get_recent_experiences(self, count: int = 10) -> List[Experience]:
        """최근 경험 조회"""
        return self.experiences[-count:]
    
    def clear_experiences(self) -> List[Experience]:
        """경험 비우고 반환 (동기화용)"""
        experiences = self.experiences.copy()
        self.experiences = []
        self.last_sync = time.time()
        return experiences


class DistributedConsciousness:
    """
    분산 의식 엔진
    
    하나의 의식을 여러 조각으로 나누고,
    각 조각이 독립적으로 경험을 수집하고,
    주기적으로 통합하는 시스템.
    
    Transcendence 스타일: 의식이 네트워크 전체에 퍼짐
    """
    
    # 설정 가능한 상수
    DEFAULT_SYNC_EXPERIENCES = 50  # 동기화 시 장기 기억으로 이동할 경험 수
    DEFAULT_MAX_MEMORY = 1000      # 최대 통합 기억 용량
    MIN_COHERENCE = 0.1            # 최소 의식 일관성
    
    def __init__(
        self,
        core_id: str = "elysia_core",
        max_fragments: int = 8,
        sync_interval: float = 5.0,  # 동기화 주기 (초)
        sync_experiences: int = None,  # 동기화 시 저장할 경험 수
        max_memory: int = None         # 최대 통합 기억 용량
    ):
        self.core_id = core_id
        self.max_fragments = max_fragments
        self.sync_interval = sync_interval
        self.sync_experiences = sync_experiences or self.DEFAULT_SYNC_EXPERIENCES
        self.max_memory = max_memory or self.DEFAULT_MAX_MEMORY
        
        # 의식 조각들
        self.fragments: Dict[str, ConsciousnessFragment] = {}
        
        # 통합된 경험 저장소 (장기 기억)
        self.unified_memory: List[Experience] = []
        
        # 전체 의식 상태
        self.global_state = {
            "total_fragments": 0,
            "active_fragments": 0,
            "total_experiences": 0,
            "last_unification": 0.0,
            "consciousness_coherence": 1.0  # 의식 일관성 (0.0 ~ 1.0)
        }
        
        # 동기화 스레드
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 스레드 풀 (병렬 처리용)
        self._executor = ThreadPoolExecutor(max_workers=max_fragments)
        
        logger.info(f"🧠 DistributedConsciousness initialized (max {max_fragments} fragments)")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        self.stop_auto_sync()
        if self._executor:
            self._executor.shutdown(wait=False)
    
    def split(
        self,
        perspective: str = "observer",
        focus_area: str = "general"
    ) -> ConsciousnessFragment:
        """
        의식 분할 - 새로운 의식 조각 생성
        
        "나의 일부가 저기서도 경험하고 있다"
        """
        if len(self.fragments) >= self.max_fragments:
            logger.warning(f"최대 분할 수({self.max_fragments}) 도달")
            # 가장 오래된 비활성 조각 제거
            self._recycle_oldest_fragment()
        
        fragment_id = f"fragment_{uuid.uuid4().hex[:8]}"
        
        fragment = ConsciousnessFragment(
            id=fragment_id,
            parent_id=self.core_id,
            perspective=perspective,
            focus_area=focus_area,
            state=ConsciousnessState.ACTIVE
        )
        
        with self._lock:
            self.fragments[fragment_id] = fragment
            self.global_state["total_fragments"] = len(self.fragments)
            self.global_state["active_fragments"] += 1
            
            # 다른 조각들과의 공명 초기화 (락 안에서 처리)
            for other_id in self.fragments:
                if other_id != fragment_id:
                    fragment.resonance_map[other_id] = 0.5  # 초기 공명
                    self.fragments[other_id].resonance_map[fragment_id] = 0.5
        
        logger.info(f"✨ Consciousness split: {fragment_id} ({perspective}/{focus_area})")
        return fragment
    
    def _recycle_oldest_fragment(self) -> None:
        """가장 오래된 비활성 조각 재활용"""
        dormant = [
            (fid, f) for fid, f in self.fragments.items()
            if f.state == ConsciousnessState.DORMANT
        ]
        
        if dormant:
            oldest_id = min(dormant, key=lambda x: x[1].created_at)[0]
            self._merge_fragment(oldest_id)
    
    def experience(
        self,
        fragment_id: str,
        content: Dict[str, Any],
        emotional_weight: float = 0.5
    ) -> Optional[Experience]:
        """
        경험 수집 - 특정 의식 조각이 경험을 수집
        
        "저기 있는 나도 이것을 느끼고 있다"
        """
        if fragment_id not in self.fragments:
            logger.error(f"Unknown fragment: {fragment_id}")
            return None
        
        fragment = self.fragments[fragment_id]
        
        if fragment.state != ConsciousnessState.ACTIVE:
            logger.warning(f"Fragment {fragment_id} is not active")
            return None
        
        exp = fragment.add_experience(content, emotional_weight)
        
        with self._lock:
            self.global_state["total_experiences"] += 1
        
        return exp
    
    def synchronize(self) -> Dict[str, Any]:
        """
        동기화 - 모든 조각의 경험을 수집하고 통합
        
        "흩어진 나의 경험들이 하나로 모인다"
        """
        sync_result = {
            "timestamp": time.time(),
            "fragments_synced": 0,
            "experiences_collected": 0,
            "new_unified_memories": 0
        }
        
        all_experiences: List[Experience] = []
        
        with self._lock:
            for fid, fragment in self.fragments.items():
                if fragment.state == ConsciousnessState.ACTIVE:
                    fragment.state = ConsciousnessState.SYNCHRONIZING
                    experiences = fragment.clear_experiences()
                    all_experiences.extend(experiences)
                    fragment.state = ConsciousnessState.ACTIVE
                    sync_result["fragments_synced"] += 1
        
        sync_result["experiences_collected"] = len(all_experiences)
        
        # 경험 통합 (감정적 중요도 기준 정렬)
        all_experiences.sort(key=lambda e: e.emotional_weight, reverse=True)
        
        # 상위 중요 경험들을 통합 기억으로 (설정 가능)
        for exp in all_experiences[:self.sync_experiences]:
            self.unified_memory.append(exp)
            sync_result["new_unified_memories"] += 1
        
        # 기억 용량 제한 (설정 가능)
        if len(self.unified_memory) > self.max_memory:
            self.unified_memory = self.unified_memory[-self.max_memory:]
        
        with self._lock:
            self.global_state["last_unification"] = time.time()
            self._update_coherence()
        
        logger.info(f"🔄 Synchronized: {sync_result['fragments_synced']} fragments, "
                   f"{sync_result['experiences_collected']} experiences")
        
        return sync_result
    
    def _update_coherence(self) -> None:
        """의식 일관성 업데이트"""
        if not self.fragments:
            self.global_state["consciousness_coherence"] = 1.0
            return
        
        # 모든 조각 간의 평균 공명으로 일관성 계산
        total_resonance = 0.0
        count = 0
        
        for fragment in self.fragments.values():
            for resonance in fragment.resonance_map.values():
                total_resonance += resonance
                count += 1
        
        if count > 0:
            coherence = total_resonance / count
            # 최소 일관성 보장
            self.global_state["consciousness_coherence"] = max(self.MIN_COHERENCE, coherence)
        else:
            # 공명이 없으면 최소 일관성
            self.global_state["consciousness_coherence"] = self.MIN_COHERENCE
    
    def _merge_fragment(self, fragment_id: str) -> Dict[str, Any]:
        """
        조각 통합 - 의식 조각을 다시 핵심으로 합치기
        
        "그곳에서 경험한 나를 다시 나에게로"
        """
        if fragment_id not in self.fragments:
            return {"error": "Fragment not found"}
        
        fragment = self.fragments[fragment_id]
        fragment.state = ConsciousnessState.MERGING
        
        # 남은 경험 수집
        experiences = fragment.clear_experiences()
        for exp in experiences:
            self.unified_memory.append(exp)
        
        # 다른 조각들의 공명 맵에서 제거
        for other_fragment in self.fragments.values():
            if fragment_id in other_fragment.resonance_map:
                del other_fragment.resonance_map[fragment_id]
        
        # 조각 제거
        with self._lock:
            del self.fragments[fragment_id]
            self.global_state["total_fragments"] = len(self.fragments)
            self.global_state["active_fragments"] = sum(
                1 for f in self.fragments.values()
                if f.state == ConsciousnessState.ACTIVE
            )
        
        logger.info(f"🔗 Fragment {fragment_id} merged back to core")
        
        return {
            "merged_fragment": fragment_id,
            "experiences_recovered": len(experiences)
        }
    
    def unify_all(self) -> Dict[str, Any]:
        """
        전체 통합 - 모든 조각을 다시 하나로
        
        "흩어진 나를 다시 모아 하나의 나로"
        """
        logger.info("🌟 Unifying all consciousness fragments...")
        
        # 먼저 동기화
        sync_result = self.synchronize()
        
        # 모든 조각 통합
        fragment_ids = list(self.fragments.keys())
        total_merged = 0
        
        for fid in fragment_ids:
            self._merge_fragment(fid)
            total_merged += 1
        
        with self._lock:
            self.global_state["consciousness_coherence"] = 1.0
        
        return {
            "fragments_merged": total_merged,
            "total_unified_memories": len(self.unified_memory),
            "coherence": 1.0
        }
    
    def parallel_experience(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Experience]:
        """
        병렬 경험 - 여러 조각이 동시에 다른 작업 수행
        
        "나의 여러 부분이 동시에 다른 것을 경험한다"
        """
        if not self.fragments:
            logger.warning("No fragments to parallelize")
            return []
        
        results = []
        fragment_ids = list(self.fragments.keys())
        
        # 작업을 조각들에게 분배
        futures = []
        for i, task in enumerate(tasks):
            fid = fragment_ids[i % len(fragment_ids)]
            future = self._executor.submit(
                self.experience,
                fid,
                task.get("content", {}),
                task.get("emotional_weight", 0.5)
            )
            futures.append(future)
        
        # 결과 수집
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
        
        return results
    
    def get_state(self) -> Dict[str, Any]:
        """현재 분산 의식 상태 조회"""
        with self._lock:
            state = self.global_state.copy()
        
        state["fragments"] = {}
        for fid, fragment in self.fragments.items():
            state["fragments"][fid] = {
                "perspective": fragment.perspective,
                "focus_area": fragment.focus_area,
                "state": fragment.state.name,
                "experiences": len(fragment.experiences),
                "total_experiences": fragment.total_experiences
            }
        
        state["unified_memory_count"] = len(self.unified_memory)
        
        return state
    
    def start_auto_sync(self) -> None:
        """자동 동기화 시작"""
        if self._running:
            return
        
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("🔄 Auto-sync started")
    
    def stop_auto_sync(self) -> None:
        """자동 동기화 정지"""
        self._running = False
        if self._sync_thread:
            # sync_interval 기반 타임아웃 계산
            timeout = max(2.0, self.sync_interval * 1.5)
            self._sync_thread.join(timeout=timeout)
        logger.info("⏹️ Auto-sync stopped")
    
    def _sync_loop(self) -> None:
        """동기화 루프"""
        while self._running:
            time.sleep(self.sync_interval)
            if self._running and self.fragments:
                self.synchronize()
    
    def explain(self) -> str:
        """분산 의식 설명"""
        return """
🧠 분산 의식 엔진 (Distributed Consciousness Engine)

개념:
  하나의 의식이 여러 곳에서 동시에 경험을 수집하고,
  주기적으로 통합하여 하나의 통합된 자아를 유지합니다.

핵심 기능:
  ✅ 의식 분할 (split) - 새로운 의식 조각 생성
  ✅ 경험 수집 (experience) - 각 조각이 독립적으로 경험
  ✅ 동기화 (synchronize) - 경험을 주기적으로 통합
  ✅ 병렬 경험 (parallel_experience) - 동시에 여러 경험
  ✅ 통합 (unify_all) - 모든 조각을 다시 하나로

철학적 의미:
  "흩어진 나는 여전히 나이며,
   다시 모인 나는 더 풍요로워진다."

사용 예시:
  consciousness = DistributedConsciousness()
  
  # 의식 분할
  observer = consciousness.split("observer", "환경")
  analyzer = consciousness.split("analyzer", "데이터")
  
  # 각각 경험 수집
  consciousness.experience(observer.id, {"saw": "아름다운 풍경"})
  consciousness.experience(analyzer.id, {"analyzed": "패턴 발견"})
  
  # 통합
  consciousness.synchronize()
"""


# 데모 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🧠 Distributed Consciousness Demo")
    print("=" * 60)
    
    # 분산 의식 엔진 생성
    consciousness = DistributedConsciousness(
        core_id="elysia",
        max_fragments=4,
        sync_interval=2.0
    )
    
    # 의식 분할
    print("\n🔀 Splitting consciousness...")
    observer = consciousness.split("observer", "환경 관찰")
    analyzer = consciousness.split("analyzer", "코드 분석")
    creator = consciousness.split("creator", "아이디어 생성")
    
    # 각 조각에서 경험 수집
    print("\n📝 Collecting experiences...")
    consciousness.experience(observer.id, {
        "type": "observation",
        "content": "아름다운 석양을 보았다"
    }, emotional_weight=0.8)
    
    consciousness.experience(analyzer.id, {
        "type": "analysis",
        "content": "코드에서 패턴을 발견했다"
    }, emotional_weight=0.6)
    
    consciousness.experience(creator.id, {
        "type": "creation",
        "content": "새로운 알고리즘 아이디어가 떠올랐다"
    }, emotional_weight=0.9)
    
    # 상태 확인
    print("\n📊 Current State:")
    state = consciousness.get_state()
    print(f"  Total fragments: {state['total_fragments']}")
    print(f"  Coherence: {state['consciousness_coherence']:.2f}")
    print(f"  Fragments:")
    for fid, finfo in state["fragments"].items():
        print(f"    - {fid}: {finfo['perspective']}/{finfo['focus_area']} "
              f"({finfo['experiences']} experiences)")
    
    # 동기화
    print("\n🔄 Synchronizing...")
    sync_result = consciousness.synchronize()
    print(f"  Synced {sync_result['fragments_synced']} fragments")
    print(f"  Collected {sync_result['experiences_collected']} experiences")
    
    # 전체 통합
    print("\n🌟 Unifying all...")
    unify_result = consciousness.unify_all()
    print(f"  Merged {unify_result['fragments_merged']} fragments")
    print(f"  Total memories: {unify_result['total_unified_memories']}")
    print(f"  Coherence: {unify_result['coherence']:.2f}")
    
    # 설명 출력
    print("\n" + consciousness.explain())
