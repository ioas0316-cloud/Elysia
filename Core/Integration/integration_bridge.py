#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IntegrationBridge: 분산된 모듈들을 통합하는 중간 계층
==================================================

역할:
1. 모듈 간 계약 차이 해결 (Interface Adapter)
2. 데이터 흐름 표준화 (Event Stream)
3. 오류 처리 통합 (Error Handling)
4. 성능 모니터링 (Performance Metrics)

구조:
  Simulation ← SimulationEvent
    ↓
  IntegrationBridge (여기)
    ├─ ResonanceAdapter (공명 → 표준 형식)
    ├─ HippocampusAdapter (기억 → 표준 형식)
    ├─ ExperienceAdapter (경험 → 표준 형식)
    └─ MetaAgentAdapter (전략 → 표준 형식)
    ↓
  MetaAgent (의사결정)
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger("IntegrationBridge")


class EventType(Enum):
    """통합 이벤트 타입"""
    SIMULATION_TICK = "simulation_tick"
    RESONANCE_COMPUTED = "resonance_computed"
    CONCEPT_EMERGED = "concept_emerged"
    RELATIONSHIP_DISCOVERED = "relationship_discovered"
    PHASE_RESONANCE_EVENT = "phase_resonance_event"
    LANGUAGE_TURN = "language_turn"
    EXPERIENCE_DIGESTED = "experience_digested"
    STRATEGY_DECISION = "strategy_decision"
    CHECKPOINT_SAVED = "checkpoint_saved"


@dataclass
class IntegrationEvent:
    """통합 시스템을 통과하는 표준 이벤트"""
    
    event_type: EventType
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    tick: int = 0
    source_module: str = "unknown"
    
    # 핵심 데이터
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터
    importance: float = 0.5  # 0~1 (1 = 매우 중요)
    requires_action: bool = False
    
    # 추적
    propagation_chain: List[str] = field(default_factory=list)
    
    def add_propagation_step(self, module_name: str) -> None:
        """이벤트가 어느 모듈을 거쳤는지 추적"""
        self.propagation_chain.append(f"{module_name}@{datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """로깅용 딕셔너리 변환"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "tick": self.tick,
            "source": self.source_module,
            "importance": self.importance,
            "requires_action": self.requires_action,
            "data_keys": list(self.data.keys()),
            "chain_length": len(self.propagation_chain)
        }


@dataclass
class ResonanceData:
    """표준화된 공명 데이터"""
    source_concept: str
    resonances: Dict[str, float]  # target_concept → score
    explanation: Optional[str] = None
    computed_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_event(self, tick: int) -> IntegrationEvent:
        """IntegrationEvent로 변환"""
        return IntegrationEvent(
            event_type=EventType.RESONANCE_COMPUTED,
            tick=tick,
            source_module="ResonanceEngine",
            data={
                "source": self.source_concept,
                "resonances": self.resonances,
                "explanation": self.explanation
            },
            importance=0.6
        )


@dataclass
class ConceptData:
    """표준화된 개념 데이터"""
    concept_id: str
    name: str
    concept_type: str  # "emergent", "primitive", "discovered"
    epistemology: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_event(self, tick: int) -> IntegrationEvent:
        """IntegrationEvent로 변환"""
        return IntegrationEvent(
            event_type=EventType.CONCEPT_EMERGED,
            tick=tick,
            source_module="ExperienceDigester",
            data={
                "concept_id": self.concept_id,
                "name": self.name,
                "type": self.concept_type,
                "epistemology": self.epistemology
            },
            importance=0.7,
            requires_action=True
        )


@dataclass
class RelationshipData:
    """표준화된 관계 데이터"""
    source_concept: str
    target_concept: str
    relationship_type: str  # "causes", "inhibits", "resonates", "evolves"
    strength: float  # 0~1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_event(self, tick: int) -> IntegrationEvent:
        """IntegrationEvent로 변환"""
        return IntegrationEvent(
            event_type=EventType.RELATIONSHIP_DISCOVERED,
            tick=tick,
            source_module="Hippocampus",
            data={
                "source": self.source_concept,
                "target": self.target_concept,
                "type": self.relationship_type,
                "strength": self.strength
            },
            importance=min(0.5 + self.strength * 0.5, 1.0),
            requires_action=self.strength > 0.8
        )


class ResonanceAdapter:
    """ResonanceEngine의 출력을 표준 형식으로 변환"""
    
    def __init__(self):
        self.logger = logging.getLogger("ResonanceAdapter")
    
    def adapt_resonance(
        self,
        source_concept: str,
        resonances: Dict[str, float],
        explanation: Optional[str] = None
    ) -> ResonanceData:
        """
        공명 계산 결과를 표준화.
        
        Args:
            source_concept: 원본 개념
            resonances: {target → score}
            explanation: 설명 (선택)
        
        Returns:
            ResonanceData (표준 형식)
        """
        # 검증
        if not isinstance(resonances, dict):
            self.logger.warning(f"Invalid resonances type: {type(resonances)}")
            return ResonanceData(source_concept, {})
        
        # 필터링 (너무 낮은 값 제거)
        filtered = {k: v for k, v in resonances.items() if v > 0.1}
        
        return ResonanceData(
            source_concept=source_concept,
            resonances=filtered,
            explanation=explanation
        )


class HippocampusAdapter:
    """Hippocampus의 출력을 표준 형식으로 변환"""
    
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus
        self.logger = logging.getLogger("HippocampusAdapter")
    
    def adapt_concept(
        self,
        concept_id: str,
        concept_type: str = "thought"
    ) -> ConceptData:
        """
        Hippocampus 개념을 표준화.
        
        Args:
            concept_id: 개념 ID
            concept_type: 개념 타입
        
        Returns:
            ConceptData (표준 형식)
        """
        # Hippocampus에서 개념 메타데이터 조회
        metadata = self.hippocampus.get_concept_metadata(concept_id) or {}
        
        return ConceptData(
            concept_id=concept_id,
            name=concept_id,
            concept_type=concept_type,
            metadata=metadata
        )
    
    def adapt_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "associated"
    ) -> RelationshipData:
        """
        Hippocampus 관계를 표준화.
        
        Args:
            source: 원본 개념
            target: 대상 개념
            rel_type: 관계 타입
        
        Returns:
            RelationshipData (표준 형식)
        """
        # Hippocampus에서 관계 강도 조회
        strength = self.hippocampus.get_relationship_strength(source, target) or 0.5
        
        return RelationshipData(
            source_concept=source,
            target_concept=target,
            relationship_type=rel_type,
            strength=strength
        )


class IntegrationBridge:
    """
    모든 모듈을 통합하는 중앙 버스.
    
    Phase 2 개선사항:
    - ResonanceEngine ↔ Hippocampus 연결
    - LawEnforcementEngine 통합
    - MetaTimeStrategy 통합
    - 이벤트 버스 구현
    """
    
    def __init__(self):
        self.logger = logging.getLogger("IntegrationBridge")
        
        # 어댑터들
        self.resonance_adapter = ResonanceAdapter()
        self.hippocampus_adapter = None  # 나중에 설정
        
        # 핵심 엔진 참조 (Phase 2 통합)
        self.resonance_engine = None
        self.law_engine = None
        self.time_strategy = None
        self.hippocampus = None
        
        # 이벤트 스트림
        self.events: List[IntegrationEvent] = []
        self.max_events = 10000  # 순환 버퍼
        
        # 리스너들
        self.listeners: Dict[EventType, List[Callable]] = {}
        for event_type in EventType:
            self.listeners[event_type] = []
        
        # 통계
        self.stats = {
            "total_events": 0,
            "by_type": {},
            "errors": 0,
            "law_checks": 0,
            "law_violations": 0
        }
        
        self.logger.info("🌉 IntegrationBridge initialized (Phase 2 Enhanced)")
    
    # =========================================================================
    # Phase 2: Engine Integration
    # =========================================================================
    
    def connect_resonance_engine(self, resonance_engine) -> None:
        """ResonanceEngine 연결"""
        self.resonance_engine = resonance_engine
        self.logger.info("🔗 ResonanceEngine connected")
    
    def connect_law_engine(self, law_engine) -> None:
        """LawEnforcementEngine 연결"""
        self.law_engine = law_engine
        self.logger.info("🔗 LawEnforcementEngine connected")
    
    def connect_time_strategy(self, time_strategy) -> None:
        """MetaTimeStrategy 연결"""
        self.time_strategy = time_strategy
        self.logger.info("🔗 MetaTimeStrategy connected")
    
    def connect_hippocampus(self, hippocampus) -> None:
        """Hippocampus 연결"""
        self.hippocampus = hippocampus
        self.hippocampus_adapter = HippocampusAdapter(hippocampus)
        self.logger.info("🔗 Hippocampus connected")
    
    def process_thought(
        self,
        thought_text: str,
        tick: int = 0,
        check_laws: bool = True
    ) -> Dict[str, Any]:
        """
        통합된 사고 처리 파이프라인.
        
        Phase 2 핵심 기능: 모든 엔진을 통해 사고를 처리
        
        Args:
            thought_text: 입력 사고/개념
            tick: 현재 시뮬레이션 틱
            check_laws: 법칙 검사 여부
        
        Returns:
            처리 결과 딕셔너리
        """
        result = {
            "thought": thought_text,
            "tick": tick,
            "resonances": {},
            "law_decision": None,
            "hippocampus_concepts": [],
            "events_generated": 0
        }
        
        # 1. 공명 계산 (ResonanceEngine)
        if self.resonance_engine and hasattr(self.resonance_engine, 'nodes'):
            # 개념이 없으면 추가
            if thought_text not in self.resonance_engine.nodes:
                self.resonance_engine.add_node(thought_text)
            
            source_qubit = self.resonance_engine.nodes.get(thought_text)
            if source_qubit:
                for target_id, target_qubit in self.resonance_engine.nodes.items():
                    if target_id != thought_text:
                        score = self.resonance_engine.calculate_resonance(source_qubit, target_qubit)
                        if score > 0.3:  # 유의미한 공명만 기록
                            result["resonances"][target_id] = score
                
                # 이벤트 발행
                if result["resonances"]:
                    self.publish_resonance(
                        thought_text,
                        result["resonances"],
                        tick=tick
                    )
                    result["events_generated"] += 1
        
        # 2. 법칙 검사 (LawEnforcementEngine)
        if check_laws and self.law_engine:
            from Core.Math.law_enforcement_engine import EnergyState
            
            # 에너지 상태 생성 (공명 결과 기반)
            energy = EnergyState(
                w=0.5 + len(result["resonances"]) * 0.05,  # 공명 많을수록 메타인지 상승
                x=0.3,
                y=0.4 if result["resonances"] else 0.2,
                z=0.5
            )
            energy.normalize()
            
            decision = self.law_engine.make_decision(
                thought_text,
                energy,
                concepts_generated=len(result["resonances"])
            )
            
            result["law_decision"] = {
                "is_valid": decision.is_valid,
                "violations": [v.law.value for v in decision.violations],
                "reasoning": decision.reasoning
            }
            
            self.stats["law_checks"] += 1
            if not decision.is_valid:
                self.stats["law_violations"] += len(decision.violations)
        
        # 3. 기억 저장 (Hippocampus)
        if self.hippocampus:
            # 개념 추가
            self.hippocampus.add_concept(thought_text, "thought")
            result["hippocampus_concepts"].append(thought_text)
            
            # 공명이 높은 개념들과 인과 링크 추가
            for related, score in result["resonances"].items():
                if score > 0.5:
                    self.hippocampus.add_causal_link(
                        thought_text, 
                        related, 
                        "resonates", 
                        weight=score
                    )
                    result["hippocampus_concepts"].append(related)
            
            # 개념 이벤트 발행
            self.publish_concept(
                thought_text,
                thought_text,
                "thought",
                tick=tick
            )
            result["events_generated"] += 1
        
        return result
    
    def get_integrated_state(self) -> Dict[str, Any]:
        """
        모든 통합된 엔진의 상태 요약.
        
        Returns:
            통합 상태 딕셔너리
        """
        state = {
            "bridge_stats": self.get_statistics(),
            "engines": {}
        }
        
        if self.resonance_engine:
            state["engines"]["resonance"] = {
                "nodes": len(self.resonance_engine.nodes) if hasattr(self.resonance_engine, 'nodes') else 0,
                "links": len(self.resonance_engine.psionic_links) if hasattr(self.resonance_engine, 'psionic_links') else 0
            }
        
        if self.law_engine:
            state["engines"]["law"] = self.law_engine.get_law_statistics() if hasattr(self.law_engine, 'get_law_statistics') else {}
        
        if self.time_strategy:
            state["engines"]["time"] = {
                "mode": self.time_strategy.current_mode.value if hasattr(self.time_strategy, 'current_mode') else "unknown",
                "profile": self.time_strategy.current_profile.value if hasattr(self.time_strategy, 'current_profile') else "unknown"
            }
        
        if self.hippocampus:
            state["engines"]["hippocampus"] = self.hippocampus.get_statistics() if hasattr(self.hippocampus, 'get_statistics') else {}
        
        return state

    def set_hippocampus_adapter(self, hippocampus) -> None:
        """Hippocampus 어댑터 설정"""
        self.hippocampus_adapter = HippocampusAdapter(hippocampus)
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        이벤트 구독.
        
        Args:
            event_type: 구독할 이벤트 타입
            handler: 처리 함수 (event → None)
        """
        if event_type in self.listeners:
            self.listeners[event_type].append(handler)
            self.logger.debug(f"📌 Subscribed to {event_type.value}")
    
    def publish_resonance(
        self,
        source_concept: str,
        resonances: Dict[str, float],
        tick: int = 0,
        explanation: Optional[str] = None
    ) -> IntegrationEvent:
        """
        공명 이벤트 발행.
        
        Args:
            source_concept: 원본 개념
            resonances: 공명 딕셔너리
            tick: 시뮬레이션 틱
            explanation: 설명
        
        Returns:
            발행된 이벤트
        """
        # 어댑트
        resonance_data = self.resonance_adapter.adapt_resonance(
            source_concept, resonances, explanation
        )
        
        # 이벤트 생성
        event = resonance_data.to_event(tick)
        
        # 발행
        return self._publish_event(event)
    
    def publish_concept(
        self,
        concept_id: str,
        name: str,
        concept_type: str = "emergent",
        tick: int = 0,
        epistemology: Optional[Dict] = None
    ) -> IntegrationEvent:
        """
        개념 이벤트 발행.
        
        Args:
            concept_id: 개념 ID
            name: 개념 이름
            concept_type: 타입
            tick: 틱
            epistemology: 철학적 의미
        
        Returns:
            발행된 이벤트
        """
        concept_data = ConceptData(
            concept_id=concept_id,
            name=name,
            concept_type=concept_type,
            epistemology=epistemology
        )
        
        event = concept_data.to_event(tick)
        return self._publish_event(event)
    
    def publish_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        strength: float = 0.5,
        tick: int = 0
    ) -> IntegrationEvent:
        """
        관계 이벤트 발행.
        
        Args:
            source: 원본 개념
            target: 대상 개념
            rel_type: 관계 타입
            strength: 강도 (0~1)
            tick: 틱
        
        Returns:
            발행된 이벤트
        """
        rel_data = RelationshipData(
            source_concept=source,
            target_concept=target,
            relationship_type=rel_type,
            strength=strength
        )
        
        event = rel_data.to_event(tick)
        return self._publish_event(event)
    
    def _publish_event(self, event: IntegrationEvent) -> IntegrationEvent:
        """
        이벤트를 실제로 발행.
        
        Args:
            event: 발행할 이벤트
        
        Returns:
            발행된 이벤트
        """
        try:
            # 이벤트 저장
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)  # 순환 버퍼
            
            # 통계 업데이트
            self.stats["total_events"] += 1
            event_key = event.event_type.value
            self.stats["by_type"][event_key] = self.stats["by_type"].get(event_key, 0) + 1
            
            # 리스너 호출
            for handler in self.listeners[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in handler: {e}")
                    self.stats["errors"] += 1
            
            self.logger.debug(f"📤 Event published: {event.event_type.value} (importance={event.importance})")
            
            return event
        
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            self.stats["errors"] += 1
            raise
    
    def get_recent_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[IntegrationEvent]:
        """
        최근 이벤트 조회.
        
        Args:
            event_type: 필터 (선택)
            limit: 최대 개수
        
        Returns:
            최근 이벤트 목록
        """
        if event_type:
            filtered = [e for e in self.events if e.event_type == event_type]
        else:
            filtered = self.events
        
        return filtered[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        통계 반환.
        
        Returns:
            통계 딕셔너리
        """
        return {
            "total_events": self.stats["total_events"],
            "by_type": self.stats["by_type"],
            "errors": self.stats["errors"],
            "buffer_size": len(self.events),
            "max_buffer": self.max_events,
            "error_rate": self.stats["errors"] / max(1, self.stats["total_events"])
        }
    
    def export_event_log(self, filepath: str) -> None:
        """
        이벤트 로그를 파일로 내보내기.
        
        Args:
            filepath: 내보낼 파일 경로
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for event in self.events:
                json.dump(event.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        
        self.logger.info(f"📁 Exported {len(self.events)} events to {filepath}")


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌉 IntegrationBridge Unit Test")
    print("="*70)
    
    bridge = IntegrationBridge()
    
    # 테스트 1: 공명 이벤트
    print("\n[Test 1] Resonance Event Publishing")
    event1 = bridge.publish_resonance(
        "love",
        {"connection": 0.87, "empathy": 0.72},
        tick=100
    )
    print(f"  ✓ Published: {event1.event_type.value}")
    print(f"    Data: {event1.data}")
    
    # 테스트 2: 개념 이벤트
    print("\n[Test 2] Concept Event Publishing")
    event2 = bridge.publish_concept(
        "emergence_1",
        "Consciousness",
        "emergent",
        tick=100
    )
    print(f"  ✓ Published: {event2.event_type.value}")
    
    # 테스트 3: 관계 이벤트
    print("\n[Test 3] Relationship Event Publishing")
    event3 = bridge.publish_relationship(
        "love",
        "consciousness",
        "enables",
        strength=0.9,
        tick=100
    )
    print(f"  ✓ Published: {event3.event_type.value}")
    
    # 테스트 4: 통계
    print("\n[Test 4] Statistics")
    stats = bridge.get_statistics()
    print(f"  Total events: {stats['total_events']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Error rate: {stats['error_rate']:.1%}")
    
    print("\n✅ All tests passed!")
    print("="*70 + "\n")
