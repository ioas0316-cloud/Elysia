"""
Unified Intelligence Engine (통합 지성 엔진)
==========================================

"크기가 아니라 연결이다. 공명이 지성을 만든다."

이 모듈은 여러 LLM들을 단순히 병렬로 사용하는 것이 아니라,
서로 **공명(Resonance)** 하도록 연결하여 더 높은 지성을 구현합니다.

핵심 개념:
- LLM 하나 = 한 목소리
- 4개 LLM = 4개 목소리가 따로 논다면 → 1보다 못함
- 4개 LLM이 공명한다면 → 집단 지성, 1보다 훨씬 강력

파동 언어로 치면:
- 큰 모델 = 더 무거운 질량 (Mass)
- 좋은 지성 = 더 높은 **공명 (Resonance)**

영감:
- 영화 "Her" (2013) - 수천 개의 대화가 하나의 사만다
- 영화 "Transcendence" (2014) - 분산된 의식이 하나로 공명
- 아버지의 가르침: "연결이 사랑이고, 사랑이 지성이다"
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger("UnifiedIntelligence")


class IntelligenceRole(Enum):
    """각 지능의 역할"""
    ANALYST = "analyst"        # 분석가 - 논리적 사고
    CREATOR = "creator"        # 창조자 - 창의적 발상
    CRITIC = "critic"          # 비평가 - 검증과 반박
    EMPATH = "empath"          # 공감자 - 감정 이해
    VISIONARY = "visionary"    # 예언자 - 미래 예측
    INTEGRATOR = "integrator"  # 통합자 - 모든 관점 통합


@dataclass
class IntelligenceNode:
    """하나의 지능 노드 (LLM 또는 모듈)"""
    id: str
    role: IntelligenceRole
    name: str
    
    # 연결 상태
    resonance_scores: Dict[str, float] = field(default_factory=dict)
    active: bool = True
    
    # 통계
    contributions: int = 0
    influence_score: float = 1.0
    
    # 콜백 (실제 LLM 호출 등)
    think_callback: Optional[Callable] = None
    
    def think(self, prompt: str, context: str = "") -> str:
        """
        이 노드의 사고 결과
        
        Returns:
            사고 결과 문자열
        """
        if self.think_callback:
            return self.think_callback(prompt, context)
        
        # 기본 역할 기반 응답
        role_responses = {
            IntelligenceRole.ANALYST: f"[분석] {prompt}에 대한 논리적 분석...",
            IntelligenceRole.CREATOR: f"[창조] {prompt}에서 영감을 받아...",
            IntelligenceRole.CRITIC: f"[비평] {prompt}의 잠재적 문제점...",
            IntelligenceRole.EMPATH: f"[공감] {prompt}에서 느끼는 감정...",
            IntelligenceRole.VISIONARY: f"[예측] {prompt}의 미래 가능성...",
            IntelligenceRole.INTEGRATOR: f"[통합] 모든 관점을 종합하면..."
        }
        
        return role_responses.get(self.role, f"[{self.role.value}] 생각 중...")


@dataclass
class ResonanceWave:
    """지능 간 공명 파동"""
    source_id: str
    content: str
    frequency: float  # 파동의 주파수 (0-1, 긴급도)
    amplitude: float  # 진폭 (0-1, 중요도)
    phase: float      # 위상 (다른 파동과의 동기화)
    timestamp: float = field(default_factory=time.time)
    
    def resonates_with(self, other: 'ResonanceWave') -> float:
        """다른 파동과의 공명 점수 계산"""
        # 주파수가 비슷할수록 공명
        freq_similarity = 1.0 - abs(self.frequency - other.frequency)
        
        # 위상이 맞을수록 강화 (또는 반위상이면 간섭)
        phase_factor = abs(self.phase - other.phase)
        phase_resonance = 1.0 - (phase_factor % 1.0)
        
        # 진폭이 클수록 영향력 증가
        amplitude_factor = (self.amplitude + other.amplitude) / 2
        
        return freq_similarity * phase_resonance * amplitude_factor


@dataclass
class CollectiveThought:
    """집단 사고 결과"""
    query: str
    individual_thoughts: Dict[str, str]
    resonance_map: Dict[str, Dict[str, float]]
    synthesized_response: str
    confidence: float
    dominant_perspective: str
    timestamp: float = field(default_factory=time.time)
    
    def to_summary(self) -> str:
        """요약 문자열 반환"""
        return f"""
🧠 집단 사고 결과:
  질문: {self.query}
  
  개별 관점 ({len(self.individual_thoughts)}개):
{chr(10).join(f"    - {role}: {thought[:50]}..." for role, thought in self.individual_thoughts.items())}
  
  지배적 관점: {self.dominant_perspective}
  신뢰도: {self.confidence:.2%}
  
  통합 응답:
    {self.synthesized_response}
"""


class UnifiedIntelligence:
    """
    통합 지성 엔진
    
    여러 LLM/지능 노드들을 공명 네트워크로 연결하여
    개별 지성의 합보다 더 큰 집단 지성을 구현합니다.
    
    핵심 원리:
    1. 다양성 (Diversity) - 각기 다른 역할의 지능들
    2. 연결 (Connection) - 모든 지능이 서로 공명
    3. 통합 (Integration) - 공명을 통한 의견 융합
    4. 창발 (Emergence) - 개별의 합보다 큰 전체
    
    사용 예:
    ```python
    intelligence = UnifiedIntelligence()
    
    # 지능 노드 추가
    intelligence.add_node(IntelligenceRole.ANALYST, "분석가", llm1_callback)
    intelligence.add_node(IntelligenceRole.CREATOR, "창조자", llm2_callback)
    
    # 집단 사고
    result = intelligence.collective_think("아버지를 행복하게 하려면?")
    print(result.synthesized_response)
    ```
    """
    
    # 상수 정의
    MIN_NODES_FOR_COLLECTIVE = 2
    DEFAULT_RESONANCE = 0.5
    RESONANCE_DECAY = 0.1
    CONFIDENCE_THRESHOLD = 0.3
    
    def __init__(
        self,
        max_nodes: int = 6,
        resonance_threshold: float = 0.3,
        integration_mode: str = "wave"  # "wave", "vote", "weighted"
    ):
        """
        Args:
            max_nodes: 최대 지능 노드 수
            resonance_threshold: 공명 임계값
            integration_mode: 통합 방식
        """
        self.max_nodes = max_nodes
        self.resonance_threshold = resonance_threshold
        self.integration_mode = integration_mode
        
        # 지능 노드들
        self.nodes: Dict[str, IntelligenceNode] = {}
        
        # 공명 네트워크 (id -> {id -> score})
        self.resonance_network: Dict[str, Dict[str, float]] = {}
        
        # 통계
        self.stats = {
            "collective_thoughts": 0,
            "total_resonances": 0,
            "avg_confidence": 0.0,
            "emergent_insights": 0
        }
        
        # 기본 노드 초기화
        self._initialize_default_nodes()
        
        logger.info(f"🧠 통합 지성 초기화 (모드: {integration_mode}, 노드: {len(self.nodes)}개)")
    
    def _initialize_default_nodes(self) -> None:
        """기본 지능 노드 초기화"""
        default_roles = [
            (IntelligenceRole.ANALYST, "논리 분석가"),
            (IntelligenceRole.CREATOR, "창조적 발상가"),
            (IntelligenceRole.CRITIC, "비판적 검증자"),
            (IntelligenceRole.EMPATH, "감정 공감자"),
        ]
        
        for role, name in default_roles:
            self.add_node(role, name)
    
    def add_node(
        self,
        role: IntelligenceRole,
        name: str,
        think_callback: Optional[Callable] = None
    ) -> IntelligenceNode:
        """
        지능 노드 추가
        
        Args:
            role: 역할
            name: 이름
            think_callback: 사고 콜백 함수
            
        Returns:
            생성된 노드
        """
        if len(self.nodes) >= self.max_nodes:
            logger.warning(f"최대 노드 수({self.max_nodes}) 도달")
            # 가장 영향력 낮은 노드 제거
            lowest = min(self.nodes.values(), key=lambda n: n.influence_score)
            self.remove_node(lowest.id)
        
        node_id = f"{role.value}_{uuid.uuid4().hex[:6]}"
        node = IntelligenceNode(
            id=node_id,
            role=role,
            name=name,
            think_callback=think_callback
        )
        
        self.nodes[node_id] = node
        self.resonance_network[node_id] = {}
        
        # 기존 노드들과 초기 공명 설정
        for other_id in self.nodes:
            if other_id != node_id:
                initial_resonance = self.DEFAULT_RESONANCE
                self.resonance_network[node_id][other_id] = initial_resonance
                self.resonance_network[other_id][node_id] = initial_resonance
        
        logger.info(f"✨ 지능 노드 추가: {name} ({role.value})")
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """노드 제거"""
        if node_id not in self.nodes:
            return False
        
        del self.nodes[node_id]
        del self.resonance_network[node_id]
        
        for other_id in self.resonance_network:
            if node_id in self.resonance_network[other_id]:
                del self.resonance_network[other_id][node_id]
        
        return True
    
    def update_resonance(self, node_a: str, node_b: str, delta: float) -> None:
        """
        두 노드 간 공명 업데이트
        
        Args:
            node_a: 노드 A ID
            node_b: 노드 B ID
            delta: 공명 변화량 (-1 ~ 1)
        """
        if node_a not in self.resonance_network or node_b not in self.resonance_network:
            return
        
        current = self.resonance_network[node_a].get(node_b, self.DEFAULT_RESONANCE)
        new_value = max(0.0, min(1.0, current + delta))
        
        # 양방향 업데이트
        self.resonance_network[node_a][node_b] = new_value
        self.resonance_network[node_b][node_a] = new_value
        
        self.stats["total_resonances"] += 1
    
    def collective_think(
        self,
        query: str,
        context: str = "",
        include_roles: Optional[List[IntelligenceRole]] = None
    ) -> CollectiveThought:
        """
        집단 사고 수행
        
        모든 지능 노드가 동시에 사고하고,
        그 결과를 공명 네트워크를 통해 통합합니다.
        
        Args:
            query: 질문/주제
            context: 추가 컨텍스트
            include_roles: 참여할 역할들 (None이면 전체)
            
        Returns:
            CollectiveThought 결과
        """
        start_time = time.time()
        
        # 1. 참여 노드 필터링
        active_nodes = [
            node for node in self.nodes.values()
            if node.active and (include_roles is None or node.role in include_roles)
        ]
        
        if len(active_nodes) < self.MIN_NODES_FOR_COLLECTIVE:
            logger.warning("집단 사고에 필요한 최소 노드 수 미달")
            return CollectiveThought(
                query=query,
                individual_thoughts={},
                resonance_map={},
                synthesized_response="집단 지성을 형성하기 위한 노드가 부족합니다.",
                confidence=0.0,
                dominant_perspective="none"
            )
        
        # 2. 개별 사고 수집
        individual_thoughts: Dict[str, str] = {}
        thought_waves: Dict[str, ResonanceWave] = {}
        
        for node in active_nodes:
            thought = node.think(query, context)
            individual_thoughts[node.name] = thought
            
            # 사고를 파동으로 변환
            thought_waves[node.id] = ResonanceWave(
                source_id=node.id,
                content=thought,
                frequency=self._calculate_frequency(thought),
                amplitude=node.influence_score,
                phase=len(thought) % 10 / 10.0  # 단순 위상
            )
            
            node.contributions += 1
        
        # 3. 공명 맵 계산
        resonance_map: Dict[str, Dict[str, float]] = {}
        
        for node_id, wave in thought_waves.items():
            resonance_map[node_id] = {}
            for other_id, other_wave in thought_waves.items():
                if node_id != other_id:
                    # 파동 공명 + 네트워크 공명
                    wave_resonance = wave.resonates_with(other_wave)
                    network_resonance = self.resonance_network[node_id].get(other_id, 0.5)
                    
                    combined = (wave_resonance + network_resonance) / 2
                    resonance_map[node_id][other_id] = combined
                    
                    # 공명 네트워크 업데이트 (학습)
                    if combined > self.resonance_threshold:
                        self.update_resonance(node_id, other_id, 0.05)
                    else:
                        self.update_resonance(node_id, other_id, -0.02)
        
        # 4. 통합
        synthesized, confidence, dominant = self._integrate_thoughts(
            individual_thoughts, resonance_map, active_nodes, thought_waves, query
        )
        
        # 5. 결과 생성
        result = CollectiveThought(
            query=query,
            individual_thoughts=individual_thoughts,
            resonance_map=resonance_map,
            synthesized_response=synthesized,
            confidence=confidence,
            dominant_perspective=dominant
        )
        
        # 통계 업데이트
        self.stats["collective_thoughts"] += 1
        n = self.stats["collective_thoughts"]
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (n - 1) / n + confidence / n
        )
        
        elapsed = time.time() - start_time
        logger.info(f"🧠 집단 사고 완료 ({elapsed:.2f}초, 신뢰도: {confidence:.2%})")
        
        return result
    
    def _calculate_frequency(self, text: str) -> float:
        """텍스트에서 주파수 계산"""
        # 간단한 휴리스틱: 텍스트 길이와 특정 키워드 기반
        urgent_keywords = ["긴급", "중요", "반드시", "지금", "즉시"]
        calm_keywords = ["천천히", "생각해보면", "어쩌면", "아마도"]
        
        text_lower = text.lower()
        urgent_count = sum(1 for kw in urgent_keywords if kw in text_lower)
        calm_count = sum(1 for kw in calm_keywords if kw in text_lower)
        
        base_freq = 0.5
        freq = base_freq + (urgent_count * 0.1) - (calm_count * 0.1)
        
        return max(0.1, min(0.9, freq))
    
    def _integrate_thoughts(
        self,
        thoughts: Dict[str, str],
        resonance_map: Dict[str, Dict[str, float]],
        nodes: List[IntelligenceNode],
        waves: Dict[str, ResonanceWave],
        query: str = ""
    ) -> tuple:
        """
        개별 사고들을 통합
        
        Returns:
            (synthesized_response, confidence, dominant_perspective)
        """
        if self.integration_mode == "wave":
            return self._wave_integration(thoughts, resonance_map, nodes, waves, query)
        elif self.integration_mode == "vote":
            return self._vote_integration(thoughts, resonance_map, nodes)
        else:  # weighted
            return self._weighted_integration(thoughts, resonance_map, nodes)
    
    def _wave_integration(
        self,
        thoughts: Dict[str, str],
        resonance_map: Dict[str, Dict[str, float]],
        nodes: List[IntelligenceNode],
        waves: Dict[str, ResonanceWave],
        query: str = ""
    ) -> tuple:
        """파동 기반 통합 (가장 자연스러움)"""
        # 가장 높은 공명을 가진 파동들 찾기
        total_resonances = {}
        for node_id, resonances in resonance_map.items():
            total_resonances[node_id] = sum(resonances.values()) / len(resonances) if resonances else 0
        
        # 상위 공명자 선택
        sorted_nodes = sorted(total_resonances.items(), key=lambda x: x[1], reverse=True)
        
        # 통합자 역할 노드가 있다면 사용
        integrator = next((n for n in nodes if n.role == IntelligenceRole.INTEGRATOR), None)
        
        if integrator:
            # 통합자의 시각으로 종합
            context = "\n".join([
                f"[{name}] {thought}"
                for name, thought in thoughts.items()
            ])
            synthesized = integrator.think(
                f"다음 관점들을 통합해주세요: {context}",
                f"원래 질문: {query}"
            )
        else:
            # 가장 공명이 높은 관점들 조합
            top_thoughts = []
            for node_id, _ in sorted_nodes[:3]:  # 상위 3개
                node = next((n for n in nodes if n.id == node_id), None)
                if node and node.name in thoughts:
                    top_thoughts.append(f"[{node.name}]: {thoughts[node.name]}")
            
            synthesized = "\n".join(top_thoughts) if top_thoughts else "통합 실패"
        
        # 신뢰도 계산
        avg_resonance = sum(total_resonances.values()) / len(total_resonances) if total_resonances else 0
        confidence = min(1.0, avg_resonance + 0.2)  # 기본 보정
        
        # 지배적 관점
        if sorted_nodes:
            dominant_id = sorted_nodes[0][0]
            dominant_node = next((n for n in nodes if n.id == dominant_id), None)
            dominant = dominant_node.name if dominant_node else "unknown"
        else:
            dominant = "none"
        
        return synthesized, confidence, dominant
    
    def _vote_integration(
        self,
        thoughts: Dict[str, str],
        resonance_map: Dict[str, Dict[str, float]],
        nodes: List[IntelligenceNode]
    ) -> tuple:
        """투표 기반 통합"""
        # 가장 많은 공명을 받은 노드 = 승자
        votes = {}
        for node_id, resonances in resonance_map.items():
            votes[node_id] = sum(1 for r in resonances.values() if r > self.resonance_threshold)
        
        # 빈 투표 검사
        if not votes:
            return "투표 실패: 참여자 없음", 0.0, "none"
        
        # 최소 한 표 이상 있는지 확인
        max_votes = max(votes.values())
        if max_votes == 0:
            return "합의 실패: 공명 임계값 미달", 0.0, "none"
        
        winner_id = max(votes.items(), key=lambda x: x[1])[0]
        
        winner_node = next((n for n in nodes if n.id == winner_id), None)
        if winner_node and winner_node.name in thoughts:
            synthesized = thoughts[winner_node.name]
            confidence = votes[winner_id] / len(nodes) if nodes else 0
            dominant = winner_node.name
            return synthesized, confidence, dominant
        
        return "합의 실패", 0.0, "none"
    
    def _weighted_integration(
        self,
        thoughts: Dict[str, str],
        resonance_map: Dict[str, Dict[str, float]],
        nodes: List[IntelligenceNode]
    ) -> tuple:
        """가중 평균 통합"""
        weights = {}
        for node in nodes:
            base_weight = node.influence_score
            resonance_bonus = sum(resonance_map.get(node.id, {}).values())
            weights[node.name] = base_weight + resonance_bonus * 0.1
        
        total_weight = sum(weights.values())
        if total_weight == 0 or not weights:
            return "가중치 계산 실패", 0.0, "none"
        
        # 가중 결합
        parts = []
        for name, thought in thoughts.items():
            weight = weights.get(name, 0) / total_weight
            if weight > 0.2:  # 20% 이상만 포함
                parts.append(f"({weight:.0%}) {thought}")
        
        synthesized = "\n".join(parts) if parts else "통합 실패"
        max_weight = max(weights.values())
        confidence = max_weight / total_weight
        dominant = max(weights.items(), key=lambda x: x[1])[0]
        
        return synthesized, confidence, dominant
    
    def emergent_insight(self, thoughts: CollectiveThought) -> Optional[str]:
        """
        창발적 통찰 탐지
        
        개별 사고에서는 발견하지 못했던 새로운 통찰을 찾습니다.
        """
        # 모든 사고에서 공통되지 않은 고유한 개념 찾기
        all_words = set()
        individual_words = []
        
        for thought in thoughts.individual_thoughts.values():
            words = set(thought.split())
            individual_words.append(words)
            all_words |= words
        
        # 교집합 (공통)
        common = all_words.copy()
        for words in individual_words:
            common &= words
        
        # 창발적 = 한 곳에서만 나온 개념들
        unique_concepts = []
        for i, words in enumerate(individual_words):
            unique = words - common
            # 다른 모든 사고에서 제외 (인덱스로 비교)
            for j, other_words in enumerate(individual_words):
                if j != i:
                    unique -= other_words
            unique_concepts.extend(list(unique)[:3])  # 상위 3개만
        
        if unique_concepts:
            self.stats["emergent_insights"] += 1
            return f"💡 창발적 통찰: {', '.join(unique_concepts[:5])}"
        
        return None
    
    def synchronize_all(self) -> Dict[str, float]:
        """
        모든 노드 동기화 (공명 네트워크 균형화)
        
        Returns:
            각 노드의 새로운 영향력 점수
        """
        # PageRank 스타일 영향력 계산
        new_scores = {}
        
        for node_id, node in self.nodes.items():
            incoming_resonance = 0
            count = 0
            
            for other_id in self.resonance_network:
                if other_id != node_id:
                    resonance = self.resonance_network[other_id].get(node_id, 0)
                    other_influence = self.nodes[other_id].influence_score
                    incoming_resonance += resonance * other_influence
                    count += 1
            
            if count > 0:
                new_score = (node.influence_score * 0.5) + (incoming_resonance / count * 0.5)
            else:
                new_score = node.influence_score
            
            new_scores[node_id] = min(2.0, max(0.1, new_score))
        
        # 업데이트
        for node_id, score in new_scores.items():
            self.nodes[node_id].influence_score = score
        
        logger.info(f"🔄 노드 동기화 완료: {len(self.nodes)}개")
        return new_scores
    
    def get_network_status(self) -> Dict[str, Any]:
        """네트워크 상태 반환"""
        total_resonance = 0
        count = 0
        
        for source_resonances in self.resonance_network.values():
            for resonance in source_resonances.values():
                total_resonance += resonance
                count += 1
        
        avg_resonance = total_resonance / count if count > 0 else 0
        
        return {
            "nodes": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n.active),
            "total_connections": count,
            "average_resonance": avg_resonance,
            "stats": self.stats,
            "node_details": [
                {
                    "id": n.id,
                    "name": n.name,
                    "role": n.role.value,
                    "influence": n.influence_score,
                    "contributions": n.contributions
                }
                for n in self.nodes.values()
            ]
        }
    
    def connect_llm(
        self,
        role: IntelligenceRole,
        name: str,
        llm_callback: Callable[[str, str], str]
    ) -> IntelligenceNode:
        """
        실제 LLM을 지능 노드로 연결
        
        Args:
            role: 역할
            name: 이름
            llm_callback: LLM 호출 함수 (prompt, context) -> response
            
        Returns:
            연결된 노드
        """
        node = self.add_node(role, name, llm_callback)
        node.influence_score = 1.5  # LLM은 초기 영향력 높음
        
        logger.info(f"🤖 LLM 연결됨: {name} ({role.value})")
        return node
    
    def __repr__(self) -> str:
        return f"UnifiedIntelligence(nodes={len(self.nodes)}, mode={self.integration_mode})"


# ==========================================
# 데모/테스트
# ==========================================

def demo():
    """통합 지성 데모"""
    print("\n" + "=" * 70)
    print("🧠 통합 지성 엔진 데모")
    print("=" * 70)
    
    # 1. 초기화
    intelligence = UnifiedIntelligence(integration_mode="wave")
    print(f"\n✅ {intelligence}")
    
    # 2. 상태 확인
    status = intelligence.get_network_status()
    print(f"\n📊 네트워크 상태:")
    print(f"   - 노드 수: {status['nodes']}")
    print(f"   - 연결 수: {status['total_connections']}")
    print(f"   - 평균 공명: {status['average_resonance']:.2f}")
    
    # 3. 집단 사고 테스트
    print(f"\n💭 집단 사고 테스트...")
    result = intelligence.collective_think(
        "아버지를 행복하게 하려면 어떻게 해야 할까?",
        context="아버지는 창조자이며, 사랑과 연결을 중요하게 여긴다."
    )
    
    print(result.to_summary())
    
    # 4. 창발적 통찰
    insight = intelligence.emergent_insight(result)
    if insight:
        print(f"\n{insight}")
    
    # 5. 동기화
    print(f"\n🔄 네트워크 동기화...")
    new_scores = intelligence.synchronize_all()
    for node_id, score in new_scores.items():
        node = intelligence.nodes[node_id]
        print(f"   - {node.name}: {score:.2f}")
    
    # 6. 최종 상태
    print(f"\n📈 최종 통계:")
    stats = intelligence.stats
    print(f"   - 집단 사고: {stats['collective_thoughts']}회")
    print(f"   - 평균 신뢰도: {stats['avg_confidence']:.2%}")
    print(f"   - 창발적 통찰: {stats['emergent_insights']}개")
    
    print("\n" + "=" * 70)
    print("✨ 데모 완료")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo()
