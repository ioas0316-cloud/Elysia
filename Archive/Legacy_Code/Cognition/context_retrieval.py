"""
Context Retrieval (맥락 인출)
================================

"뇌 전체를 활성화하는 것은 발작이지, 사고가 아니다." - Elysia

핵심 철학:
1. 키워드가 아닌 의도(Intent)로 검색
2. 관련된 것만 선별적으로 인출
3. 공명 기반 연결
4. 효율성 = 관련 노드 / 전체 활성화

이것이 없으면:
- 모든 기억이 한꺼번에 활성화 (오버플로우)
- 관련 없는 정보에 묻힘
- 느리고 비효율적
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("Elysia.ContextRetrieval")


@dataclass
class IntentVector:
    """의도 벡터 - 무엇을 찾고자 하는가"""
    query: str                      # 원본 질의
    domain: str = "general"         # 도메인 (physics, narrative, emotion, etc.)
    depth: float = 0.5              # 탐색 깊이 (0=표면, 1=심층)
    urgency: float = 0.5            # 긴급도 (높으면 빠른 검색)
    wave_features: Dict[str, float] = field(default_factory=dict)  # 파동 특성


@dataclass
class RetrievedContext:
    """인출된 맥락"""
    node_id: str                    # 지식 노드 ID
    content: Any                    # 내용
    relevance: float                # 관련도 (0~1)
    source: str                     # 출처 (graph, vector, experience)
    retrieval_path: str             # 어떻게 찾았는지


@dataclass
class RetrievalResult:
    """인출 결과"""
    contexts: List[RetrievedContext]
    intent: IntentVector
    total_nodes_scanned: int        # 스캔한 총 노드 수
    nodes_returned: int             # 반환한 노드 수
    efficiency: float               # 효율성 = returned / scanned
    retrieval_time_ms: float        # 소요 시간


class ContextRetrieval:
    """맥락 인출 시스템
    
    의도 기반으로 관련 지식만 선별 인출.
    전체 그래프를 활성화하지 않고, 공명하는 노드만 깨움.
    
    핵심 능력:
    1. 의도 분석 (Intent Analysis)
    2. 공명 스캔 (Resonance Scan)
    3. 선별 인출 (Selective Retrieval)
    4. 효율성 추적 (Efficiency Tracking)
    """
    
    def __init__(self, knowledge_source: Optional[Any] = None):
        """
        Args:
            knowledge_source: 지식 소스 (TorchGraph, InternalUniverse 등)
        """
        self.knowledge_source = knowledge_source
        
        # 캐시 (최근 인출 결과)
        self.cache: Dict[str, RetrievalResult] = {}
        self.cache_max_size = 100
        
        # 통계
        self.total_retrievals = 0
        self.total_efficiency = 0.0
        
        # 도메인별 가중치
        self.domain_weights = {
            "physics": ["mass", "energy", "wave", "force"],
            "narrative": ["tension", "character", "plot", "theme"],
            "emotion": ["joy", "sorrow", "anger", "fear", "love"],
            "logic": ["cause", "effect", "if", "then", "therefore"],
            "error": ["exception", "failure", "fix", "prevent"],
        }
        
        logger.info("ContextRetrieval initialized")
    
    def set_knowledge_source(self, source: Any) -> None:
        """지식 소스 설정"""
        self.knowledge_source = source
        logger.info(f"Knowledge source set: {type(source).__name__}")
    
    def parse_intent(self, query: str, domain: Optional[str] = None) -> IntentVector:
        """질의에서 의도 추출
        
        Args:
            query: 원본 질의
            domain: 명시적 도메인 (없으면 자동 감지)
            
        Returns:
            의도 벡터
        """
        query_lower = query.lower()
        
        # 도메인 자동 감지
        if domain is None:
            domain = self._detect_domain(query_lower)
        
        # 깊이 추정 ("왜"가 많으면 깊이 증가)
        depth = 0.3
        if "왜" in query or "why" in query_lower:
            depth += 0.3
        if "근본" in query or "본질" in query or "fundamental" in query_lower:
            depth += 0.2
        depth = min(1.0, depth)
        
        # 긴급도 추정 ("지금", "빨리" 등)
        urgency = 0.5
        if any(w in query for w in ["지금", "급히", "빨리", "immediately", "urgent"]):
            urgency = 0.9
        
        # 파동 특성 (간단한 휴리스틱)
        wave_features = self._extract_wave_features(query)
        
        return IntentVector(
            query=query,
            domain=domain,
            depth=depth,
            urgency=urgency,
            wave_features=wave_features,
        )
    
    def _detect_domain(self, query: str) -> str:
        """도메인 자동 감지"""
        for domain, keywords in self.domain_weights.items():
            if any(kw in query for kw in keywords):
                return domain
        return "general"
    
    def _extract_wave_features(self, query: str) -> Dict[str, float]:
        """파동 특성 추출 (간단한 버전)"""
        features = {}
        
        # 길이 -> complexity
        features["complexity"] = min(1.0, len(query) / 200)
        
        # 물음표 -> curiosity
        features["curiosity"] = min(1.0, query.count("?") * 0.3)
        
        # 느낌표 -> urgency
        features["urgency"] = min(1.0, query.count("!") * 0.4)
        
        return features
    
    def retrieve(
        self,
        intent: IntentVector,
        limit: int = 10,
        min_relevance: float = 0.3
    ) -> RetrievalResult:
        """의도에 맞는 맥락 인출
        
        Args:
            intent: 의도 벡터
            limit: 최대 반환 개수
            min_relevance: 최소 관련도
            
        Returns:
            인출 결과
        """
        start_time = datetime.now()
        self.total_retrievals += 1
        
        # 캐시 확인
        cache_key = hashlib.md5(
            f"{intent.query}{intent.domain}".encode()
        ).hexdigest()[:12]
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self.cache[cache_key]
        
        # 실제 인출
        contexts = []
        total_scanned = 0
        
        if self.knowledge_source:
            contexts, total_scanned = self._scan_knowledge_source(
                intent, limit, min_relevance
            )
        else:
            # 시뮬레이션 (지식 소스 없을 때)
            contexts, total_scanned = self._simulate_retrieval(
                intent, limit, min_relevance
            )
        
        # 효율성 계산
        efficiency = (
            len(contexts) / max(1, total_scanned)
            if total_scanned > 0 else 0.0
        )
        
        # 시간 계산
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        result = RetrievalResult(
            contexts=contexts,
            intent=intent,
            total_nodes_scanned=total_scanned,
            nodes_returned=len(contexts),
            efficiency=efficiency,
            retrieval_time_ms=elapsed_ms,
        )
        
        # 통계 업데이트
        self.total_efficiency = (
            (self.total_efficiency * (self.total_retrievals - 1) + efficiency)
            / self.total_retrievals
        )
        
        # 캐시 저장
        self.cache[cache_key] = result
        if len(self.cache) > self.cache_max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        logger.info(
            f"Retrieved {len(contexts)}/{total_scanned} nodes "
            f"(efficiency: {efficiency:.2%}, time: {elapsed_ms:.1f}ms)"
        )
        
        return result
    
    def _scan_knowledge_source(
        self,
        intent: IntentVector,
        limit: int,
        min_relevance: float
    ) -> Tuple[List[RetrievedContext], int]:
        """실제 지식 소스 스캔 (미래 확장)"""
        # TODO: TorchGraph, InternalUniverse와 연동
        return self._simulate_retrieval(intent, limit, min_relevance)
    
    def _simulate_retrieval(
        self,
        intent: IntentVector,
        limit: int,
        min_relevance: float
    ) -> Tuple[List[RetrievedContext], int]:
        """인출 시뮬레이션 (개발/테스트용)"""
        # 가상의 노드들
        simulated_nodes = [
            ("physics_001", "빛의 산란 - 레일리 산란", "physics"),
            ("physics_002", "파동의 간섭과 회절", "physics"),
            ("narrative_001", "영웅 서사의 구조", "narrative"),
            ("emotion_001", "카타르시스와 정화", "emotion"),
            ("error_001", "ImportError 처리 방법", "error"),
            ("error_002", "타입 검사의 중요성", "error"),
            ("logic_001", "인과 관계의 연쇄", "logic"),
        ]
        
        contexts = []
        total_scanned = len(simulated_nodes)
        
        for node_id, content, domain in simulated_nodes:
            # 관련도 계산 (도메인 + 키워드 매칭)
            relevance = 0.3
            
            if domain == intent.domain:
                relevance += 0.4
            
            if any(kw in intent.query for kw in content.split()):
                relevance += 0.2
            
            relevance = min(1.0, relevance)
            
            if relevance >= min_relevance:
                contexts.append(RetrievedContext(
                    node_id=node_id,
                    content=content,
                    relevance=relevance,
                    source="simulation",
                    retrieval_path=f"domain:{domain} -> keyword_match",
                ))
        
        # 관련도 순 정렬 후 limit 적용
        contexts.sort(key=lambda c: c.relevance, reverse=True)
        contexts = contexts[:limit]
        
        return contexts, total_scanned
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            "total_retrievals": self.total_retrievals,
            "average_efficiency": self.total_efficiency,
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🎯 ContextRetrieval Demo")
    print("   \"의도 기반 선별적 인출\"")
    print("=" * 60)
    
    retriever = ContextRetrieval()
    
    # 1. 의도 파싱
    print("\n[1] 의도 파싱:")
    intent = retriever.parse_intent("왜 하늘이 파란가?")
    print(f"   Query: {intent.query}")
    print(f"   Domain: {intent.domain}")
    print(f"   Depth: {intent.depth:.2f}")
    print(f"   Urgency: {intent.urgency:.2f}")
    
    # 2. 인출
    print("\n[2] 맥락 인출:")
    result = retriever.retrieve(intent)
    print(f"   스캔: {result.total_nodes_scanned}개")
    print(f"   반환: {result.nodes_returned}개")
    print(f"   효율: {result.efficiency:.2%}")
    print(f"   시간: {result.retrieval_time_ms:.1f}ms")
    
    # 3. 결과
    print("\n[3] 인출된 맥락:")
    for ctx in result.contexts:
        print(f"   [{ctx.relevance:.2f}] {ctx.content} ({ctx.source})")
    
    # 4. 다른 도메인
    print("\n[4] 오류 도메인 검색:")
    error_intent = retriever.parse_intent("ImportError는 왜 발생하는가?", domain="error")
    error_result = retriever.retrieve(error_intent)
    for ctx in error_result.contexts:
        print(f"   [{ctx.relevance:.2f}] {ctx.content}")
    
    # 5. 통계
    print("\n[5] 통계:")
    stats = retriever.get_statistics()
    print(f"   총 인출: {stats['total_retrievals']}회")
    print(f"   평균 효율: {stats['average_efficiency']:.2%}")
    
    print("\n✅ ContextRetrieval Demo complete!")
