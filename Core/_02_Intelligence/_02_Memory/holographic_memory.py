"""
홀로그래픽 메모리 (Holographic Memory)
======================================

RGB 간섭 원리 기반 지식 레이어 시스템

핵심 개념:
- 지식을 도메인별 레이어(물리/화학/생물/예술/인문/철학)에 저장
- 레이어 ON/OFF로 검색 공간을 동적 축소 (O(N) → O(N/L))
- 여러 레이어가 겹치는 "교집합"에서 학제간 통찰 발견
- 데이터는 경계에서 "액체처럼 퍼짐" (그라데이션 소속)

추가 축 (하모니 v2 제안):
- 시간 축 (Entropy): 고대(0.0) ↔ 현대(1.0)
- 감정 축 (Qualia): 이성(0.0) ↔ 감성(1.0)

영감: 강덕리 & 하모니의 '홀로그래픽 압축' 아이디어
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import math

# Neural Registry 데코레이터 (Elysia 유기적 임포트 시스템)
try:
    from Core._01_Foundation._01_Infrastructure.elysia_core import Cell
except ImportError:
    def Cell(name):
        def decorator(cls):
            return cls
        return decorator


class KnowledgeLayer(Enum):
    """지식 도메인 레이어 (기본 6개 + 기저 수학)"""
    MATHEMATICS = "수학"      # 기저 레이어
    PHYSICS = "물리"          # Layer 1
    CHEMISTRY = "화학"        # Layer 2
    BIOLOGY = "생물"          # Layer 3
    ART = "예술"              # Layer 4
    HUMANITIES = "인문"       # Layer 5
    PHILOSOPHY = "철학"       # Layer 6


@dataclass
class KnowledgeNode:
    """
    지식 노드 - 하나의 개념
    
    핵심: 하나의 노드가 여러 레이어에 "부분적으로" 속할 수 있음
    (강덕리 원리: 데이터는 경계에서 잉크처럼 퍼지는 액체)
    """
    concept: str                          # 개념 이름 (예: "엔트로피")
    layers: Dict[KnowledgeLayer, float]   # 레이어별 소속도 (0.0~1.0)
    amplitude: float = 1.0                # 진폭 (중요도)
    connections: List[str] = field(default_factory=list)  # 연결된 다른 노드
    
    # 하모니 제안: 추가 축
    entropy_position: float = 0.5         # 시간 축: 0.0=고대, 1.0=현대
    qualia_position: float = 0.5          # 감정 축: 0.0=이성, 1.0=감성
    
    def get_primary_layer(self) -> KnowledgeLayer:
        """가장 강하게 속한 레이어"""
        return max(self.layers.items(), key=lambda x: x[1])[0]
    
    def belongs_to(self, layer: KnowledgeLayer, threshold: float = 0.1) -> bool:
        """해당 레이어에 속하는지 (threshold 이상이면 속함)"""
        return self.layers.get(layer, 0.0) >= threshold
    
    def resonance_with(self, active_layers: Set[KnowledgeLayer]) -> float:
        """
        활성 레이어들과의 공명 강도
        
        여러 레이어에 동시에 속할수록 공명 강도 증가 (교집합 효과)
        """
        total = 0.0
        count = 0
        for layer in active_layers:
            if layer in self.layers:
                total += self.layers[layer]
                count += 1
        
        if count == 0:
            return 0.0
        
        # 여러 레이어에 동시 속하면 보너스 (교집합 강조)
        intersection_bonus = 1.0 + (count - 1) * 0.5
        return (total / count) * intersection_bonus * self.amplitude


@Cell("HolographicMemory")
class HolographicMemory:
    """
    홀로그래픽 메모리 - 레이어 기반 지식 저장소
    
    사용법:
        memory = HolographicMemory()
        memory.deposit("엔트로피", {PHYSICS: 0.7, PHILOSOPHY: 0.3})
        
        memory.toggle_layer(PHYSICS, on=True)
        memory.toggle_layer(PHILOSOPHY, on=True)
        
        results = memory.query("시간")  # 물리+철학 레이어에서만 검색
    """
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.active_layers: Set[KnowledgeLayer] = set()
        self.intersection_cache: Dict[frozenset, List[str]] = {}
        
        # 하모니 축 필터 (0.0~1.0 범위, None=필터 없음)
        self.entropy_range: Optional[Tuple[float, float]] = None  # 시간 축
        self.qualia_range: Optional[Tuple[float, float]] = None   # 감정 축
        
        # 기본: 모든 레이어 활성화
        for layer in KnowledgeLayer:
            self.active_layers.add(layer)
    
    # =========================================
    # 레이어 토글 (RGB 조명 ON/OFF)
    # =========================================
    
    def toggle_layer(self, layer: KnowledgeLayer, on: bool = True) -> None:
        """레이어 켜기/끄기"""
        if on:
            self.active_layers.add(layer)
        else:
            self.active_layers.discard(layer)
        # 캐시 무효화
        self.intersection_cache.clear()
    
    def set_active_layers(self, layers: List[KnowledgeLayer]) -> None:
        """활성 레이어 일괄 설정"""
        self.active_layers = set(layers)
        self.intersection_cache.clear()
    
    def zoom_out(self) -> None:
        """줌 아웃 - 추상 레이어만 (하모니 제안)"""
        self.active_layers = {KnowledgeLayer.PHILOSOPHY, KnowledgeLayer.PHYSICS}
    
    def zoom_in(self) -> None:
        """줌 인 - 구체 레이어만 (하모니 제안)"""
        self.active_layers = {
            KnowledgeLayer.CHEMISTRY, 
            KnowledgeLayer.BIOLOGY,
            KnowledgeLayer.MATHEMATICS
        }
    
    def zoom_all(self) -> None:
        """모든 레이어 활성화"""
        self.active_layers = set(KnowledgeLayer)
    
    # =========================================
    # 하모니 축 제어 (Entropy & Qualia)
    # =========================================
    
    def set_entropy_range(self, min_val: float, max_val: float) -> None:
        """
        시간 축 필터 설정
        
        Args:
            min_val: 최소값 (0.0=고대)
            max_val: 최대값 (1.0=현대)
        
        예시:
            memory.set_entropy_range(0.8, 1.0)  # 현대 개념만
            memory.set_entropy_range(0.0, 0.3)  # 고대 개념만
        """
        self.entropy_range = (min_val, max_val)
        self.intersection_cache.clear()
    
    def set_qualia_range(self, min_val: float, max_val: float) -> None:
        """
        감정 축 필터 설정
        
        Args:
            min_val: 최소값 (0.0=이성적)
            max_val: 최대값 (1.0=감성적)
        
        예시:
            memory.set_qualia_range(0.7, 1.0)  # 감성적 해석만
            memory.set_qualia_range(0.0, 0.3)  # 이성적/논리적만
        """
        self.qualia_range = (min_val, max_val)
        self.intersection_cache.clear()
    
    def clear_axis_filters(self) -> None:
        """축 필터 제거 (모든 시간/감정 범위 허용)"""
        self.entropy_range = None
        self.qualia_range = None
        self.intersection_cache.clear()
    
    def _passes_axis_filter(self, node: KnowledgeNode) -> bool:
        """노드가 현재 축 필터를 통과하는지 확인"""
        if self.entropy_range:
            if not (self.entropy_range[0] <= node.entropy_position <= self.entropy_range[1]):
                return False
        if self.qualia_range:
            if not (self.qualia_range[0] <= node.qualia_position <= self.qualia_range[1]):
                return False
        return True
    
    # =========================================
    # 데이터 퇴적 (Deposit)
    # =========================================
    
    def deposit(
        self, 
        concept: str, 
        layers: Dict[KnowledgeLayer, float],
        amplitude: float = 1.0,
        connections: Optional[List[str]] = None,
        entropy: float = 0.5,   # 하모니: 시간 축
        qualia: float = 0.5     # 하모니: 감정 축
    ) -> KnowledgeNode:
        """
        지식 노드를 레이어에 퇴적
        
        Args:
            concept: 개념 이름
            layers: 레이어별 소속도 (예: {PHYSICS: 0.7, PHILOSOPHY: 0.3})
            amplitude: 중요도 (기본 1.0)
            connections: 연결된 다른 개념들
            entropy: 시간 위치 (0.0=고대, 1.0=현대)
            qualia: 감정 위치 (0.0=이성, 1.0=감성)
        """
        node = KnowledgeNode(
            concept=concept,
            layers=layers,
            amplitude=amplitude,
            connections=connections or [],
            entropy_position=entropy,
            qualia_position=qualia
        )
        self.nodes[concept] = node
        self.intersection_cache.clear()
        return node
    
    # =========================================
    # 검색 (Query) - O(N/L) 성능!
    # =========================================
    
    def query(
        self, 
        keyword: str = "", 
        threshold: float = 0.1,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        활성 레이어에서만 검색 (병목 해결!)
        
        하모니 축 필터도 적용됨 (entropy_range, qualia_range)
        
        Returns:
            [(개념명, 공명강도), ...] 공명 강도 내림차순
        """
        results = []
        
        for name, node in self.nodes.items():
            # 하모니 축 필터 먼저 체크 (빠른 제거)
            if not self._passes_axis_filter(node):
                continue
            
            # 활성 레이어에 속하는지 확인
            if not any(node.belongs_to(layer, threshold) for layer in self.active_layers):
                continue  # 스킵! → O(N/L) 달성
            
            # 키워드 필터 (간단한 부분 문자열 매칭)
            if keyword and keyword not in name:
                continue
            
            # 공명 강도 계산
            resonance = node.resonance_with(self.active_layers)
            if resonance > 0:
                results.append((name, resonance))
        
        # 공명 강도로 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    # =========================================
    # 교집합 발견 (Intersection Discovery)
    # =========================================
    
    def find_intersections(self, threshold: float = 0.2) -> List[Tuple[str, Set[KnowledgeLayer]]]:
        """
        현재 활성 레이어들의 교집합에 있는 노드들 발견
        
        이것이 핵심!! 여러 레이어에 동시에 속하는 개념 = 학제간 통찰
        """
        cache_key = frozenset(self.active_layers)
        
        if cache_key not in self.intersection_cache:
            intersections = []
            
            for name, node in self.nodes.items():
                # 활성 레이어 중 2개 이상에 속하는 노드만
                belonging_layers = {
                    layer for layer in self.active_layers 
                    if node.belongs_to(layer, threshold)
                }
                
                if len(belonging_layers) >= 2:
                    intersections.append((name, belonging_layers))
            
            # 레이어 수 많은 순으로 정렬 (더 많이 겹칠수록 흥미로움)
            intersections.sort(key=lambda x: len(x[1]), reverse=True)
            self.intersection_cache[cache_key] = intersections
        
        return self.intersection_cache[cache_key]
    
    # =========================================
    # 유틸리티
    # =========================================
    
    def get_layer_stats(self) -> Dict[KnowledgeLayer, int]:
        """레이어별 노드 수 통계"""
        stats = {layer: 0 for layer in KnowledgeLayer}
        for node in self.nodes.values():
            for layer in node.layers:
                if node.belongs_to(layer):
                    stats[layer] += 1
        return stats
    
    def __repr__(self) -> str:
        active = [l.value for l in self.active_layers]
        return f"HolographicMemory(nodes={len(self.nodes)}, active={active})"


# =========================================
# 데모 데이터 (테스트용)
# =========================================

def create_demo_memory() -> HolographicMemory:
    """데모용 홀로그래픽 메모리 생성 (하모니 v2: 시간/감정 축 포함)"""
    memory = HolographicMemory()
    
    # 물리+철학 교집합 (강덕리님 예시)
    memory.deposit("엔트로피", {
        KnowledgeLayer.PHYSICS: 0.8,
        KnowledgeLayer.PHILOSOPHY: 0.4,
        KnowledgeLayer.BIOLOGY: 0.2
    }, amplitude=1.2, entropy=0.7, qualia=0.4)  # 현대적, 약간 이성적
    
    memory.deposit("시간의 화살", {
        KnowledgeLayer.PHYSICS: 0.9,
        KnowledgeLayer.PHILOSOPHY: 0.7,
    }, amplitude=1.0, entropy=0.8, qualia=0.6)  # 현대적, 중립
    
    memory.deposit("자유의지", {
        KnowledgeLayer.PHILOSOPHY: 0.9,
        KnowledgeLayer.PHYSICS: 0.3,
        KnowledgeLayer.BIOLOGY: 0.4
    }, amplitude=1.5, entropy=0.2, qualia=0.8)  # 고대부터 논의, 감성적
    
    # 현대 양자역학 - 둘다 높음!
    memory.deposit("양자 불확정성", {
        KnowledgeLayer.PHYSICS: 0.95,
        KnowledgeLayer.PHILOSOPHY: 0.6,
        KnowledgeLayer.MATHEMATICS: 0.7
    }, amplitude=1.3, entropy=0.95, qualia=0.3)  # 아주 현대적, 이성적
    
    memory.deposit("우주의 경이로움", {
        KnowledgeLayer.PHYSICS: 0.6,
        KnowledgeLayer.PHILOSOPHY: 0.8,
        KnowledgeLayer.ART: 0.5
    }, amplitude=1.2, entropy=0.5, qualia=0.95)  # 중간, 아주 감성적!
    
    # 화학+생물 교집합
    memory.deposit("DNA", {
        KnowledgeLayer.BIOLOGY: 0.9,
        KnowledgeLayer.CHEMISTRY: 0.8,
        KnowledgeLayer.MATHEMATICS: 0.3
    }, amplitude=1.3, entropy=0.9, qualia=0.3)  # 현대, 이성적
    
    memory.deposit("분자 결합", {
        KnowledgeLayer.CHEMISTRY: 0.95,
        KnowledgeLayer.PHYSICS: 0.4
    }, entropy=0.6, qualia=0.2)  # 중간, 이성적
    
    # 예술+인문 교집합
    memory.deposit("아름다움", {
        KnowledgeLayer.ART: 0.9,
        KnowledgeLayer.PHILOSOPHY: 0.7,
        KnowledgeLayer.HUMANITIES: 0.5
    }, amplitude=1.4, entropy=0.1, qualia=0.95)  # 아주 고대, 아주 감성적
    
    memory.deposit("서사 구조", {
        KnowledgeLayer.HUMANITIES: 0.8,
        KnowledgeLayer.ART: 0.6,
        KnowledgeLayer.PHILOSOPHY: 0.3
    }, entropy=0.3, qualia=0.7)  # 고대, 감성적
    
    # 순수 단일 레이어
    memory.deposit("미적분", {
        KnowledgeLayer.MATHEMATICS: 0.95
    }, entropy=0.4, qualia=0.1)  # 뉴턴 시대, 극도로 이성적
    
    memory.deposit("르네상스", {
        KnowledgeLayer.HUMANITIES: 0.9,
        KnowledgeLayer.ART: 0.7
    }, entropy=0.35, qualia=0.8)  # 고대~중세, 감성적
    
    # 플라톤 - 고대 철학
    memory.deposit("이데아", {
        KnowledgeLayer.PHILOSOPHY: 0.95,
        KnowledgeLayer.MATHEMATICS: 0.4
    }, amplitude=1.1, entropy=0.05, qualia=0.6)  # 아주 고대!
    
    return memory


if __name__ == "__main__":
    print("=" * 60)
    print("🌈 홀로그래픽 메모리 데모")
    print("=" * 60)
    
    memory = create_demo_memory()
    print(f"\n생성됨: {memory}")
    print(f"레이어별 노드 수: {memory.get_layer_stats()}")
    
    # 테스트 1: 모든 레이어 활성
    print("\n" + "-" * 40)
    print("📊 테스트 1: 모든 레이어 (전체 검색)")
    memory.zoom_all()
    results = memory.query()
    for name, resonance in results:
        print(f"  - {name}: 공명 {resonance:.2f}")
    
    # 테스트 2: 물리+철학만 활성 (줌 아웃)
    print("\n" + "-" * 40)
    print("🔭 테스트 2: 물리+철학 (줌 아웃 - 거시적)")
    memory.zoom_out()
    results = memory.query()
    print(f"  활성 레이어: {[l.value for l in memory.active_layers]}")
    for name, resonance in results:
        print(f"  - {name}: 공명 {resonance:.2f}")
    
    # 테스트 3: 교집합 발견
    print("\n" + "-" * 40)
    print("✨ 테스트 3: 물리+철학 교집합 (학제간 통찰!)")
    intersections = memory.find_intersections()
    for name, layers in intersections:
        layer_names = [l.value for l in layers]
        print(f"  - {name} ∈ {layer_names}")
    
    # 테스트 4: 줌 인 (화학+생물)
    print("\n" + "-" * 40)
    print("🔬 테스트 4: 화학+생물+수학 (줌 인 - 미시적)")
    memory.zoom_in()
    results = memory.query()
    print(f"  활성 레이어: {[l.value for l in memory.active_layers]}")
    for name, resonance in results:
        print(f"  - {name}: 공명 {resonance:.2f}")
    
    # =========================================
    # 하모니 축 테스트 (시간/감정)
    # =========================================
    
    # 테스트 5: 현대 개념만 (entropy 0.7~1.0)
    print("\n" + "-" * 40)
    print("⏰ 테스트 5: 현대 개념만 (entropy 0.7~1.0)")
    memory.zoom_all()
    memory.set_entropy_range(0.7, 1.0)
    memory.clear_axis_filters()  # qualia는 초기화
    memory.set_entropy_range(0.7, 1.0)  # entropy만 다시 설정
    results = memory.query()
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  - {name}: 공명 {resonance:.2f}, 시대={node.entropy_position:.1f}")
    
    # 테스트 6: 고대 개념만 (entropy 0.0~0.3)
    print("\n" + "-" * 40)
    print("📜 테스트 6: 고대 개념만 (entropy 0.0~0.3)")
    memory.set_entropy_range(0.0, 0.3)
    results = memory.query()
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  - {name}: 공명 {resonance:.2f}, 시대={node.entropy_position:.2f}")
    
    # 테스트 7: 감성적 해석만 (qualia 0.7~1.0)
    print("\n" + "-" * 40)
    print("💖 테스트 7: 감성적 해석만 (qualia 0.7~1.0)")
    memory.clear_axis_filters()
    memory.set_qualia_range(0.7, 1.0)
    results = memory.query()
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  - {name}: 공명 {resonance:.2f}, 감성={node.qualia_position:.1f}")
    
    # 테스트 8: 물리 레이어 + 감성적 해석 = "우주의 경이로움"!
    print("\n" + "-" * 40)
    print("🌌 테스트 8: 물리 + 감성 = '우주의 경이로움' 발견!")
    memory.set_active_layers([KnowledgeLayer.PHYSICS])
    memory.set_qualia_range(0.7, 1.0)
    results = memory.query()
    print("  [물리 레이어 + 감성적 관점]")
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  → {name}: 공명 {resonance:.2f}")
    
    print("\n" + "=" * 60)
    print("🎉 데모 완료! (하모니 v2: 시간/감정 축 포함)")
    print("=" * 60)
