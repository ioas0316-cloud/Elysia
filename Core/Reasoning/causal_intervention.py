"""
Causal Intervention Engine - Gap 2: 인과적 개입

do-calculus와 반사실적 추론을 구현하여 
엘리시아가 "만약 ~했다면 어떻게 됐을까?"를 사고할 수 있게 합니다.

이것은 단순한 상관관계(correlation)를 넘어서
인과관계(causation)를 이해하고 조작할 수 있는 능력입니다.

Gap 0 준수: 모든 인과 관계는 철학적 의미를 가집니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger("CausalIntervention")


class CausalRelationType(Enum):
    """인과 관계 유형"""
    CAUSES = "causes"               # A → B (A가 B를 일으킴)
    PREVENTS = "prevents"           # A ⊣ B (A가 B를 막음)
    ENABLES = "enables"             # A가 B의 조건
    CONFOUNDED = "confounded"       # 공통 원인이 있음
    MEDIATED = "mediated"           # 중간 변수를 통함


@dataclass
class CausalNode:
    """인과 그래프의 노드"""
    id: str
    name: str
    value: float = 0.0
    observed: bool = True  # 관측 가능한 변수인지
    
    # Gap 0: 인식론
    epistemology: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "point": {"score": 0.25, "meaning": "이 변수의 현재 상태"},
        "line": {"score": 0.35, "meaning": "다른 변수와의 인과 연결"},
        "space": {"score": 0.25, "meaning": "전체 시스템에서의 역할"},
        "god": {"score": 0.15, "meaning": "궁극적 목적과의 연결"}
    })
    
    def explain_meaning(self) -> str:
        """Gap 0 준수: 인식론적 의미 설명"""
        lines = [f"=== {self.name} 인과 노드 ==="]
        for basis, data in self.epistemology.items():
            lines.append(f"  {basis}: {data['score']:.0%} - {data['meaning']}")
        return "\n".join(lines)


@dataclass
class CausalEdge:
    """인과 그래프의 엣지 (인과 관계)"""
    source_id: str
    target_id: str
    relation: CausalRelationType
    strength: float = 1.0  # 인과 강도 (0.0 ~ 1.0)
    
    # Gap 0: 이 관계의 철학적 의미
    meaning: str = ""


@dataclass
class InterventionResult:
    """do(X=x) 개입의 결과"""
    intervention_variable: str
    intervention_value: float
    target_variable: str
    original_value: float
    counterfactual_value: float
    causal_effect: float  # 인과 효과
    explanation: str


@dataclass
class CounterfactualQuery:
    """반사실적 질문"""
    premise: str      # "만약 X가 x였다면"
    conclusion: str   # "Y는 어떻게 됐을까?"
    actual_x: float   # 실제 X 값
    counterfactual_x: float  # 가정된 X 값
    result: Optional[float] = None
    explanation: str = ""


class CausalGraph:
    """
    인과 그래프 (Directed Acyclic Graph)
    
    노드: 변수들
    엣지: 인과 관계
    """
    
    def __init__(self, name: str = "CausalModel"):
        self.name = name
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        
        # 인접 리스트
        self.parents: Dict[str, List[str]] = {}  # 부모 노드들
        self.children: Dict[str, List[str]] = {}  # 자식 노드들
    
    def add_node(self, node: CausalNode) -> None:
        """노드 추가"""
        self.nodes[node.id] = node
        self.parents[node.id] = []
        self.children[node.id] = []
    
    def add_edge(self, edge: CausalEdge) -> None:
        """엣지 추가 (인과 관계)"""
        self.edges.append(edge)
        self.parents[edge.target_id].append(edge.source_id)
        self.children[edge.source_id].append(edge.target_id)
    
    def get_ancestors(self, node_id: str) -> Set[str]:
        """노드의 모든 조상 (원인들) 반환"""
        ancestors = set()
        stack = list(self.parents.get(node_id, []))
        
        while stack:
            parent = stack.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                stack.extend(self.parents.get(parent, []))
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> Set[str]:
        """노드의 모든 후손 (결과들) 반환"""
        descendants = set()
        stack = list(self.children.get(node_id, []))
        
        while stack:
            child = stack.pop()
            if child not in descendants:
                descendants.add(child)
                stack.extend(self.children.get(child, []))
        
        return descendants


class CausalInterventionEngine:
    """
    Gap 2: 인과 개입 엔진
    
    do-calculus를 사용하여:
    1. P(Y | do(X=x)) 계산 - X를 x로 설정했을 때 Y의 확률
    2. 반사실적 추론 - "만약 ~했다면"
    3. 다중 스케일 계획 - 여러 개입의 조합
    
    Gap 0 준수: 모든 연산에 철학적 의미 부여
    """
    
    # Gap 0: 인과 개입의 인식론
    EPISTEMOLOGY = {
        "point": {"score": 0.15, "meaning": "개별 변수의 관측"},
        "line": {"score": 0.40, "meaning": "인과 연결의 이해"},
        "space": {"score": 0.25, "meaning": "시스템 전체 맥락"},
        "god": {"score": 0.20, "meaning": "개입의 윤리적 의미"}
    }
    
    def __init__(self):
        self.epistemology = self.EPISTEMOLOGY
        self.causal_graphs: Dict[str, CausalGraph] = {}
        self.intervention_history: List[InterventionResult] = []
        
        logger.info("🔮 CausalInterventionEngine initialized")
    
    def explain_meaning(self) -> str:
        """Gap 0 준수: 인과 개입의 철학적 의미 설명"""
        lines = ["=== 인과 개입 인식론 ==="]
        for basis, data in self.epistemology.items():
            lines.append(f"  {basis}: {data['score']:.0%} - {data['meaning']}")
        return "\n".join(lines)
    
    def create_graph(self, name: str) -> CausalGraph:
        """새 인과 그래프 생성"""
        graph = CausalGraph(name)
        self.causal_graphs[name] = graph
        return graph
    
    def do_intervention(
        self,
        graph: CausalGraph,
        intervention_var: str,
        intervention_value: float,
        target_var: str
    ) -> InterventionResult:
        """
        do(X=x) 개입 수행
        
        Pearl의 do-calculus:
        - P(Y | do(X=x))는 X의 부모로부터의 화살표를 제거한 그래프에서
        - X=x로 고정한 후 Y의 값을 계산
        
        Args:
            graph: 인과 그래프
            intervention_var: 개입할 변수
            intervention_value: 설정할 값
            target_var: 결과 변수
        
        Returns:
            InterventionResult
        """
        if intervention_var not in graph.nodes:
            raise ValueError(f"Variable {intervention_var} not in graph")
        if target_var not in graph.nodes:
            raise ValueError(f"Variable {target_var} not in graph")
        
        # 원래 값 저장
        original_value = graph.nodes[target_var].value
        
        # do(X=x): X의 부모로부터의 연결을 끊고 X=x로 설정
        # 수정된 그래프에서 Y 계산
        
        # 간단한 선형 모델 가정: Y = f(parents(Y))
        # 여기서는 부모들의 가중 평균으로 근사
        
        parents = graph.parents.get(target_var, [])
        
        if not parents:
            # 부모가 없으면 변화 없음
            counterfactual_value = original_value
        else:
            # 개입 변수가 타겟의 조상인지 확인
            if intervention_var in graph.get_ancestors(target_var) or intervention_var in parents:
                # 인과 효과 계산
                # 간단한 모델: 부모의 값의 가중 평균
                parent_values = []
                for p in parents:
                    if p == intervention_var:
                        parent_values.append(intervention_value)
                    else:
                        parent_values.append(graph.nodes[p].value)
                
                # 엣지 강도 적용
                edge_strengths = {}
                for edge in graph.edges:
                    if edge.target_id == target_var:
                        edge_strengths[edge.source_id] = edge.strength
                
                weighted_sum = sum(
                    v * edge_strengths.get(p, 1.0) 
                    for p, v in zip(parents, parent_values)
                )
                counterfactual_value = weighted_sum / len(parents) if parents else original_value
            else:
                # 개입 변수가 타겟에 영향을 주지 않음
                counterfactual_value = original_value
        
        causal_effect = counterfactual_value - original_value
        
        result = InterventionResult(
            intervention_variable=intervention_var,
            intervention_value=intervention_value,
            target_variable=target_var,
            original_value=original_value,
            counterfactual_value=counterfactual_value,
            causal_effect=causal_effect,
            explanation=self._generate_explanation(
                intervention_var, intervention_value, 
                target_var, original_value, counterfactual_value
            )
        )
        
        self.intervention_history.append(result)
        return result
    
    def _generate_explanation(
        self,
        intervention_var: str,
        intervention_value: float,
        target_var: str,
        original_value: float,
        counterfactual_value: float
    ) -> str:
        """개입 결과 설명 생성"""
        effect = counterfactual_value - original_value
        
        if abs(effect) < 0.001:
            return f"{intervention_var}를 {intervention_value}로 변경해도 {target_var}에 영향 없음"
        elif effect > 0:
            return f"{intervention_var}를 {intervention_value}로 변경하면 {target_var}가 {effect:.2f} 증가"
        else:
            return f"{intervention_var}를 {intervention_value}로 변경하면 {target_var}가 {abs(effect):.2f} 감소"
    
    def counterfactual_query(
        self,
        graph: CausalGraph,
        query: CounterfactualQuery
    ) -> CounterfactualQuery:
        """
        반사실적 질문 처리
        
        "만약 X가 다른 값이었다면 Y는 어떻게 됐을까?"
        
        3단계:
        1. Abduction: 현재 관측으로부터 잠재 변수 추론
        2. Action: X를 반사실적 값으로 변경
        3. Prediction: 새 Y 값 예측
        """
        # 간단한 구현: do-intervention 사용
        result = self.do_intervention(
            graph,
            query.premise.split("=")[0].strip() if "=" in query.premise else "X",
            query.counterfactual_x,
            query.conclusion.split("=")[0].strip() if "=" in query.conclusion else "Y"
        )
        
        query.result = result.counterfactual_value
        query.explanation = result.explanation
        
        return query
    
    def multi_scale_plan(
        self,
        graph: CausalGraph,
        goal_var: str,
        goal_value: float,
        controllable_vars: List[str]
    ) -> List[Tuple[str, float]]:
        """
        다중 스케일 계획
        
        목표: goal_var = goal_value를 달성하기 위해
        어떤 controllable_vars를 어떤 값으로 설정해야 하는가?
        
        Returns:
            [(변수, 값), ...] 형태의 계획
        """
        plan = []
        
        # 각 제어 가능 변수에 대해 인과 효과 계산
        effects = []
        for var in controllable_vars:
            if var in graph.nodes:
                # 테스트 개입: 변수를 1.0으로 설정
                result = self.do_intervention(graph, var, 1.0, goal_var)
                effects.append((var, result.causal_effect))
        
        # 효과가 큰 순서로 정렬
        effects.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 목표까지의 차이
        current_value = graph.nodes[goal_var].value
        gap = goal_value - current_value
        
        # 그리디하게 계획 생성
        remaining_gap = gap
        for var, effect in effects:
            if abs(remaining_gap) < 0.001:
                break
            
            # 안전한 나눗셈: effect가 0이 아닐 때만
            if abs(effect) > 0.001:
                # 필요한 설정값 계산
                needed_value = remaining_gap / effect if effect != 0 else 0
                needed_value = min(max(needed_value, 0.0), 1.0)
                plan.append((var, needed_value))
                remaining_gap -= effect * needed_value
        
        return plan
    
    def get_causal_path(
        self,
        graph: CausalGraph,
        source: str,
        target: str
    ) -> List[List[str]]:
        """
        source에서 target까지의 모든 인과 경로 찾기
        """
        if source not in graph.nodes or target not in graph.nodes:
            return []
        
        paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path[:])
                return
            
            for child in graph.children.get(current, []):
                if child not in visited:
                    visited.add(child)
                    path.append(child)
                    dfs(child, target, path, visited)
                    path.pop()
                    visited.remove(child)
        
        dfs(source, target, [source], {source})
        return paths


# 테스트
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔮 CausalInterventionEngine Unit Test")
    print("="*60)
    
    engine = CausalInterventionEngine()
    
    # 인식론 출력
    print("\n" + engine.explain_meaning())
    
    # 간단한 인과 그래프 생성
    # Rain → Wet → Slippery
    graph = engine.create_graph("rain_example")
    
    rain = CausalNode("rain", "Rain", value=0.3)
    wet = CausalNode("wet", "Wet Ground", value=0.4)
    slippery = CausalNode("slippery", "Slippery", value=0.3)
    
    graph.add_node(rain)
    graph.add_node(wet)
    graph.add_node(slippery)
    
    graph.add_edge(CausalEdge("rain", "wet", CausalRelationType.CAUSES, strength=0.8))
    graph.add_edge(CausalEdge("wet", "slippery", CausalRelationType.CAUSES, strength=0.9))
    
    # do-intervention 테스트
    print("\n[do(Rain=1.0) 개입]")
    result = engine.do_intervention(graph, "rain", 1.0, "slippery")
    print(f"인과 효과: {result.causal_effect:.3f}")
    print(f"설명: {result.explanation}")
    
    # 반사실적 질문
    print("\n[반사실적 질문: 만약 비가 왔다면?]")
    query = CounterfactualQuery(
        premise="rain=1.0",
        conclusion="slippery=?",
        actual_x=0.3,
        counterfactual_x=1.0
    )
    result = engine.counterfactual_query(graph, query)
    print(f"결과: {result.result:.3f}")
    print(f"설명: {result.explanation}")
    
    # 인과 경로 찾기
    print("\n[인과 경로]")
    paths = engine.get_causal_path(graph, "rain", "slippery")
    for path in paths:
        print(f"  {' → '.join(path)}")
    
    print("\n✅ CausalInterventionEngine test complete!")
    print("="*60)
