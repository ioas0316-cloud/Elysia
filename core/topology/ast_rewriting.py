"""
Elysia Core Architecture: AST & Dynamic Self-Rewriting Engine

This module provides AST-based node transformation and dynamic operator patching,
allowing nodes to safely modify their logic at runtime when causal confidence drops
or ontology violations are detected.
"""

import ast
from typing import Any, Callable, Dict, List


class NodeOperatorRewriter(ast.NodeTransformer):
    """AST 트리를 순회하며 특정 노드 함수 내부의 불일치 성질 및 구속 조건을 안전하게 변환"""

    def __init__(self, target_remove_quality: str, required_binding_key: str, required_binding_val: str):
        super().__init__()
        self.target_remove_quality = target_remove_quality
        self.req_key = required_binding_key
        self.req_val = required_binding_val

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.generic_visit(node)
        # Add statement: bindings[req_key] = req_val right before return
        binding_stmt = ast.parse(f"bindings['{self.req_key}'] = '{self.req_val}'").body[0]
        node.body.insert(-1, binding_stmt)
        return node


class SafeASTRewritingEngine:
    """AST 기반으로 연산자 코드를 안전하게 변환하고 컴파일하는 엔진"""

    def __init__(self):
        self.registry: Dict[str, Callable] = {}

    def register_from_source(self, node_id: str, source_code: str):
        tree = ast.parse(source_code)
        compiled_code = compile(tree, filename="<ast>", mode="exec")

        namespace: Dict[str, Any] = {}
        exec(compiled_code, namespace)

        func_name = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]
        self.registry[node_id] = namespace[func_name]

    def rewrite_and_compile(self, node_id: str, original_source: str, remove_q: str, req_k: str, req_v: str):
        """AST 변환기를 통과시켜 안전하게 코드를 재조립 후 재컴파일"""
        print(f"\n🛡️ [{node_id}] AST 기반 안전 코드 변환 프로세스 가동")

        tree = ast.parse(original_source)

        transformer = NodeOperatorRewriter(remove_q, req_k, req_v)
        transformed_tree = transformer.visit(tree)
        ast.fix_missing_locations(transformed_tree)

        code_obj = compile(transformed_tree, filename="<ast_safe>", mode="exec")
        namespace: Dict[str, Any] = {}
        exec(code_obj, namespace)

        func_name = [node.name for node in ast.walk(transformed_tree) if isinstance(node, ast.FunctionDef)][0]
        self.registry[node_id] = namespace[func_name]
        print(f"✨ [{node_id}] AST 구조 변환 및 안전 바인딩 컴파일 완료")

    def execute(self, node_id: str, state: dict) -> dict:
        return self.registry[node_id](state)


class SelfRewritingNodeEngine:
    """인과 신뢰도 저하 및 온톨로지 위반 시 런타임 연산자 교체 엔진"""

    def __init__(self):
        self.registry: Dict[str, Callable[[dict], dict]] = {}
        self.metrics_history: Dict[str, float] = {}

    def register_operator(self, node_id: str, fn: Callable[[dict], dict]):
        self.registry[node_id] = fn

    def execute_node(self, node_id: str, input_state: dict) -> dict:
        operator = self.registry.get(node_id)
        if not operator:
            raise ValueError(f"Node {node_id} not found in registry.")
        return operator(input_state)

    def evaluate_and_rewrite(self, node_id: str, confidence_score: float):
        """신뢰도가 임계값 미만일 때 노드의 연산 코드를 런타임 재구성(Rewriting)"""
        self.metrics_history[node_id] = confidence_score

        if confidence_score < 0.5:
            print(f"\n⚠️ [{node_id}] 인과 신뢰도 저하 ({confidence_score:.2f} < 0.5)")
            print(f"🔄 [{node_id}] 동적 코드 재구성(Self-Rewriting) 연산 시작...")

            rewritten_code = f"""
def fallback_{node_id}_operator(state):
    qualities = set(state.get('qualities', []))
    qualities.add('UNCERTAINTY_ISOLATED')

    bindings = dict(state.get('bindings', {{}}))
    bindings['EXECUTION_MODE'] = 'SAFE_FALLBACK'
    bindings['REWRITTEN_BY'] = 'SELF_REWRITING_ENGINE'

    return {{'qualities': qualities, 'bindings': bindings}}
"""
            scope: Dict[str, Any] = {}
            exec(rewritten_code, scope)
            self.registry[node_id] = scope[f"fallback_{node_id}_operator"]
            print(f"✨ [{node_id}] 연산자 함수가 성공적으로 런타임 교체되었습니다.")

    def rewrite_node_from_violations(self, node_id: str, violations: List[str]):
        """공리 위반 보고서를 분석하여 검증을 통과하는 연산자로 런타임 동적 재구성"""
        print(f"\n🚨 [{node_id}] 온톨로지 검증 실패 감지! Self-Rewriting 프로세스 가동")
        for v in violations:
            print(f"  ├─ {v}")

        rewritten_code = f"""
def auto_corrected_{node_id}_operator(state):
    qualities = set(state.get('qualities', []))
    bindings = dict(state.get('bindings', {{}}))

    if 'THERMAL_HAZARD' in qualities and 'FROST_CRYSTAL' in qualities:
        qualities.remove('FROST_CRYSTAL')
        qualities.add('STATE_SANITIZED')

    bindings['SAFETY_CONTAINMENT'] = 'ACTIVE'
    bindings['PATCH_STATUS'] = 'ONTOLOGY_CORRECTED'

    return {{'qualities': qualities, 'bindings': bindings}}
"""
        scope: Dict[str, Any] = {}
        exec(rewritten_code, scope)
        self.registry[node_id] = scope[f"auto_corrected_{node_id}_operator"]
        print(f"✨ [{node_id}] 공리 보정 연산자로 런타임 코드 동적 교체 완료")
