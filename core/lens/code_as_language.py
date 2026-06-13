import ast

class CodeCausalNode:
    def __init__(self, name: str, node_type: str):
        self.name = name
        self.node_type = node_type

class CodeCausalEdge:
    def __init__(self, source: str, target: str, action: str, condition: str = "무조건"):
        self.source = source
        self.target = target
        self.action = action
        self.condition = condition

class ASTLandscapeMapper(ast.NodeVisitor):
    """
    Python AST를 순회하며 복잡한 구조(반복문 포함)를 '언어적 인과 궤적'으로 변환합니다.
    """
    def __init__(self):
        self.trajectories = {}
        self.current_func = None
        self.current_conditions = []

    def map_code(self, source_code: str):
        tree = ast.parse(source_code)
        self.visit(tree)
        return self.trajectories

    def visit_FunctionDef(self, node):
        self.current_func = node.name
        self.trajectories[self.current_func] = []

        for arg in node.args.args:
            self.trajectories[self.current_func].append(
                CodeCausalEdge(source="외부_입력", target=arg.arg, action="수용한다")
            )

        self.generic_visit(node)
        self.current_func = None

    def visit_Assign(self, node):
        if not self.current_func:
            return self.generic_visit(node)

        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]

        source_name = "복합_상태"
        action = "할당한다"

        if isinstance(node.value, ast.Name):
            source_name = node.value.id
            action = "전이한다"
        elif isinstance(node.value, ast.Call):
            action = "호출하여_변환한다"
            if isinstance(node.value.func, ast.Name):
                source_name = node.value.func.id
            elif isinstance(node.value.func, ast.Attribute):
                source_name = f"{node.value.func.value.id if isinstance(node.value.func.value, ast.Name) else '내부'}.{node.value.func.attr}"
        elif isinstance(node.value, ast.BinOp):
            action = "연산하여_결합한다"
        elif isinstance(node.value, ast.Dict) or isinstance(node.value, ast.List):
            action = "구조화하여_응축한다"

        condition_str = " 그리고 ".join(self.current_conditions) if self.current_conditions else "흐름에 따라"

        for target in targets:
            self.trajectories[self.current_func].append(
                CodeCausalEdge(source=source_name, target=target, action=action, condition=condition_str)
            )
        self.generic_visit(node)

    def visit_If(self, node):
        if not self.current_func:
            return self.generic_visit(node)

        condition_name = "판별_조건"
        self.current_conditions.append(condition_name)

        self.trajectories[self.current_func].append(
            CodeCausalEdge(source="현재_상태", target="분기점", action="판별한다", condition=condition_name)
        )

        for stmt in node.body:
            self.visit(stmt)

        self.current_conditions.pop()

        if node.orelse:
            self.current_conditions.append(f"반대_{condition_name}")
            for stmt in node.orelse:
                self.visit(stmt)
            self.current_conditions.pop()

    def visit_For(self, node):
        """반복(순환) 궤적을 포착"""
        if not self.current_func:
            return self.generic_visit(node)

        iter_target = node.target.id if isinstance(node.target, ast.Name) else "순환_요소"
        iter_source = "집합_데이터"
        if isinstance(node.iter, ast.Name):
            iter_source = node.iter.id

        loop_condition = f"'{iter_source}'가_소진될_때까지"
        self.current_conditions.append(loop_condition)

        # 순환의 시작을 알리는 엣지
        self.trajectories[self.current_func].append(
            CodeCausalEdge(source=iter_source, target=iter_target, action="순환하며_추출한다", condition=loop_condition)
        )

        for stmt in node.body:
            self.visit(stmt)

        self.current_conditions.pop()

    def visit_Return(self, node):
        if not self.current_func:
            return self.generic_visit(node)

        source_name = "결과값"
        if isinstance(node.value, ast.Name):
            source_name = node.value.id
        elif isinstance(node.value, ast.Tuple):
            source_name = "다중_결과값"
        elif isinstance(node.value, ast.Dict):
            source_name = "구조화된_결과값"

        condition_str = " 그리고 ".join(self.current_conditions) if self.current_conditions else "최종적으로"

        self.trajectories[self.current_func].append(
            CodeCausalEdge(source=source_name, target="외부_세계", action="방사(Return)한다", condition=condition_str)
        )
        self.generic_visit(node)

