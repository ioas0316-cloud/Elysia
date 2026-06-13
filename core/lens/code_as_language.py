import ast

class CodeCausalNode:
    """코드 안의 상태(변수, 데이터 구조 등)를 나타냅니다."""
    def __init__(self, name: str, node_type: str):
        self.name = name
        self.node_type = node_type # 예: Variable, Argument, Return

class CodeCausalEdge:
    """코드 안의 작용(함수 호출, 연산, 할당 등)을 나타냅니다."""
    def __init__(self, source: str, target: str, action: str, condition: str = "무조건"):
        self.source = source
        self.target = target
        self.action = action      # 예: 할당한다, 호출한다, 반환한다
        self.condition = condition # 예: if x > 0 인 경우

class ASTLandscapeMapper(ast.NodeVisitor):
    """
    Python AST를 순회하며 코드를 '언어적 인과 궤적'으로 변환합니다.
    """
    def __init__(self):
        self.trajectories = {} # 함수명 -> [CodeCausalEdge, ...]
        self.current_func = None
        self.current_conditions = []

    def map_code(self, source_code: str):
        tree = ast.parse(source_code)
        self.visit(tree)
        return self.trajectories

    def visit_FunctionDef(self, node):
        self.current_func = node.name
        self.trajectories[self.current_func] = []

        # 인자들을 초기 상태로 기록
        for arg in node.args.args:
            self.trajectories[self.current_func].append(
                CodeCausalEdge(source="외부_입력", target=arg.arg, action="수용한다")
            )

        self.generic_visit(node)
        self.current_func = None

    def visit_Assign(self, node):
        if not self.current_func:
            return self.generic_visit(node)

        # 타겟(할당받는 변수)
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]

        # 소스(할당하는 값이나 연산)
        source_name = "미상_연산"
        action = "할당한다"

        if isinstance(node.value, ast.Name):
            source_name = node.value.id
            action = "전이한다"
        elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            source_name = node.value.func.id
            action = "호출하여_변환한다"
        elif isinstance(node.value, ast.BinOp):
            action = "연산하여_결합한다"
            # 단순화를 위해 왼쪽 피연산자만 추적
            if isinstance(node.value.left, ast.Name):
                source_name = node.value.left.id

        condition_str = " 그리고 ".join(self.current_conditions) if self.current_conditions else "흐름에 따라"

        for target in targets:
            self.trajectories[self.current_func].append(
                CodeCausalEdge(source=source_name, target=target, action=action, condition=condition_str)
            )

        self.generic_visit(node)

    def visit_If(self, node):
        if not self.current_func:
            return self.generic_visit(node)

        # 조건식을 텍스트로 단순화 (ex: ast.Compare 등 파싱은 복잡하므로 단순화)
        condition_name = "특정_조건_충족시"
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name):
            condition_name = f"'{node.test.left.id}'의_상태에_따라"

        self.current_conditions.append(condition_name)

        # 분기 기록
        self.trajectories[self.current_func].append(
            CodeCausalEdge(source="현재_상태", target="분기점", action="판별한다", condition=condition_name)
        )

        for stmt in node.body:
            self.visit(stmt)

        self.current_conditions.pop()

        # Else 처리
        if node.orelse:
            self.current_conditions.append(f"반대_조건_{condition_name}")
            for stmt in node.orelse:
                self.visit(stmt)
            self.current_conditions.pop()

    def visit_Return(self, node):
        if not self.current_func:
            return self.generic_visit(node)

        source_name = "결과값"
        if isinstance(node.value, ast.Name):
            source_name = node.value.id

        condition_str = " 그리고 ".join(self.current_conditions) if self.current_conditions else "최종적으로"

        self.trajectories[self.current_func].append(
            CodeCausalEdge(source=source_name, target="외부_세계", action="방사(Return)한다", condition=condition_str)
        )
        self.generic_visit(node)

