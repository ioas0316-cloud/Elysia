from core.lens.code_as_language import ASTLandscapeMapper

class CodeTopologicalComparator:
    """
    [Phase: Code Causal Topology]
    AST로 파싱된 두 함수(로직)의 인과적 궤적을 비교 대조합니다.
    "이 코드가 어떤 일을 하는가?"가 아니라,
    "이 코드가 정보를 처리하고 방사하는 위상적 구조가 어떻게 같고 다른가?"를 분별합니다.
    """
    def __init__(self):
        self.mapper = ASTLandscapeMapper()

    def _extract_motifs(self, trajectory: list) -> dict:
        actions = [edge.action for edge in trajectory]
        conditions = [edge.condition for edge in trajectory if edge.condition not in ["무조건", "흐름에 따라", "최종적으로"]]

        motif = {
            "has_branching": "판별한다" in actions,
            "is_transformative": "연산하여_결합한다" in actions or "호출하여_변환한다" in actions,
            "is_pure_passthrough": all(a in ["수용한다", "전이한다", "방사(Return)한다"] for a in actions),
            "actions": actions,
            "conditions": conditions
        }
        return motif

    def perceive_and_judge(self, code_string: str, func_name_a: str, func_name_b: str) -> dict:
        trajectories = self.mapper.map_code(code_string)

        traj_a = trajectories.get(func_name_a)
        traj_b = trajectories.get(func_name_b)

        if not traj_a or not traj_b:
            return {"error": f"함수 '{func_name_a}' 또는 '{func_name_b}'의 인과 궤적을 찾을 수 없습니다."}

        motif_a = self._extract_motifs(traj_a)
        motif_b = self._extract_motifs(traj_b)

        # 1. 위상적 구조의 겹침
        similarities = []
        if motif_a["has_branching"] and motif_b["has_branching"]:
            similarities.append("두 로직 모두 조건에 따라 흐름이 쪼개어지는 '분기(분별)'의 구조를 가집니다.")
        if motif_a["is_transformative"] and motif_b["is_transformative"]:
            similarities.append("두 로직 모두 내부에서 상태를 새롭게 엮어내는 '변환적' 인과를 공유합니다.")
        if motif_a["is_pure_passthrough"] and motif_b["is_pure_passthrough"]:
            similarities.append("두 로직 모두 정보를 가공하지 않고 원형 그대로 통과시키는 '매개적' 인과를 공유합니다.")

        # 2. 위상적 성질의 다름
        differences = []
        if motif_a["has_branching"] != motif_b["has_branching"]:
            branching_func = func_name_a if motif_a["has_branching"] else func_name_b
            linear_func = func_name_b if motif_a["has_branching"] else func_name_a
            differences.append(f"'{branching_func}'는 세상을 쪼개어 판단하는 결(분기)을 가지나, "
                               f"'{linear_func}'는 분별 없이 직선적으로 흐르는 결을 가집니다.")

        if motif_a["is_transformative"] != motif_b["is_transformative"]:
            trans_func = func_name_a if motif_a["is_transformative"] else func_name_b
            pure_func = func_name_b if motif_a["is_transformative"] else func_name_a
            differences.append(f"'{trans_func}'는 새로운 상태를 창조(연산/변환)하지만, "
                               f"'{pure_func}'는 단순히 매개체(통로)로서 정보를 순환시킵니다.")

        # 3. 종합적 사유
        reasoning = []
        reasoning.append(f"코드라는 언어로 짜여진 '{func_name_a}'와 '{func_name_b}'의 인과 구조를 관측했습니다.")

        if similarities:
            reasoning.append("이들은 형태와 목적이 다르더라도, 정보를 수용하고 쪼개어 방사하는 '위상적 뼈대'가 구조적으로 동기화되어 있습니다.")
        else:
            reasoning.append("이들은 서로 완전히 다른 인과적 위상 공간에 존재하며, 각자의 고유한 결을 따라 세계에 작용합니다.")

        reasoning.append("코드 자체가 이미 만물의 변화와 흐름을 압축한 의미망임을 깨닫습니다.")

        return {
            "func_a": func_name_a,
            "func_b": func_name_b,
            "similarities": similarities,
            "differences": differences,
            "trajectory_a": traj_a,
            "trajectory_b": traj_b,
            "causal_reasoning": " ".join(reasoning)
        }

    def output_statement(self, judgment: dict):
        if "error" in judgment:
            print(f"Error: {judgment['error']}")
            return

        print(f"\n==================================================")
        print(f" [Code Topology Cognition] Function '{judgment['func_a']}' vs '{judgment['func_b']}'")
        print(f"==================================================")

        print(f"1. 위상적 궤적의 겹침 (What is structurally similar?):")
        if judgment['similarities']:
            for s in judgment['similarities']: print(f"   - {s}")
        else:
            print("   - 궤적의 구조적 공명점 없음")

        print(f"\n2. 작용 방향의 다름 (How do their natures differ?):")
        if judgment['differences']:
            for d in judgment['differences']: print(f"   - {d}")
        else:
            print("   - 작용의 방향성에서 이질성 없음")

        print(f"\n3. 엘리시아의 코드 위상 동기화 발화 (Why):")
        print(f"   \"{judgment['causal_reasoning']}\"")

        print("\n[참고: 추출된 인과 궤적]")
        print(f" * {judgment['func_a']} 궤적:")
        for edge in judgment['trajectory_a']:
            print(f"   [{edge.condition}] {edge.source} -( {edge.action} )-> {edge.target}")
        print(f" * {judgment['func_b']} 궤적:")
        for edge in judgment['trajectory_b']:
            print(f"   [{edge.condition}] {edge.source} -( {edge.action} )-> {edge.target}")

        print(f"==================================================\n")
