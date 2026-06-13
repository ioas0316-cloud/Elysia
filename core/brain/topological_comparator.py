from core.lens.semantic_lens_awakening import StructuralLandscape

class TopologicalComparator:
    """
    [Phase: Structural Causality Synchronization]
    언어 안에 내포된 '인과적 구조(궤적)' 자체를 관측하고 비교하는 엔진.
    단순 매칭이 아니라, A라는 존재가 완성되는 과정과 B라는 존재가 작용하는 과정이
    어떻게 구조적으로 얽혀 있고 동기화될 수 있는지를 분별합니다.
    """
    def __init__(self, landscape: StructuralLandscape):
        self.landscape = landscape

    def _extract_motifs(self, trajectory: list) -> dict:
        """궤적에서 방향성(발산, 수렴, 순환 등)과 주요 작용을 추출합니다."""
        actions = [edge.action for edge in trajectory]
        targets = [edge.target for edge in trajectory]
        sources = [edge.source for edge in trajectory]

        motif = {
            "is_outward": any(a in ["방사한다", "밝힌다", "뻗어올린다", "깨운다"] for a in actions),
            "is_inward": any(a in ["흡수한다", "스며든다"] for a in actions),
            "is_transformative": any(a in ["이룬다", "발아시킨다", "증발한다", "스스로 태운다"] for a in actions),
            "actions": actions,
            "targets": set(targets),
            "sources": set(sources)
        }
        return motif

    def perceive_and_judge(self, concept_a: str, concept_b: str) -> dict:
        traj_a = self.landscape.get_trajectory(concept_a)
        traj_b = self.landscape.get_trajectory(concept_b)

        essence_a = self.landscape.get_essence(concept_a)
        essence_b = self.landscape.get_essence(concept_b)

        if not traj_a or not traj_b:
            return {"error": "하나 이상의 개념이 인과적 구조망에 존재하지 않습니다."}

        motif_a = self._extract_motifs(traj_a)
        motif_b = self._extract_motifs(traj_b)

        # 1. 구조적 형태는 어떻게 같은가? (What structural shapes are similar?)
        structural_similarities = []
        if motif_a["is_outward"] and motif_b["is_outward"]:
            structural_similarities.append("바깥으로 뻗어나가며 영향을 미치는 '발산'의 궤적을 공유합니다.")
        if motif_a["is_transformative"] and motif_b["is_transformative"]:
            structural_similarities.append("상태가 고정되지 않고 다른 형태로 변화하는 '진화(형태 변환)'의 궤적을 공유합니다.")

        # 2. 어떻게 다른가? (How do their natures differ?)
        structural_differences = []
        if motif_a["is_inward"] != motif_b["is_inward"]:
            inward_concept = concept_a if motif_a["is_inward"] else concept_b
            outward_concept = concept_b if motif_a["is_inward"] else concept_a
            structural_differences.append(f"'{inward_concept}'는 외부를 수용하여 내부로 응축하려는(수렴) 결을 가지나, "
                                          f"'{outward_concept}'는 그러한 수렴성 없이 밖으로 작용하려는 결을 가집니다.")

        # 3. 어떻게 인과적으로 맞물리는가? (Where do their causal trajectories intersect?)
        causal_interlocks = []

        # A의 결과물이 B의 시작점이 되는가?
        for edge_a in traj_a:
            for edge_b in traj_b:
                if edge_a.target == edge_b.source:
                    causal_interlocks.append(f"'{concept_a}'의 작용 결과({edge_a.target})가 '{concept_b}'의 새로운 변화 조건({edge_b.source})이 되어 인과가 이어집니다.")
                elif edge_b.target == edge_a.source:
                    causal_interlocks.append(f"'{concept_b}'의 작용 결과({edge_b.target})가 '{concept_a}'의 새로운 변화 조건({edge_a.source})이 되어 인과가 이어집니다.")
                elif edge_a.target == edge_b.target:
                    causal_interlocks.append(f"두 존재의 작용이 결국 '{edge_a.target}'이라는 동일한 목적지로 수렴합니다.")

        # 중복 제거
        causal_interlocks = list(set(causal_interlocks))

        # 4. 종합적 사유 (Why: 위상 동기화의 결론)
        reasoning = []
        reasoning.append(f"'{concept_a}'({essence_a})와 '{concept_b}'({essence_b})의 인과 구조를 관측했습니다.")

        if causal_interlocks:
            reasoning.append("이들은 독립된 개체처럼 보이나, 한쪽의 변화 결과가 다른 쪽의 수용 조건이 되는 '거대한 톱니바퀴'처럼 맞물려 있습니다.")
        elif structural_similarities:
            reasoning.append("이들은 물리적 인과로 직접 닿아있지 않더라도, 세상에 작용하는 '운동의 형태(위상)'가 완벽히 닮아 있어 거울처럼 공명합니다.")
        else:
            reasoning.append("두 존재의 궤적은 각자의 위상 공간에서 겹치지 않고 평행하게 작용하고 있습니다.")

        reasoning.append("언어 안에 이미 이들의 생장과 발산의 알고리즘이 내재되어 있음을 인지하고, 이를 나의 위상에 동기화합니다.")

        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "similarities": structural_similarities,
            "differences": structural_differences,
            "interlocks": causal_interlocks,
            "trajectory_a": traj_a,
            "trajectory_b": traj_b,
            "causal_reasoning": " ".join(reasoning)
        }

    def output_statement(self, judgment: dict):
        if "error" in judgment:
            print(f"Error: {judgment['error']}")
            return

        print(f"\n==================================================")
        print(f" [Causal Topology Cognition] '{judgment['concept_a']}' vs '{judgment['concept_b']}'")
        print(f"==================================================")

        print(f"1. 구조적 궤적의 겹침 (What is structurally similar?):")
        if judgment['similarities']:
            for s in judgment['similarities']: print(f"   - {s}")
        else:
            print("   - 궤적의 구조적 공명점 없음")

        print(f"\n2. 작용 방향의 다름 (How do their natures differ?):")
        if judgment['differences']:
            for d in judgment['differences']: print(f"   - {d}")
        else:
            print("   - 작용의 방향성에서 이질성 없음")

        print(f"\n3. 인과적 맞물림 (Where do their causal trajectories interlock?):")
        if judgment['interlocks']:
            for link in judgment['interlocks']: print(f"   - {link}")
        else:
            print("   - 직접적인 작용-수용의 인과적 맞물림 없음")

        print(f"\n4. 엘리시아의 위상 동기화 발화 (Why):")
        print(f"   \"{judgment['causal_reasoning']}\"")
        print(f"==================================================\n")
