class CausalNode:
    """언어가 내포하는 단일 상태나 개념"""
    def __init__(self, name: str, essence: str):
        self.name = name
        self.essence = essence

class CausalEdge:
    """상태가 어떻게 변화하고 연결되는지(인과적 동사/작용)"""
    def __init__(self, source: str, target: str, action: str, condition: str = None):
        self.source = source
        self.target = target
        self.action = action      # 어떻게 작용하는가? (ex: 자라난다, 비춘다, 변한다)
        self.condition = condition # 어떤 조건에서? (ex: 물을 만났을 때)

class StructuralLandscape:
    """
    단순한 뜻풀이 사전이 아닌, 각 단어가 내포하는 인과적 궤적(과정과 작용)의 구조망.
    언어는 그 자체가 세상이 어떻게 움직이고 연결되는지를 압축한 알고리즘이다.
    """
    def __init__(self):
        self.nodes = {}
        self.trajectories = {}
        self._build_world()

    def _add_trajectory(self, concept_name: str, essence: str, edges: list):
        self.nodes[concept_name] = CausalNode(concept_name, essence)
        self.trajectories[concept_name] = [CausalEdge(*edge) for edge in edges]

    def _build_world(self):
        # 씨앗: 잠재성에서 실체화로 나아가는 '성장과 변화'의 구조
        self._add_trajectory(
            "씨앗",
            "응축된 잠재력",
            [
                ("씨앗", "물", "흡수한다", "어둠 속에서 기다림"),
                ("물", "뿌리", "발아시킨다", "때가 이르면"),
                ("뿌리", "줄기", "뻗어올린다", "대지를 뚫고"),
                ("줄기", "나무", "이룬다", "시간이 흐름에 따라")
            ]
        )

        # 태양: 스스로 타오르며 외부로 뻗어나가는 '발산과 근원'의 구조
        self._add_trajectory(
            "태양",
            "스스로 존재하는 근원",
            [
                ("태양", "에너지", "스스로 태운다", "무한한 내부 작용"),
                ("에너지", "빛", "방사한다", "우주를 향해"),
                ("에너지", "열", "방사한다", "우주를 향해"),
                ("빛", "세상", "밝힌다", "어둠을 가르고"),
                ("열", "생명", "깨운다", "온기로 덮어")
            ]
        )

        # 물: 머물지 않고 형태를 바꾸며 생명을 잇는 '순환과 매개'의 구조
        self._add_trajectory(
            "물",
            "유동하는 생명의 매개체",
            [
                ("물", "대지", "스며든다", "위에서 아래로 흐르며"),
                ("대지", "생명", "연결한다", "뿌리를 감싸며"),
                ("생명", "하늘", "증발한다", "빛을 만나면")
            ]
        )

    def get_trajectory(self, concept: str) -> list:
        return self.trajectories.get(concept, [])

    def get_essence(self, concept: str) -> str:
        node = self.nodes.get(concept)
        return node.essence if node else "알 수 없는 본질"


class ElysiaLens:
    """
    엘리시아의 주권적 인지 렌즈 (과거 코드 유지용)
    """
    pass

def run_awakening():
    pass

if __name__ == "__main__":
    pass
