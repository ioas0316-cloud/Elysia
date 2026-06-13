from core.lens.semantic_lens_awakening import SemanticLandscape

class TopologicalComparator:
    """
    [Phase: Topological Synchronization]
    정보화된 원리를 실질적인 알고리즘으로 구현한 모듈.
    단순한 키워드 매칭이 아니라, 두 개념 사이의 '같고 다름(What)', '연결 지점(Where/How)',
    그리고 '인과적 이유(Why)'를 분별하여 사유(Reasoning)를 출력합니다.
    """
    def __init__(self, landscape: SemanticLandscape):
        self.landscape = landscape

    def _extract_core_nodes(self, word: str) -> set:
        """
        [추출의 원리] 단어의 뜻풀이 속에서 다른 개념(Node)들을 추출합니다.
        """
        definition = self.landscape.get_definition(word)
        if not definition:
            return set()

        nodes = set()
        for dict_word in self.landscape.get_all_words():
            if dict_word != word and dict_word in definition:
                nodes.add(dict_word)
        return nodes

    def perceive_and_judge(self, concept_a: str, concept_b: str) -> dict:
        """
        [위상적 판단의 공리] 두 개념을 인지하고 비교 대조하여 분별 결과를 반환합니다.
        """
        nodes_a = self._extract_core_nodes(concept_a)
        nodes_b = self._extract_core_nodes(concept_b)

        # 1. 무엇이 같은가? (What is the same?)
        intersection = nodes_a.intersection(nodes_b)

        # 2. 무엇이 다른가? (What is different?)
        unique_a = nodes_a - nodes_b
        unique_b = nodes_b - nodes_a

        # 3. 어디가 어떻게 연결되는가? (Where and How do they connect?)
        direct_link = []
        if concept_b in self.landscape.get_definition(concept_a):
            direct_link.append(f"'{concept_a}'의 위상 안에 '{concept_b}'가 내재되어 있습니다.")
        if concept_a in self.landscape.get_definition(concept_b):
            direct_link.append(f"'{concept_b}'의 위상 안에 '{concept_a}'가 내재되어 있습니다.")

        # 4. 어째서 그러한가? (Why? - 인과적 사유의 정립)
        reasoning = []
        if intersection:
            shared_str = ", ".join(intersection)
            reasoning.append(f"두 개념은 [{shared_str}](이)라는 공통된 매개체를 품고 있습니다. "
                             f"이는 '{concept_a}'와 '{concept_b}'가 표면적으로는 다른 형태를 띠더라도, "
                             f"근원적으로는 [{shared_str}]의 장력 아래서 위상적으로 동기화되는 동일한 이치를 공유함을 뜻합니다.")

        if direct_link:
            reasoning.append(f"또한 이들은 서로를 정의하는 인과적 그물망에 직접적으로 얽혀 있습니다. "
                             f"한쪽의 존재가 다른 쪽의 존재를 증명하는 거울과 같은 관계입니다.")

        if unique_a or unique_b:
            diff_a_str = ", ".join(unique_a) if unique_a else "없음"
            diff_b_str = ", ".join(unique_b) if unique_b else "없음"
            reasoning.append(f"하지만 이들의 쓰임과 방향성은 분별되어야 합니다. "
                             f"'{concept_a}'는 [{diff_a_str}]의 결을 따라 작용하며, "
                             f"'{concept_b}'는 [{diff_b_str}]의 결을 따라 작용합니다. "
                             f"즉, 기원이나 연결성은 같을지라도 세상을 쪼개어 나타나는 양태(다름)는 분명히 구별됩니다.")

        if not intersection and not direct_link:
            reasoning.append(f"현재의 인지망 안에서 '{concept_a}'와 '{concept_b}' 사이의 직접적 인과나 "
                             f"공명점은 발견되지 않았습니다. 이들은 완전히 독립된 위상에 존재하거나, "
                             f"아직 내가 알지 못하는 미지의 연결고리를 통해 이어져 있을 것입니다.")

        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "intersection": list(intersection),
            "difference_a": list(unique_a),
            "difference_b": list(unique_b),
            "direct_links": direct_link,
            "causal_reasoning": " ".join(reasoning)
        }

    def output_statement(self, judgment: dict):
        """
        [발화와 정보화의 원리] 분별된 결과를 엘리시아의 발화로 정립합니다.
        """
        print(f"\n==================================================")
        print(f" [Topological Cognition] '{judgment['concept_a']}' vs '{judgment['concept_b']}'")
        print(f"==================================================")
        print(f"1. 무엇이 같은가?: {judgment['intersection'] if judgment['intersection'] else '공통점 없음'}")

        diff_a = judgment['difference_a'] if judgment['difference_a'] else "없음"
        diff_b = judgment['difference_b'] if judgment['difference_b'] else "없음"
        print(f"2. 무엇이 다른가?: '{judgment['concept_a']}'는 {diff_a}, '{judgment['concept_b']}'는 {diff_b}의 고유성을 가짐")

        if judgment['direct_links']:
            print(f"3. 어디가 연결되는가?:")
            for link in judgment['direct_links']:
                print(f"   - {link}")
        else:
            print(f"3. 어디가 연결되는가?: 직접적인 내재적 연결 없음")

        print(f"\n4. 엘리시아의 인과적 사유 (Why):")
        print(f"   \"{judgment['causal_reasoning']}\"")
        print(f"==================================================\n")
