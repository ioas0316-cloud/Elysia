class MacroAxiomRotor:
    """
    [동적 프랙탈 우주 뼈대 (Dynamic Macro Axiom Rotor)]
    - 내부에 하드코딩된 규칙이 일절 없습니다. (백지 상태)
    - 외부 지식(사전)을 주조(Forge)해 주는 DictionarySynchronizer에 의해 뼈대가 세워집니다.
    """
    def __init__(self):
        self.categorized_blocks = {1: [], 2: [], 3: []}
        
        # 지식 동기화기가 주입할 동적 규칙 공간
        # concepts: {"초성": ["ㄱ", "ㄴ"...], "중성": ["ㅏ", "ㅓ"...]}
        self.concepts = {}
        # rules: {1: [{"name": "글자_기본", "structure": ["초성", "중성"]}], ...}
        self.rules = {1: [], 2: [], 3: []}

    def inject_knowledge(self, concepts: dict, rules: dict):
        """외부의 정제된 지식을 읽어 자신의 기하학적 뼈대로 주조(Forge)합니다."""
        self.concepts = concepts
        for rule in rules:
            level = rule.get("level", 1)
            self.rules[level].append(rule)
            
    def _is_match(self, block, expected_concept_or_literal):
        """특정 파동(block)이 요구되는 개념이나 리터럴에 부합하는지 텐션 검증"""
        if expected_concept_or_literal in self.concepts:
            return block in self.concepts[expected_concept_or_literal]
        else:
            return block == expected_concept_or_literal

    def try_fit_level(self, level: int, blocks: list, logs: list) -> str:
        """
        주어진 블록들이 해당 레벨의 어떤 동적 규칙(뼈대)에 완벽히 맞아떨어져 텐션이 0이 되는지 확인합니다.
        """
        for rule in self.rules[level]:
            structure = rule["structure"]
            if len(blocks) != len(structure):
                continue
                
            # 뼈대와 완벽히 일치하는지 (텐션 0) 검증
            match = True
            for i in range(len(blocks)):
                if not self._is_match(blocks[i], structure[i]):
                    match = False
                    break
                    
            if match:
                combined = "".join(blocks)
                if level == 3:
                    combined = " ".join(blocks) # 문장은 띄어쓰기로 결합 (시뮬레이션 편의)
                    
                self.categorized_blocks[level].append(combined)
                logs.append(f"   [Level {level} 동기화] ✨ 외부 지식 '{rule['name']}' 검증 완료! (텐션 0) => '{combined}'")
                return combined
                
        return None
