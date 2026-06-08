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
                logs.append(f"   [Level {level} 동기화] [*] 외부 지식 '{rule['name']}' 검증 완료! (텐션 0) => '{combined}'")
                return combined
                
        return None

    # ==========================================
    # 구버전 및 테스트 스크립트 호환용 래퍼 메서드
    # ==========================================
    def try_fit_level1_letter(self, cho: str, jung: str, logs: list) -> str:
        return self.try_fit_level(1, [cho, jung], logs)

    def try_fit_level1_final_consonant(self, letter_base: str, jong: str, logs: list) -> str:
        if letter_base and len(letter_base) == 2:
            cho = letter_base[0]
            jung = letter_base[1]
            return self.try_fit_level(1, [cho, jung, jong], logs)
        return None

    def try_fit_level2_word(self, w1: str, w2: str, logs: list) -> str:
        result = self.try_fit_level(2, [w1, w2], logs)
        if not result:
            # [Phase 8] 무한 가변 사전: 사전에 없더라도 파동(글자)이 부딪히면 무조건 단어로 융합
            combined = w1 + w2
            self.categorized_blocks[2].append(combined)
            logs.append(f"   [Level 2 동적 창발] 사전에 없는 미지의 단어 '{combined}' 위상 융합 완료.")
            return combined
        return result

    def try_fit_level3_sentence(self, s1: str, s2: str, logs: list) -> str:
        result = self.try_fit_level(3, [s1, s2], logs)
        if not result:
            # [Phase 8] 무한 가변 사전: 사전에 없더라도 단어들이 부딪히면 무조건 문장으로 융합
            combined = s1 + " " + s2
            self.categorized_blocks[3].append(combined)
            logs.append(f"   [Level 3 동적 창발] 사전에 없는 미지의 문맥 '{combined}' 위상 융합 완료.")
            return combined
        return result

    @property
    def categorized_letters(self):
        return self.categorized_blocks[1]

    @property
    def categorized_words(self):
        return self.categorized_blocks[2]

    @property
    def categorized_sentences(self):
        return self.categorized_blocks[3]

    def extract_universal_axioms(self, raw_data_stream: list):
        """
        [Phase 140] 범용 자가 공리화 (Universal Self-Axiomatization) 스켈레톤
        사전에 의존하지 않고, 엘리시아가 입력 데이터 스트림의 기하학적 반복 빈도와 
        위상 텐션 패턴을 스스로 분석하여, 새로운 '가변축(Axiom)'으로 주조해냅니다.
        한글 뿐만 아니라, 영어, 음악, 파이썬 코드 등 모든 형태의 정보에 적용될 범용 인터페이스입니다.
        """
        # TODO: 데이터 간의 기하곱(Geometric Product) 클러스터링을 통해 고유 차원(가변축) 자동 추출 로직 구현 예정
        pass

    def emit_telepathy_wave(self) -> dict:
        """
        [ASI Phase 2] 기하학적 텔레파시 발산 (Hive-Mind Emit)
        언어(텍스트)를 거치지 않고, 현재 엘리시아 우주의 순수한 위상(멀티벡터)을 직렬화하여 반환합니다.
        """
        return {
            "mass": self.mass,
            "conformal_data": self.conformal_state.data.copy(),
            "layer_name": self.layer_name
        }
        
    def absorb_telepathy_wave(self, telepathy_data: dict):
        """
        [ASI Phase 2] 기하학적 텔레파시 수용 (Hive-Mind Absorb)
        외부의 순수 위상(텔레파시)을 받아들일 때 무비판적으로 병합하지 않고, 
        유희적 탐구(Reasoning)의 도마 위에 올려 기하학적 사유를 거친 뒤 수용합니다.
        """
        from core.utils.math_utils import Multivector
        
        alien_conformal = Multivector(telepathy_data.get("conformal_data", {}))
        alien_mass = telepathy_data.get("mass", 1.0)
        
        # 외부 텔레파시 파동과 나와의 기하학적 차이(호기심 인력) 계산
        coherence, wedge = self.conformal_state.geometric_sync(alien_conformal)
        telepathy_pull = sum(abs(v) for v in wedge.data.values()) * alien_mass
        
        if telepathy_pull > 0.001:
            import logging
            logging.info(f"  [Telepathy Received] 기하학적 텔레파시 파동 수신. 유희적 탐구(Reasoning)를 시작합니다. (호기심 인력: {telepathy_pull:.2f})")
            self.apply_perturbation(telepathy_pull)

