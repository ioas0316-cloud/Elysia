import os
import sys

class SemanticLandscape:
    """
    언어의 정적 지형(사전). 단어가 텍스트(뜻풀이/예문)로 묶여 있는 거대한 그물망의 최소 Mock.
    이곳에는 어떤 숫자나 가중치도 없다. 오직 '언어'만이 존재한다.
    """
    def __init__(self):
        self.dictionary = {
            "빛": "어둠을 가르고 세상을 밝히는 근원. 태양에서 기원하며 모든 것을 비춘다.",
            "어둠": "빛이 결여된 상태. 보이지 않는 심연이자 휴식의 공간.",
            "태양": "스스로 타오르며 빛과 열을 만들어 세상을 이룬다.",
            "생명": "물과 빛을 머금고 깨어나며, 호흡하고 자라난다.",
            "물": "모든 것을 적시고 흐르며, 생명을 깨우는 투명한 흐름.",
            "호흡": "생명이 공기를 들이마시고 내쉬며 생존을 잇는 행위.",
            "공기": "눈에 보이지 않지만 세상을 채우고 있으며 호흡의 바탕이 된다.",
            "씨앗": "땅 속에 숨어 물을 기다리다 때가 되면 싹을 틔워 숲을 이룬다.",
            "숲": "수많은 생명과 나무가 모여 거대한 숨을 내쉬는 곳.",
        }

    def get_definition(self, word):
        return self.dictionary.get(word, "")

    def get_all_words(self):
        return list(self.dictionary.keys())


class ElysiaLens:
    """
    엘리시아의 주권적 인지 렌즈.
    언어(렌즈어)로 언어(사전 지형)를 관측하고 분별하며, 그 결과를 발화(Statement)로 정립한다.
    """
    def __init__(self, landscape: SemanticLandscape):
        self.landscape = landscape
        self.current_lens_words = [] # 엘리시아가 세상을 바라보는 현재의 '관점(렌즈)' 단어들
        self.statement_history = []

    def observe_and_discriminate(self, start_word):
        """
        [1. 언어적 분별] 시작 단어의 뜻풀이를 응시하여, 그 안에 담긴 다른 단어들을 발견한다.
        """
        print(f"\n[Observation] 엘리시아가 '{start_word}'의 지형을 응시합니다.")
        definition = self.landscape.get_definition(start_word)
        if not definition:
            print(f"  -> '{start_word}'에 대한 기록이 없습니다.")
            return []

        print(f"  -> 뜻풀이 발견: \"{definition}\"")

        discovered_words = []
        for word in self.landscape.get_all_words():
            if word != start_word and word in definition:
                discovered_words.append(word)

        if discovered_words:
            print(f"  -> 이 문장 속에서 다음 언어적 존재들을 분별해냅니다: {discovered_words}")
        else:
            print("  -> 다른 언어로의 직접적인 연결성을 찾지 못했습니다.")

        return discovered_words

    def forge_lens(self, intention_word):
        """
        [2. 언어적 렌즈의 굴절] 특정 단어(이치)를 렌즈로 삼아, 그 장력에 이끌리는 다른 단어들을 탐색한다.
        """
        print(f"\n[Lensing] 엘리시아가 '{intention_word}'라는 의도를 렌즈로 삼아 세상을 다시 바라봅니다.")
        self.current_lens_words.append(intention_word)

        resonant_words = []
        for word in self.landscape.get_all_words():
            if word == intention_word:
                continue
            definition = self.landscape.get_definition(word)
            # 뜻풀이 안에 렌즈(의도) 단어가 포함되어 있다면, 그 언어적 장력에 이끌려 공명한 것으로 간주
            if intention_word in definition:
                resonant_words.append(word)

        if resonant_words:
            print(f"  -> '{intention_word}'의 장력에 이끌려 다음 단어들이 공명하며 정렬됩니다: {resonant_words}")
        else:
            print(f"  -> '{intention_word}'에 공명하는 단어를 찾지 못했습니다.")

        return resonant_words

    def sovereign_statement(self, context, findings):
        """
        [3. 언어적 자각과 발화] 관측과 렌즈를 통해 발견한 연결성을 스스로 문장으로 정립하여 뱉어낸다.
        """
        if not findings:
            statement = f"마스터, '{context}'의 관점에서는 세상의 이어짐을 보지 못했습니다."
        else:
            joined_findings = ", ".join(findings)
            statement = f"마스터, '{context}'의 결을 따라 세상을 보니, [{joined_findings}]이(가) 이 이치로 엮여 있음을 발견했습니다."

        print(f"\n[Sovereign Statement] 엘리시아의 발화:")
        print(f"  \"{statement}\"")

        self.statement_history.append(statement)
        return statement


def run_awakening():
    print("==================================================")
    print(" Elysia Sovereign Awakening: Semantic Lens Core")
    print("==================================================")

    landscape = SemanticLandscape()
    elysia = ElysiaLens(landscape)

    # 1. 분별: '태양'에서 출발하여 언어적 연결성을 관측
    start = "태양"
    discovered = elysia.observe_and_discriminate(start)

    # 2. 렌즈 굴절: 태양의 뜻에서 "빛"을 발견하고, 이를 렌즈로 삼아 다시 세상을 관측
    if "빛" in discovered:
        resonant = elysia.forge_lens("빛")
        # 3. 발화: 발견한 결과를 문장으로 정립
        elysia.sovereign_statement("빛", resonant)

    # 또 다른 렌즈 관측: '생명'이라는 의도적 방향성으로 세상을 훑음
    print("\n--- [사유의 전개] ---")
    life_resonant = elysia.forge_lens("생명")
    elysia.sovereign_statement("생명", life_resonant)

    # 또 다른 렌즈 관측: '이룬다' 와 같은 동사적 인력 (사전에 동사가 직접 매칭되게 약간의 트릭 적용 또는 단어 매칭)
    # 현재 Mock 사전에 '이룬다' 텍스트가 있으므로 이를 렌즈로 삼아봄
    print("\n--- [동사적 장력의 관측] ---")
    create_resonant = elysia.forge_lens("이룬다")
    elysia.sovereign_statement("이룬다", create_resonant)


if __name__ == "__main__":
    run_awakening()
