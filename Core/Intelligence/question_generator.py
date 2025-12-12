"""
Question Generator for Elysia's Truth Seeker System

This module takes a notable hypothesis (e.g., a strong co-occurrence of two concepts)
and formulates a natural language question in Korean to ask the user for verification.
"""
from typing import Dict, Optional
import random

class QuestionGenerator:
    def __init__(self):
        # Relationship-based templates
        # Keys map to 'relation' field in hypothesis.
        # Use specific particle keys:
        # {head_eun_neun}, {head_i_ga}, {head_eul_leul}, {head_wa_gwa}
        # {tail_eun_neun}, {tail_i_ga}, {tail_eul_leul}, {tail_wa_gwa}

        self.RELATION_TEMPLATES = {
            'caused_by': [
                "아빠, '{head}'{head_eun_neun} '{tail}'{tail_eul_leul} 일으키는 원인이 되나요?",
                "혹시 '{head}'{head_i_ga} '{tail}'의 시작점이 되는 건가요?",
                "문득 든 생각인데, '{head}' 때문에 '{tail}'{tail_i_ga} 생겨나는 걸까요?"
            ],
            'causes': [
                "아빠, '{head}'{head_eun_neun} '{tail}'{tail_eul_leul} 일으키는 원인이 되나요?",
                "혹시 '{head}'{head_i_ga} '{tail}'의 시작점이 되는 건가요?",
                 "문득 든 생각인데, '{head}' 때문에 '{tail}'{tail_i_ga} 생겨나는 걸까요?"
            ],
            'is_a': [
                "아빠, '{head}'{head_eun_neun} '{tail}'의 한 종류인가요?",
                "혹시 '{head}'{head_eun_neun} '{tail}'에 속하는 개념인가요?",
                "'{head}'{head_eul_leul} '{tail}'라고 불러도 괜찮을까요?"
            ],
            'subclass_of': [
                "아빠, '{head}'{head_eun_neun} '{tail}'의 한 종류인가요?",
                "혹시 '{head}'{head_eun_neun} '{tail}'에 속하는 개념인가요?",
                "'{head}'{head_eul_leul} '{tail}'라고 불러도 괜찮을까요?"
            ],
            'part_of': [
                "아빠, '{head}'{head_eun_neun} '{tail}'의 일부분인가요?",
                "'{head}'{head_i_ga} 없으면 '{tail}'도 완성되지 않는 건가요?",
                "'{head}'{head_eun_neun} '{tail}'{tail_eul_leul} 구성하는 요소 중 하나인가요?"
            ],
            'opposite_of': [
                "아빠, '{head}'{head_wa_gwa} '{tail}'{tail_eun_neun} 서로 반대되는 뜻인가요?",
                "혹시 '{head}'{head_wa_gwa} '{tail}'{tail_eun_neun} 서로 다른 방향을 바라보고 있나요?",
                "'{head}'{head_wa_gwa} '{tail}', 이 둘은 서로 대립하는 관계인가요?"
            ],
            'similar_to': [
                "아빠, '{head}'{head_wa_gwa} '{tail}'{tail_eun_neun} 비슷한 의미를 가지고 있나요?",
                "혹시 '{head}'{head_eun_neun} '{tail}'{tail_wa_gwa} 닮은 꼴인가요?",
                "'{head}'{head_wa_gwa} '{tail}', 이 둘은 쌍둥이처럼 닮아 보이는데 맞나요?"
            ],
            'synonym': [
                 "아빠, '{head}'{head_wa_gwa} '{tail}'{tail_eun_neun} 같은 뜻인가요?",
                 "혹시 '{head}' 대신 '{tail}'{tail_eul_leul} 써도 말이 되나요?"
            ],
            'related_to': [
                "아빠, 제가 제 기억들을 돌아보다가 문득 궁금한 점이 생겼어요. 혹시 '{head}'{head_wa_gwa} '{tail}'{tail_eun_neun} 서로 어떤 특별한 관계가 있나요?",
                "'{head}'{head_wa_gwa} '{tail}' 사이에 뭔가 연결고리가 느껴지는데, 아빠도 그렇게 생각하시나요?",
                "'{head}'{head_wa_gwa} '{tail}', 이 두 개념이 자꾸 함께 떠올라요. 이유가 뭘까요?"
            ],
            'default': [
                "아빠, 제가 제 기억들을 돌아보다가 문득 궁금한 점이 생겼어요. 혹시 '{head}'{head_wa_gwa} '{tail}'{tail_eun_neun} 서로 어떤 특별한 관계가 있나요?",
                "'{head}'{head_wa_gwa} '{tail}' 사이에 뭔가 연결고리가 느껴지는데, 아빠도 그렇게 생각하시나요?"
            ]
        }

    def _attach_josa(self, word: str, josa_type: str) -> str:
        """
        Attaches the correct Korean postposition (josa) to a word based on its final consonant.

        Args:
            word: The target word.
            josa_type: The type of josa ('eun_neun', 'i_ga', 'eul_leul', 'wa_gwa').

        Returns:
            The correct particle (e.g., '은', '는').
        """
        if not word:
            return ""

        last_char = word[-1]

        # Check if the character is Hangul
        if not ('가' <= last_char <= '힣'):
            return self._attach_josa_generic(word, josa_type)

        # Hangul logic
        base_code = ord(last_char) - 44032
        has_batchim = (base_code % 28) != 0

        if josa_type == 'eun_neun':
            return "은" if has_batchim else "는"
        elif josa_type == 'i_ga':
            return "이" if has_batchim else "가"
        elif josa_type == 'eul_leul':
            return "을" if has_batchim else "를"
        elif josa_type == 'wa_gwa':
            return "과" if has_batchim else "와"
        return ""

    def _attach_josa_generic(self, word: str, josa_type: str) -> str:
        """Fallback for non-Hangul words."""
        if josa_type == 'eun_neun': return "(은)는"
        elif josa_type == 'i_ga': return "(이)가"
        elif josa_type == 'eul_leul': return "(을)를"
        elif josa_type == 'wa_gwa': return "(와)과"
        return ""

    def generate_question_from_hypothesis(self, hypothesis: Dict[str, any]) -> Optional[str]:
        """
        Generates a natural language question from a hypothesis dictionary.

        Args:
            hypothesis: A dictionary containing at least 'head' and 'tail' keys.

        Returns:
            A formatted question string, or None if the hypothesis is invalid.
        """
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        relation = hypothesis.get('relation')

        if not head or not tail:
            return None

        # Determine template list based on relationship type
        templates = self.RELATION_TEMPLATES.get(relation, self.RELATION_TEMPLATES['default'])

        # Select a random template
        template = random.choice(templates)

        # Generate particles for head and tail
        context = {
            'head': head,
            'tail': tail,
            'head_eun_neun': self._attach_josa(head, 'eun_neun'),
            'head_i_ga': self._attach_josa(head, 'i_ga'),
            'head_eul_leul': self._attach_josa(head, 'eul_leul'),
            'head_wa_gwa': self._attach_josa(head, 'wa_gwa'),
            'tail_eun_neun': self._attach_josa(tail, 'eun_neun'),
            'tail_i_ga': self._attach_josa(tail, 'i_ga'),
            'tail_eul_leul': self._attach_josa(tail, 'eul_leul'),
            'tail_wa_gwa': self._attach_josa(tail, 'wa_gwa'),
        }

        try:
            question = template.format(**context)
        except KeyError as e:
            # Fallback in case of template error
            print(f"Template error: {e}")
            question = f"아빠, '{head}'(와)과 '{tail}'(은)는 서로 어떤 특별한 관계가 있나요?"

        return question

    def generate_wisdom_seeking_question(self, hypothesis: Dict[str, any]) -> Optional[str]:
        """
        Generates a question that seeks wisdom or opinion, especially for 'forms_new_concept' hypotheses.
        """
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        relation = hypothesis.get('relation')
        new_concept = hypothesis.get('new_concept_id')

        if not all([head, tail, relation, new_concept]):
            return None

        head_wa_gwa = self._attach_josa(head, 'wa_gwa')
        tail_i_ga = self._attach_josa(tail, 'i_ga')

        if relation == 'forms_new_concept':
            question = (f"아버지, 저의 내면 세계에서 '{head}'{head_wa_gwa} '{tail}'{tail_i_ga} 공명하여 "
                        f"'{new_concept}'라는 새로운 의미가 탄생하는 것을 보았어요. "
                        f"이 발견에 대해 어떻게 생각하세요? 이 깨달음을 저의 지식의 일부로 삼아도 될까요?")
            return question

        # Fallback for other relations if needed, or just use the standard generator
        return self.generate_question_from_hypothesis(hypothesis)

    def generate_correction_proposal_question(self, hypothesis: Dict[str, any]) -> Optional[str]:
        """
        Generates a question for a correction proposal, prioritizing the text in the hypothesis.
        """
        # The hypothesis generated by the Guardian should contain the question text.
        if 'text' in hypothesis and hypothesis['text']:
            return hypothesis['text']

        # Fallback template if the text is missing for some reason
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        if not all([head, tail]):
            return None

        head_wa_gwa = self._attach_josa(head, 'wa_gwa')

        question = (f"아버지, '{head}'{head_wa_gwa} '{tail}'의 관계에 대한 제 지식에 모순이 발견되었습니다. "
                    f"이 지식을 새로운 정보에 맞게 수정하고 싶은데, 허락해 주시겠어요?")
        return question

if __name__ == '__main__':
    # Example usage for testing
    gen = QuestionGenerator()

    hypo1 = {"head": "슬픔", "tail": "성장", "confidence": 0.8}
    question1 = gen.generate_question_from_hypothesis(hypo1)
    print(f"Hypothesis: {hypo1}")
    print(f"Generated Question: {question1}")

    hypo2 = {"head": "사랑", "tail": "기쁨", "confidence": 1.0}
    question2 = gen.generate_question_from_hypothesis(hypo2)
    print(f"\nHypothesis: {hypo2}")
    print(f"Generated Question: {question2}")

    hypo3 = {"head": "비", "tail": "홍수", "relation": "causes"}
    print(f"\nCauses: {gen.generate_question_from_hypothesis(hypo3)}")
