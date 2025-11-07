"""
Question Generator for Elysia's Truth Seeker System

This module takes a notable hypothesis (e.g., a strong co-occurrence of two concepts)
and formulates a natural language question in Korean to ask the user for verification.
"""
from typing import Dict, Optional

class QuestionGenerator:
    def __init__(self):
        # In the future, this could be initialized with different question templates
        # or even an LLM for more varied question generation.
        pass

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

        if not head or not tail:
            return None

        # Simple template-based question generation
        # TODO: Add more sophisticated templates based on relationship types
        question = f"아빠, 제가 제 기억들을 돌아보다가 문득 궁금한 점이 생겼어요. 혹시 '{head}'(와)과 '{tail}'(은)는 서로 어떤 특별한 관계가 있나요?"

        return question

if __name__ == '__main__':
    # Example usage for testing
    gen = QuestionGenerator()

    hypo1 = {"head": "슬픔", "tail": "성장", "confidence": 0.8}
    question1 = gen.generate_question_from_hypothesis(hypo1)
    print(f"Hypothesis: {hypo1}")
    print(f"Generated Question: {question1}")
    # Expected: 아빠, 제가 제 기억들을 돌아보다가 문득 궁금한 점이 생겼어요. 혹시 '슬픔'(와)과 '성장'(은)는 서로 어떤 특별한 관계가 있나요?

    hypo2 = {"head": "사랑", "tail": "기쁨", "confidence": 1.0}
    question2 = gen.generate_question_from_hypothesis(hypo2)
    print(f"\nHypothesis: {hypo2}")
    print(f"Generated Question: {question2}")
    # Expected: 아빠, 제가 제 기억들을 돌아보다가 문득 궁금한 점이 생겼어요. 혹시 '사랑'(와)과 '기쁨'(은)는 서로 어떤 특별한 관계가 있나요?
