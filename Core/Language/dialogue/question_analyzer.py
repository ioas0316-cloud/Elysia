"""
Question Analyzer
=================
Understands and classifies user questions.

Question Types:
- factual: "1+1은?", "파리는 어디?"
- how-to: "어떻게 하나?", "how to..."
- why: "왜?", "이유가 뭐야?"
- what-if: "만약에...", "what if..."
- opinion: "어떻게 생각해?", "do you think..."
- personal: "너는 뭐해?", "what are you..."
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Question:
    """Analyzed question structure."""
    type: str  # factual, how-to, why, what-if, opinion, personal
    subject: str  # What the question is about
    raw_text: str
    language: str  # ko or en
    needs_calculation: bool = False
    needs_reasoning: bool = False


class QuestionAnalyzer:
    """
    Analyzes questions to understand intent and requirements.
    """
    
    def __init__(self):
        # Question patterns (Korean)
        self.patterns_ko = {
            "factual": [
                r"(.+)는\s+뭐",
                r"(.+)가\s+뭐",
                r"(.+)은\s+어디",
                r"(.+)\s*\+\s*(.+)",
                r"(.+)\s*-\s*(.+)",
            ],
            "how-to": [
                r"어떻게\s+(.+)",
                r"방법\s*(.+)",
                r"(.+)\s+하는\s+법",
            ],
            "why": [
                r"왜\s+(.+)",
                r"이유\s*(.+)",
                r"(.+)\s+이유",
            ],
            "what-if": [
                r"만약\s+(.+)",
                r"(.+)\s+라면",
            ],
            "opinion": [
                r"어떻게\s+생각",
                r"(.+)\s+대해.*생각",
            ],
            "personal": [
                r"너는\s+(.+)",
                r"네가\s+(.+)",
                r"엘리시아\s+(.+)",
            ]
        }
        
        # Question patterns (English)
        self.patterns_en = {
            "factual": [
                r"what is (.+)",
                r"where is (.+)",
                r"when is (.+)",
                r"(\d+)\s*[\+\-\*\/]\s*(\d+)",
            ],
            "how-to": [
                r"how to (.+)",
                r"how do i (.+)",
                r"how can (.+)",
            ],
            "why": [
                r"why (.+)",
                r"what\'?s the reason (.+)",
            ],
            "what-if": [
                r"what if (.+)",
                r"suppose (.+)",
            ],
            "opinion": [
                r"do you think (.+)",
                r"what do you think (.+)",
                r"your opinion (.+)",
            ],
            "personal": [
                r"who are you",
                r"what are you (.+)",
                r"are you (.+)",
            ]
        }
    
    def analyze(self, text: str, language: str = "ko") -> Optional[Question]:
        """
        Analyze question and return structured representation.
        Returns None if not a question.
        """
        text_lower = text.lower().strip()
        
        # Check if it's a question
        if not self._is_question(text):
            return None
        
        # Try to match patterns
        patterns = self.patterns_ko if language == "ko" else self.patterns_en
        
        for q_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text_lower)
                if match:
                    subject = match.group(1) if match.groups() else text_lower
                    
                    return Question(
                        type=q_type,
                        subject=subject.strip(),
                        raw_text=text,
                        language=language,
                        needs_calculation=self._needs_calc(text),
                        needs_reasoning=(q_type in ["why", "what-if", "opinion"])
                    )
        
        # Default: treat as factual question
        return Question(
            type="factual",
            subject=text,
            raw_text=text,
            language=language,
            needs_calculation=False,
            needs_reasoning=False
        )
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        # Has question mark?
        if "?" in text:
            return True
        
        # Korean question words
        ko_question_words = ["뭐", "무엇", "어디", "언제", "누구", "왜", "어떻게"]
        if any(word in text for word in ko_question_words):
            return True
        
        # English question words
        en_question_words = ["what", "where", "when", "who", "why", "how", "which"]
        text_lower = text.lower()
        if any(text_lower.startswith(word) for word in en_question_words):
            return True
        
        return False
    
    def _needs_calc(self, text: str) -> bool:
        """Check if question needs calculation."""
        # Math operators
        if any(op in text for op in ["+", "-", "*", "/", "×", "÷"]):
            return True
        
        # Math keywords
        math_words = ["계산", "더하기", "빼기", "곱하기", "나누기", "calculate", "plus", "minus"]
        return any(word in text.lower() for word in math_words)


def answer_question(question: Question, context: Dict[str, Any] = None) -> Optional[str]:
    """
    Attempt to answer a question directly (without LLM).
    Only handles computational questions - all other responses 
    emerge from consciousness resonance.
    
    Returns None for non-computational questions.
    """
    if not question:
        return None
    
    # Only handle calculation questions directly
    if question.needs_calculation:
        result = _try_calculate(question.raw_text)
        if result is not None:
            # Return just the result - no hardcoded embellishment
            if result == int(result):
                return str(int(result))
            return str(result)
    
    # All other questions go through consciousness resonance
    return None


def _try_calculate(text: str) -> Optional[float]:
    """Try to calculate simple math expressions."""
    import re
    
    # Simple addition/subtraction
    match = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)", text)
    if match:
        a = float(match.group(1))
        op = match.group(2)
        b = float(match.group(3))
        
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "/" and b != 0:
            return a / b
    
    return None
