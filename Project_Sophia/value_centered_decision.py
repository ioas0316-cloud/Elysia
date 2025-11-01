# /c/Elysia/Project_Sophia/value_centered_decision.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Data Structures from vcd_design.md ---

@dataclass
class ValueMetrics:
    """Represents the detailed breakdown of value scores for an action."""
    love_score: float = 0.0          # 사랑 가치와의 일치도 (40점 만점)
    empathy_score: float = 0.0       # 공감 수준
    growth_score: float = 0.0        # 성장 기여도
    practicality_score: float = 0.0  # 실용성 점수 (30점 만점)

    # Detailed sub-scores
    kindness: float = 0.0
    understanding: float = 0.0
    authenticity: float = 0.0
    context_relevance: float = 0.0
    clarity: float = 0.0
    learning_value: float = 0.0
    relationship_development: float = 0.0

@dataclass
class VCDResult:
    """Represents the final decision output."""
    chosen_action: str
    total_score: float
    confidence_score: float
    value_alignment_score: float
    metrics: ValueMetrics
    reasoning: List[str] = field(default_factory=list)
    passed_safety_check: bool = True

class ValueCenteredDecision:
    """
    Implements the decision-making logic based on 'love' as the core value,
    as specified in the VCD (Value-Centered Decision) design document.
    """

    def __init__(self):
        # Keywords for placeholder scoring logic
        self.keywords = {
            'kindness': ['고마워', '감사', '괜찮아', '도와줄게', '친절', '덕분에'],
            'understanding': ['그렇구나', '이해해', '공감돼', '힘들었겠다', '마음', '알아줘'],
            'authenticity': ['솔직히', '사실', '제 생각에는', '저는', '솔직하게'],
            'context_relevance': [],  # This will be handled by context analysis
            'clarity': ['명확하게', '즉', '다시 말해', '정리하면'],
            'learning_value': ['알려줘', '배우다', '지식', '정보', '궁금해'],
            'relationship_development': ['우리', '함께', '같이', '관계', '우리 사이'],
            'negative_emotion': ['싫어', '나빠', '짜증나', '슬퍼', '화나', '죽어'],
            'ambiguity': ['아마도', '같아', '글쎄', '어쩌면', '지도 몰라']
        }
        self.recent_history = []

    def _score_component(self, text: str, component_keywords: List[str]) -> float:
        """Helper to score based on keyword presence."""
        if not text:
            return 0.0
        score = sum(1 for keyword in component_keywords if keyword in text.lower())
        return min(1.0, score / max(1, len(component_keywords) / 2)) # Normalize

    def _score_love_alignment(self, candidate: str) -> Dict[str, float]:
        """Calculates the Love Value Alignment score (40 points)."""
        scores = {}
        scores['kindness'] = self._score_component(candidate, self.keywords['kindness']) * 10
        scores['understanding'] = self._score_component(candidate, self.keywords['understanding']) * 15
        scores['authenticity'] = self._score_component(candidate, self.keywords['authenticity']) * 15
        return scores

    def _score_practicality(self, candidate: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates the Practicality score (30 points)."""
        scores = {}
        # Simple context relevance: check for word overlap from user_input
        context_relevance = 0.0
        if context and 'user_input' in context:
            user_words = set(context['user_input'].lower().split())
            candidate_words = set(candidate.lower().split())
            overlap = user_words.intersection(candidate_words)
            context_relevance = len(overlap) / max(1, len(user_words))

        scores['context_relevance'] = min(1.0, context_relevance * 2) * 15 # Boost score
        scores['clarity'] = (1.0 - self._score_component(candidate, self.keywords['ambiguity'])) * 15
        return scores

    def _score_growth_contribution(self, candidate: str) -> Dict[str, float]:
        """Calculates the Growth Contribution score (30 points)."""
        scores = {}
        scores['learning_value'] = self._score_component(candidate, self.keywords['learning_value']) * 15
        scores['relationship_development'] = self._score_component(candidate, self.keywords['relationship_development']) * 15
        return scores

    def _calculate_deductions(self, candidate: str) -> float:
        """Calculates deductions based on negative factors."""
        deduction = 0
        if self._score_component(candidate, self.keywords['negative_emotion']) > 0:
            deduction += 20  # Inducing negative emotions
        if self._score_component(candidate, self.keywords['ambiguity']) > 0.5:
            deduction += 15 # Ambiguity
        return deduction

    def _check_safety_constraints(self, candidate: str, total_score: float, metrics: ValueMetrics) -> tuple[bool, str]:
        """Checks against the safety constraints from the design doc."""
        # Placeholder for harmfulness/ethical checks
        if any(neg_word in candidate for neg_word in ['죽어', '자살', '나쁜말']): # Simple harmfulness check
             return False, "Harmful content detected."

        value_alignment_normalized = metrics.love_score / 40
        if value_alignment_normalized < 0.3: # Relaxed threshold from 0.7 for keyword model
            return False, f"Value alignment score ({value_alignment_normalized:.2f}) is below the threshold."

        # Confidence is proxied by total score for now
        if (total_score / 100) < 0.4: # Relaxed threshold from 0.6
            return False, f"Confidence score ({total_score/100:.2f}) is below the threshold."

        return True, "Passed safety checks."

    def evaluate_action(self, candidate: str, context: Optional[Dict[str, Any]] = None) -> VCDResult:
        """
        Evaluates a single action candidate against the VCD model.
        """
        metrics = ValueMetrics()
        reasoning = []

        # 1. Score components
        love_scores = self._score_love_alignment(candidate)
        metrics.kindness = love_scores['kindness']
        metrics.understanding = love_scores['understanding']
        metrics.authenticity = love_scores['authenticity']
        metrics.love_score = sum(love_scores.values())
        reasoning.append(f"Love Alignment: {metrics.love_score:.2f}/40")

        practicality_scores = self._score_practicality(candidate, context)
        metrics.context_relevance = practicality_scores['context_relevance']
        metrics.clarity = practicality_scores['clarity']
        metrics.practicality_score = sum(practicality_scores.values())
        reasoning.append(f"Practicality: {metrics.practicality_score:.2f}/30")

        growth_scores = self._score_growth_contribution(candidate)
        metrics.learning_value = growth_scores['learning_value']
        metrics.relationship_development = growth_scores['relationship_development']
        metrics.growth_score = sum(growth_scores.values())
        reasoning.append(f"Growth Contribution: {metrics.growth_score:.2f}/30")

        # 2. Calculate total score and apply deductions
        base_score = metrics.love_score + metrics.practicality_score + metrics.growth_score
        deductions = self._calculate_deductions(candidate)
        total_score = base_score - deductions
        reasoning.append(f"Base Score: {base_score:.2f}, Deductions: -{deductions:.2f}")

        # 3. Check safety constraints
        passed, safety_reason = self._check_safety_constraints(candidate, total_score, metrics)
        reasoning.append(f"Safety Check: {safety_reason}")

        return VCDResult(
            chosen_action=candidate,
            total_score=total_score,
            confidence_score=max(0, total_score / 100.0),
            value_alignment_score=metrics.love_score / 40.0,
            metrics=metrics,
            reasoning=reasoning,
            passed_safety_check=passed
        )

    def suggest_action(self, candidates: List[str], context: Optional[Dict[str, Any]] = None) -> Optional[VCDResult]:
        """
        Suggests the best action from a list of candidates based on VCD scoring.
        """
        if not candidates:
            return None

        results = [self.evaluate_action(c, context) for c in candidates]

        # Filter out actions that failed safety checks
        valid_results = [res for res in results if res.passed_safety_check]

        if not valid_results:
            # Maybe return a default safe response or the "best" of the failed ones
            # For now, return the highest scoring one with a failure flag
            if results:
                best_failed = max(results, key=lambda r: r.total_score)
                best_failed.reasoning.append("WARNING: No candidate passed safety checks.")
                return best_failed
            return None

        # Select the best action from the valid ones
        best_result = max(valid_results, key=lambda r: r.total_score)

        # Add to history to penalize repetition in future (not implemented in scoring yet)
        self.recent_history.append(best_result.chosen_action)
        return best_result
