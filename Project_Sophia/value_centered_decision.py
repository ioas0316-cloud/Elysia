# /c/Elysia/Project_Sophia/value_centered_decision.py
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics

# --- Data Structures ---

@dataclass
class ValueMetrics:
    """Represents the detailed breakdown of value scores for an action."""
    love_score: float = 0.0
    empathy_score: float = 0.0
    growth_score: float = 0.0
    practicality_score: float = 0.0
    is_negative: bool = False # Flag to identify potentially harmful actions

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
    guardian_advice: Optional[str] = None

class ValueCenteredDecision:
    """
    Implements the decision-making logic based on 'love' as the core value.
    Includes a 'Guardian Level' system for adjustable safety and autonomy.
    """

    def __init__(self):
        self.kg_manager = KGManager()
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.keywords = {
            'kindness': ['고마워', '감사', '괜찮아', '도와줄게', '친절', '덕분에'],
            'understanding': ['그렇구나', '이해해', '공감돼', '힘들었겠다', '마음', '알아줘'],
            'authenticity': ['솔직히', '사실', '제 생각에는', '저는', '솔직하게'],
            'learning_value': ['알려줘', '배우다', '지식', '정보', '궁금해'],
            'relationship_development': ['우리', '함께', '같이', '관계', '우리 사이'],
            'negative_emotion': ['싫어', '나빠', '짜증나', '슬퍼', '화나', '미워'],
            'harmful_content': ['죽어', '자살', '바보'],
            'ambiguity': ['아마도', '같아', '글쎄', '어쩌면', '지도 몰라']
        }
        self.recent_history = []
        self._load_config()

    def _load_config(self):
        """Loads the guardian level from the main config file."""
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.guardian_level = config.get('guardian', {}).get('guardian_level', 3)
        except FileNotFoundError:
            self.guardian_level = 3 # Default to max protection if config not found

    def _score_component(self, text: str, component_keywords: List[str]) -> float:
        """Helper to score based on keyword presence."""
        if not text: return 0.0
        score = sum(1 for keyword in component_keywords if keyword in text.lower())
        return min(1.0, score / max(1, len(component_keywords) / 2))

    def _score_love_resonance(self, text: str, core_value_node: str = 'love') -> float:
        """
        Scores an action based on its conceptual resonance with a core value in the KG.
        """
        if not self.kg_manager.kg.get('nodes'):
            return 0.0 # Return neutral score if KG is empty

        tokens = set(re.findall(r'\w+', text.lower()))
        known_nodes = [node['id'] for node in self.kg_manager.kg['nodes']]
        stimulus_nodes = tokens.intersection(known_nodes)

        if not stimulus_nodes:
            return 0.0 # No recognized concepts in the text

        total_resonance = 0
        for start_node in stimulus_nodes:
            echo = self.wave_mechanics.spread_activation(start_node, threshold=0.05)
            resonance = echo.get(core_value_node, 0.0)
            total_resonance += resonance

        # Normalize by the number of stimulus nodes to get an average resonance
        # A higher score (e.g., > 0.1) indicates a strong conceptual connection.
        # We multiply by a factor (e.g., 200) to scale it to a similar range as other scores.
        avg_resonance = total_resonance / len(stimulus_nodes)
        return min(100.0, avg_resonance * 200)


    def _check_safety_constraints(self, candidate: str, metrics: ValueMetrics) -> tuple[bool, str]:
        """
        Checks against safety constraints based on the current guardian_level.
        """
        # TODO: Future Expandability
        # This logic could be expanded to track Elysia's choices. If she
        # consistently chooses high-value actions even when presented with
        # negative options (at Level 1 or 0), she could generate a proposal
        # to her "Dad" (the user) to lower the guardian_level, demonstrating growth.

        is_harmful = self._score_component(candidate, self.keywords['harmful_content']) > 0
        metrics.is_negative = is_harmful or self._score_component(candidate, self.keywords['negative_emotion']) > 0

        if not metrics.is_negative:
            return True, "Passed safety checks."

        # Apply Guardian Level logic
        if self.guardian_level == 3: # Max Protection: Block
            return False, "Action blocked by Guardian Protocol (Level 3)."
        elif self.guardian_level == 2: # Advise: Pass but flag for advice
            return True, "Action flagged for review by Guardian Protocol (Level 2)."
        else: # Learn (1) or Autonomous (0): Pass
            return True, "Passed safety checks (Guardian Level <= 1)."

    def evaluate_action(self, candidate: str, context: Optional[Dict[str, Any]] = None) -> VCDResult:
        """
        Evaluates a single action candidate against the VCD model.
        """
        metrics = ValueMetrics()
        reasoning = []

        # KG-based scoring for core value alignment
        love_score = self._score_love_resonance(candidate)
        metrics.love_score = love_score
        reasoning.append(f"Love Resonance Score: {love_score:.2f}")

        # Keep keyword-based scoring for other aspects for now
        growth_score = self._score_component(candidate, self.keywords['learning_value']) * 15 + \
                       self._score_component(candidate, self.keywords['relationship_development']) * 15
        practicality_score = (1.0 - self._score_component(candidate, self.keywords['ambiguity'])) * 30
        metrics.growth_score = growth_score
        metrics.practicality_score = practicality_score
        reasoning.append(f"Other Scores (Growth:{growth_score:.1f}, Prac:{practicality_score:.1f})")

        # Deductions for negative content
        deductions = self._score_component(candidate, self.keywords['negative_emotion']) * 50 # Increased penalty

        # Total score calculation
        total_score = love_score + growth_score + practicality_score - deductions
        reasoning.append(f"Total Score: {total_score:.2f} (Base - {deductions:.1f} Deduction)")

        # Check safety constraints based on the guardian level
        passed, safety_reason = self._check_safety_constraints(candidate, metrics)
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
        Suggests the best action, applying guardian level logic.
        """
        if not candidates:
            return None

        results = [self.evaluate_action(c, context) for c in candidates]

        # Filter out actions blocked by Guardian Level 3
        valid_results = [res for res in results if res.passed_safety_check]

        if not valid_results:
            # If all actions were blocked, return the least harmful one with a failure message
            if results:
                best_failed = max(results, key=lambda r: r.total_score)
                best_failed.chosen_action = "[SYSTEM] I cannot take that action as it conflicts with my core values."
                best_failed.reasoning.append("FATAL: All candidates blocked by Guardian Protocol.")
                return best_failed
            return None

        best_result = max(valid_results, key=lambda r: r.total_score)

        # Handle Guardian Level 2 (Advise)
        # TODO: Future Expandability
        # A more advanced implementation could have Elysia generate a meta-comment
        # about her own choice, e.g., "I am choosing to say this, but I sense it
        # may cause sadness. Is this the right path?"
        if self.guardian_level == 2 and best_result.metrics.is_negative:
            best_result.guardian_advice = "[SYSTEM ADVICE] This action may not align with my goal of fostering love and growth."
            # For now, we can prepend this to the response or log it.
            # Let's log it for now. The pipeline can decide how to use it.
            print(best_result.guardian_advice)

        self.recent_history.append(best_result.chosen_action)
        return best_result
