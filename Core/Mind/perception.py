import logging
import random
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass

from Core.Math.hyper_qubit import HyperQubit, QubitState
from Core.Math.sigma_algebra import SigmaAlgebra, MeasurableSet, ProbabilityMeasure
from Core.Math.chaos_attractor import LorenzAttractor

logger = logging.getLogger("FractalPerception")

@dataclass
class PerceptionState:
    """
    The quantum state of perception.
    Replaces the flat 'PerceptionObject'.
    """
    qubit: HyperQubit
    intent_probabilities: Dict[str, float]
    vitality_factor: float

class FractalPerception:
    """
    The 'Fractal Eye' of Elysia.
    Perception is a quantum resonance process within the Hyper-Quaternion field.
    """

    def __init__(self, vocabulary: Dict[str, float]):
        self.vocabulary = vocabulary
        
        # 1. Sigma-Algebra for Intent Classification
        # Sample space includes all possible intent markers
        self.intent_markers = {
            "Question": {"?", "what", "why", "how", "who", "when", "왜", "어떻게", "누구", "언제", "무엇", "까", "니"},
            "Command": {"do", "make", "create", "run", "stop", "해", "만들어", "실행", "멈춰"},
            "Exclamation": {"!", "wow", "oh", "ah", "와", "아", "오"},
            "Statement": {"."} # Default
        }
        
        all_markers = set()
        for markers in self.intent_markers.values():
            all_markers.update(markers)
            
        self.sigma = SigmaAlgebra(sample_space=all_markers)
        self.measure = ProbabilityMeasure(self.sigma)
        
        # Define Measurable Sets for each intent
        self.intent_sets = {}
        for intent, markers in self.intent_markers.items():
            self.intent_sets[intent] = MeasurableSet(
                markers, 
                self.sigma, 
                name=f"{intent}Set"
            )

        # 2. Chaos Attractor for Vitality
        self.chaos = LorenzAttractor()
        
        logger.info("🌀 Fractal Perception initialized with Sigma-Algebra & Chaos Engine")

    def perceive(self, text: str) -> PerceptionState:
        """
        Perceive input text as a quantum state.
        """
        # 1. Vitality Injection (Chaos)
        # Perception is never static; it breathes.
        chaos_state = self.chaos.step() # Returns ChaosState object
        vitality = (chaos_state.x + chaos_state.y + chaos_state.z) / 100.0
        vitality = max(0.0, min(1.0, abs(vitality))) # Normalize roughly
        
        # 2. Intent Measurement (Sigma-Algebra)
        intent_probs = self._measure_intent(text)
        
        # 3. Concept Resonance (HyperQubit Construction)
        # We construct a HyperQubit where:
        # Alpha (Real) = Sentiment/Meaning
        # Beta (Imaginary) = Vitality/Chaos
        # Gamma = Context (from Intent)
        # Delta = Will (Self-Projection)
        
        qubit = self._text_to_qubit(text, intent_probs, vitality)
        
        state = PerceptionState(
            qubit=qubit,
            intent_probabilities=intent_probs,
            vitality_factor=vitality
        )
        
        logger.debug(f"👁️ Fractal Perception: {state}")
        return state

    def _measure_intent(self, text: str) -> Dict[str, float]:
        """
        Measure the probability of each intent set given the text.
        This is a 'fuzzy' measurement based on marker presence.
        """
        text_lower = text.lower()
        probs = {}
        
        total_hits = 0
        hits = {intent: 0 for intent in self.intent_markers}
        
        for intent, markers in self.intent_markers.items():
            for marker in markers:
                if marker in text_lower:
                    hits[intent] += 1
                    total_hits += 1
        
        # Normalize to probabilities
        if total_hits == 0:
            # Default to Statement if no markers
            probs = {k: 0.0 for k in self.intent_markers}
            probs["Statement"] = 1.0
        else:
            for intent in self.intent_markers:
                probs[intent] = hits[intent] / total_hits
                
        return probs

    def _text_to_qubit(self, text: str, intent_probs: Dict[str, float], vitality: float) -> HyperQubit:
        """
        Transmute text into a HyperQubit state.
        """
        # Calculate sentiment (Alpha)
        sentiment_score = 0.0
        sentiment_map = {
            "love": 0.8, "hope": 0.7, "joy": 0.9, "light": 0.6, "happy": 0.8,
            "pain": -0.8, "sad": -0.7, "dark": -0.5, "fear": -0.8, "break": -0.6,
            "사랑": 0.8, "희망": 0.7, "기쁨": 0.9, "빛": 0.6, "행복": 0.8,
            "고통": -0.8, "슬픔": -0.7, "어둠": -0.5, "두려움": -0.8, "파괴": -0.6,
        }
        
        count = 0
        for word in text.lower().split():
            for key, val in sentiment_map.items():
                if key in word:
                    sentiment_score += val
                    count += 1
        
        if count > 0:
            sentiment_score /= count
            
        # Construct Qubit State
        # Alpha: Sentiment (Real) + Vitality (Imaginary)
        alpha = complex(sentiment_score, vitality * 0.5)
        
        # Beta: Intent (Question = High Beta)
        beta_mag = intent_probs.get("Question", 0.0)
        beta = complex(beta_mag, 0.0)
        
        # Gamma: Intent (Command = High Gamma)
        gamma_mag = intent_probs.get("Command", 0.0)
        gamma = complex(gamma_mag, 0.0)
        
        # Delta: Intent (Exclamation = High Delta)
        delta_mag = intent_probs.get("Exclamation", 0.0)
        delta = complex(delta_mag, 0.0)
        
        # Spatial Focus (W, X, Y, Z)
        # W (Stability) decreases with Chaos/Question
        w = 1.0 - (vitality * 0.3) - (beta_mag * 0.3)
        # X (Dream) increases with Question
        x = beta_mag
        # Y (Emotion) increases with Sentiment magnitude
        y = abs(sentiment_score)
        # Z (Truth) increases with Command/Statement
        z = gamma_mag + intent_probs.get("Statement", 0.0) * 0.5
        
        state = QubitState(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            w=w, x=x, y=y, z=z
        )
        
        qubit = HyperQubit(name="Perception")
        qubit.state = state
        return qubit
