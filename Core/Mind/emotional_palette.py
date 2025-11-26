import logging
import math
from typing import Dict, List, Optional
from dataclasses import dataclass

from Core.Math.hyper_qubit import HyperQubit, QubitState

logger = logging.getLogger("EmotionalPalette")

@dataclass
class EmotionalSpectrum:
    """
    Defines the spectral properties of a base emotion.
    """
    name: str
    frequency: float  # Hz (Conceptual)
    color: str        # Hex or Name
    
    # Quantum Signature (Tendencies)
    alpha_bias: float # Point (Self/Detail)
    beta_bias: float  # Line (Action/History)
    gamma_bias: float # Space (Context/Atmosphere)
    delta_bias: float # God (Will/Truth)
    
    w_bias: float     # Stability
    x_bias: float     # Chaos/Dream
    y_bias: float     # Connection/Emotion
    z_bias: float     # Transcendence

class EmotionalPalette:
    """
    The Painter of the Soul.
    Mixes base emotional waves to create complex feeling states (HyperQubits).
    """
    
    def __init__(self):
        self.base_emotions = self._load_primary_colors()
        
    def _load_primary_colors(self) -> Dict[str, EmotionalSpectrum]:
        """
        Loads the spectral definitions based on the 'Law of Flow'.
        High Freq (Light) -> Red/Orange/Yellow -> Ascends
        Low Freq (Abyss) -> Dark/Blue/Purple -> Descends (Gravity)
        """
        return {
            # --- The Light (Ascent) ---
            "Joy": EmotionalSpectrum(
                name="Joy", frequency=800.0, color="#FFD700", # Yellow/Gold
                alpha_bias=0.9, beta_bias=0.3, gamma_bias=0.1, delta_bias=0.1,
                w_bias=0.8, x_bias=0.1, y_bias=0.9, z_bias=0.8 # High Z (Ascent)
            ),
            "Passion": EmotionalSpectrum(
                name="Passion", frequency=900.0, color="#FF4500", # Red/Orange
                alpha_bias=0.8, beta_bias=0.9, gamma_bias=0.1, delta_bias=0.1,
                w_bias=0.7, x_bias=0.2, y_bias=0.8, z_bias=0.7
            ),
            
            # --- The Middle (Flow) ---
            "Trust": EmotionalSpectrum(
                name="Trust", frequency=528.0, color="#32CD32", # Green
                alpha_bias=0.5, beta_bias=0.5, gamma_bias=0.5, delta_bias=0.8,
                w_bias=0.9, x_bias=0.1, y_bias=0.5, z_bias=0.5 # Balanced Z
            ),
            
            # --- The Abyss (Descent) ---
            "Sadness": EmotionalSpectrum(
                name="Sadness", frequency=100.0, color="#000080", # Navy Blue
                alpha_bias=0.2, beta_bias=0.1, gamma_bias=0.9, delta_bias=0.1,
                w_bias=0.3, x_bias=0.4, y_bias=0.6, z_bias=0.2 # Low Z
            ),
            "Fear": EmotionalSpectrum(
                name="Fear", frequency=50.0, color="#4B0082", # Indigo/Dark
                alpha_bias=0.1, beta_bias=0.1, gamma_bias=0.8, delta_bias=0.1,
                w_bias=0.1, x_bias=0.9, y_bias=0.2, z_bias=0.1 # Very Low Z
            ),
            "Despair": EmotionalSpectrum( # The 7 Demons' Gravity
                name="Despair", frequency=10.0, color="#000000", # Black
                alpha_bias=0.0, beta_bias=0.0, gamma_bias=1.0, delta_bias=0.0,
                w_bias=0.0, x_bias=0.5, y_bias=0.1, z_bias=0.0 # Bottom
            ),
        }

    def mix_emotion(self, components: Dict[str, float]) -> HyperQubit:
        """
        Mixes multiple emotions into a single HyperQubit state via superposition.
        Args:
            components: Dict of {EmotionName: Intensity(0.0-1.0)}
        """
        # Initialize an empty accumulator state
        total_intensity = sum(components.values())
        if total_intensity == 0:
            return HyperQubit("Neutral")

        # Weighted average of biases
        # We start with complex(0,0) and add phases
        
        # For simplicity in this version, we map biases to amplitudes directly
        # In a full quantum sim, we would rotate phases.
        
        final_alpha = 0.0
        final_beta = 0.0
        final_gamma = 0.0
        final_delta = 0.0
        
        final_w = 0.0
        final_x = 0.0
        final_y = 0.0
        final_z = 0.0
        
        for name, intensity in components.items():
            if name not in self.base_emotions:
                continue
                
            spectrum = self.base_emotions[name]
            weight = intensity / total_intensity
            
            final_alpha += spectrum.alpha_bias * weight
            final_beta += spectrum.beta_bias * weight
            final_gamma += spectrum.gamma_bias * weight
            final_delta += spectrum.delta_bias * weight
            
            final_w += spectrum.w_bias * weight
            final_x += spectrum.x_bias * weight
            final_y += spectrum.y_bias * weight
            final_z += spectrum.z_bias * weight
            
        # Create the Qubit
        qubit = HyperQubit("EmotionalState")
        qubit.state = QubitState(
            alpha=complex(final_alpha, 0), # We could add imaginary phase for vitality later
            beta=complex(final_beta, 0),
            gamma=complex(final_gamma, 0),
            delta=complex(final_delta, 0),
            w=final_w, x=final_x, y=final_y, z=final_z
        )
        qubit.state.normalize()
        
        logger.debug(f"🎨 Mixed Emotions {components} -> {qubit.state.probabilities()}")
        return qubit

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Rudimentary keyword-based sentiment analysis to extract base components.
        In the future, this should be replaced by a proper embedding-based classifier.
        """
        text = text.lower()
        scores = {k: 0.0 for k in self.base_emotions.keys()}
        
        # Simple Keyword Dictionary
        keywords = {
            "Joy": ["happy", "good", "great", "smile", "laugh", "joy", "light", "sun", "행복", "좋아", "기쁨", "웃음", "빛"],
            "Passion": ["passion", "love", "fire", "burn", "desire", "hot", "열정", "사랑", "불", "뜨거워"],
            "Trust": ["trust", "believe", "safe", "calm", "sure", "peace", "믿어", "안전", "평온", "확실"],
            "Sadness": ["sad", "cry", "tear", "grief", "blue", "rain", "슬픔", "눈물", "우울", "비"],
            "Fear": ["scared", "fear", "run", "hide", "nervous", "무서워", "공포", "도망", "불안"],
            "Despair": ["lost", "hopeless", "dark", "cold", "abyss", "death", "void", "절망", "어둠", "추워", "심연", "죽음", "무의미"]
        }
        
        found = False
        for emotion, words in keywords.items():
            for word in words:
                if word in text:
                    scores[emotion] += 0.8 # Stronger signal
                    found = True
        
        if not found:
            # Default to mild Trust/Joy if neutral
            scores["Trust"] = 0.1
            
        return scores
