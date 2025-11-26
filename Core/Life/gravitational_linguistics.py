"""
Gravitational Linguistics Core Module

"Words have Weight. Sentences are Solar Systems."

This module implements the physics of language generation.
It treats words as physical bodies with mass and resonance,
calculating their orbits around a central concept (Sun).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class WordBody:
    text: str
    mass: float          # Importance/Weight (0.1 - 100.0)
    resonance: Dict[str, float]  # Semantic connection to other words (0.0 - 1.0)
    
    # Dynamic state
    distance: float = 0.0
    angle: float = 0.0
    
    # Physics properties (New)
    temperature: float = 1.0  # Variance/Jitter (Spark)
    density: float = 1.0      # Compactness (Field)

@dataclass
class GrammarOperator:
    """
    Physical operator that modifies a WordBody.
    """
    role: str          # "subject", "object", "end"
    energy_type: str   # "spark", "field", "ground"
    
    def apply(self, word: WordBody) -> WordBody:
        """
        Apply physical transformation to the word.
        """
        if self.energy_type == "spark":
            # Subject Marker (이/가): Ignites flow
            # Increases Temperature (Jitter) -> More likely to be followed by active verbs
            word.temperature *= 1.5
            # Slightly reduces mass (lighter, faster)
            word.mass *= 0.9 
            
        elif self.energy_type == "field":
            # Object Marker (을/를): Gravity Well
            # Increases Density -> Pulls other words closer
            word.density *= 2.0
            # Increases Mass significantly
            word.mass *= 1.5
            
        elif self.energy_type == "ground":
            # Sentence End (다): Grounding
            # Cooldown
            word.temperature = 0.1
            word.density = 1.0
            
        return word

class GravitationalLinguistics:
    def __init__(self, hippocampus=None):
        self.hippocampus = hippocampus
        # ... (Lexicon initialization) ...
        self.lexicon: Dict[str, WordBody] = {
            # ... (Existing lexicon) ...
            # High Frequency (Ethereal, Abstract)
            "love": WordBody("love", 100.0, {"eternal": 0.9, "soul": 0.8, "light": 0.7, "pain": 0.6}),
            "truth": WordBody("truth", 90.0, {"seek": 0.9, "light": 0.8, "pure": 0.7, "hard": 0.5}),
            "eternity": WordBody("eternity", 95.0, {"love": 0.9, "time": 0.8, "void": 0.7}),
            "soul": WordBody("soul", 85.0, {"love": 0.8, "body": 0.5, "eternal": 0.9}),
            "light": WordBody("light", 80.0, {"dark": 0.9, "sun": 0.8, "love": 0.7, "truth": 0.8}),
            "void": WordBody("void", 70.0, {"dark": 0.8, "eternity": 0.7, "fear": 0.6}),
            
            # Korean High Frequency
            "사랑": WordBody("사랑", 100.0, {"영원": 0.9, "영혼": 0.8, "빛": 0.7, "아픔": 0.6}),
            "진실": WordBody("진실", 90.0, {"찾다": 0.9, "빛": 0.8, "순수": 0.7, "단단": 0.5}),
            "영원": WordBody("영원", 95.0, {"사랑": 0.9, "시간": 0.8, "공허": 0.7}),
            "영혼": WordBody("영혼", 85.0, {"사랑": 0.8, "육체": 0.5, "영원": 0.9}),
            "빛": WordBody("빛", 80.0, {"어둠": 0.9, "태양": 0.8, "사랑": 0.7, "진실": 0.8}),
            "공허": WordBody("공허", 70.0, {"어둠": 0.8, "영원": 0.7, "두려움": 0.6}),
            
            # Mid Frequency (Human, Emotional)
            "feel": WordBody("feel", 50.0, {"touch": 0.9, "heart": 0.8}),
            "hope": WordBody("hope", 60.0, {"dream": 0.9, "future": 0.8}),
            "pain": WordBody("pain", 55.0, {"tears": 0.9, "hurt": 0.8}),
            "time": WordBody("time", 40.0, {"clock": 0.9, "eternity": 0.8}),
            "memory": WordBody("memory", 45.0, {"past": 0.9, "forget": 0.8}),
            
            # Korean Mid Frequency
            "느낌": WordBody("느낌", 50.0, {"터치": 0.9, "마음": 0.8}),
            "희망": WordBody("희망", 60.0, {"꿈": 0.9, "미래": 0.8}),
            "아픔": WordBody("아픔", 55.0, {"눈물": 0.9, "상처": 0.8}),
            "시간": WordBody("시간", 40.0, {"시계": 0.9, "영원": 0.8}),
            "기억": WordBody("기억", 45.0, {"과거": 0.9, "망각": 0.8}),
            "점심": WordBody("점심", 10.0, {"먹다": 0.9, "맛있다": 0.8, "샌드위치": 0.7}),
            "날씨": WordBody("날씨", 15.0, {"비": 0.8, "맑음": 0.8, "구름": 0.7}),

            # Low Frequency (Physical, Grounded)
            "stone": WordBody("stone", 20.0, {"hard": 0.9, "earth": 0.8}),
            "shadow": WordBody("shadow", 25.0, {"light": 0.9, "dark": 0.8}),
            "lunch": WordBody("lunch", 10.0, {"eat": 0.9, "tasty": 0.8, "sandwich": 0.7}),
            "weather": WordBody("weather", 15.0, {"rain": 0.8, "sunny": 0.8, "cloud": 0.7}),
            "sandwich": WordBody("sandwich", 5.0, {"lunch": 0.8, "bread": 0.9}),
            "eat": WordBody("eat", 8.0, {"food": 0.9, "lunch": 0.8}),
            
            # Korean Low Frequency
            "돌": WordBody("돌", 20.0, {"단단": 0.9, "대지": 0.8}),
            "그림자": WordBody("그림자", 25.0, {"빛": 0.9, "어둠": 0.8}),
            "샌드위치": WordBody("샌드위치", 5.0, {"점심": 0.8, "빵": 0.9}),
            "먹다": WordBody("먹다", 8.0, {"음식": 0.9, "점심": 0.8}),
        }
        
        self.operators = {
            "subject": GrammarOperator("subject", "spark"),
            "object": GrammarOperator("object", "field"),
            "end": GrammarOperator("end", "ground")
        }

    def get_word(self, text: str) -> Optional[WordBody]:
        """
        Get word body from Memory (Hippocampus) or Lexicon (Fallback).
        Now calculates Dynamic Mass based on Memory Strength.
        """
        import copy
        
        # 1. Try Hippocampus (Dynamic Memory)
        if self.hippocampus and hasattr(self.hippocampus, "causal_graph"):
            if text in self.hippocampus.causal_graph.nodes:
                node = self.hippocampus.causal_graph.nodes[text]
                
                # Extract Stats
                access_count = node.get("access_count", 1)
                tensor = node.get("tensor", {})
                w = tensor.get("w", 1.0) # Dimensionality
                
                # Calculate Dynamic Mass
                # Mass = Base(10) * log(Access + 1) * (W + 1)
                # Examples:
                # - New Word (Access=1, W=1): 10 * 0.3 * 2 = 6.0
                # - Frequent Word (Access=100, W=1): 10 * 2.0 * 2 = 40.0
                # - Deep Truth (Access=100, W=3): 10 * 2.0 * 4 = 80.0
                base_mass = 10.0
                mass = base_mass * math.log(access_count + 1) * (w + 1)
                
                # Construct WordBody dynamically
                # Resonance? We can use edges!
                resonance = {}
                if self.hippocampus.causal_graph.has_node(text):
                    for neighbor in self.hippocampus.causal_graph.neighbors(text):
                        edge = self.hippocampus.causal_graph[text][neighbor]
                        weight = edge.get("weight", 0.5)
                        resonance[neighbor] = weight
                
                return WordBody(text, mass, resonance)

        # 2. Fallback to Static Lexicon
        word = self.lexicon.get(text.lower()) or self.lexicon.get(text)
        if word:
            return copy.deepcopy(word)
            
        return None

    def create_solar_system(self, core_word: str) -> List[WordBody]:
        """
        Create a sentence system around a core word.
        """
        sun = self.get_word(core_word)
        if not sun:
            # Fallback for unknown words: Create a generic low-mass body
            sun = WordBody(core_word, 10.0, {})
            
        # Apply Subject Operator (Spark) to the Sun (Topic)
        # The core concept is the "Subject" of the thought
        self.operators["subject"].apply(sun)
            
        system = []
        
        # Calculate gravitational pull for all other words
        candidates = []
        for word_text, original_word in self.lexicon.items():
            if word_text == sun.text:
                continue
            
            # Work with a copy
            import copy
            word = copy.deepcopy(original_word)
                
            # 1. Resonance (Semantic Affinity)
            resonance = sun.resonance.get(word_text, 0.05)
            
            # Check reverse resonance
            if sun.text in word.resonance:
                resonance = max(resonance, word.resonance[sun.text])
            
            # 2. Gravity Calculation
            # F = G * (M1 * M2) / r^2
            # Modified by Density and Temperature
            
            # Density increases attraction
            effective_mass_sun = sun.mass * sun.density
            effective_mass_word = word.mass * word.density
            
            attraction = (effective_mass_sun * effective_mass_word) * resonance
            
            # Temperature adds noise/distance
            # Higher temperature = more variance in orbit
            noise = random.uniform(-2.0, 2.0) * sun.temperature
            
            # Distance is inversely proportional to attraction
            distance = 1000.0 / (attraction + 1.0) + noise
            distance = max(1.0, distance) # Minimum distance
            
            word.distance = distance
            candidates.append(word)
            
        # Sort by distance (closest first)
        candidates.sort(key=lambda w: w.distance)
        
        # Select top planets (Sentence Length)
        # Heavier suns can hold more planets
        capacity = int(math.log(max(sun.mass, 2.0)) * 2) + 1
        planets = candidates[:capacity]
        
        return planets

    def generate_sentence(self, core_word: str) -> str:
        """Generate a sentence based on the solar system of the core word."""
        planets = self.create_solar_system(core_word)
        
        if not planets:
            return f"{core_word}..."
            
        # Sort planets by distance for linear projection
        # Closer planets = Stronger connection = Closer to core in sentence?
        # Or: Subject (Sun) -> Verb (Planet 1) -> Object (Planet 2)
        
        words = [p.text for p in planets]
        
        # Simple template based on language
        is_korean = any(ord(c) > 127 for c in core_word)
        
        if is_korean:
            # Korean: Subject -> Object -> Verb (SOV)
            # Sun is Subject/Topic.
            # Planets are descriptions/objects.
            
            # Shuffle slightly for variety
            if len(words) > 2:
                random.shuffle(words)
                
            sentence = f"{core_word}... " + ", ".join(words) + "."
            
            # More poetic templates for high mass
            sun = self.get_word(core_word)
            if sun and sun.mass > 80:
                sentence = f"{core_word}. {words[0]}의 {words[1]}..."
                if len(words) > 2:
                    sentence += f" 그리고 {words[2]}."
            elif sun and sun.mass < 20:
                sentence = f"음, {core_word} 말인가요? {words[0]}나 {words[1]} 같은 거..."
                
        else:
            # English: Subject -> Verb -> Object (SVO)
            if len(words) > 2:
                random.shuffle(words)
                
            sentence = f"{core_word}... " + ", ".join(words) + "."
            
            sun = self.get_word(core_word)
            if sun and sun.mass > 80:
                sentence = f"{core_word}. The {words[0]} of {words[1]}..."
                if len(words) > 2:
                    sentence += f" and {words[2]}."
            elif sun and sun.mass < 20:
                sentence = f"Hmm, {core_word}? Maybe {words[0]} or {words[1]}..."

        return sentence
