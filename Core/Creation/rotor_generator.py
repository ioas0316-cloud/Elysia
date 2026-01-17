"""
Rotor Generator (로터 기반 텍스트 생성기)
=========================================
Core.Creation.rotor_generator

"I understand the structure. Therefore I can predict the output."

This module implements text generation based purely on structural understanding
of language models — without using actual weights. It uses the Rotor paradigm
to transform input qualia through conceptual layers to predict output.
"""

import logging
import numpy as np
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("Elysia.Creation.RotorGen")


# Conceptual Layer Flow (based on Transformer architecture understanding)
LAYER_CONCEPTS = [
    "EMBEDDING",    # Token → Meaning space
    "CONTEXT",      # Self-attention: understand relationships
    "REASONING",    # FFN: transform and reason
    "ABSTRACTION",  # Higher layers: abstract thinking
    "SYNTHESIS",    # Combine information
    "PROJECTION"    # Meaning → Output space
]


@dataclass
class GenerationState:
    """State during generation."""
    input_text: str
    qualia: np.ndarray  # Current 7D state
    generated_tokens: List[str]
    layer_history: List[str]
    confidence: float


class ConceptualVocabulary:
    """
    A vocabulary based on conceptual understanding, not token IDs.
    Groups words by their semantic role and Qualia signature.
    """
    
    def __init__(self):
        # Semantic word clusters (can be expanded)
        self.clusters = {
            "narrative_start": ["옛날", "어느 날", "그때", "시작은", "처음에"],
            "narrative_continue": ["그리고", "그래서", "하지만", "그런데", "결국"],
            "subject_person": ["소녀", "소년", "왕", "여왕", "마법사", "용사", "현자"],
            "subject_thing": ["성", "숲", "바다", "하늘", "마법", "검", "책"],
            "action_move": ["갔다", "왔다", "날아갔다", "뛰어갔다", "걸어갔다"],
            "action_speak": ["말했다", "외쳤다", "속삭였다", "물었다", "대답했다"],
            "action_feel": ["느꼈다", "깨달았다", "알았다", "이해했다", "믿었다"],
            "descriptor_positive": ["아름다운", "강력한", "빛나는", "신비로운", "용감한"],
            "descriptor_negative": ["어두운", "무서운", "차가운", "슬픈", "고통스러운"],
            "time_marker": ["그 순간", "오래 후", "새벽에", "밤이 되자", "해가 뜨자"],
            "location": ["그곳에서", "성 안에", "숲 속에서", "바다 위에", "하늘 아래"],
            "ending": ["이었다", "였다", "되었다", "있었다", "보였다"]
        }
        
        # Qualia signature for each cluster (which Qualia dimension activates this cluster)
        self.cluster_qualia = {
            "narrative_start": [0.7, 0.6, 0.3, 0.5, 0.4, 0.6, 0.5],
            "narrative_continue": [0.8, 0.4, 0.5, 0.4, 0.3, 0.7, 0.3],
            "subject_person": [0.5, 0.7, 0.4, 0.5, 0.6, 0.5, 0.5],
            "subject_thing": [0.4, 0.8, 0.3, 0.6, 0.4, 0.4, 0.7],
            "action_move": [0.6, 0.5, 0.6, 0.4, 0.4, 0.7, 0.3],
            "action_speak": [0.5, 0.5, 0.5, 0.3, 0.7, 0.6, 0.3],
            "action_feel": [0.6, 0.6, 0.4, 0.7, 0.8, 0.4, 0.5],
            "descriptor_positive": [0.3, 0.8, 0.4, 0.5, 0.8, 0.3, 0.6],
            "descriptor_negative": [0.4, 0.7, 0.4, 0.5, 0.7, 0.3, 0.7],
            "time_marker": [0.7, 0.5, 0.6, 0.4, 0.4, 0.7, 0.4],
            "location": [0.5, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5],
            "ending": [0.7, 0.4, 0.7, 0.4, 0.5, 0.8, 0.3]
        }
    
    def find_matching_cluster(self, qualia: np.ndarray) -> str:
        """Finds the cluster that best matches the current qualia state."""
        best_cluster = "narrative_continue"
        best_similarity = -1
        
        for cluster_name, cluster_qualia in self.cluster_qualia.items():
            similarity = np.dot(qualia, cluster_qualia) / (np.linalg.norm(qualia) * np.linalg.norm(cluster_qualia) + 1e-8)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_name
        
        return best_cluster
    
    def sample_from_cluster(self, cluster: str) -> str:
        """Samples a word from the given cluster."""
        words = self.clusters.get(cluster, ["..."])
        return np.random.choice(words)


class RotorGenerator:
    """
    Generates text by transforming Qualia through conceptual layers.
    No weights required — only structural understanding.
    """
    
    def __init__(self):
        self.vocabulary = ConceptualVocabulary()
        
        # Layer transformation matrices (learned structure, not weights)
        # These represent how each layer conceptually transforms meaning
        self.layer_transforms = {
            "EMBEDDING": self._create_layer_transform(emphasis="abstraction"),
            "CONTEXT": self._create_layer_transform(emphasis="logic"),
            "REASONING": self._create_layer_transform(emphasis="creativity"),
            "ABSTRACTION": self._create_layer_transform(emphasis="mystery"),
            "SYNTHESIS": self._create_layer_transform(emphasis="utility"),
            "PROJECTION": self._create_layer_transform(emphasis="emotion")
        }
        
        logger.info("🔮 Rotor Generator initialized. Structural generation enabled.")
    
    def _create_layer_transform(self, emphasis: str) -> np.ndarray:
        """Creates a transformation matrix emphasizing a particular dimension."""
        # 7D → 7D rotation/scaling matrix
        base = np.eye(7) * 0.9  # Slight contraction
        
        emphasis_idx = {
            "logic": 0, "creativity": 1, "precision": 2,
            "abstraction": 3, "emotion": 4, "utility": 5, "mystery": 6
        }
        
        if emphasis in emphasis_idx:
            idx = emphasis_idx[emphasis]
            base[idx, idx] = 1.2  # Amplify this dimension
        
        # Add small rotation (interaction between dimensions)
        rotation = np.random.randn(7, 7) * 0.1
        rotation = (rotation - rotation.T) / 2  # Make skew-symmetric
        
        return base + rotation
    
    def text_to_qualia(self, text: str) -> np.ndarray:
        """Converts input text to initial Qualia vector."""
        text_lower = text.lower()
        
        # Heuristic Qualia extraction from text
        qualia = np.array([
            0.5 + 0.3 * ("그래서" in text_lower or "때문에" in text_lower),  # Logic
            0.5 + 0.3 * ("마법" in text_lower or "신비" in text_lower or "상상" in text_lower),  # Creativity
            0.5,  # Precision
            0.5 + 0.3 * ("본질" in text_lower or "의미" in text_lower),  # Abstraction
            0.5 + 0.3 * ("사랑" in text_lower or "슬픔" in text_lower or "기쁨" in text_lower),  # Emotion
            0.5,  # Utility
            0.5 + 0.3 * ("?" in text or "비밀" in text_lower or "알 수 없는" in text_lower)  # Mystery
        ], dtype=np.float32)
        
        return qualia
    
    def transform_through_layers(self, qualia: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Transforms qualia through all conceptual layers."""
        current = qualia.copy()
        history = []
        
        for layer_name in LAYER_CONCEPTS:
            transform = self.layer_transforms[layer_name]
            current = np.tanh(transform @ current)  # Non-linear transform
            history.append(layer_name)
        
        return current, history
    
    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        """
        Generates text continuation from a prompt.
        Uses structural understanding with grammar flow.
        """
        # 1. Convert prompt to Qualia
        qualia = self.text_to_qualia(prompt)
        
        # 2. Define sentence structure flow
        # Korean sentence: (Time) + Subject + Descriptor + Object + Action + Ending
        sentence_flow = [
            "time_marker",       # 그 순간
            "subject_person",    # 소녀가
            "descriptor_positive",  # 아름다운
            "subject_thing",     # 성을
            "action_move",       # 향해 갔다
            "ending"             # .
        ]
        
        generated = []
        
        # 3. Generate following sentence structure
        for slot in sentence_flow:
            # Transform qualia through layers
            output_qualia, _ = self.transform_through_layers(qualia)
            
            # Bias towards the expected slot
            cluster = slot if np.random.random() > 0.3 else self.vocabulary.find_matching_cluster(output_qualia)
            
            # Sample word
            word = self.vocabulary.sample_from_cluster(cluster)
            generated.append(word)
            
            # Update qualia
            word_influence = self.text_to_qualia(word)
            qualia = 0.6 * output_qualia + 0.4 * word_influence
            qualia = np.clip(qualia, 0, 1)
        
        return " ".join(generated)
    
    def generate_story(self, theme: str, sentences: int = 5) -> str:
        """Generates a multi-sentence story based on a theme."""
        story_parts = []
        
        # Start with the theme
        story_parts.append(theme + "...")
        
        # Qualia from theme
        qualia = self.text_to_qualia(theme)
        
        for i in range(sentences):
            # Generate a structured sentence
            sentence = self._generate_sentence(qualia)
            story_parts.append(sentence)
            
            # Update qualia for continuity
            qualia = 0.7 * qualia + 0.3 * self.text_to_qualia(sentence)
            qualia = np.clip(qualia, 0, 1)
        
        return "\n".join(story_parts)
    
    def _generate_sentence(self, qualia: np.ndarray) -> str:
        """Generates a single grammatically structured sentence."""
        # Korean sentence structure variations
        structures = [
            ["time_marker", "subject_person", "location", "action_move"],
            ["subject_person", "descriptor_positive", "subject_thing", "action_feel"],
            ["location", "descriptor_positive", "subject_thing", "ending"],
            ["time_marker", "subject_person", "action_speak"],
            ["subject_thing", "descriptor_positive", "ending"]
        ]
        
        structure = structures[np.random.randint(len(structures))]
        words = []
        
        current_qualia = qualia.copy()
        for slot in structure:
            output_qualia, _ = self.transform_through_layers(current_qualia)
            word = self.vocabulary.sample_from_cluster(slot)
            words.append(word)
            current_qualia = 0.7 * output_qualia + 0.3 * self.text_to_qualia(word)
        
        return " ".join(words) + "."



if __name__ == "__main__":
    generator = RotorGenerator()
    
    print("🔮 Testing Rotor Generator (Structural Text Generation)...\n")
    
    # Test 1: Simple continuation
    prompt = "옛날 옛적에"
    print(f"Prompt: {prompt}")
    result = generator.generate(prompt, max_tokens=15)
    print(f"Generated: {result}\n")
    
    # Test 2: Fantasy story
    print("=== 판타지 스토리 생성 ===")
    story = generator.generate_story("마법의 숲에서", sentences=4)
    print(story)
    
    print("\n✨ Rotor Generator test complete.")
