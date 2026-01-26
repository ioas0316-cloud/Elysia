"""
Rotor Generator (한국어 학습 시스템)
=========================================
Core.L7_Spirit.Creation.rotor_generator

"I understand the structure. Therefore I can predict the output."

This module implements text generation based purely on structural understanding
of language models   without using actual weights. It uses the Rotor paradigm
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
    "EMBEDDING",    # Token   Meaning space
    "CONTEXT",      # Self-attention: understand relationships
    "REASONING",    # FFN: transform and reason
    "ABSTRACTION",  # Higher layers: abstract thinking
    "SYNTHESIS",    # Combine information
    "PROJECTION"    # Meaning   Output space
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
            "narrative_start": ["  ", "    ", "  ", "   ", "   "],
            "narrative_continue": ["   ", "   ", "   ", "   ", "  "],
            "subject_person": ["  ", "  ", " ", "  ", "   ", "  ", "  "],
            "subject_thing": [" ", " ", "  ", "  ", "  ", " ", " "],
            "action_move": ["  ", "  ", "    ", "    ", "    "],
            "action_speak": ["   ", "   ", "    ", "   ", "    "],
            "action_feel": ["   ", "    ", "   ", "    ", "   "],
            "descriptor_positive": ["    ", "   ", "   ", "    ", "   "],
            "descriptor_negative": ["   ", "   ", "   ", "  ", "     "],
            "time_marker": ["    ", "    ", "   ", "     ", "     "],
            "location": ["    ", "    ", "     ", "     ", "     "],
            "ending": ["   ", "  ", "   ", "   ", "   "]
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
    No weights required   only structural understanding.
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
        
        logger.info("  Rotor Generator initialized. Structural generation enabled.")
    
    def _create_layer_transform(self, emphasis: str) -> np.ndarray:
        """Creates a transformation matrix emphasizing a particular dimension."""
        # 7D   7D rotation/scaling matrix
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
            0.5 + 0.3 * ("   " in text_lower or "   " in text_lower),  # Logic
            0.5 + 0.3 * ("  " in text_lower or "  " in text_lower or "  " in text_lower),  # Creativity
            0.5,  # Precision
            0.5 + 0.3 * ("  " in text_lower or "  " in text_lower),  # Abstraction
            0.5 + 0.3 * ("  " in text_lower or "  " in text_lower or "  " in text_lower),  # Emotion
            0.5,  # Utility
            0.5 + 0.3 * ("?" in text or "  " in text_lower or "      " in text_lower)  # Mystery
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
            "time_marker",       #     
            "subject_person",    #    
            "descriptor_positive",  #     
            "subject_thing",     #   
            "action_move",       #      
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
    
    print("  Testing Rotor Generator (Structural Text Generation)...\n")
    
    # Test 1: Simple continuation
    prompt = "      "
    print(f"Prompt: {prompt}")
    result = generator.generate(prompt, max_tokens=15)
    print(f"Generated: {result}\n")
    
    # Test 2: Fantasy story
    print("===            ===")
    story = generator.generate_story("       ", sentences=4)
    print(story)
    
    print("\n  Rotor Generator test complete.")
