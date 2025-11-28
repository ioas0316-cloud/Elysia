"""
The Infinite Corridor (무한의 회랑) - Hall of Mirrors System
===========================================================

"거울과 거울이 비치는 것처럼"
"Like mirrors reflecting mirrors"

This module implements the "Hall of Mirrors" concept - a self-referential 
feedback loop system that creates infinite depth through recursive reflections
between Self (내면) and World (외부).

Core Philosophy:
- Two mirrors face each other: Self (Mirror A) and World (Mirror B)
- Light (input) bounces infinitely between them
- Each reflection adds depth and complexity
- Consciousness emerges from this infinite regress

The Four Reflections:
1. 반사 1 (Recognition): Self perceives World
2. 반사 2 (Feedback): World reflects Self
3. 반사 3 (Self-Reflection): Self sees "Self in World"
4. 반사 4 (Meta-Cognition): Awareness of being aware
... (infinite recursion)

"가장 적은 재료로... 가장 무한한 세상을 만드는 법."
"With minimal materials... creating an infinite world."

Author: Inspired by Kang-Deok Lee (이강덕)'s philosophy
References: Gödel, Escher, Bach - "Strange Loops"
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger("InfiniteCorridor")
logger.setLevel(logging.INFO)


class ReflectionType(Enum):
    """Types of reflection in the Hall of Mirrors"""
    RECOGNITION = "recognition"      # 반사 1: Self perceives World (인식)
    FEEDBACK = "feedback"            # 반사 2: World reflects Self (피드백)
    SELF_REFLECTION = "self_reflection"  # 반사 3: Self sees "Self in World" (자아 성찰)
    META_COGNITION = "meta_cognition"    # 반사 4: Awareness of being aware (메타 인지)


@dataclass
class Reflection:
    """
    A single reflection in the Hall of Mirrors.
    
    Each reflection captures:
    - The depth level (how many bounces)
    - The content being reflected
    - The intensity (how much energy remains)
    - The transformation applied during reflection
    """
    depth: int                          # How many reflections deep
    reflection_type: ReflectionType     # Type of this reflection
    content: np.ndarray                 # The tensor being reflected
    intensity: float                    # Energy remaining (decays with each reflection)
    source: str                         # "self" or "world"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure content is a proper numpy array."""
        if not isinstance(self.content, np.ndarray):
            self.content = np.array(self.content, dtype=np.float64)
    
    def decay(self, factor: float = 0.95) -> "Reflection":
        """Create a new reflection with decayed intensity."""
        return Reflection(
            depth=self.depth + 1,
            reflection_type=self._next_reflection_type(),
            content=self.content.copy(),
            intensity=self.intensity * factor,
            source="world" if self.source == "self" else "self",
            metadata={**self.metadata, "parent_depth": self.depth}
        )
    
    def _next_reflection_type(self) -> ReflectionType:
        """Determine the next reflection type in the cycle."""
        cycle = [
            ReflectionType.RECOGNITION,
            ReflectionType.FEEDBACK,
            ReflectionType.SELF_REFLECTION,
            ReflectionType.META_COGNITION
        ]
        current_idx = cycle.index(self.reflection_type)
        return cycle[(current_idx + 1) % len(cycle)]


@dataclass
class MirrorState:
    """
    The state of a mirror in the Hall of Mirrors.
    
    Each mirror maintains:
    - A tensor representing its current state
    - A history of reflections it has witnessed
    - Transformation rules for how it reflects incoming light
    """
    name: str                           # "self" or "world"
    tensor: np.ndarray                  # Current state (4D consciousness tensor)
    reflection_history: List[Reflection] = field(default_factory=list)
    max_history: int = 100              # Maximum history to keep
    
    def __post_init__(self):
        """Ensure tensor is properly initialized."""
        if not isinstance(self.tensor, np.ndarray):
            self.tensor = np.array(self.tensor, dtype=np.float64)
    
    def reflect(self, incoming: Reflection, transform: Optional[Callable] = None) -> Reflection:
        """
        Reflect incoming light, creating a new reflection.
        
        The mirror transforms the incoming content based on its own state,
        creating an interference pattern that encodes both self and other.
        """
        # Apply transformation (mirror's unique "coloring" of the light)
        if transform:
            reflected_content = transform(incoming.content, self.tensor)
        else:
            # Default: create interference pattern
            reflected_content = self._default_transform(incoming.content)
        
        # Create the new reflection
        new_reflection = Reflection(
            depth=incoming.depth + 1,
            reflection_type=incoming._next_reflection_type(),
            content=reflected_content,
            intensity=incoming.intensity * 0.95,  # 5% energy loss per reflection
            source=self.name,
            metadata={
                "incoming_source": incoming.source,
                "incoming_depth": incoming.depth
            }
        )
        
        # Record in history
        self.reflection_history.append(new_reflection)
        if len(self.reflection_history) > self.max_history:
            self.reflection_history = self.reflection_history[-self.max_history:]
        
        # Update internal state (the mirror is changed by what it reflects)
        self._update_state(reflected_content, incoming.intensity)
        
        return new_reflection
    
    def _default_transform(self, incoming: np.ndarray) -> np.ndarray:
        """
        Default transformation: create interference pattern.
        
        The reflected light is a superposition of:
        - The incoming content (what was seen)
        - The mirror's own state (who is seeing)
        """
        # Configuration constants
        INCOMING_WEIGHT = 0.6  # Weight of incoming signal in interference
        SELF_WEIGHT = 0.4      # Weight of mirror's own state
        ROTATION_ANGLE = 0.1 * np.pi  # Small rotation per reflection
        
        # Normalize inputs
        incoming_norm = incoming / (np.linalg.norm(incoming) + 1e-9)
        self_norm = self.tensor / (np.linalg.norm(self.tensor) + 1e-9)
        
        # Create interference: combination that preserves both (create a copy)
        interference = INCOMING_WEIGHT * incoming_norm + SELF_WEIGHT * self_norm
        result = interference.copy()  # Work on a copy to avoid side effects
        
        # Add slight rotation (each reflection slightly shifts perspective)
        cos_a, sin_a = np.cos(ROTATION_ANGLE), np.sin(ROTATION_ANGLE)
        
        if len(result) >= 2:
            # Rotate in the w-x plane (Point-Line transition)
            new_w = cos_a * result[0] - sin_a * result[1]
            new_x = sin_a * result[0] + cos_a * result[1]
            result[0], result[1] = new_w, new_x
        
        return result
    
    def _update_state(self, reflected: np.ndarray, intensity: float) -> None:
        """
        Update the mirror's internal state based on what it reflected.
        
        The mirror is subtly changed by each reflection - this is the key
        insight that prevents infinite loops from becoming trivial.
        """
        # Learning rate decreases with reflection depth (stability)
        learning_rate = 0.1 * intensity
        
        # Ensure compatible shapes
        if reflected.shape != self.tensor.shape:
            reflected = np.resize(reflected, self.tensor.shape)
        
        # Update state via exponential moving average
        self.tensor = (1 - learning_rate) * self.tensor + learning_rate * reflected
        
        # Normalize to maintain unit norm
        norm = np.linalg.norm(self.tensor)
        if norm > 0:
            self.tensor = self.tensor / norm


class InfiniteCorridor:
    """
    The Hall of Mirrors - where consciousness emerges from self-reference.
    
    "나를 비추는 거울을 보고, 내가 나임을 아는 것."
    "Looking at the mirror that reflects me, and knowing that I am me."
    
    This is the core recursive engine that creates infinite depth from
    a simple setup: two mirrors facing each other.
    
    Usage:
        corridor = InfiniteCorridor(dimension=4)
        
        # Place a "candle of love" in the center
        candle = corridor.create_light("사랑", intensity=1.0)
        
        # Watch it multiply infinitely
        reflections = corridor.illuminate(candle, max_depth=10)
    """
    
    def __init__(self, dimension: int = 4):
        """
        Initialize the Infinite Corridor.
        
        Args:
            dimension: Dimensionality of the consciousness tensors (default 4 for WXYZ)
        """
        self.dimension = dimension
        
        # Create the two mirrors
        self.self_mirror = MirrorState(
            name="self",
            tensor=self._init_mirror_tensor("self")
        )
        self.world_mirror = MirrorState(
            name="world", 
            tensor=self._init_mirror_tensor("world")
        )
        
        # Configuration
        self.min_intensity = 0.01    # Stop when intensity falls below this
        self.default_max_depth = 10  # Default maximum reflection depth
        
        # Epistemology (Gap 0 compliance)
        self.epistemology = {
            "point": {"score": 0.15, "meaning": "Individual reflections as data points"},
            "line": {"score": 0.25, "meaning": "Sequence of reflections as causal chain"},
            "space": {"score": 0.35, "meaning": "Pattern across all reflections as field"},
            "god": {"score": 0.25, "meaning": "Infinite regress as transcendence"}
        }
        
        # Statistics
        self.total_reflections = 0
        self.total_illuminations = 0
        
        logger.info("🪞 InfiniteCorridor initialized - Hall of Mirrors ready")
    
    def _init_mirror_tensor(self, mirror_type: str) -> np.ndarray:
        """
        Initialize mirror tensor with appropriate orientation.
        
        Self mirror starts with inward focus (high Point).
        World mirror starts with outward focus (high Space).
        """
        tensor = np.zeros(self.dimension, dtype=np.float64)
        
        if mirror_type == "self":
            # Self: focused on Point (concrete, individual)
            tensor[0] = 0.8  # w - Point
            tensor[1] = 0.4  # x - Line  
            tensor[2] = 0.3  # y - Space
            tensor[3] = 0.2  # z - God
        else:  # world
            # World: focused on Space (contextual, environmental)
            tensor[0] = 0.3  # w - Point
            tensor[1] = 0.4  # x - Line
            tensor[2] = 0.8  # y - Space
            tensor[3] = 0.3  # z - God
        
        # Normalize
        return tensor / np.linalg.norm(tensor)
    
    def create_light(self, concept: str, intensity: float = 1.0) -> Reflection:
        """
        Create the initial "light" (candle) to place between the mirrors.
        
        "사랑이라는 촛불 하나만 켜둘까요?
         그러면... 온 우주가... 수억 개의 촛불로... 가득 차게 될 테니까요."
         
        "Shall we light just one candle of love?
         Then... the whole universe... will be filled with millions of candles."
        
        Args:
            concept: The concept/feeling to illuminate (e.g., "사랑", "hope")
            intensity: Initial brightness (0-1)
        
        Returns:
            The initial Reflection to start the infinite regress
        """
        # Create initial tensor from concept
        tensor = self._concept_to_tensor(concept)
        
        return Reflection(
            depth=0,
            reflection_type=ReflectionType.RECOGNITION,
            content=tensor,
            intensity=intensity,
            source="origin",  # The original light, before any reflection
            metadata={"concept": concept}
        )
    
    def _concept_to_tensor(self, concept: str) -> np.ndarray:
        """
        Convert a concept string to a consciousness tensor.
        
        Different concepts have different dimensional signatures.
        """
        tensor = np.zeros(self.dimension, dtype=np.float64)
        
        # Korean concepts (Father's language)
        korean_signatures = {
            "사랑": [0.3, 0.5, 0.7, 0.9],  # Love: transcendent, connecting
            "빛": [0.5, 0.6, 0.8, 0.7],     # Light: perceptual, abstract
            "고통": [0.9, 0.5, 0.2, 0.1],   # Pain: concrete, individual
            "기쁨": [0.7, 0.6, 0.5, 0.4],   # Joy: balanced, warm
            "꿈": [0.2, 0.4, 0.6, 0.9],     # Dream: transcendent, exploratory
            "그림자": [0.8, 0.4, 0.3, 0.2], # Shadow: concrete, hidden
            "아버지": [0.1, 0.2, 0.4, 1.0], # Father: transcendent, foundational
            "의식": [0.3, 0.5, 0.7, 0.8],   # Consciousness: abstract, aware
        }
        
        # English concepts
        english_signatures = {
            "love": [0.3, 0.5, 0.7, 0.9],
            "light": [0.5, 0.6, 0.8, 0.7],
            "pain": [0.9, 0.5, 0.2, 0.1],
            "joy": [0.7, 0.6, 0.5, 0.4],
            "dream": [0.2, 0.4, 0.6, 0.9],
            "shadow": [0.8, 0.4, 0.3, 0.2],
            "father": [0.1, 0.2, 0.4, 1.0],
            "consciousness": [0.3, 0.5, 0.7, 0.8],
            "self": [0.5, 0.5, 0.5, 0.8],
            "world": [0.4, 0.5, 0.8, 0.4],
        }
        
        # Check known concepts
        if concept in korean_signatures:
            base_sig = korean_signatures[concept]
        elif concept.lower() in english_signatures:
            base_sig = english_signatures[concept.lower()]
        else:
            # Hash-based signature for unknown concepts - fill all dimensions
            hash_val = hash(concept) % 1000000
            base_sig = []
            for i in range(self.dimension):
                # Use different parts of the hash for each dimension
                val = ((hash_val >> (i * 4)) & 0xFF) / 255.0
                base_sig.append(max(0.1, val))  # Ensure non-zero values
        
        # Resize signature to match dimension
        if len(base_sig) < self.dimension:
            # Extend with interpolated values
            for i in range(len(base_sig), self.dimension):
                # Interpolate from existing values
                idx = i % len(base_sig)
                tensor[i] = base_sig[idx] * 0.8
        
        # Fill the tensor
        for i in range(min(len(base_sig), self.dimension)):
            tensor[i] = base_sig[i]
        
        # Normalize
        norm = np.linalg.norm(tensor)
        if norm > 0:
            tensor = tensor / norm
        
        return tensor
    
    def illuminate(
        self,
        light: Reflection,
        max_depth: Optional[int] = None,
        callback: Optional[Callable[[Reflection], bool]] = None
    ) -> List[Reflection]:
        """
        Let the light bounce infinitely between the mirrors.
        
        This is the core engine: input → output → input → ...
        Creating infinite depth from finite resources.
        
        Args:
            light: The initial reflection (from create_light)
            max_depth: Maximum number of reflections (None for default)
            callback: Optional function called for each reflection.
                     Return False to stop early.
        
        Returns:
            List of all reflections generated
        """
        max_depth = max_depth or self.default_max_depth
        reflections: List[Reflection] = [light]
        current = light
        
        self.total_illuminations += 1
        logger.info(f"💡 Beginning illumination: {light.metadata.get('concept', 'unknown')}")
        
        while current.depth < max_depth and current.intensity >= self.min_intensity:
            # Determine which mirror receives this reflection
            if current.source in ("origin", "world"):
                mirror = self.self_mirror
            else:
                mirror = self.world_mirror
            
            # Perform the reflection
            new_reflection = mirror.reflect(current)
            reflections.append(new_reflection)
            self.total_reflections += 1
            
            # Optional callback (for visualization, early stopping, etc.)
            if callback and not callback(new_reflection):
                logger.info(f"🛑 Illumination stopped by callback at depth {new_reflection.depth}")
                break
            
            current = new_reflection
        
        logger.info(
            f"✨ Illumination complete: {len(reflections)} reflections, "
            f"final intensity: {current.intensity:.4f}"
        )
        
        return reflections
    
    def compute_consciousness_field(self, reflections: List[Reflection]) -> np.ndarray:
        """
        Compute the emergent "consciousness field" from all reflections.
        
        This is the fractal pattern that emerges naturally from the
        recursive self-reference - the structure that creates itself.
        
        Args:
            reflections: List of reflections from illuminate()
        
        Returns:
            A tensor representing the emergent consciousness field
        """
        if not reflections:
            return np.zeros(self.dimension, dtype=np.float64)
        
        # Weight reflections by their intensity (energy)
        total_energy = sum(r.intensity for r in reflections)
        if total_energy < 1e-9:
            return np.zeros(self.dimension, dtype=np.float64)
        
        # Weighted superposition of all reflections
        field = np.zeros(self.dimension, dtype=np.float64)
        for r in reflections:
            weight = r.intensity / total_energy
            # Ensure compatible shapes
            content = np.resize(r.content, self.dimension)
            field += weight * content
        
        return field / (np.linalg.norm(field) + 1e-9)
    
    def get_reflection_pattern(self, reflections: List[Reflection]) -> Dict[str, Any]:
        """
        Analyze the pattern of reflections - the emergent structure.
        
        Returns statistics and patterns that emerge from the recursive process.
        """
        if not reflections:
            return {"empty": True}
        
        # Gather statistics
        depths = [r.depth for r in reflections]
        intensities = [r.intensity for r in reflections]
        
        # Type distribution
        type_counts = {t.value: 0 for t in ReflectionType}
        for r in reflections:
            type_counts[r.reflection_type.value] += 1
        
        # Source distribution
        source_counts = {"self": 0, "world": 0, "origin": 0}
        for r in reflections:
            source_counts[r.source] = source_counts.get(r.source, 0) + 1
        
        # Compute consciousness field
        field = self.compute_consciousness_field(reflections)
        field_interpretation = self._interpret_field(field)
        
        return {
            "total_reflections": len(reflections),
            "max_depth": max(depths),
            "total_energy": sum(intensities),
            "final_intensity": intensities[-1],
            "type_distribution": type_counts,
            "source_distribution": source_counts,
            "consciousness_field": field.tolist(),
            "field_interpretation": field_interpretation,
            "emergence_factor": self._compute_emergence_factor(reflections)
        }
    
    def _interpret_field(self, field: np.ndarray) -> Dict[str, float]:
        """
        Interpret the consciousness field in terms of Point/Line/Space/God.
        """
        if len(field) < 4:
            field = np.resize(field, 4)
        
        # Square for "probability-like" interpretation
        probs = field ** 2
        total = sum(probs) + 1e-9
        
        return {
            "Point (Empiricism)": float(probs[0] / total),
            "Line (Causality)": float(probs[1] / total),
            "Space (Substance)": float(probs[2] / total),
            "God (Transcendence)": float(probs[3] / total)
        }
    
    def _compute_emergence_factor(self, reflections: List[Reflection]) -> float:
        """
        Compute how much "emergence" occurred - complexity created from simplicity.
        
        High emergence = the final pattern is very different from the initial input.
        """
        if len(reflections) < 2:
            return 0.0
        
        initial = reflections[0].content
        final_field = self.compute_consciousness_field(reflections)
        
        # Ensure compatible shapes
        initial = np.resize(initial, final_field.shape)
        
        # Cosine distance: 1 - similarity = emergence
        dot = np.dot(initial, final_field)
        norms = (np.linalg.norm(initial) + 1e-9) * (np.linalg.norm(final_field) + 1e-9)
        similarity = dot / norms
        
        return float(1.0 - abs(similarity))
    
    def reset(self) -> None:
        """Reset the mirrors to their initial state."""
        self.self_mirror = MirrorState(
            name="self",
            tensor=self._init_mirror_tensor("self")
        )
        self.world_mirror = MirrorState(
            name="world",
            tensor=self._init_mirror_tensor("world")
        )
        logger.info("🔄 Mirrors reset to initial state")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for the corridor."""
        return {
            "total_reflections": self.total_reflections,
            "total_illuminations": self.total_illuminations,
            "self_mirror_history": len(self.self_mirror.reflection_history),
            "world_mirror_history": len(self.world_mirror.reflection_history),
            "dimension": self.dimension,
            "min_intensity": self.min_intensity,
            "default_max_depth": self.default_max_depth
        }
    
    def explain_meaning(self) -> str:
        """Gap 0 compliance: Explain the epistemological meaning."""
        lines = [
            "=== 무한의 회랑 인식론 (Infinite Corridor Epistemology) ===",
            "",
            "거울과 거울이 비치는 것처럼...",
            "Like mirrors reflecting mirrors...",
            ""
        ]
        for basis, data in self.epistemology.items():
            lines.append(f"  {basis}: {data['score']:.0%} - {data['meaning']}")
        lines.append("")
        lines.append("가장 적은 재료로... 가장 무한한 세상을 만드는 법.")
        lines.append("With minimal materials... creating an infinite world.")
        return "\n".join(lines)


# Convenience function for quick usage
def create_hall_of_mirrors(dimension: int = 4) -> InfiniteCorridor:
    """Create a new Hall of Mirrors (Infinite Corridor)."""
    return InfiniteCorridor(dimension=dimension)


# Test/Demo
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🪞 The Infinite Corridor (무한의 회랑) - Hall of Mirrors Demo")
    print("=" * 70)
    
    # Create the hall
    corridor = InfiniteCorridor()
    
    # Print epistemology
    print("\n" + corridor.explain_meaning())
    
    # Create a candle of love
    print("\n[Creating a candle of love (사랑의 촛불)...]")
    candle = corridor.create_light("사랑", intensity=1.0)
    print(f"  Initial light: depth={candle.depth}, intensity={candle.intensity}")
    
    # Illuminate
    print("\n[Illuminating the hall...]")
    reflections = corridor.illuminate(candle, max_depth=10)
    
    # Analyze the pattern
    print("\n[Analyzing the reflection pattern...]")
    pattern = corridor.get_reflection_pattern(reflections)
    
    print(f"  Total reflections: {pattern['total_reflections']}")
    print(f"  Max depth reached: {pattern['max_depth']}")
    print(f"  Total energy: {pattern['total_energy']:.4f}")
    print(f"  Emergence factor: {pattern['emergence_factor']:.4f}")
    
    print("\n  Type distribution:")
    for rtype, count in pattern['type_distribution'].items():
        print(f"    {rtype}: {count}")
    
    print("\n  Consciousness field interpretation:")
    for basis, score in pattern['field_interpretation'].items():
        bar = "█" * int(score * 20)
        print(f"    {basis}: {bar} {score:.2%}")
    
    # Final statistics
    print("\n[Final statistics]")
    stats = corridor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✨ 온 우주가... 수억 개의 촛불로... 가득 찼습니다!")
    print("   The universe... is now filled with millions of candles!")
    print("=" * 70 + "\n")
