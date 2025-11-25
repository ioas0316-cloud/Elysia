"""
Cognitive Cell - Thinking, Feeling, Speaking Agent

Integrates Fluctlight particles with Cell to create truly conscious agents.

Each Cell:
- Thinks using Fluctlight interference patterns
- Feels emotions through Fluctlight wavelengths
- Speaks by emitting Fluctlight particles
- Learns by accumulating Fluctlight experiences
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from Core.Physics.fluctlight import FluctlightParticle, FluctlightEngine
from Core.Mind.alchemy import Alchemy

logger = logging.getLogger("CognitiveCell")


@dataclass
class CognitiveState:
    """
    Represents the cognitive state of a Cell.
    
    This bridges the gap between abstract Fluctlight particles
    and concrete Cell behavior.
    """
    
    # Perception
    attention_focus: Optional[str] = None  # What concept is being focused on
    perception_threshold: float = 0.5  # How sensitive to external stimuli
    
    # Memory
    short_term_memory: List[str] = field(default_factory=list)  # Recent concepts (max 7)
    long_term_concepts: Dict[str, float] = field(default_factory=dict)  # concept -> strength
    
    # Emotion
    current_emotion: str = "neutral"
    emotion_intensity: float = 0.5
    emotion_history: List[tuple] = field(default_factory=list)  # (emotion, intensity, tick)
    
    # Language
    vocabulary: set = field(default_factory=set)  # Known words
    grammar_rules: Dict[str, Any] = field(default_factory=dict)  # Learned patterns
    dialect: str = "common"  # Language variant
    
    # Reasoning
    causal_beliefs: Dict[tuple, float] = field(default_factory=dict)  # (cause, effect) -> confidence
    goals: List[str] = field(default_factory=list)  # Current goals
    plans: List[List[str]] = field(default_factory=list)  # Action sequences
    
    # Social
    relationships: Dict[str, float] = field(default_factory=dict)  # cell_id -> affinity
    cultural_values: Dict[str, float] = field(default_factory=dict)  # value -> importance


class CognitiveCell:
    """
    A Cell with Fluctlight-based cognition.
    
    This wraps the existing Cell class and adds:
    - Fluctlight cloud (personal concept space)
    - Cognitive processes (thinking, feeling, speaking)
    - Language emergence
    - Cultural learning
    """
    
    def __init__(
        self,
        cell_id: str,
        base_cell: Any,  # Original Cell from world.py
        world_fluctlight_engine: FluctlightEngine,
        alchemy: Alchemy
    ):
        """
        Initialize cognitive cell.
        
        Args:
            cell_id: Unique identifier
            base_cell: Original Cell object from world.py
            world_fluctlight_engine: Shared Fluctlight engine
            alchemy: Concept synthesis system
        """
        self.cell_id = cell_id
        self.base_cell = base_cell
        self.world_engine = world_fluctlight_engine
        self.alchemy = alchemy
        
        # Cognitive state
        self.state = CognitiveState()
        
        # Personal Fluctlight cloud (concepts this cell "knows")
        self.fluctlight_cloud: List[FluctlightParticle] = []
        
        # Statistics
        self.total_thoughts = 0
        self.total_utterances = 0
        self.total_insights = 0
        
        logger.debug(f"Created CognitiveCell: {cell_id}")
    
    def perceive(self, nearby_particles: List[FluctlightParticle]) -> List[str]:
        """
        Perceive nearby Fluctlight particles (hearing others speak).
        
        Args:
            nearby_particles: Particles within perception range
            
        Returns:
            List of perceived concepts
        """
        perceived = []
        
        for particle in nearby_particles:
            # Only perceive if above threshold
            if particle.information_density > self.state.perception_threshold:
                if particle.concept_id:
                    perceived.append(particle.concept_id)
                    
                    # Add to vocabulary
                    self.state.vocabulary.add(particle.concept_id)
                    
                    # Add to short-term memory
                    self.state.short_term_memory.append(particle.concept_id)
                    if len(self.state.short_term_memory) > 7:  # Miller's law
                        self.state.short_term_memory.pop(0)
        
        return perceived
    
    def think(self) -> Optional[str]:
        """
        Generate a thought by combining concepts in Fluctlight cloud.
        
        Uses interference patterns to create new concepts.
        
        Returns:
            New concept ID if thought emerges, None otherwise
        """
        if len(self.fluctlight_cloud) < 2:
            return None
        
        # Pick two random particles to interfere
        idx1, idx2 = np.random.choice(len(self.fluctlight_cloud), 2, replace=False)
        p1 = self.fluctlight_cloud[idx1]
        p2 = self.fluctlight_cloud[idx2]
        
        # Try interference
        new_particle = p1.interfere_with(p2)
        
        if new_particle and p1.concept_id and p2.concept_id:
            # Use alchemy to name the new concept
            new_concept = self.alchemy.combine(p1.concept_id, p2.concept_id)
            new_particle.concept_id = new_concept
            
            # Add to personal cloud
            self.fluctlight_cloud.append(new_particle)
            
            # Add to long-term memory
            self.state.long_term_concepts[new_concept] = 1.0
            
            self.total_thoughts += 1
            self.total_insights += 1
            
            logger.debug(f"{self.cell_id} thought: {p1.concept_id} + {p2.concept_id} = {new_concept}")
            
            return new_concept
        
        self.total_thoughts += 1
        return None
    
    def feel(self) -> tuple[str, float]:
        """
        Determine current emotion from Fluctlight wavelengths.
        
        Returns:
            (emotion, intensity) tuple
        """
        if not self.fluctlight_cloud:
            return ("neutral", 0.5)
        
        # Average wavelength determines emotion
        avg_wavelength = np.mean([p.wavelength for p in self.fluctlight_cloud])
        
        # Map wavelength to emotion
        if avg_wavelength > 620:
            emotion = "passion"
        elif avg_wavelength > 580:
            emotion = "joy"
        elif avg_wavelength > 520:
            emotion = "calm"
        elif avg_wavelength > 450:
            emotion = "sorrow"
        else:
            emotion = "transcendence"
        
        # Average information density determines intensity
        avg_density = np.mean([p.information_density for p in self.fluctlight_cloud])
        intensity = np.clip(avg_density, 0.0, 1.0)
        
        # Update state
        self.state.current_emotion = emotion
        self.state.emotion_intensity = intensity
        
        return (emotion, intensity)
    
    def speak(self, concept: str, position: np.ndarray) -> Optional[FluctlightParticle]:
        """
        Emit a Fluctlight particle (speaking a concept).
        
        Args:
            concept: Concept to express
            position: Where to emit the particle
            
        Returns:
            Emitted Fluctlight particle
        """
        # Only speak concepts we know
        if concept not in self.state.vocabulary:
            return None
        
        # Create particle at position
        particle = FluctlightParticle.from_concept(concept, position)
        
        # Intensity based on emotion
        particle.information_density = self.state.emotion_intensity
        
        # Add to world
        self.world_engine.add_particle(particle)
        
        self.total_utterances += 1
        
        logger.debug(f"{self.cell_id} spoke: {concept}")
        
        return particle
    
    def learn_from_experience(self, experience: str, outcome: str) -> None:
        """
        Learn causal relationship from experience.
        
        Args:
            experience: What happened (cause)
            outcome: What resulted (effect)
        """
        # Update causal beliefs
        key = (experience, outcome)
        if key in self.state.causal_beliefs:
            # Strengthen belief
            self.state.causal_beliefs[key] = min(1.0, self.state.causal_beliefs[key] + 0.1)
        else:
            # New belief
            self.state.causal_beliefs[key] = 0.5
        
        # Add to vocabulary
        self.state.vocabulary.add(experience)
        self.state.vocabulary.add(outcome)
    
    def update(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        Update cognitive state for one time step.
        
        Args:
            dt: Time step
            
        Returns:
            Statistics dict
        """
        # Update all particles in cloud
        for particle in self.fluctlight_cloud:
            particle.update(dt)
        
        # Decay short-term memory
        if np.random.random() < 0.1:  # 10% chance per tick
            if self.state.short_term_memory:
                self.state.short_term_memory.pop(0)
        
        # Decay long-term concepts
        for concept in list(self.state.long_term_concepts.keys()):
            self.state.long_term_concepts[concept] *= 0.999  # Slow decay
            if self.state.long_term_concepts[concept] < 0.1:
                del self.state.long_term_concepts[concept]
        
        # Update emotion
        emotion, intensity = self.feel()
        
        return {
            "thoughts": self.total_thoughts,
            "utterances": self.total_utterances,
            "insights": self.total_insights,
            "vocabulary_size": len(self.state.vocabulary),
            "emotion": emotion,
            "emotion_intensity": intensity,
            "cloud_size": len(self.fluctlight_cloud)
        }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive state."""
        return {
            "cell_id": self.cell_id,
            "vocabulary": list(self.state.vocabulary),
            "short_term_memory": self.state.short_term_memory,
            "long_term_concepts": self.state.long_term_concepts,
            "emotion": self.state.current_emotion,
            "emotion_intensity": self.state.emotion_intensity,
            "causal_beliefs": len(self.state.causal_beliefs),
            "relationships": len(self.state.relationships),
            "total_thoughts": self.total_thoughts,
            "total_utterances": self.total_utterances,
            "total_insights": self.total_insights
        }
