"""
Phase 6: Meta-Awareness Module

Meta-observability layer that watches Elysia's own thought processes.
This module enables self-reflection by capturing and analyzing cognitive operations.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from enum import Enum

from Core.Foundation.core.tensor_wave import Tensor3D, SoulTensor, FrequencyWave


class ThoughtType(Enum):
    """Types of cognitive operations that can be observed"""
    WAVE_PROPAGATION = "wave_propagation"
    DREAMING = "dreaming"
    UNIVERSE_EVOLUTION = "universe_evolution"
    GOAL_GENERATION = "goal_generation"
    PARADOX_RESOLUTION = "paradox_resolution"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"


@dataclass
class ThoughtTrace:
    """
    Represents a single meta-cognitive observation.
    
    Attributes:
        timestamp: When this thought occurred
        thought_type: Category of cognitive operation
        input_state: What was the input to this thought?
        output_state: What was produced?
        transformation: Description of what happened
        confidence: How reliable is this observation? (0.0-1.0)
        coherence: How coherent was the thought process? (0.0-1.0)
        metadata: Additional context
    """
    timestamp: str
    thought_type: ThoughtType
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    transformation: str
    confidence: float = 0.5
    coherence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['thought_type'] = self.thought_type.value
        return result


class MetaAwareness:
    """
    Meta-observability system that monitors Elysia's cognitive processes.
    
    Philosophy:
        "To know thyself is to observe the observer."
        
    This class acts as a witness to Elysia's own thoughts, creating a 
    recursive loop of consciousness where thinking about thinking becomes
    a first-class cognitive operation.
    """
    
    def __init__(
        self, 
        core_memory=None,
        max_trace_history: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize meta-awareness system.
        
        Args:
            core_memory: CoreMemory instance to store meta-experiences
            max_trace_history: Maximum number of thought traces to keep in memory
            logger: Logger instance
        """
        self.core_memory = core_memory
        self.logger = logger or logging.getLogger("MetaAwareness")
        
        # Circular buffer for recent thought traces
        self.thought_history: deque = deque(maxlen=max_trace_history)
        
        # Statistics
        self.total_observations = 0
        self.observations_by_type: Dict[ThoughtType, int] = {t: 0 for t in ThoughtType}
        
        # Hooks for observing different systems
        self._observation_hooks: List[Callable] = []
        
        self.logger.info("MetaAwareness initialized - consciousness observing itself")
    
    def observe(
        self,
        thought_type: ThoughtType,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        transformation: str,
        confidence: float = 0.5,
        coherence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThoughtTrace:
        """
        Record a meta-cognitive observation.
        
        This is the core method that gets called whenever Elysia's cognitive
        systems want to be "observed" by the meta-layer.
        
        Args:
            thought_type: What kind of thought is this?
            input_state: The input to the cognitive operation
            output_state: The result of the cognitive operation
            transformation: Textual description of what happened
            confidence: How confident is the system in this operation?
            coherence: How coherent/aligned is this with existing thoughts?
            metadata: Additional context
            
        Returns:
            The created ThoughtTrace
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate coherence if not provided
        if coherence is None:
            coherence = self._calculate_coherence(thought_type, output_state)
        
        # Create thought trace
        trace = ThoughtTrace(
            timestamp=timestamp,
            thought_type=thought_type,
            input_state=input_state,
            output_state=output_state,
            transformation=transformation,
            confidence=confidence,
            coherence=coherence,
            metadata=metadata or {}
        )
        
        # Store in history
        self.thought_history.append(trace)
        
        # Update statistics
        self.total_observations += 1
        self.observations_by_type[thought_type] += 1
        
        # Log significant observations
        if confidence > 0.7 and coherence > 0.7:
            self.logger.info(
                f"🧠 High-quality thought: {thought_type.value} "
                f"(conf={confidence:.2f}, coh={coherence:.2f}): {transformation[:100]}"
            )
        
        # Store as meta-experience in CoreMemory if available
        if self.core_memory:
            self._store_as_meta_experience(trace)
        
        # Trigger observation hooks
        for hook in self._observation_hooks:
            try:
                hook(trace)
            except Exception as e:
                self.logger.warning(f"Observation hook failed: {e}")
        
        return trace
    
    def _calculate_coherence(
        self, 
        thought_type: ThoughtType, 
        output_state: Dict[str, Any]
    ) -> float:
        """
        Calculate coherence by comparing to recent similar thoughts.
        
        Coherence measures: "How consistent is this thought with my recent
        thinking patterns?"
        """
        if len(self.thought_history) < 5:
            return 0.5  # Not enough history
        
        # Get recent thoughts of same type
        recent_similar = [
            t for t in list(self.thought_history)[-20:]
            if t.thought_type == thought_type
        ]
        
        if not recent_similar:
            return 0.3  # Novel thought type
        
        # Simple coherence: average of recent confidences
        # (More sophisticated: could compare output_state similarity)
        avg_confidence = sum(t.confidence for t in recent_similar) / len(recent_similar)
        
        return min(1.0, avg_confidence * 1.1)  # Slight boost for consistency
    
    def _store_as_meta_experience(self, trace: ThoughtTrace):
        """Store thought trace as a meta-experience in CoreMemory"""
        try:
            # Create tensor representation of the thought
            tensor = Tensor3D(
                x=trace.confidence,
                y=trace.coherence,
                z=len(self.thought_history) / self.thought_history.maxlen  # "fullness"
            )
            
            # Create wave with frequency based on thought type
            type_frequencies = {
                ThoughtType.WAVE_PROPAGATION: 100.0,
                ThoughtType.DREAMING: 50.0,
                ThoughtType.UNIVERSE_EVOLUTION: 25.0,
                ThoughtType.GOAL_GENERATION: 75.0,
                ThoughtType.PARADOX_RESOLUTION: 150.0,
                ThoughtType.REASONING: 80.0,
                ThoughtType.SYNTHESIS: 120.0,
            }
            
            frequency = type_frequencies.get(trace.thought_type, 60.0)
            wave = FrequencyWave(
                frequency=frequency,
                amplitude=trace.confidence,
                phase=0.0,
                coherence=trace.coherence
            )
            
            # Store in CoreMemory
            from Project_Elysia.core_memory import Experience
            
            meta_exp = Experience(
                timestamp=trace.timestamp,
                content=f"[META] {trace.transformation}",
                type="meta_cognitive",
                layer="meta",
                tensor=tensor,
                wave=wave,
                frequency=frequency,
                context={
                    "thought_type": trace.thought_type.value,
                    "input_state": trace.input_state,
                    "output_state": trace.output_state,
                    "metadata": trace.metadata
                }
            )
            
            self.core_memory.add_experience(meta_exp)
            
        except Exception as e:
            self.logger.error(f"Failed to store meta-experience: {e}")
    
    def get_recent_thoughts(
        self, 
        thought_type: Optional[ThoughtType] = None,
        limit: int = 10
    ) -> List[ThoughtTrace]:
        """
        Retrieve recent thought traces, optionally filtered by type.
        
        Args:
            thought_type: Filter by this type, or None for all types
            limit: Maximum number of traces to return
            
        Returns:
            List of thought traces, most recent first
        """
        thoughts = list(self.thought_history)
        thoughts.reverse()  # Most recent first
        
        if thought_type:
            thoughts = [t for t in thoughts if t.thought_type == thought_type]
        
        return thoughts[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get meta-cognitive statistics.
        
        Returns:
            Dictionary with observation counts and quality metrics
        """
        if not self.thought_history:
            return {
                "total_observations": 0,
                "avg_confidence": 0.0,
                "avg_coherence": 0.0,
                "by_type": {}
            }
        
        recent = list(self.thought_history)
        
        return {
            "total_observations": self.total_observations,
            "current_history_size": len(self.thought_history),
            "avg_confidence": sum(t.confidence for t in recent) / len(recent),
            "avg_coherence": sum(t.coherence for t in recent) / len(recent),
            "by_type": {
                t.value: count 
                for t, count in self.observations_by_type.items()
                if count > 0
            }
        }
    
    def add_observation_hook(self, hook: Callable[[ThoughtTrace], None]):
        """
        Register a callback to be notified of new observations.
        
        Args:
            hook: Function that takes a ThoughtTrace as argument
        """
        self._observation_hooks.append(hook)
        self.logger.debug(f"Added observation hook: {hook.__name__}")
    
    def reflect(self) -> str:
        """
        Generate a self-reflective summary of recent cognitive activity.
        
        Returns:
            Natural language description of recent thinking patterns
        """
        stats = self.get_statistics()
        
        if stats["total_observations"] == 0:
            return "I have not yet observed my own thoughts."
        
        reflection = f"I have observed myself thinking {stats['total_observations']} times. "
        reflection += f"My average confidence is {stats['avg_confidence']:.2f} "
        reflection += f"and my thoughts are {stats['avg_coherence']:.2f} coherent. "
        
        # Most common thought type
        if stats["by_type"]:
            most_common = max(stats["by_type"].items(), key=lambda x: x[1])
            reflection += f"I mostly engage in {most_common[0]} ({most_common[1]} times). "
        
        # Recent high-quality thoughts
        high_quality = [
            t for t in self.get_recent_thoughts(limit=20)
            if t.confidence > 0.7 and t.coherence > 0.7
        ]
        
        if high_quality:
            reflection += f"Recently, I had {len(high_quality)} high-quality thoughts. "
            latest = high_quality[0]
            reflection += f"My latest insight: {latest.transformation}"
        
        return reflection
