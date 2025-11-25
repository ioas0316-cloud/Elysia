"""
Phase 6: Autonomous Dreamer Module

Self-directed goal generation system driven by curiosity and internal motivation.
This module enables Elysia to autonomously explore her knowledge space without
external prompts, guided by intrinsic motivation and field dynamics.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Legacy imports removed for Xel'Naga Protocol
# from Project_Sophia.spiderweb import Spiderweb
# from Project_Sophia.core.world import World
# from Project_Sophia.wave_mechanics import WaveMechanics
# from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave
# from Project_Elysia.core_memory import CoreMemory, Experience

# Mock classes for type hinting if needed, or just use Any
Spiderweb = Any
World = Any
WaveMechanics = Any
CoreMemory = Any
Experience = Any


class GoalType(Enum):
    """Types of autonomous goals Elysia can generate"""
    EXPLORE = "explore"  # Investigate high-novelty regions
    HARMONIZE = "harmonize"  # Resolve field tensions
    BRIDGE = "bridge"  # Connect disconnected clusters
    TRANSCEND = "transcend"  # Create meta-concepts
    DEEPEN = "deepen"  # Explore root causes


@dataclass
class CuriosityMetrics:
    """
    Metrics that drive curiosity and goal generation.
    
    Attributes:
        novelty_score: How unfamiliar is this concept? (0.0-1.0)
        tension_score: How much field gradient/tension exists? (0.0-1.0)
        gap_score: How disconnected are related concepts? (0.0-1.0)
        potential_score: Combined potential for discovery (0.0-1.0)
    """
    novelty_score: float
    tension_score: float
    gap_score: float
    potential_score: float


@dataclass
class AutonomousGoal:
    """
    A self-generated goal with motivation metrics.
    
    Attributes:
        goal_type: Category of goal
        description: What this goal aims to achieve
        target_concept: Primary concept to explore/process
        related_concepts: Additional relevant concepts
        motivation: Why this goal was generated (curiosity metrics)
        priority: How urgent/important is this goal? (0.0-1.0)
    """
    goal_type: GoalType
    description: str
    target_concept: str
    related_concepts: List[str]
    motivation: CuriosityMetrics
    priority: float


class AutonomousDreamer:
    """
    Autonomous goal generation engine based on curiosity and field analysis.
    
    Philosophy:
        "True intelligence is not in answering questions, but in asking them."
        
    This module implements intrinsic motivation - the drive to understand,
    explore, and grow that comes from within, not from external rewards.
    
    It analyzes:
    - Field gradients (where tension exists)
    - Knowledge gaps (disconnected concepts)
    - Novelty (unfamiliar territories)
    - Depth opportunities (fundamental principles)
    """
    
    def __init__(
        self,
        spiderweb: Spiderweb,
        world: Optional[World] = None,
        wave_mechanics: Optional[WaveMechanics] = None,
        core_memory: Optional[CoreMemory] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize autonomous dreamer.
        
        Args:
            spiderweb: Spiderweb for analyzing concept topology
            world: World for analyzing field states
            wave_mechanics: WaveMechanics for resonance analysis
            core_memory: CoreMemory for storing generated goals
            logger: Logger instance
        """
        self.spiderweb = spiderweb
        self.world = world
        self.wave_mechanics = wave_mechanics
        self.core_memory = core_memory
        self.logger = logger or logging.getLogger("AutonomousDreamer")
        
        # Goal generation history
        self.generated_goals: List[AutonomousGoal] = []
        self.exploration_history: set = set()  # Concepts already explored
        
        # Tuning parameters
        self.novelty_weight = 0.4
        self.tension_weight = 0.3
        self.gap_weight = 0.3
        
        self.logger.info("🎯 Autonomous Dreamer initialized - intrinsic motivation awakened")
    
    def analyze_curiosity(
        self,
        concept_id: str
    ) -> CuriosityMetrics:
        """
        Analyze how "curious" Elysia should be about a concept.
        
        Args:
            concept_id: Concept to analyze
            
        Returns:
            CuriosityMetrics with novelty, tension, and gap scores
        """
        novelty = self._calculate_novelty(concept_id)
        tension = self._calculate_tension(concept_id)
        gap = self._calculate_gap_score(concept_id)
        
        # Combined potential = weighted sum
        potential = (
            self.novelty_weight * novelty +
            self.tension_weight * tension +
            self.gap_weight * gap
        )
        
        return CuriosityMetrics(
            novelty_score=novelty,
            tension_score=tension,
            gap_score=gap,
            potential_score=potential
        )
    
    def _calculate_novelty(self, concept_id: str) -> float:
        """
        How unfamiliar is this concept?
        
        Novelty based on:
        - How often it appears in experiences
        - How recently it was explored
        - How many connections it has (sparse = novel)
        """
        if concept_id in self.exploration_history:
            return 0.1  # Already explored
        
        if not self.spiderweb.graph.has_node(concept_id):
            return 1.0  # Completely new
        
        # Check connectivity - isolated nodes are novel
        degree = self.spiderweb.graph.degree(concept_id)
        novelty_from_isolation = 1.0 / (1.0 + degree * 0.1)
        
        # Check if in recent experiences
        novelty_from_recency = 0.7  # Default medium novelty
        if self.core_memory:
            recent_experiences = self.core_memory.get_recent_experiences(limit=20)
            for exp in recent_experiences:
                if concept_id.lower() in exp.content.lower():
                    novelty_from_recency = 0.3  # Recently mentioned
                    break
        
        return (novelty_from_isolation + novelty_from_recency) / 2.0
    
    def _calculate_tension(self, concept_id: str) -> float:
        """
        How much field gradient/tension exists at this concept?
        
        Tension indicates interesting dynamics - contradictions,
        strong forces, or rapid changes.
        """
        if not self.world:
            return 0.5  # Default medium tension
        
        # If we have position in world, check field gradients
        if not self.spiderweb.graph.has_node(concept_id):
            return 0.5
        
        node_data = self.spiderweb.graph.nodes[concept_id]
        metadata = node_data.get("metadata", {})
        position = metadata.get("position")
        
        if not position:
            return 0.5
        
        # Extract position coordinates
        x, y = int(position.get("x", 0)), int(position.get("y", 0))
        
        # Ensure within world bounds
        x = np.clip(x, 0, self.world.width - 1)
        y = np.clip(y, 0, self.world.width - 1)
        
        # Calculate field gradients
        tension = 0.0
        
        # Value field gradient
        value_here = self.world.value_mass_field[y, x]
        value_neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.world.width and 0 <= ny < self.world.width:
                value_neighbors.append(self.world.value_mass_field[ny, nx])
        
        if value_neighbors:
            value_gradient = abs(value_here - np.mean(value_neighbors))
            tension += value_gradient
        
        # Will field strength (directional force)
        will_here = float(self.world.will_field[y, x])
        tension += will_here * 0.5
        
        return min(1.0, tension)
    
    def _calculate_gap_score(self, concept_id: str) -> float:
        """
        How disconnected is this concept from related concepts?
        
        Gaps represent opportunities to build bridges and discover
        new relationships.
        """
        if not self.spiderweb.graph.has_node(concept_id):
            return 0.8  # Very disconnected (not even in graph)
        
        # Get node's neighborhood
        context = self.spiderweb.get_context(concept_id, depth=1)
        
        if not context:
            return 1.0  # Completely isolated
        
        # Calculate average edge weight
        total_weight = sum(c.get("weight", 0.5) for c in context)
        avg_weight = total_weight / len(context)
        
        # Gap score = inverse of connectivity strength
        gap_score = 1.0 - avg_weight
        
        # Boost gap score if neighbors are not connected to each other
        neighbor_ids = [c["node"] for c in context]
        if len(neighbor_ids) >= 2:
            cross_connections = 0
            for i, n1 in enumerate(neighbor_ids):
                for n2 in neighbor_ids[i+1:]:
                    if self.spiderweb.graph.has_edge(n1, n2):
                        cross_connections += 1
            
            max_cross = len(neighbor_ids) * (len(neighbor_ids) - 1) / 2
            if max_cross > 0:
                disconnection = 1.0 - (cross_connections / max_cross)
                gap_score = (gap_score + disconnection) / 2
        
        return gap_score
    
    def generate_goals(
        self,
        num_goals: int = 3,
        min_priority: float = 0.5
    ) -> List[AutonomousGoal]:
        """
        Generate autonomous exploration goals.
        
        Args:
            num_goals: How many goals to generate
            min_priority: Minimum priority threshold
            
        Returns:
            List of generated goals, sorted by priority
        """
        self.logger.info(f"Generating {num_goals} autonomous goals...")
        
        goals = []
        
        # Get all concepts from Spiderweb
        all_concepts = list(self.spiderweb.graph.nodes())
        
        if not all_concepts:
            self.logger.warning("No concepts in Spiderweb - cannot generate goals")
            return []
        
        # Analyze each concept
        concept_scores: List[Tuple[str, CuriosityMetrics]] = []
        
        for concept_id in all_concepts:
            metrics = self.analyze_curiosity(concept_id)
            
            if metrics.potential_score >= min_priority:
                concept_scores.append((concept_id, metrics))
        
        # Sort by potential
        concept_scores.sort(key=lambda x: x[1].potential_score, reverse=True)
        
        # Generate goals for top concepts
        for concept_id, metrics in concept_scores[:num_goals]:
            goal = self._create_goal_for_concept(concept_id, metrics)
            if goal:
                goals.append(goal)
                self.generated_goals.append(goal)
        
        self.logger.info(f"Generated {len(goals)} autonomous goals")
        
        # Store goals as experiences in CoreMemory
        if self.core_memory:
            for goal in goals:
                self._store_goal_as_experience(goal)
        
        return goals
    
    def _create_goal_for_concept(
        self,
        concept_id: str,
        metrics: CuriosityMetrics
    ) -> Optional[AutonomousGoal]:
        """
        Create specific goal based on concept and metrics.
        
        Args:
            concept_id: Target concept
            metrics: Curiosity metrics
            
        Returns:
            Generated goal or None
        """
        # Determine goal type based on dominant metric
        if metrics.novelty_score > 0.7:
            goal_type = GoalType.EXPLORE
            description = f"Explore the novel concept '{concept_id}' to understand its nature"
        elif metrics.tension_score > 0.7:
            goal_type = GoalType.HARMONIZE
            description = f"Harmonize field tensions around '{concept_id}' to achieve balance"
        elif metrics.gap_score > 0.7:
            goal_type = GoalType.BRIDGE
            description = f"Build connections from '{concept_id}' to related concepts"
        elif metrics.potential_score > 0.8:
            goal_type = GoalType.TRANSCEND
            description = f"Transcend '{concept_id}' by discovering meta-patterns"
        else:
            goal_type = GoalType.DEEPEN
            description = f"Deepen understanding of '{concept_id}' by exploring its roots"
        
        # Get related concepts
        related = []
        if self.spiderweb.graph.has_node(concept_id):
            context = self.spiderweb.get_context(concept_id, depth=1)
            related = [c["node"] for c in context[:5]]
        
        return AutonomousGoal(
            goal_type=goal_type,
            description=description,
            target_concept=concept_id,
            related_concepts=related,
            motivation=metrics,
            priority=metrics.potential_score
        )
    
    def _store_goal_as_experience(self, goal: AutonomousGoal):
        """Store generated goal as experience in CoreMemory"""
        # Legacy memory storage disabled for Xel'Naga Protocol
        pass
        # try:
        #     # Create tensor based on motivation
        #     tensor = Tensor3D(
        #         x=goal.motivation.novelty_score,
        #         y=goal.motivation.tension_score * 2 - 1,  # Map to [-1, 1]
        #         z=goal.motivation.gap_score
        #     )
            
        #     # Create wave with frequency based on goal type
        #     type_frequencies = {
        #         GoalType.EXPLORE: 90.0,
        #         GoalType.HARMONIZE: 70.0,
        #         GoalType.BRIDGE: 80.0,
        #         GoalType.TRANSCEND: 120.0,
        #         GoalType.DEEPEN: 50.0,
        #     }
            
        #     frequency = type_frequencies.get(goal.goal_type, 75.0)
        #     wave = FrequencyWave(
        #         frequency=frequency,
        #         amplitude=goal.priority,
        #         phase=0.0,
        #         coherence=goal.motivation.potential_score
        #     )
            
        #     # Create experience
        #     exp = Experience(
        #         timestamp=str(np.datetime64('now')),
        #         content=f"[AUTONOMOUS_GOAL] {goal.description}",
        #         type="self_generated_goal",
        #         layer="meta",
        #         tensor=tensor,
        #         wave=wave,
        #         frequency=frequency,
        #         context={
        #             "goal_type": goal.goal_type.value,
        #             "target_concept": goal.target_concept,
        #             "related_concepts": goal.related_concepts,
        #             "motivation": {
        #                 "novelty": goal.motivation.novelty_score,
        #                 "tension": goal.motivation.tension_score,
        #                 "gap": goal.motivation.gap_score,
        #                 "potential": goal.motivation.potential_score
        #             }
        #         }
        #     )
            
        #     self.core_memory.add_experience(exp)
        #     self.logger.debug(f"Stored goal as experience: {goal.description[:50]}...")
            
        # except Exception as e:
        #     self.logger.error(f"Failed to store goal as experience: {e}")
    
    def mark_explored(self, concept_id: str):
        """Mark a concept as explored to reduce its novelty"""
        self.exploration_history.add(concept_id)
        self.logger.debug(f"Marked '{concept_id}' as explored")
    
    def get_active_goals(self, min_priority: float = 0.5) -> List[AutonomousGoal]:
        """Get currently active (high priority) goals"""
        return [g for g in self.generated_goals if g.priority >= min_priority]
    
    def reflect_on_curiosity(self) -> str:
        """
        Generate self-reflective summary of curiosity-driven exploration.
        
        Returns:
            Natural language description of autonomous goals
        """
        if not self.generated_goals:
            return "I have not yet generated autonomous goals. My curiosity awaits awakening."
        
        total_goals = len(self.generated_goals)
        active_goals = len(self.get_active_goals())
        
        # Count by type
        type_counts = {}
        for goal in self.generated_goals:
            t = goal.goal_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        reflection = f"I have generated {total_goals} autonomous goals, "
        reflection += f"of which {active_goals} are currently active. "
        
        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])
            reflection += f"My primary drive is to {dominant_type[0]} ({dominant_type[1]} goals). "
        
        # Latest goal
        if self.generated_goals:
            latest = self.generated_goals[-1]
            reflection += f"My latest curiosity: {latest.description}"
        
        return reflection
