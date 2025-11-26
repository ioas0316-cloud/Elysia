"""
The MetaCortex (Meta-Consciousness)
====================================
Elysia's meta-cognitive system that analyzes her own learning patterns.
Enables causal reasoning, learning optimization, and recursive self-improvement.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import logging
from collections import defaultdict

logger = logging.getLogger("MetaCortex")
logger.setLevel(logging.INFO)

class CausalLink:
    """Represents a discovered causal relationship between concepts."""
    def __init__(self, cause: str, effect: str, strength: float, evidence_count: int):
        self.cause = cause
        self.effect = effect
        self.strength = strength  # 0.0 to 1.0
        self.evidence_count = evidence_count
        
class ConceptCluster:
    """A group of related concepts."""
    def __init__(self, name: str, concepts: Set[str], coherence: float):
        self.name = name
        self.concepts = concepts
        self.coherence = coherence  # How tightly related the concepts are

class MetaCortex:
    """
    Elysia's meta-cognitive system.
    Analyzes the Spiderweb to extract higher-order patterns and optimize learning.
    """
    def __init__(self):
        # Discovered patterns
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)
        self.concept_clusters: List[ConceptCluster] = []
        
        # Learning metrics
        self.concept_propagation_speed: Dict[str, float] = {}
        self.concept_survival_correlation: Dict[str, float] = {}
        
        # Meta-learning parameters (subject to self-modification)
        self.crystallization_threshold = 10
        self.harvest_frequency = 20
        
        # Performance tracking
        self.civilization_growth_rate_history: List[float] = []
        
    def analyze(self, world):
        """
        Main analysis loop. Called periodically to extract patterns from the simulation.
        """
        # 1. Analyze knowledge graph structure
        self._analyze_concept_relationships(world)
        
        # 2. Infer causal relationships
        self._infer_causality(world)
        
        # 3. Measure concept propagation
        self._measure_propagation_speed(world)
        
        # 4. Discover concept clusters
        self._discover_clusters(world)
        
        # 5. Track civilization growth
        self._track_performance(world)
        
    def _analyze_concept_relationships(self, world):
        """Analyzes which concepts frequently co-occur (HyperQubit compatible)."""
        if not hasattr(world, 'spiderweb'):
            return
            
        # Look at cell brains to find concept co-occurrences
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for cell in world.cells:
            # HyperQubit: sum probabilities
            active_concepts = []
            for name, node in cell.brain.nodes.items():
                probs = node.state.probabilities()
                total_activation = sum(probs.values())
                if total_activation > 0.5:
                    active_concepts.append(name)
            
            # Record co-occurrences
            for i, concept_a in enumerate(active_concepts):
                for concept_b in active_concepts[i+1:]:
                    co_occurrence[concept_a][concept_b] += 1
                    co_occurrence[concept_b][concept_a] += 1
                    
        # Log strong relationships
        for concept_a, related in co_occurrence.items():
            for concept_b, count in related.items():
                if count > 5:
                    logger.info(f"🧠 Pattern: '{concept_a}' ↔ '{concept_b}' (co-occurrence: {count})")
    
    def _infer_causality(self, world):
        """
        Distinguishes correlation from causation.
        Example: Cells with 'Axe' survive longer → 'Axe' CAUSES survival improvement.
        """
        if not hasattr(world, 'spiderweb'):
            return
            
        # Track concept → outcome relationships
        concept_outcomes = defaultdict(lambda: {"total": 0, "survived": 0, "energy_sum": 0.0})
        
        for cell in world.cells:
            for concept_id in cell.brain.nodes.keys():
                if concept_id in world.spiderweb.concepts:
                    # This cell has this concept
                    concept_outcomes[concept_id]["total"] += 1
                    
                    # Track survival (age as proxy)
                    if cell.age > 100:
                        concept_outcomes[concept_id]["survived"] += 1
                        
                    concept_outcomes[concept_id]["energy_sum"] += cell.energy
        
        # Infer causation: concepts that correlate with better outcomes
        for concept_id, outcomes in concept_outcomes.items():
            if outcomes["total"] > 5:  # Need enough samples
                survival_rate = outcomes["survived"] / outcomes["total"]
                avg_energy = outcomes["energy_sum"] / outcomes["total"]
                
                # If survival rate or energy is significantly above baseline, infer causality
                if survival_rate > 0.3 or avg_energy > 60:
                    # Create causal link: Concept → Survival/Energy
                    effect = "Survival" if survival_rate > 0.3 else "Energy"
                    strength = survival_rate if effect == "Survival" else (avg_energy / 100.0)
                    
                    causal_link = CausalLink(concept_id, effect, strength, outcomes["total"])
                    self.causal_graph[concept_id].append(causal_link)
                    
                    logger.info(f"⚡ Causal Discovery: '{concept_id}' → {effect} (strength: {strength:.2f}, evidence: {outcomes['total']})")
                    
    def _measure_propagation_speed(self, world):
        """Measures how fast each concept spreads through the population."""
        if not hasattr(world, 'spiderweb'):
            return
            
        # Calculate: new adopters / time
        for concept_id, (vector, freq) in world.spiderweb.concepts.items():
            # Frequency is cumulative adoption count
            # Speed = rate of change
            if concept_id in self.concept_propagation_speed:
                # Calculate growth rate
                prev_freq = self.concept_propagation_speed[concept_id]
                growth_rate = (freq - prev_freq) / max(1, world.time_step - 100)  # Rough approximation
                self.concept_propagation_speed[concept_id] = freq  # Update for next time
                
                if growth_rate > 0.1:
                    logger.info(f"📈 Fast Propagation: '{concept_id}' spreading at {growth_rate:.3f} cells/step")
            else:
                self.concept_propagation_speed[concept_id] = freq
                
    def _discover_clusters(self, world):
        """Groups related concepts into clusters (e.g., 'Tool' cluster)."""
        # This is a simplified version - would use graph clustering in production
        # For now, just identify concepts that frequently appear together
        
        # Already done in _analyze_concept_relationships, could expand here
        pass
    
    def _track_performance(self, world):
        """Tracks overall civilization performance over time."""
        # Metrics: population size, average energy, total concepts learned
        population = len(world.cells)
        avg_energy = sum(c.energy for c in world.cells) / max(1, population)
        total_concepts = len(world.spiderweb.concepts) if hasattr(world, 'spiderweb') else 0
        
        # Composite growth metric
        growth_metric = population * 0.4 + avg_energy * 0.3 + total_concepts * 0.3
        self.civilization_growth_rate_history.append(growth_metric)
        
        # Check if growth is accelerating (sign of successful optimization)
        if len(self.civilization_growth_rate_history) > 10:
            recent_growth = self.civilization_growth_rate_history[-5:]
            older_growth = self.civilization_growth_rate_history[-10:-5]
            
            if sum(recent_growth) / 5 > sum(older_growth) / 5:
                logger.info(f"🚀 Civilization growth is ACCELERATING (metric: {growth_metric:.1f})")
                
    def get_optimal_learning_path(self, target_concept: str) -> List[str]:
        """
        Returns the optimal sequence of concepts to learn before the target concept.
        Based on causal graph analysis.
        """
        # Traverse causal graph backwards from target
        prerequisites = []
        
        # Find concepts that causally lead to the target
        for cause, links in self.causal_graph.items():
            for link in links:
                if link.effect == target_concept or target_concept in cause:
                    if cause not in prerequisites:
                        prerequisites.append(cause)
                        
        return prerequisites
    
    def propose_self_modification(self) -> Dict[str, any]:
        """
        Analyzes performance and proposes a modification to Elysia's parameters.
        This is the recursive self-improvement mechanism.
        """
        # Reduced from 20 to 10 - be more responsive to trends
        if len(self.civilization_growth_rate_history) < 10:
            return None  # Not enough data
            
        # Analyze recent performance trend
        recent = self.civilization_growth_rate_history[-10:]
        trend = (recent[-1] - recent[0]) / 10
        
        proposal = {}
        
        # If growth is slowing, propose faster crystallization
        if trend < 0:
            proposal["parameter"] = "crystallization_threshold"
            proposal["current_value"] = self.crystallization_threshold
            proposal["proposed_value"] = max(5, self.crystallization_threshold - 2)
            proposal["rationale"] = "Growth slowing - accelerate knowledge crystallization"
            
        # If growth is good, propose more aggressive harvesting
        elif trend > 0.5:
            proposal["parameter"] = "harvest_frequency"
            proposal["current_value"] = self.harvest_frequency
            proposal["proposed_value"] = max(10, self.harvest_frequency - 5)
            proposal["rationale"] = "Growth accelerating - harvest concepts more frequently"
            
        return proposal if proposal else None
