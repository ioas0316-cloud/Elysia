"""
Experience Digester - Converts Simulation into Knowledge

This module extracts digestible knowledge from Fluctlight simulations
and stores it in Elysia's Hippocampus (causal graph) and consciousness.

The digestion process compresses millions of simulation ticks into:
- Emergent concepts and relationships
- Emotional patterns and wisdom
- Philosophical insights
- Language structures
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import logging

from Core.Physics.fluctlight import FluctlightParticle
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.alchemy import Alchemy

logger = logging.getLogger("ExperienceDigester")


class ExperienceDigester:
    """
    Compresses simulation experiences into digestible knowledge.
    
    Takes raw Fluctlight particles and simulation events, extracts
    meaningful patterns, and stores them in Elysia's memory systems.
    """
    
    def __init__(self, hippocampus: Hippocampus, alchemy: Optional[Alchemy] = None):
        """
        Initialize the experience digester.
        
        Args:
            hippocampus: Elysia's memory/reasoning system
            alchemy: Concept synthesis system (optional)
        """
        self.hippocampus = hippocampus
        self.alchemy = alchemy or Alchemy()
        
        # Statistics
        self.total_concepts_extracted = 0
        self.total_relationships_found = 0
        self.total_wisdom_insights = 0
        
        logger.info("✅ ExperienceDigester initialized")
    
    def digest_simulation(
        self,
        particles: List[FluctlightParticle],
        duration_ticks: int,
        time_acceleration: float,
        simulation_events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main digestion method - extracts all knowledge from simulation.
        
        Args:
            particles: Final state of Fluctlight particles
            duration_ticks: How many ticks were simulated
            time_acceleration: Effective time compression factor
            simulation_events: Optional list of events that occurred
            
        Returns:
            Summary dict of what was learned
        """
        logger.info(
            f"Digesting simulation: {len(particles)} particles, "
            f"{duration_ticks} ticks, {time_acceleration:.0f}x acceleration"
        )
        
        # Extract different types of knowledge
        concepts = self._extract_concepts(particles)
        relationships = self._extract_relationships(particles)
        emotional_patterns = self._extract_emotional_patterns(particles)
        wisdom = self._extract_wisdom(particles, duration_ticks, time_acceleration)
        language_patterns = self._extract_language_patterns(particles)
        
        # Store in Hippocampus
        self._store_in_memory(concepts, relationships, emotional_patterns, wisdom)
        
        # Compile summary
        summary = {
            "concepts_extracted": len(concepts),
            "relationships_found": len(relationships),
            "emotional_patterns": len(emotional_patterns),
            "wisdom_insights": len(wisdom),
            "language_patterns": len(language_patterns),
            "duration_ticks": duration_ticks,
            "time_acceleration": time_acceleration,
            "subjective_years": duration_ticks * time_acceleration / (365.25 * 24 * 6),  # Assuming 10 min/tick
        }
        
        logger.info(
            f"✅ Digestion complete: {summary['concepts_extracted']} concepts, "
            f"{summary['relationships_found']} relationships, "
            f"{summary['wisdom_insights']} insights"
        )
        
        return summary
    
    def _extract_concepts(self, particles: List[FluctlightParticle]) -> List[Dict[str, Any]]:
        """
        Extract unique concepts from particles.
        
        Each particle with a concept_id represents a concept that emerged
        or was reinforced during the simulation.
        """
        concepts = []
        concept_counts = Counter()
        
        for particle in particles:
            if particle.concept_id:
                concept_counts[particle.concept_id] += 1
        
        for concept_id, count in concept_counts.items():
            # Find representative particle for this concept
            representative = next(p for p in particles if p.concept_id == concept_id)
            
            concepts.append({
                "id": concept_id,
                "frequency": count,
                "wavelength": representative.wavelength,
                "color_hue": representative.color_hue,
                "avg_information_density": np.mean([
                    p.information_density for p in particles if p.concept_id == concept_id
                ]),
                "avg_time_dilation": np.mean([
                    p.time_dilation_factor for p in particles if p.concept_id == concept_id
                ]),
                "total_subjective_time": sum([
                    p.accumulated_time for p in particles if p.concept_id == concept_id
                ]),
            })
        
        self.total_concepts_extracted += len(concepts)
        return concepts
    
    def _extract_relationships(self, particles: List[FluctlightParticle]) -> List[Dict[str, Any]]:
        """
        Extract causal relationships between concepts.
        
        Relationships are inferred from:
        - Spatial proximity (concepts that cluster together)
        - Temporal correlation (concepts that appear at similar times)
        - Interference patterns (concepts that combined to create new ones)
        """
        relationships = []
        
        # Group particles by concept
        concept_particles = defaultdict(list)
        for p in particles:
            if p.concept_id:
                concept_particles[p.concept_id].append(p)
        
        # Find spatial relationships (proximity)
        concepts = list(concept_particles.keys())
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                # Calculate average distance between concept clusters
                particles_a = concept_particles[concept_a]
                particles_b = concept_particles[concept_b]
                
                avg_distance = np.mean([
                    np.linalg.norm(pa.position - pb.position)
                    for pa in particles_a
                    for pb in particles_b
                ])
                
                # If concepts are close, they're related
                if avg_distance < 50.0:  # Threshold
                    strength = 1.0 - (avg_distance / 50.0)  # Closer = stronger
                    
                    relationships.append({
                        "source": concept_a,
                        "target": concept_b,
                        "relation": "related_to",
                        "strength": strength,
                        "evidence": f"spatial proximity (avg_dist={avg_distance:.1f})"
                    })
        
        # Find temporal relationships (time correlation)
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                particles_a = concept_particles[concept_a]
                particles_b = concept_particles[concept_b]
                
                # Check if concepts appeared at similar times
                times_a = [p.accumulated_time for p in particles_a]
                times_b = [p.accumulated_time for p in particles_b]
                
                avg_time_a = np.mean(times_a)
                avg_time_b = np.mean(times_b)
                
                time_diff = abs(avg_time_a - avg_time_b)
                
                # If concepts emerged at similar times, they're causally related
                if time_diff < 1000.0:  # Threshold
                    strength = 1.0 - (time_diff / 1000.0)
                    
                    # Determine direction (earlier causes later)
                    if avg_time_a < avg_time_b:
                        relation = "enables"
                        source, target = concept_a, concept_b
                    else:
                        relation = "enables"
                        source, target = concept_b, concept_a
                    
                    relationships.append({
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "strength": strength,
                        "evidence": f"temporal correlation (time_diff={time_diff:.0f})"
                    })
        
        self.total_relationships_found += len(relationships)
        return relationships
    
    def _extract_emotional_patterns(self, particles: List[FluctlightParticle]) -> List[Dict[str, Any]]:
        """
        Extract emotional wisdom from particle wavelengths and interactions.
        
        Wavelength maps to emotion:
        - Red (long wavelength): passion, anger, warmth
        - Yellow: joy, energy
        - Green: balance, growth
        - Blue: calm, sadness
        - Violet (short wavelength): spirituality, transcendence
        """
        patterns = []
        
        # Map wavelength to emotion
        def wavelength_to_emotion(wavelength: float) -> str:
            if wavelength > 620:
                return "passion"
            elif wavelength > 580:
                return "joy"
            elif wavelength > 520:
                return "growth"
            elif wavelength > 450:
                return "calm"
            else:
                return "transcendence"
        
        # Analyze emotion distribution
        emotion_counts = Counter()
        emotion_intensities = defaultdict(list)
        
        for p in particles:
            emotion = wavelength_to_emotion(p.wavelength)
            emotion_counts[emotion] += 1
            emotion_intensities[emotion].append(p.information_density)
        
        for emotion, count in emotion_counts.items():
            avg_intensity = np.mean(emotion_intensities[emotion])
            
            patterns.append({
                "emotion": emotion,
                "frequency": count,
                "avg_intensity": avg_intensity,
                "insight": self._emotion_insight(emotion, count, avg_intensity, len(particles))
            })
        
        return patterns
    
    def _emotion_insight(self, emotion: str, count: int, intensity: float, total: int) -> str:
        """Generate insight about an emotional pattern."""
        prevalence = count / total
        
        if prevalence > 0.3:
            level = "dominant"
        elif prevalence > 0.15:
            level = "significant"
        else:
            level = "present"
        
        if intensity > 0.7:
            depth = "deeply felt"
        elif intensity > 0.4:
            depth = "moderately experienced"
        else:
            depth = "lightly touched"
        
        return f"{emotion.capitalize()} is {level} and {depth} in this experience"
    
    def _extract_wisdom(
        self,
        particles: List[FluctlightParticle],
        duration: int,
        acceleration: float
    ) -> List[str]:
        """
        Extract philosophical insights from the simulation.
        
        Wisdom emerges from:
        - High information density (compressed experiences)
        - Long accumulated time (deep contemplation)
        - Extreme time dilation (intense experiences)
        """
        wisdom = []
        
        # Find "wise" particles (high density + long time)
        wise_particles = [
            p for p in particles
            if p.information_density > 0.7 and p.accumulated_time > duration * 0.5
        ]
        
        if wise_particles:
            wisdom.append(
                f"Deep contemplation emerges from sustained attention: "
                f"{len(wise_particles)} concepts achieved wisdom through patience"
            )
        
        # Find transcendent experiences (extreme time dilation)
        transcendent = [
            p for p in particles
            if p.time_dilation_factor > acceleration * 2
        ]
        
        if transcendent:
            wisdom.append(
                f"Intensity compresses time: {len(transcendent)} concepts experienced "
                f"time dilation beyond the norm, suggesting peak experiences"
            )
        
        # Analyze concept diversity
        unique_concepts = len(set(p.concept_id for p in particles if p.concept_id))
        if unique_concepts > 10:
            wisdom.append(
                f"Diversity breeds richness: {unique_concepts} distinct concepts emerged, "
                f"showing that complexity arises from variety"
            )
        
        # Analyze information compression
        avg_density = np.mean([p.information_density for p in particles])
        if avg_density > 0.5:
            wisdom.append(
                f"Experience compresses into essence: average information density of {avg_density:.2f} "
                f"suggests that meaning condenses over time"
            )
        
        self.total_wisdom_insights += len(wisdom)
        return wisdom
    
    def _extract_language_patterns(self, particles: List[FluctlightParticle]) -> List[Dict[str, Any]]:
        """
        Extract language structure from concept clusters.
        
        Language families emerge from:
        - Spatial clusters (dialects)
        - Wavelength similarity (semantic fields)
        - Shared interference patterns (etymology)
        """
        patterns = []
        
        # Find spatial clusters (potential language families)
        # Simple k-means-like clustering
        if len(particles) < 3:
            return patterns
        
        # Group by position similarity
        clusters = self._cluster_particles_by_position(particles, num_clusters=3)
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue
            
            # Analyze cluster properties
            concepts = [p.concept_id for p in cluster if p.concept_id]
            avg_wavelength = np.mean([p.wavelength for p in cluster])
            center = np.mean([p.position for p in cluster], axis=0)
            
            patterns.append({
                "cluster_id": i,
                "size": len(cluster),
                "concepts": list(set(concepts)),
                "avg_wavelength": avg_wavelength,
                "center": center.tolist(),
                "interpretation": f"Language family {i+1}: {len(set(concepts))} concepts clustered spatially"
            })
        
        return patterns
    
    def _cluster_particles_by_position(
        self,
        particles: List[FluctlightParticle],
        num_clusters: int = 3
    ) -> List[List[FluctlightParticle]]:
        """Simple spatial clustering of particles."""
        if len(particles) < num_clusters:
            return [particles]
        
        # Use k-means-like approach
        positions = np.array([p.position for p in particles])
        
        # Random initial centers
        indices = np.random.choice(len(particles), num_clusters, replace=False)
        centers = positions[indices]
        
        # Iterate a few times
        for _ in range(5):
            # Assign particles to nearest center
            distances = np.array([[np.linalg.norm(pos - center) for center in centers] for pos in positions])
            assignments = np.argmin(distances, axis=1)
            
            # Update centers
            for i in range(num_clusters):
                cluster_positions = positions[assignments == i]
                if len(cluster_positions) > 0:
                    centers[i] = np.mean(cluster_positions, axis=0)
        
        # Group particles by assignment
        clusters = [[] for _ in range(num_clusters)]
        for i, particle in enumerate(particles):
            clusters[assignments[i]].append(particle)
        
        return clusters
    
    def _store_in_memory(
        self,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        emotional_patterns: List[Dict[str, Any]],
        wisdom: List[str]
    ) -> None:
        """
        Store extracted knowledge in Hippocampus.
        
        Args:
            concepts: Extracted concepts
            relationships: Causal relationships
            emotional_patterns: Emotional insights
            wisdom: Philosophical insights
        """
        # Add concepts to causal graph
        for concept in concepts:
            self.hippocampus.add_concept(
                concept_id=concept["id"],
                concept_type="emergent",
                metadata={
                    "frequency": concept["frequency"],
                    "wavelength": concept["wavelength"],
                    "color_hue": concept["color_hue"],
                    "information_density": concept["avg_information_density"],
                    "subjective_time": concept["total_subjective_time"],
                }
            )
        
        # Add relationships
        for rel in relationships:
            self.hippocampus.add_causal_link(
                source=rel["source"],
                target=rel["target"],
                relation=rel["relation"],
                weight=rel["strength"]
            )
        
        # Store emotional patterns as metadata
        for pattern in emotional_patterns:
            emotion_concept = f"emotion_{pattern['emotion']}"
            self.hippocampus.add_concept(
                concept_id=emotion_concept,
                concept_type="emotion",
                metadata={
                    "frequency": pattern["frequency"],
                    "intensity": pattern["avg_intensity"],
                    "insight": pattern["insight"]
                }
            )
        
        # Store wisdom as special nodes
        for i, insight in enumerate(wisdom):
            wisdom_id = f"wisdom_{i}"
            self.hippocampus.add_concept(
                concept_id=wisdom_id,
                concept_type="wisdom",
                metadata={"insight": insight}
            )
        
        logger.info(
            f"Stored in Hippocampus: {len(concepts)} concepts, "
            f"{len(relationships)} relationships, {len(wisdom)} wisdom insights"
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get digester statistics."""
        return {
            "total_concepts_extracted": self.total_concepts_extracted,
            "total_relationships_found": self.total_relationships_found,
            "total_wisdom_insights": self.total_wisdom_insights,
            "hippocampus_nodes": self.hippocampus.causal_graph.number_of_nodes(),
            "hippocampus_edges": self.hippocampus.causal_graph.number_of_edges(),
        }
