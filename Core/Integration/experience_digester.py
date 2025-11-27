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
import json
import os
import time

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
        
        # Detect phase-resonance wave patterns
        resonance_events = self._detect_phase_resonance_events(particles, duration_ticks)
        
        # Store in Hippocampus
        self._store_in_memory(concepts, relationships, emotional_patterns, wisdom)
        
        # Compile summary
        summary = {
            "concepts_extracted": len(concepts),
            "relationships_found": len(relationships),
            "emotional_patterns": len(emotional_patterns),
            "wisdom_insights": len(wisdom),
            "language_patterns": len(language_patterns),
            "resonance_events_detected": len(resonance_events),
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
                concept=concept["id"],
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
    
    def _detect_phase_resonance_events(
        self,
        particles: List[FluctlightParticle],
        duration_ticks: int
    ) -> List[Dict[str, Any]]:
        """
        Detect phase-resonance wave patterns in particle distribution.
        
        Phase-resonance occurs when:
        1. Information density shows non-uniform (multimodal) distribution
        2. Wavelength clustering suggests standing wave patterns
        3. Particles exhibit synchronized behavior (similar accumulated_time ranges)
        
        Returns:
            List of detected resonance events with metadata
        """
        events = []
        
        if len(particles) < 5:
            return events
        
        # --- Analysis 1: Information Density Distribution ---
        densities = np.array([p.information_density for p in particles])
        density_mean = np.mean(densities)
        density_std = np.std(densities)
        density_kurtosis = self._compute_kurtosis(densities)
        
        # High kurtosis indicates multimodal (peak-valley) distribution = resonance pattern
        if density_kurtosis > 0.5:  # Threshold for non-Gaussian distribution
            events.append({
                "type": "density_multimodality",
                "severity": "high" if density_kurtosis > 1.0 else "medium",
                "metric": density_kurtosis,
                "interpretation": f"Information density shows {density_kurtosis:.2f} kurtosis (peaked, non-uniform)",
                "involved_particles": len([p for p in particles if abs(p.information_density - density_mean) > 2*density_std])
            })
            logger.info(f"🌀 Detected phase-resonance: density multimodality (kurtosis={density_kurtosis:.2f})")
        
        # --- Analysis 2: Wavelength Clustering (Standing Waves) ---
        wavelengths = np.array([p.wavelength for p in particles])
        wavelength_mean = np.mean(wavelengths)
        wavelength_std = np.std(wavelengths)
        
        # Count particles in tight wavelength clusters
        tight_clusters = []
        cluster_threshold = wavelength_std * 0.5
        for w in np.unique(wavelengths):
            cluster_particles = np.sum(np.abs(wavelengths - w) < cluster_threshold)
            if cluster_particles >= max(3, len(particles) // 10):  # At least 10% or 3 particles
                tight_clusters.append((w, cluster_particles))
        
        if len(tight_clusters) >= 2:
            events.append({
                "type": "wavelength_standing_waves",
                "severity": "high" if len(tight_clusters) >= 3 else "medium",
                "clusters": tight_clusters,
                "interpretation": f"Standing wave pattern detected with {len(tight_clusters)} distinct wavelength clusters",
                "involved_particles": sum(c[1] for c in tight_clusters)
            })
            logger.info(f"🌊 Detected phase-resonance: standing waves ({len(tight_clusters)} clusters)")
        
        # --- Analysis 3: Temporal Synchronization ---
        times = np.array([p.accumulated_time for p in particles])
        time_ranges = np.max(times) - np.min(times)
        
        if time_ranges > 0:
            # Partition into temporal bands
            num_bands = max(3, len(particles) // 5)
            time_bands = np.linspace(np.min(times), np.max(times), num_bands)
            band_counts = np.histogram(times, bins=time_bands)[0]
            
            # Compute coefficient of variation of band populations
            band_mean = np.mean(band_counts)
            band_std = np.std(band_counts)
            band_cv = band_std / (band_mean + 1e-9)
            
            # High CV = uneven temporal distribution = synchronization patterns
            if band_cv > 0.8:
                sync_severity = "high" if band_cv > 1.2 else "medium"
                events.append({
                    "type": "temporal_synchronization",
                    "severity": sync_severity,
                    "coefficient_of_variation": band_cv,
                    "interpretation": f"Particles show synchronized temporal clustering (CV={band_cv:.2f})",
                    "involved_particles": len(particles)
                })
                logger.info(f"⏰ Detected phase-resonance: temporal synchronization (CV={band_cv:.2f})")
        
        # --- Analysis 4: Time Dilation Resonance ---
        dilations = np.array([p.time_dilation_factor for p in particles])
        dilation_mean = np.mean(dilations)
        dilation_std = np.std(dilations)
        dilation_kurtosis = self._compute_kurtosis(dilations)
        
        # Detect outlier groups (extreme dilation = intense experiences)
        outlier_threshold = dilation_mean + 2.5 * dilation_std
        outliers = [p for p in particles if p.time_dilation_factor > outlier_threshold]
        
        if len(outliers) >= max(2, len(particles) // 10):
            events.append({
                "type": "extreme_time_dilation_burst",
                "severity": "high" if len(outliers) > len(particles) // 5 else "medium",
                "burst_magnitude": dilation_mean + 2.5 * dilation_std,
                "involved_count": len(outliers),
                "interpretation": f"{len(outliers)} particles experienced extreme time dilation (intensity peaks)",
            })
            logger.info(f"💥 Detected phase-resonance: time dilation burst ({len(outliers)} particles)")
        
        # --- Persist to JSONL ---
        if events:
            self._persist_resonance_events(events, duration_ticks)
        
        return events
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute excess kurtosis (normalized, 0 for Gaussian)."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        m4 = np.mean((data - mean) ** 4)
        m2 = std ** 2
        return (m4 / (m2 ** 2)) - 3  # Excess kurtosis
    
    def _persist_resonance_events(self, events: List[Dict[str, Any]], duration_ticks: int):
        """Write resonance events to JSONL log for external analysis."""
        try:
            os.makedirs("logs", exist_ok=True)
            record = {
                "timestamp": time.time(),
                "duration_ticks": duration_ticks,
                "event_count": len(events),
                "events": events
            }
            with open(os.path.join("logs", "resonance_events.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"✅ Persisted {len(events)} phase-resonance events to resonance_events.jsonl")
        except Exception:
            logger.exception("Failed to persist resonance events")
