"""
Concept Evolution Engine
========================
Enables Elysia to autonomously discover new concepts from Fluctlight interference patterns.
Ported and modernized from Legacy/Project_Elysia/learning/self_evolution.py.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from Core.Physics.fluctlight import FluctlightEngine, FluctlightParticle
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.alchemy import Alchemy

logger = logging.getLogger("ConceptEvolution")

class ConceptEvolution:
    """
    Analyzes Fluctlight interference to discover and name new concepts.
    Integrates with Hippocampus to store discoveries.
    """
    
    def __init__(self, hippocampus: Hippocampus, alchemy: Alchemy):
        self.hippocampus = hippocampus
        self.alchemy = alchemy
        self.discovered_concepts: List[Dict] = []
        
    def evolve_concepts(self, engine: FluctlightEngine) -> List[str]:
        """
        Main entry point: Analyze engine state and evolve new concepts.
        Returns list of newly discovered concept names.
        """
        # 1. Detect strong interference clusters
        clusters = self._detect_interference_clusters(engine)
        
        new_concepts = []
        for cluster in clusters:
            # 2. Synthesize new concept from cluster
            concept_data = self._synthesize_concept(cluster)
            
            if concept_data:
                # 3. Register in Hippocampus
                self._integrate_discovery(concept_data)
                new_concepts.append(concept_data['name'])
                
        return new_concepts
    
    def _detect_interference_clusters(self, engine: FluctlightEngine) -> List[List[FluctlightParticle]]:
        """
        Finds groups of particles that are interacting strongly.
        Simplified spatial clustering for now.
        """
        # In a real implementation, we'd use DBSCAN or similar on particle positions
        # For now, we'll look for particles very close to each other
        
        particles = engine.particles
        if len(particles) < 2:
            return []
            
        clusters = []
        processed = set()
        
        # Simple O(N^2) clustering (limit N in engine)
        for i, p1 in enumerate(particles):
            if i in processed: continue
            
            cluster = [p1]
            processed.add(i)
            
            for j, p2 in enumerate(particles):
                if j in processed: continue
                
                dist = np.linalg.norm(p1.position - p2.position)
                if dist < 10.0:  # Interaction radius
                    cluster.append(p2)
                    processed.add(j)
            
            if len(cluster) >= 2:
                clusters.append(cluster)
                
        return clusters

    def _synthesize_concept(self, cluster: List[FluctlightParticle]) -> Optional[Dict]:
        """
        Creates a new concept from a cluster of particles.
        """
        # Extract source concepts
        sources = list(set(p.concept_id for p in cluster if p.concept_id is not None))
        if len(sources) < 2:
            return None
            
        # Generate name using Alchemy
        # Try pairs first
        name = self.alchemy.combine(sources[0], sources[1])
        
        # If alchemy returned a simple combination, try to be more creative
        if "-" in name and len(sources) > 2:
             name = f"{sources[0]}_{sources[1]}_complex"

        # Check if already known
        if self.hippocampus.causal_graph.has_node(name):
            return None # Already known
            
        # Calculate properties
        avg_pos = np.mean([p.position for p in cluster], axis=0)
        avg_wavelength = np.mean([p.wavelength for p in cluster])
        
        # Update unnamed particles in the cluster with the new name
        for p in cluster:
            if p.concept_id is None:
                p.concept_id = name
        
        return {
            "name": name,
            "sources": sources,
            "position": avg_pos,
            "wavelength": avg_wavelength,
            "strength": len(cluster)  # More particles = stronger concept
        }
        
    def _integrate_discovery(self, data: Dict):
        """
        Adds the new concept to the Hippocampus.
        """
        name = data['name']
        sources = data['sources']
        
        # Add node
        self.hippocampus.add_concept(name)
        
        # Add causal links to parents
        for source in sources:
            self.hippocampus.add_causal_link(source, name, relation="emerged_to", weight=0.8)
            
        logger.info(f"✨ EVOLUTION: Discovered concept '{name}' from {sources}")
        self.discovered_concepts.append(data)
