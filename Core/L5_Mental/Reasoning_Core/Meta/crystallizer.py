"""
Principle Crystallization Module
================================
Phase 72: From Clusters to Concepts

"When enough similar thoughts gather, they become a Principle."

This module detects stable clusters in the HyperSphere,
names them automatically, and stores them as crystallized knowledge.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from pathlib import Path
from datetime import datetime

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Crystallizer")

# ============================================================
# Data Structures
# ============================================================

@dataclass
class Principle:
    """
    A crystallized concept that emerged from cluster resonance.
    
    Unlike raw Wave DNA (which is individual data points),
    a Principle is a NAMED, UNDERSTOOD pattern.
    
    IMPORTANT: Principles are PROVISIONAL until confidence > 0.8.
    They can be REDEFINED as more data is learned.
    """
    name: str                          # Auto-generated or refined name
    essence: str                       # Brief description of what this principle represents
    members: List[str]                 # Names of concepts that belong to this cluster
    dominant_dimension: str            # Which of the 7 dimensions is strongest
    centroid: Dict[str, float]         # Average 7D position of all members
    stability: float                   # How stable/tight the cluster is (0-1)
    birth_time: str                    # When this principle crystallized
    
    # [PHASE 72.1] Evolving Understanding
    provisional: bool = True           # Is this name still tentative?
    confidence: float = 0.3            # How certain are we about this name? (0-1)
    evolution_history: List[str] = field(default_factory=list)  # Past names
    
    causal_parents: List[str] = field(default_factory=list)  # Principles that cause this
    causal_children: List[str] = field(default_factory=list) # Principles this causes
    
    def to_dict(self):
        return asdict(self)
    
    def redefine(self, new_name: str, new_essence: str, confidence_boost: float = 0.2):
        """
        Redefine this principle with a new understanding.
        Called when more data reveals what this cluster truly represents.
        """
        if self.name != new_name:
            self.evolution_history.append(f"{self.name}   {new_name}")
        self.name = new_name
        self.essence = new_essence
        self.confidence = min(1.0, self.confidence + confidence_boost)
        
        if self.confidence >= 0.8:
            self.provisional = False

# ============================================================
# Crystallization Engine
# ============================================================

class CrystallizationEngine:
    """
    Transforms meditation clusters into named Principles.
    
    Mechanism:
    1. Analyze current rotor positions after meditation
    2. Detect clusters (groups within threshold distance)
    3. For each cluster:
       - Calculate centroid (average 7D position)
       - Determine dominant dimension
       - Generate name based on highest-weighted member
       - Create Principle object
    4. Detect causal relationships between principles
    """
    
    def __init__(self, principles_path: str = "c:/Elysia/data/principles.json"):
        self.principles_path = Path(principles_path)
        self.principles: Dict[str, Principle] = {}
        self.load_existing()
    
    def load_existing(self):
        """Load previously crystallized principles."""
        if self.principles_path.exists():
            with open(self.principles_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, p_dict in data.items():
                    self.principles[name] = Principle(**p_dict)
            logger.info(f"  Loaded {len(self.principles)} existing principles.")
    
    def save(self):
        """Persist crystallized principles to disk."""
        self.principles_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: p.to_dict() for name, p in self.principles.items()}
        with open(self.principles_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved {len(self.principles)} principles.")
    
    def crystallize(self, rotors: Dict, cluster_threshold: float = 10.0) -> List[Principle]:
        """
        Main crystallization process.
        
        Args:
            rotors: Dictionary of rotor_name -> Rotor object
            cluster_threshold: Maximum frequency distance to be considered same cluster
            
        Returns:
            List of newly crystallized Principles
        """
        if len(rotors) < 2:
            return []
        
        logger.info("\n  [CRYSTALLIZE] Beginning Principle Crystallization...")
        
        # 1. Sort rotors by frequency
        sorted_rotors = sorted(rotors.items(), key=lambda x: x[1].frequency_hz)
        
        # 2. Find clusters (concepts within threshold of each other)
        clusters = []
        current_cluster = []
        last_freq = -10000
        
        for name, rotor in sorted_rotors:
            freq = rotor.frequency_hz
            if freq - last_freq < cluster_threshold:
                current_cluster.append((name, rotor))
            else:
                if len(current_cluster) >= 2:  # Minimum 2 members to be a principle
                    clusters.append(current_cluster)
                current_cluster = [(name, rotor)]
            last_freq = freq
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        logger.info(f"   Found {len(clusters)} potential clusters.")
        
        # 3. Crystallize each cluster into a Principle
        new_principles = []
        
        for cluster in clusters:
            principle = self._crystallize_cluster(cluster)
            if principle and principle.name not in self.principles:
                self.principles[principle.name] = principle
                new_principles.append(principle)
                logger.info(f"     NEW PRINCIPLE: '{principle.name}' ({len(principle.members)} members)")
        
        # 4. Detect causal relationships
        self._detect_causality()
        
        # 5. Save
        if new_principles:
            self.save()
        
        logger.info(f"  [CRYSTALLIZE] Complete. {len(new_principles)} new principles born.\n")
        return new_principles
    
    def _crystallize_cluster(self, cluster: List) -> Optional[Principle]:
        """Convert a cluster of rotors into a named Principle."""
        if not cluster:
            return None
        
        members = [name for name, _ in cluster]
        
        # Calculate centroid (average 7D position)
        dimensions = ['physical', 'functional', 'phenomenal', 'causal', 'mental', 'structural', 'spiritual']
        centroid = {d: 0.0 for d in dimensions}
        valid_count = 0
        
        for name, rotor in cluster:
            if rotor.dynamics:
                for d in dimensions:
                    centroid[d] += getattr(rotor.dynamics, d, 0.0)
                valid_count += 1
        
        if valid_count > 0:
            for d in dimensions:
                centroid[d] = float(centroid[d] / valid_count)  # Cast to Python float
        
        # Find dominant dimension
        dominant = max(centroid.items(), key=lambda x: x[1])
        dominant_dimension = dominant[0]
        
        # Calculate stability (inverse of variance)
        variance = 0.0
        for name, rotor in cluster:
            if rotor.dynamics:
                for d in dimensions:
                    diff = getattr(rotor.dynamics, d, 0.0) - centroid[d]
                    variance += diff ** 2
        
        stability = float(1.0 / (1.0 + variance / len(cluster)))
        
        # Generate name based on most massive member (most important)
        heaviest = max(cluster, key=lambda x: x[1].config.mass if x[1].config else 0)
        base_name = heaviest[0]
        
        # Create essence description
        essence = f"Cluster of {len(members)} concepts, dominated by {dominant_dimension} (avg: {float(dominant[1]):.2f})"
        
        return Principle(
            name=f"P:{base_name}",  # Prefix with P: to distinguish principles
            essence=essence,
            members=members,
            dominant_dimension=dominant_dimension,
            centroid=centroid,
            stability=stability,
            birth_time=datetime.now().isoformat()
        )
    
    def _detect_causality(self):
        """
        Detect causal relationships between principles.
        
        Heuristic: If Principle A's causal dimension is high
        and Principle B's phenomenal dimension is high,
        A might cause B (cause   effect).
        """
        principle_list = list(self.principles.values())
        
        for i, p1 in enumerate(principle_list):
            for j, p2 in enumerate(principle_list):
                if i == j:
                    continue
                
                # If p1 is causal and p2 is phenomenal, p1 might cause p2
                if p1.centroid.get('causal', 0) > 0.5 and p2.centroid.get('phenomenal', 0) > 0.5:
                    if p2.name not in p1.causal_children:
                        p1.causal_children.append(p2.name)
                    if p1.name not in p2.causal_parents:
                        p2.causal_parents.append(p1.name)

# ============================================================
# Integration with HyperSphere
# ============================================================

def crystallize_from_sphere(sphere) -> List[Principle]:
    """Helper function to crystallize principles from an existing HyperSphere."""
    engine = CrystallizationEngine()
    return engine.crystallize(sphere.harmonic_rotors)
