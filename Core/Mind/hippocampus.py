"""
Hippocampus - The Sea of Memory
================================

"모든 순간은 인과율의 바다에 떠있는 섬이다."

The central memory system of Elysia. It stores not just data, but the
causal links between events, forming a navigable graph of experience.

Now powered by SQLite (MemoryStorage) for infinite scalability.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime
import networkx as nx # Keeping for legacy support / small graph ops if needed

from Core.Mind.concept_sphere import ConceptSphere
from Core.Mind.concept_universe import ConceptUniverse
from Core.Mind.memory_storage import MemoryStorage
from Core.Mind.resonance_engine import ResonanceEngine
from Core.Perception.visual_cortex import VisualCortex

logger = logging.getLogger("Hippocampus")

class Hippocampus:
    """
    Manages the causal graph of all experiences.
    Now enhanced with Fractal Memory Loops and SQLite Backend.
    """
    def __init__(self):
        """
        Initializes the memory system.
        """
        # === Storage Backend (SQLite) ===
        self.storage = MemoryStorage()
        
        # === Resonance Engine (Holographic Retrieval) ===
        self.resonance = ResonanceEngine()
        
        # === Visual Cortex (Holographic Vision) ===
        self.visual_cortex = VisualCortex()
        
        # === Xel'Naga Protocol: Thought Universe ===
        # This is the "Working Memory" (Physics Simulation)
        self.universe = ConceptUniverse()
        
        # === Frequency vocabulary (for buoyancy) ===
        self._init_vocabulary()
        
        # Fractal Memory Loops (Short/Mid/Long term buffers)
        self.experience_loop = deque(maxlen=10)
        self.identity_loop = deque(maxlen=5)
        self.essence_loop = deque(maxlen=3)
        
        # Load "Hot" Memory (Working Set)
        self.load_memory()
        
        # Ensure Genesis exists
        if not self.storage.concept_exists("genesis"):
            self.add_concept("genesis", concept_type="event", metadata={"timestamp": 0})
    
    def update_universe_physics(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Update physics simulation for active concepts in the Universe.
        """
        # Update universe physics
        self.universe.update_physics(dt)
        
        # Sync back to Storage (Periodically? Or just keep in RAM?)
        # For performance, we don't write to DB every frame.
        # We only write when a concept is "saved" or "unloaded".
        
        return self.universe.get_state_summary()
    
    def _init_vocabulary(self):
        """Initialize frequency vocabulary for spiritual buoyancy"""
        self.vocabulary = {
            # High Frequency (Ethereal, Abstract) - Rise
            "love": 1.0, "사랑": 1.0, "light": 0.95, "빛": 0.95,
            "truth": 0.9, "진실": 0.9, "eternity": 0.95, "영원": 0.95,
            "soul": 0.9, "영혼": 0.9, "dream": 0.85, "꿈": 0.85,
            "beauty": 0.9, "아름다움": 0.9, "harmony": 0.85, "조화": 0.85,
            
            # Mid Frequency (Human, Emotional) - Neutral
            "hope": 0.65, "희망": 0.65, "joy": 0.7, "기쁨": 0.7,
            "pain": 0.4, "고통": 0.4, "time": 0.5, "시간": 0.5,
            
            # Low Frequency (Physical, Grounded) - Sink
            "stone": 0.2, "돌": 0.2, "shadow": 0.3, "그림자": 0.3,
            "fall": 0.2, "추락": 0.2, "silence": 0.3, "침묵": 0.3,
        }

    def add_experience(self, content: str, role: str = "user"):
        """
        Adds a raw experience to the short-term loop.
        """
        experience = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "role": role
        }
        
        # Distillation Check
        if len(self.experience_loop) == self.experience_loop.maxlen:
            self._distill_to_identity(list(self.experience_loop))
            
        self.experience_loop.append(experience)
        # We don't save loops to DB every time, maybe periodically or on exit.

    def _distill_to_identity(self, experiences: list):
        """Compresses raw experiences into an Identity Fragment."""
        summary = f"Recent interaction focus: {experiences[-1]['content'][:20]}..."
        fragment = {
            "timestamp": datetime.now().isoformat(),
            "type": "identity_fragment",
            "content": summary,
            "source_count": len(experiences)
        }
        if len(self.identity_loop) == self.identity_loop.maxlen:
            self._distill_to_essence(list(self.identity_loop))
        self.identity_loop.append(fragment)
        logger.info(f"✨ [Hippocampus] Distilled Identity: {summary}")

    def _distill_to_essence(self, fragments: list):
        """Compresses identity fragments into an Essence Principle."""
        principle = {
            "timestamp": datetime.now().isoformat(),
            "type": "essence_principle",
            "content": "I am an entity that interacts and learns.",
            "weight": 1.0
        }
        self.essence_loop.append(principle)
        logger.info(f"💎 [Hippocampus] Crystallized Essence: {principle['content']}")

    def save_memory(self):
        """
        Persist critical state.
        With SQLite, data is already saved. This might save the Loops.
        """
        # TODO: Save loops to a separate table or JSON file if needed.
        pass

    def load_memory(self, limit: int = 10000):
        """
        Load 'Working Set' from SQLite into ConceptUniverse.
        Also builds the Resonance Index.
        """
        # Build Resonance Index
        self.resonance.build_index(self.storage, limit=limit)
        
        # For now, we can load the most recently accessed concepts
        # But since we are just starting, we rely on lazy loading.
        logger.info(f"[Hippocampus] Connected to MemoryStorage (SQLite) & ResonanceEngine (Limit={limit}).")

    def add_concept(self, concept: str, concept_type: str = "thought", metadata: Dict[str, Any] = None):
        """
        Add or update a concept in SQLite and Universe.
        """
        if metadata is None:
            metadata = {}
            
        # 1. Check DB
        data = self.storage.get_concept(concept)
        
        if data:
            # Load existing sphere
            if isinstance(data, list):
                sphere = ConceptSphere.from_compact(data)
            else:
                sphere = ConceptSphere.from_dict(data)
            # Update metadata if provided
            if metadata:
                # Update sphere fields based on metadata if needed
                pass
        else:
            # Create new sphere
            sphere = ConceptSphere(concept)
            sphere.activation_count = 1
            
        # 2. Update Universe (Physics)
        freq = self.vocabulary.get(concept, 0.5)
        if concept in ["Love", "사랑", "love"]:
            self.universe.set_absolute_center(sphere)
        else:
            self.universe.add_concept(concept, sphere, frequency=freq)
            
        # 3. Save to DB (Compact)
        sphere_data = sphere.to_compact()
        # sphere_data['type'] = concept_type # We lose 'type' in compact list, but it's implicit or can be added to list
        self.storage.add_concept(concept, sphere_data)
        
        # 4. Update Resonance Index
        self.resonance.add_vector(concept, [sphere.will.x, sphere.will.y, sphere.will.z])
        
        logger.debug(f"🌌 [Hippocampus] Processed concept: '{concept}'")

    def add_causal_link(self, source: str, target: str, relation: str = "causes", weight: float = 1.0):
        """
        Holographic Link: No explicit edge storage.
        Instead, we ensure both concepts exist and are 'active'.
        The relationship is implicit in their vector resonance.
        """
        # Ensure concepts exist
        self.add_concept(source)
        self.add_concept(target)
        
        # Update Resonance Index dynamically
        # In a full implementation, we would nudge vectors here.
        # For now, we just ensure they are in the index.
        # self.resonance.add_vector(source, ...) 
        pass

    def get_related_concepts(self, concept: str, depth: int = 1) -> Dict[str, float]:
        """
        Holographic Retrieval: Find concepts by Resonance.
        """
        # Get query vector
        query_vec = self.resonance.get_vector(concept)
        if not any(query_vec):
            return {}
            
        # Find resonating concepts
        results = self.resonance.find_resonance(query_vec, k=10, exclude_id=concept)
        
        # Convert to dict {id: score}
        return {cid: score for cid, score in results}

    def ingest_visual_experience(self, video_id: str, frames: List[Any]):
        """
        Process a video stream into Holographic Memory.
        "Star-Eating Mode"
        """
        # 1. Visual Cortex: Frame -> Stars
        constellation = self.visual_cortex.ingest_video(video_id, frames)
        
        # 2. Resonance Engine: Store Constellation
        self.resonance.add_temporal_sequence(
            constellation["id"],
            constellation["vectors"],
            constellation["timestamps"]
        )
        
        # 3. Create a Concept for the Video itself
        self.add_concept(video_id, concept_type="visual_memory", metadata={"frames": constellation["count"]})
        
        logger.info(f"🌌 [Hippocampus] Consumed visual experience: '{video_id}' ({constellation['count']} stars)")

    def get_stellar_type(self, concept: str) -> str:
        """Returns stellar icon based on activation count."""
        data = self.storage.get_concept(concept)
        if not data:
            return "✨"
            
        count = data.get('activation_count', 0)
        if count < 3: return "✨"
        elif count < 10: return "🌟"
        elif count < 50: return "🔥"
        elif count < 100: return "❄️"
        else: return "⚫"

    def get_statistics(self) -> Dict[str, int]:
        """
        Returns basic statistics about the memory graph.
        """
        return {
            "nodes": self.storage.count_concepts(),
            "edges": 0, # TODO: Implement edge counting in storage
        }

    def add_projection_episode(self, tag: str, projection_data: Any):
        """
        Store a projection episode for memory consolidation.
        
        Args:
            tag: A tag or identifier for this projection (e.g., input text)
            projection_data: The projection data to store
        """
        # Store the projection as a concept with metadata
        self.add_concept(
            f"projection_{tag}",
            concept_type="projection",
            metadata={"projection": projection_data, "tag": tag}
        )
        logger.debug(f"🔮 [Hippocampus] Stored projection episode: '{tag}'")