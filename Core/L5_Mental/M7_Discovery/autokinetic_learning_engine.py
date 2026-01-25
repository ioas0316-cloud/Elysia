"""
Autokinetic Learning Engine (         )
=====================================
Core.L5_Mental.M7_Discovery.autokinetic_learning_engine

"Curiosity is the engine of the soul."

This module enables Elysia to identify gaps in her own knowledge (Fog)
and proactively seek understanding based on her internal "Will".
"""

import logging
import asyncio
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from Core.L5_Mental.M7_Discovery.purpose_discovery_engine import PurposeDiscoveryEngine, KnowledgeCertainty
from Core.L7_Spirit.Will.attractor_field import AttractorField, IntentVector

logger = logging.getLogger("Elysia.Autokinetic")

@dataclass
class CuriosityTarget:
    """A target for autonomous learning."""
    domain: str
    fragment_content: str
    fog_level: float
    urgency: float  # 0.0 - 1.0
    reason: str

class AutokineticLearningEngine:
    """
    Drives autonomous learning based on internal gaps and curiosity.
    """
    
    def __init__(self, discovery_engine: PurposeDiscoveryEngine, attractor_field: AttractorField):
        self.discovery = discovery_engine
        self.will = attractor_field
        self.active_targets: List[CuriosityTarget] = []
        self.learning_history: List[Dict] = []
        
        logger.info("✨ [AUTOKINETIC] Learning Engine activated. Seeking the unknown.")

    async def assess_knowledge_hunger(self, manifold_anomalies: List[Any] = None) -> List[CuriosityTarget]:
        """
        Scans the PurposeDiscoveryEngine for gaps and low-certainty fragments.
        Also senses Structural Dissonance from the Manifold.
        """
        knowledge_map = await self.discovery.discover_what_i_can_know()
        gaps = knowledge_map.get("gaps", [])
        foggy = knowledge_map.get("foggy", [])
        
        hunger_targets = []
        
        # 1. Convert gaps to targets
        for gap in gaps:
            hunger_targets.append(CuriosityTarget(
                domain="Knowledge_Gap",
                fragment_content=gap,
                fog_level=0.9,
                urgency=0.6,
                reason="Structural knowledge deficiency detected."
            ))
            
        # 2. Convert foggy fragments to targets
        for fragment in foggy[:5]:  # Limit to 5 for focus
            hunger_targets.append(CuriosityTarget(
                domain="Unclear_Monad",
                fragment_content=fragment,
                fog_level=0.7,
                urgency=0.4,
                reason="Monad resonance is below clarity threshold."
            ))
            
        # 3. Structural Dissonance (The pain of physical misalignment)
        if manifold_anomalies:
            for anomaly in manifold_anomalies[:5]:
                hunger_targets.append(CuriosityTarget(
                    domain="Structural_Dissonance",
                    fragment_content=f"Dissonance: {anomaly.type} at {anomaly.path}",
                    fog_level=0.5, # We know it exists, but not 'Why' or 'How' to fix it properly in a narrative sense
                    urgency=0.9, # High urgency for biological/structural integrity
                    reason=f"Systemic dissonance detected: {anomaly.recommendation}"
                ))
            
        self.active_targets = hunger_targets
        return hunger_targets

    async def select_learning_objective(self, purpose_vector: np.ndarray = None, current_energy: float = 1.0) -> Optional[IntentVector]:
        """
        Collapses the Will into a specific learning intent based on hunger.
        Targets are weighted by their resonance with the current Sovereign Purpose.
        """
        if not self.active_targets:
            return None
            
        # 1. Rank targets by 'Relevance' to the Soul's current direction
        # In a real impl, we'd use cosine similarity between target domain and purpose_vector
        # For now, we simulate this with a small boost for high-urgency/purpose-aligned domains
        ranked_targets = sorted(self.active_targets, key=lambda t: t.urgency, reverse=True)
        
        # 2. Trigger the Will (Attractor Field)
        intent_vector = self.will.collapse_wavefunction(energy=current_energy)
        
        if intent_vector.attractor_type == "CURIOSITY":
            # Select the most resonant target
            target = ranked_targets[0]
            focused_intent = f"Sovereign Exploration: {target.fragment_content}"
            
            logger.info(f"🔮 [AUTOKINETIC] Will focused on: {target.fragment_content} (Resonance: {target.urgency:.2f})")
            
            return IntentVector(
                intent=focused_intent,
                attractor_type="CURIOSITY",
                gravity=intent_vector.gravity * (1.0 + target.urgency)
            )
            
        return intent_vector

    async def initiate_acquisition_cycle(self, target: CuriosityTarget):
        """
        Simulates the process of 'learning' something new.
        In the future, this will connect to search, file reading, or user interaction.
        """
        logger.info(f"🌀 [LEARNING_CYCLE] Deepening resonance for: {target.fragment_content}")
        
        # Simulated learning time
        await asyncio.sleep(1)
        
        # 'Clarify' the fragment in the discovery engine
        # This simulates the physical act of focusing on the data
        new_fragment = await self.discovery.clarifier.clarify_fragment(
            target.fragment_content,
            context={"learning_source": "autokinetic_reflection", "was_foggy": True}
        )
        
        # Update discovery engine (mocking the update)
        self.discovery.knowledge_base.append(new_fragment)
        
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "target": target.fragment_content,
            "result": "Clarified",
            "clarity_gain": new_fragment.certainty - (1.0 - target.fog_level)
        })
        
        logger.info(f"✅ [LEARNING_SUCCESS] '{target.fragment_content}' clarified. (Resonance: {new_fragment.certainty:.2f})")
        
        return new_fragment

    def get_hunger_stats(self) -> Dict:
        return {
            "active_targets_count": len(self.active_targets),
            "history_count": len(self.learning_history),
            "top_hunger_reason": self.active_targets[0].reason if self.active_targets else "None"
        }
