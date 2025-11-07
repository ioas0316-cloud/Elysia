from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging
import os
import json

from .core_memory import CoreMemory
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager
from Project_Sophia.core.world import World
from Project_Elysia.core_memory_base import Memory
from Project_Sophia.emotional_engine import EmotionalEngine, EmotionalState
from Project_Sophia.response_styler import ResponseStyler
# --- Import the new InsightSynthesizer ---
from Project_Sophia.insight_synthesizer import InsightSynthesizer
import re

class CognitionPipeline:
    def __init__(self, cellular_world: Optional[World] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # --- Centralized Component Initialization ---
        self.kg_manager = KGManager()
        self.core_memory = CoreMemory()
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.reasoner = LogicalReasoner(kg_manager=self.kg_manager, cellular_world=cellular_world)
        self.cellular_world = cellular_world
        self.emotional_engine = EmotionalEngine()
        self.response_styler = ResponseStyler()
        
        # --- Initialize the new InsightSynthesizer ---
        self.insight_synthesizer = InsightSynthesizer()

        self.current_emotional_state = self.emotional_engine.get_current_state()
        # Other components would be initialized here...

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], EmotionalState]:
        """Processes an incoming message through the cognitive pipeline."""
        try:
            # This simplified path focuses on the internal response generation
            response, emotional_state = self._generate_internal_response(message, self.current_emotional_state, context or {})
            
            memory = Memory(timestamp=datetime.now().isoformat(), content=message, emotional_state=self.current_emotional_state)
            self.core_memory.add_experience(memory)

            return response, emotional_state

        except Exception as e:
            self.logger.error(f"Critical error in process_message: {e}", exc_info=True)
            return {"type": "text", "text": "A critical error occurred."}, self.current_emotional_state

    def _generate_internal_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any]) -> Tuple[Dict[str, Any], EmotionalState]:
        """
        Generates a response using internal capabilities, now enhanced with the InsightSynthesizer.
        """
        try:
            # 0. Wave→Cell bridge: inject activation energy as nutrients into the Cellular World
            try:
                if self.cellular_world:
                    node_ids = [n.get('id') for n in self.kg_manager.kg.get('nodes', []) if n.get('id')]
                    mentioned = []
                    for nid in node_ids:
                        try:
                            if nid and re.search(re.escape(nid), message, re.IGNORECASE):
                                mentioned.append(nid)
                        except re.error:
                            continue
                    mentioned = list(set(mentioned))
                    for start in mentioned:
                        activated = self.wave_mechanics.spread_activation(
                            start_node_id=start,
                            initial_energy=1.0,
                            decay_factor=0.9,
                            threshold=0.2,
                            top_k=6,
                        )
                        for node_id, energy in activated.items():
                            cell = self.cellular_world.get_cell(node_id)
                            if cell and cell.is_alive:
                                cell.add_energy(float(energy))
            except Exception as e:
                self.logger.error(f"Wave→Cell bridge error: {e}", exc_info=True)
            # 1. Deduce facts using the LogicalReasoner as before.
            facts = self.reasoner.deduce_facts(message)

            # 2. Synthesize these facts into a coherent insight.
            #    Instead of just joining strings, we now create a meaningful narrative.
            insightful_text = self.insight_synthesizer.synthesize(facts)

            # 3. (Optional) Style the synthesized insight based on the current emotional state.
            final_response = self.response_styler.style_response(insightful_text, emotional_state)

            return {"type": "text", "text": final_response}, emotional_state

        except Exception as e:
            self.logger.error(f"Error during internal response generation: {e}", exc_info=True)
            # Safe fallback in case of any error.
            return {"type": "text", "text": "생각을 정리하는 데 어려움을 겪고 있어요."}, emotional_state
