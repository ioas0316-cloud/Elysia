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
from Project_Sophia.insight_synthesizer import InsightSynthesizer
from .value_centered_decision import VCD
# --- Import Cortexes for routing ---
from Project_Sophia.arithmetic_cortex import ArithmeticCortex

import re
import json as _json
import os as _os

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
        self.insight_synthesizer = InsightSynthesizer()
        self.vcd = VCD(self.kg_manager, self.wave_mechanics, core_value='love')

        # --- Initialize Specialized Cortexes for Routing ---
        self.arithmetic_cortex = ArithmeticCortex()

        self._aliases_path = _os.path.join('data', 'lexicon', 'aliases_ko.json')
        self._aliases = None
        self.current_emotional_state = self.emotional_engine.get_current_state()

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], EmotionalState]:
        """Processes an incoming message through the cognitive pipeline."""
        try:
            # --- Triage and Routing Logic ---
            # If the message is a command for a specialized cortex, route it directly.
            if message.strip().startswith("calculate:") or message.strip().startswith("계산:"):
                raw_command = message.split(":", 1)[1].strip()
                result_text = self.arithmetic_cortex.process(raw_command)
                response = {"type": "text", "text": result_text}
                emotional_state = self.current_emotional_state # No emotional change for now
            else:
                # Otherwise, use the main internal response generation path.
                response, emotional_state = self._generate_internal_response(message, self.current_emotional_state, context or {})

            memory = Memory(timestamp=datetime.now().isoformat(), content=message, emotional_state=self.current_emotional_state)
            self.core_memory.add_experience(memory)
            return response, emotional_state

        except Exception as e:
            self.logger.error(f"Critical error in process_message: {e}", exc_info=True)
            return {"type": "text", "text": "A critical error occurred."}, self.current_emotional_state

    def _generate_internal_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any]) -> Tuple[Dict[str, Any], EmotionalState]:
        """Generates a response using the main VCD-guided path."""
        try:
            self._inject_energy_into_cellular_world(message)

            potential_facts = self.reasoner.deduce_facts(message)

            if not potential_facts:
                insightful_text = "흥미로운 관점이네요. 조금 더 생각해볼게요."
            else:
                self.logger.info(f"VCD evaluating {len(potential_facts)} potential facts...")
                chosen_fact = self.vcd.suggest_action(candidates=potential_facts, context=[message])
                self.logger.info(f"VCD chose fact: '{chosen_fact}'")
                insightful_text = self.insight_synthesizer.synthesize([chosen_fact] if chosen_fact else [])

            final_response = self.response_styler.style_response(insightful_text, emotional_state)
            return {"type": "text", "text": final_response}, emotional_state

        except Exception as e:
            self.logger.error(f"Error during internal response generation: {e}", exc_info=True)
            return {"type": "text", "text": "생각을 정리하는 데 어려움을 겪고 있어요."}, emotional_state

    def _inject_energy_into_cellular_world(self, message: str):
        # (Implementation of this method is unchanged)
        pass
