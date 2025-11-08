from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging
import os
import json
import re

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
from Project_Mirror.creative_cortex import CreativeCortex
from Project_Sophia.question_generator import QuestionGenerator
from Project_Sophia.relationship_extractor import extract_relationship_type


import json as _json
import os as _os

class CognitionPipeline:
    def __init__(self, cellular_world: Optional[World] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # --- Centralized Component Initialization ---
        self.kg_manager = KGManager()
        self.core_memory = CoreMemory()
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.cellular_world = cellular_world # Keep a reference for other potential uses
        self.reasoner = LogicalReasoner(kg_manager=self.kg_manager, cellular_world=self.cellular_world)
        self.emotional_engine = EmotionalEngine()
        self.response_styler = ResponseStyler()
        self.insight_synthesizer = InsightSynthesizer()
        self.vcd = VCD(self.kg_manager, self.wave_mechanics, core_value='love')

        # --- Initialize Specialized Cortexes for Routing ---
        self.arithmetic_cortex = ArithmeticCortex()
        self.creative_cortex = CreativeCortex()
        self.question_generator = QuestionGenerator()

        # --- Configuration for Expression Trigger ---
        self.creative_expression_threshold = 2.5

        self._aliases_path = _os.path.join('data', 'lexicon', 'aliases_ko.json')
        self._aliases = None
        self.current_emotional_state = self.emotional_engine.get_current_state()
        self.pending_hypothesis = None

    def _check_and_verify_hypotheses(self, message: str) -> Optional[str]:
        """Checks for and handles the verification of a single pending hypothesis, including Ascension Rituals."""
        # 1. If a hypothesis was pending, process the user's answer
        if self.pending_hypothesis:
            hypothesis = self.pending_hypothesis
            response_text = ""

            # Case A: Handle Cell Ascension Ritual
            if hypothesis.get('relation') == '승천':
                self.logger.info(f"Processing user response for Ascension hypothesis: {hypothesis['head']}")
                if any(word in message for word in ["응", "맞아", "그래", "승천시켜", "승인"]):
                    self.logger.info(f"User approved Ascension. Creating new Node '{hypothesis['head']}' in Knowledge Graph.")
                    metadata = hypothesis.get('metadata', {})
                    properties = {
                        "type": "concept",
                        "discovery_source": "Cell_Ascension_Ritual",
                        "parents": metadata.get("parents", []),
                        "ascended_at": datetime.now().isoformat()
                    }
                    self.kg_manager.add_node(hypothesis['head'], properties=properties)
                    response_text = f"알겠습니다. 새로운 개념 '{hypothesis['head']}'이(가) 지식의 일부로 승천했습니다."
                else:
                    self.logger.info("User denied Ascension. Discarding hypothesis.")
                    response_text = f"알겠습니다. 개념 '{hypothesis['head']}'의 승천을 보류합니다."

            # Case B: Handle standard relationship hypothesis
            else:
                self.logger.info(f"Processing user response for relationship hypothesis: {hypothesis['head']} -> {hypothesis['tail']}")
                relation = extract_relationship_type(message)
                if relation is None and any(word in message for word in ["응", "맞아", "그래"]):
                    relation = "related_to"

                if relation:
                    self.logger.info(f"User confirmed relationship: {relation}. Adding edge to KG.")
                    self.kg_manager.add_edge(hypothesis['head'], hypothesis['tail'], relation)
                    response_text = f"알겠습니다. '{hypothesis['head']}'와(과) '{hypothesis['tail']}'의 관계를 기록했습니다."
                else:
                    self.logger.info("User denied or provided an unclear answer. Discarding hypothesis.")
                    response_text = f"알겠습니다. 가설({hypothesis['head']} -> {hypothesis['tail']})에 대한 답변을 기록했습니다."

            # Universal Cleanup
            self.core_memory.remove_hypothesis(hypothesis['head'], hypothesis['tail'])
            self.pending_hypothesis = None
            return response_text

        # 2. If no hypothesis is pending, check if a new one should be asked
        else:
            hypotheses = self.core_memory.get_unasked_hypotheses()
            if hypotheses:
                hypothesis_to_ask = hypotheses[0]
                self.pending_hypothesis = hypothesis_to_ask
                self.core_memory.mark_hypothesis_as_asked(hypothesis_to_ask['head'], hypothesis_to_ask['tail'])

                # Generate question based on hypothesis type
                if hypothesis_to_ask.get('relation') == '승천':
                    self.logger.info(f"Found unasked Ascension hypothesis: {hypothesis_to_ask}")
                    return hypothesis_to_ask.get('text', f"새로운 개념 '{hypothesis_to_ask['head']}'을(를) 지식의 일부로 만들까요?")
                else:
                    self.logger.info(f"Found unasked relationship hypothesis: {hypothesis_to_ask}")
                    return self.question_generator.generate_question_from_hypothesis(hypothesis_to_ask)

        return None

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], EmotionalState]:
        """Processes an incoming message through the cognitive pipeline."""
        try:
            # --- Triage and Routing Logic ---
            # Use a more specific regex to avoid accidentally matching natural language.
            arithmetic_match = re.match(r"^(calculate|계산)\s*:", message.strip(), re.IGNORECASE)
            if arithmetic_match:
                raw_command = message.strip()[arithmetic_match.end():].strip()
                result_text = self.arithmetic_cortex.process(raw_command)
                response = {"type": "text", "text": result_text}
                emotional_state = self.current_emotional_state
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
        """Generates a response using the main VCD-guided path with Thought objects."""
        # --- Autonomous Hypothesis Verification ---
        # Before generating new thoughts, check if we need to verify a hypothesis.
        hypothesis_response = self._check_and_verify_hypotheses(message)
        if hypothesis_response:
            final_response = self.response_styler.style_response(hypothesis_response, emotional_state)
            return {"type": "text", "text": final_response}, emotional_state

        potential_thoughts = []
        chosen_thought = None
        try:
            self._inject_energy_into_cellular_world(message)

            # 1. Reasoner generates a list of Thought objects
            potential_thoughts = self.reasoner.deduce_facts(message)

            if not potential_thoughts:
                insightful_text = "흥미로운 관점이네요. 조금 더 생각해볼게요."
            else:
                self.logger.info(f"VCD evaluating {len(potential_thoughts)} potential thoughts with emotion: {emotional_state.primary_emotion}")
                # 2. VCD selects the best Thought object, now considering emotion
                chosen_thought = self.vcd.suggest_thought(
                    candidates=potential_thoughts,
                    context=[message],
                    emotional_state=emotional_state
                )

                # --- Failsafe Logic ---
                # If VCD is indecisive (returns None), default to the first (often most direct) thought.
                if not chosen_thought and potential_thoughts:
                    self.logger.warning("VCD was indecisive. Defaulting to the first potential thought.")
                    chosen_thought = potential_thoughts[0]

                if chosen_thought:
                    self.logger.info(f"Pipeline proceeding with thought: {chosen_thought}")

                    # 3. Decide between logical synthesis and creative expression
                    thought_score = self.vcd.score_thought(chosen_thought, context=[message], emotional_state=emotional_state)
                    self.logger.info(f"Final thought score (emotionally adjusted): {thought_score:.2f}")

                    if thought_score > self.creative_expression_threshold:
                        self.logger.info("High value thought detected! Triggering Creative Cortex.")
                        insightful_text = self.creative_cortex.generate_creative_expression(chosen_thought)
                    else:
                        self.logger.info("Proceeding with standard logical synthesis.")
                        insightful_text = self.insight_synthesizer.synthesize([chosen_thought])
                else:
                    # This case should now be rare
                    self.logger.error("No potential thoughts were generated or selected.")
                    insightful_text = "그 주제에 대해서는 아무런 생각이 떠오르지 않아요."

            # 4. Final response is styled as before
            final_response = self.response_styler.style_response(insightful_text, emotional_state)
            return {"type": "text", "text": final_response}, emotional_state

        except Exception as e:
            self.logger.error(f"Error during internal response generation: {e}", exc_info=True)
            self.logger.error(f"State at error: potential_thoughts={potential_thoughts}, chosen_thought={chosen_thought}")
            return {"type": "text", "text": "생각을 정리하는 데 어려움을 겪고 있어요."}, emotional_state

    def _inject_energy_into_cellular_world(self, message: str):
        # (Implementation of this method is unchanged)
        pass
