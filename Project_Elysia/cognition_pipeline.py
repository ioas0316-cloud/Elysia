import logging
from typing import Dict, Any, Optional, Tuple

# --- New Architecture Dependencies ---
from Project_Elysia.architecture.context import ConversationContext
from Project_Elysia.architecture.cortex_registry import CortexRegistry
from Project_Elysia.architecture.event_bus import EventBus
from Project_Elysia.architecture.handlers import (
    HypothesisHandler, CommandWordHandler, DefaultReasoningHandler
)

# --- Existing Component Dependencies (for dependency injection) ---
from .core_memory import CoreMemory
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager
from Project_Sophia.core.world import World
from Project_Sophia.emotional_engine import EmotionalEngine, EmotionalState
from Project_Sophia.response_styler import ResponseStyler
from Project_Sophia.insight_synthesizer import InsightSynthesizer
from .value_centered_decision import ValueCenteredDecision
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Mirror.creative_cortex import CreativeCortex
from Project_Sophia.question_generator import QuestionGenerator
from Project_Mirror.perspective_cortex import PerspectiveCortex

class CognitionPipeline:
    """
    A stateless pipeline that processes messages using a Central Dispatch model.
    It analyzes incoming messages and routes them to the appropriate handler
    (e.g., for hypotheses, commands, or default reasoning).
    """
    def __init__(
        self,
        kg_manager: KGManager,
        core_memory: CoreMemory,
        wave_mechanics: WaveMechanics,
        cellular_world: Optional[World],
        emotional_engine: EmotionalEngine, # Explicitly injected
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.event_bus = EventBus()
        self.cortex_registry = CortexRegistry()
        self.conversation_context = ConversationContext() # Manages conversation state
        self.emotional_engine = emotional_engine # Store for later use
        self.last_reason = "Idle"

        # --- Instantiate Components (dependencies for handlers) ---
        # Allow injecting mocks for testing, otherwise create real instances.
        # This is a common pattern to make code more testable without a full DI framework.
        response_styler = kwargs.get('response_styler') or ResponseStyler()
        insight_synthesizer = kwargs.get('insight_synthesizer') or InsightSynthesizer()
        question_generator = kwargs.get('question_generator') or QuestionGenerator()
        reasoner = kwargs.get('reasoner') or LogicalReasoner(kg_manager=kg_manager, cellular_world=cellular_world)
        vcd = kwargs.get('vcd') or ValueCenteredDecision(kg_manager=kg_manager, wave_mechanics=wave_mechanics, core_value='love')
        creative_cortex = kwargs.get('creative_cortex') or CreativeCortex()

        # --- Register Cortexes ---
        self.cortex_registry.register("arithmetic", ArithmeticCortex())
        self.cortex_registry.register("creative", creative_cortex)

        # --- Instantiate the new PerspectiveCortex with clear dependencies ---
        perspective_cortex = PerspectiveCortex(
            logger=self.logger, core_memory=core_memory,
            wave_mechanics=wave_mechanics, kg_manager=kg_manager,
            emotional_engine=self.emotional_engine
        )

        # --- Store Handlers and Cortexes directly as members ---
        self.hypothesis_handler = HypothesisHandler(
            core_memory=core_memory, kg_manager=kg_manager,
            question_generator=question_generator, response_styler=response_styler, logger=self.logger
        )
        self.command_handler = CommandWordHandler(
            cortex_registry=self.cortex_registry, logger=self.logger
        )
        self.default_reasoning_handler = DefaultReasoningHandler(
            reasoner=reasoner, vcd=vcd, synthesizer=insight_synthesizer,
            creative_cortex=creative_cortex, styler=response_styler,
            logger=self.logger, perspective_cortex=perspective_cortex,
            question_generator=question_generator, emotional_engine=self.emotional_engine
        )

        self.logger.info("CognitionPipeline initialized with Central Dispatch Model.")
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Subscribe to events for logging and telemetry."""
        self.event_bus.subscribe("message_processed", lambda result: self.logger.info(f"Event: Message processing completed. Result: {result.get('text', 'N/A')}"))
        self.event_bus.subscribe("error_occurred", lambda error: self.logger.error(f"Event: An error occurred: {error}"))

    def process_message(self, message: str) -> Tuple[Dict[str, Any], EmotionalState]:
        """
        Processes a message using the Central Dispatch model.
        It analyzes the message and explicitly routes it to the correct handler.
        """
        current_emotional_state = self.emotional_engine.get_current_state()
        result = None

        try:
            self.logger.debug(f"Processing message: '{message}' with context: {self.conversation_context}")

            # 1. --- Analysis and Routing ---
            if self.conversation_context.pending_hypothesis:
                self.logger.info("Routing to HypothesisHandler (pending hypothesis).")
                result = self.hypothesis_handler.handle_response(message, self.conversation_context, current_emotional_state)

            # Check for command words (e.g., "계산:")
            elif self.command_handler.can_handle(message):
                self.logger.info("Routing to CommandWordHandler.")
                result = self.command_handler.handle(message, self.conversation_context, current_emotional_state)

            # Check if a new hypothesis should be asked
            elif self.hypothesis_handler.should_ask_new_hypothesis():
                 self.logger.info("Routing to HypothesisHandler (ask new hypothesis).")
                 result = self.hypothesis_handler.handle_ask(self.conversation_context, current_emotional_state)

            # Default to general reasoning
            else:
                self.logger.info("Routing to DefaultReasoningHandler.")
                result = self.default_reasoning_handler.handle(message, self.conversation_context, current_emotional_state)

            # --- Finalization ---
            if not result:
                self.logger.error("No handler returned a result.")
                result = {"type": "text", "text": "I'm not sure how to respond to that."}
                self.event_bus.publish("error_occurred", "No result from handlers")

            reason_text = result.get("reason") or result.get("text")
            if reason_text:
                self.last_reason = reason_text

            self.event_bus.publish("message_processed", result)
            return result, current_emotional_state

        except Exception as e:
            self.logger.error(f"Critical error in CognitionPipeline: {e}", exc_info=True)
            error_response = {"type": "text", "text": "A critical error occurred in my thought process."}
            self.event_bus.publish("error_occurred", str(e))
            return error_response, current_emotional_state
