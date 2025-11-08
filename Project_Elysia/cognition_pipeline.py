import logging
from typing import Dict, Any, Optional, Tuple

# --- New Architecture Dependencies ---
from Project_Elysia.architecture.context import ConversationContext
from Project_Elysia.architecture.cortex_registry import CortexRegistry
from Project_Elysia.architecture.event_bus import EventBus
from Project_Elysia.architecture.handlers import (
    Handler, HypothesisHandler, CommandWordHandler, DefaultReasoningHandler
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
from .value_centered_decision import VCD
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Mirror.creative_cortex import CreativeCortex
from Project_Sophia.question_generator import QuestionGenerator

class CognitionPipeline:
    """
    A refactored, stateless pipeline that processes messages using a
    Chain of Responsibility and Event Bus architecture.
    """
    def __init__(
        self,
        kg_manager: KGManager,
        core_memory: CoreMemory,
        wave_mechanics: WaveMechanics,
        cellular_world: Optional[World],
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.event_bus = EventBus()
        self.cortex_registry = CortexRegistry()
        self.conversation_context = ConversationContext() # Manages conversation state

        # --- Instantiate Components (dependencies for handlers) ---
        emotional_engine = EmotionalEngine()
        response_styler = ResponseStyler()
        insight_synthesizer = InsightSynthesizer()
        question_generator = QuestionGenerator()
        reasoner = LogicalReasoner(kg_manager=kg_manager, cellular_world=cellular_world)
        vcd = VCD(kg_manager=kg_manager, wave_mechanics=wave_mechanics, core_value='love')
        creative_cortex = CreativeCortex()

        # --- Register Cortexes ---
        self.cortex_registry.register("arithmetic", ArithmeticCortex())
        # Note: creative_cortex is used directly by a handler, but could be registered for other uses.
        self.cortex_registry.register("creative", creative_cortex)


        # --- Build the Chain of Responsibility ---
        # The chain is built in reverse order: the last handler is created first.
        default_handler = DefaultReasoningHandler(
            reasoner=reasoner, vcd=vcd, synthesizer=insight_synthesizer,
            creative_cortex=creative_cortex, styler=response_styler, logger=self.logger,
            question_generator=question_generator, emotional_engine=emotional_engine
        )
        command_handler = CommandWordHandler(
            successor=default_handler, cortex_registry=self.cortex_registry, logger=self.logger
        )
        self.entry_handler: Handler = HypothesisHandler(
            successor=command_handler, core_memory=core_memory, kg_manager=kg_manager,
            question_generator=question_generator, response_styler=response_styler, logger=self.logger
        )

        self.logger.info("CognitionPipeline initialized with Chain of Responsibility.")
        # Setup event logging
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Subscribe to events to log them."""
        # This is a placeholder for where more complex event handling could go
        self.event_bus.subscribe("message_processed", lambda result: self.logger.info(f"Event: Message processing completed. Result: {result.get('text', 'N/A')}"))
        self.event_bus.subscribe("error_occurred", lambda error: self.logger.error(f"Event: An error occurred in a handler: {error}"))

    def process_message(self, message: str) -> Tuple[Dict[str, Any], EmotionalState]:
        """
        Processes a message by passing it through the handler chain.
        The pipeline itself is now stateless.
        """
        # In a real scenario, emotional_state would also be part of the context
        # For now, we'll fetch it here.
        current_emotional_state = EmotionalEngine().get_current_state()

        try:
            self.logger.debug(f"Processing message: '{message}' with context: {self.conversation_context}")

            # The message and context are passed to the first handler
            result = self.entry_handler.handle(message, self.conversation_context, current_emotional_state)

            if result:
                self.event_bus.publish("message_processed", result)
                return result, current_emotional_state
            else:
                # This should ideally not be reached if the default handler always returns something
                self.logger.error("Handler chain completed without generating a response.")
                error_response = {"type": "text", "text": "I'm not sure how to respond to that."}
                self.event_bus.publish("error_occurred", "No response from handler chain")
                return error_response, current_emotional_state

        except Exception as e:
            self.logger.error(f"Critical error in CognitionPipeline: {e}", exc_info=True)
            error_response = {"type": "text", "text": "A critical error occurred in my thought process."}
            self.event_bus.publish("error_occurred", str(e))
            return error_response, current_emotional_state
