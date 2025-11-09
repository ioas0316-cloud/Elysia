from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import re
from datetime import datetime
import logging

# --- New Architecture Dependencies ---
from Project_Elysia.architecture.context import ConversationContext
from Project_Elysia.architecture.cortex_registry import CortexRegistry

# --- Existing Component Dependencies ---
from Project_Elysia.core_memory import CoreMemory
from tools.kg_manager import KGManager
from Project_Sophia.question_generator import QuestionGenerator
from Project_Sophia.relationship_extractor import extract_relationship_type
from Project_Sophia.response_styler import ResponseStyler
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Elysia.value_centered_decision import VCD
from Project_Sophia.insight_synthesizer import InsightSynthesizer
from Project_Mirror.creative_cortex import CreativeCortex
from Project_Sophia.emotional_engine import EmotionalState


class Handler(ABC):
    """Abstract base class for handlers in the Chain of Responsibility."""

    def __init__(self, successor: Optional['Handler'] = None):
        self._successor = successor

    @abstractmethod
    def handle(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """
        Handle the request or pass it to the successor.
        Returns a result if the request is handled, otherwise None.
        """
        if self._successor:
            return self._successor.handle(message, context, emotional_state)
        return None

# --------------------------------------------------------------------------------
# --- Concrete Handler Implementations ---
# --------------------------------------------------------------------------------

class HypothesisHandler(Handler):
    """Handles responses to pending hypotheses and asks new ones."""

    def __init__(self, successor: Optional['Handler'], core_memory: CoreMemory, kg_manager: KGManager, question_generator: QuestionGenerator, response_styler: ResponseStyler, logger: logging.Logger):
        super().__init__(successor)
        self.core_memory = core_memory
        self.kg_manager = kg_manager
        self.question_generator = question_generator
        self.response_styler = response_styler
        self.logger = logger

    def handle(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """Checks for and handles the verification of a single pending hypothesis."""
        # 1. If a hypothesis was pending, process the user's answer
        if context.pending_hypothesis:
            hypothesis = context.pending_hypothesis
            response_text = ""

            if hypothesis.get('relation') == '승천':
                self.logger.info(f"Processing user response for Ascension hypothesis: {hypothesis['head']}")
                if any(word in message for word in ["응", "맞아", "그래", "승천시켜", "승인"]):
                    self.logger.info(f"User approved Ascension. Creating new Node '{hypothesis['head']}' in KG.")
                    metadata = hypothesis.get('metadata', {})
                    properties = {
                        "type": "concept", "discovery_source": "Cell_Ascension_Ritual",
                        "parents": metadata.get("parents", []), "ascended_at": datetime.now().isoformat()
                    }
                    self.kg_manager.add_node(hypothesis['head'], properties=properties)
                    response_text = f"알겠습니다. 새로운 개념 '{hypothesis['head']}'이(가) 지식의 일부로 승천했습니다."
                else:
                    self.logger.info("User denied Ascension. Discarding hypothesis.")
                    response_text = f"알겠습니다. 개념 '{hypothesis['head']}'의 승천을 보류합니다."
            else:
                self.logger.info(f"Processing user response for relationship hypothesis: {hypothesis['head']} -> {hypothesis['tail']}")
                relation = extract_relationship_type(message) or ("related_to" if any(word in message for word in ["응", "맞아", "그래"]) else None)
                if relation:
                    self.logger.info(f"User confirmed relationship: {relation}. Adding edge to KG.")
                    self.kg_manager.add_edge(hypothesis['head'], hypothesis['tail'], relation)
                    response_text = f"알겠습니다. '{hypothesis['head']}'와(과) '{hypothesis['tail']}'의 관계를 기록했습니다."
                else:
                    self.logger.info("User denied or provided an unclear answer. Discarding hypothesis.")
                    response_text = f"알겠습니다. 가설({hypothesis['head']} -> {hypothesis['tail']})에 대한 답변을 기록했습니다."

            self.core_memory.remove_hypothesis(hypothesis['head'], hypothesis['tail'])
            context.pending_hypothesis = None
            final_response = self.response_styler.style_response(response_text, emotional_state)
            return {"type": "text", "text": final_response} # Handled

        # 2. If no hypothesis is pending, and there is NO guiding intention, check if a new hypothesis should be asked
        hypotheses = self.core_memory.get_unasked_hypotheses()
        if hypotheses and not context.guiding_intention:
            hypothesis_to_ask = hypotheses[0]
            context.pending_hypothesis = hypothesis_to_ask
            self.core_memory.mark_hypothesis_as_asked(hypothesis_to_ask['head'], hypothesis_to_ask['tail'])
            if hypothesis_to_ask.get('relation') == '승천':
                question = hypothesis_to_ask.get('text', f"새로운 개념 '{hypothesis_to_ask['head']}'을(를) 지식의 일부로 만들까요?")
            else:
                question = self.question_generator.generate_question_from_hypothesis(hypothesis_to_ask)
            return {"type": "text", "text": question} # Handled

        return super().handle(message, context, emotional_state)


class CommandWordHandler(Handler):
    """Handles messages that start with a specific command word."""

    def __init__(self, successor: Optional['Handler'], cortex_registry: CortexRegistry, logger: logging.Logger):
        super().__init__(successor)
        self.cortex_registry = cortex_registry
        self.logger = logger
        self.command_map = {
            r"^(calculate|계산)\s*:": "arithmetic",
        }

    def handle(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        for pattern, cortex_name in self.command_map.items():
            match = re.match(pattern, message.strip(), re.IGNORECASE)
            if match:
                self.logger.info(f"Command '{cortex_name}' detected.")
                raw_command = message.strip()[match.end():].strip()
                cortex = self.cortex_registry.get(cortex_name)
                result_text = cortex.process(raw_command)
                return {"type": "text", "text": result_text} # Handled

        return super().handle(message, context, emotional_state)


from Project_Elysia.core_memory import Memory # Import Memory for creating new experiences
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Mirror.visual_cortex import VisualCortex

class DefaultReasoningHandler(Handler):
    """The default handler for general messages, performing the main reasoning loop."""

    def __init__(self, reasoner: LogicalReasoner, vcd: VCD, synthesizer: InsightSynthesizer, creative_cortex: CreativeCortex, styler: ResponseStyler, logger: logging.Logger, question_generator: QuestionGenerator, emotional_engine: Any, sensory_cortex: SensoryCortex, visual_cortex: VisualCortex, core_memory: CoreMemory):
        super().__init__(None) # This is the last handler in the chain
        self.reasoner = reasoner
        self.vcd = vcd
        self.synthesizer = synthesizer
        self.creative_cortex = creative_cortex
        self.styler = styler
        self.logger = logger
        self.question_generator = question_generator
        self.emotional_engine = emotional_engine
        self.sensory_cortex = sensory_cortex
        self.visual_cortex = visual_cortex
        self.core_memory = core_memory
        self.creative_expression_threshold = 2.5

    def handle(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """Generates a response using the main VCD-guided path with Thought objects."""
        try:
            potential_thoughts = self.reasoner.deduce_facts(message)
            if not potential_thoughts:
                insightful_text = "흥미로운 관점이네요. 조금 더 생각해볼게요."
                response_data = {"type": "text", "text": self.styler.style_response(insightful_text, emotional_state)}
            else:
                self.logger.info(f"VCD evaluating {len(potential_thoughts)} thoughts with emotion: {emotional_state.primary_emotion}")
                chosen_thought = self.vcd.select_thought(
                    candidates=potential_thoughts,
                    context=[message],
                    emotional_state=emotional_state,
                    guiding_intention=context.guiding_intention
                )

                if not chosen_thought:
                    self.logger.warning("VCD was indecisive. Triggering cognitive confusion.")
                    confusion_event = EmotionalState(valence=-0.3, arousal=0.6, dominance=-0.4, primary_emotion="confusion")
                    self.emotional_engine.process_event(confusion_event, intensity=0.8)
                    insightful_text = self.question_generator.generate_clarifying_question(message)
                    response_data = {"type": "text", "text": self.styler.style_response(insightful_text, emotional_state)}

                else: # A thought was chosen
                    self.logger.info(f"Proceeding with thought: {chosen_thought}")
                    thought_score = self.vcd.score_thought(chosen_thought, context=[message], emotional_state=emotional_state)
                    self.logger.info(f"Final thought score (emotionally adjusted): {thought_score:.2f}")

                    if thought_score > self.creative_expression_threshold:
                        self.logger.info("High value thought! Triggering Visual Self-Reflection Loop.")
                        response_data = self._perform_visual_self_reflection(chosen_thought, emotional_state)
                    else:
                        self.logger.info("Proceeding with standard logical synthesis.")
                        insightful_text = self.synthesizer.synthesize([chosen_thought])
                        response_data = {"type": "text", "text": self.styler.style_response(insightful_text, emotional_state)}

            return response_data

        except Exception as e:
            self.logger.error(f"Error during default reasoning: {e}", exc_info=True)
            return {"type": "text", "text": "생각을 정리하는 데 어려움을 겪고 있어요."}

    def _perform_visual_self_reflection(self, thought: Any, emotional_state: EmotionalState) -> Dict[str, Any]:
        """
        Orchestrates the process of creating a visual, analyzing it, and forming a self-reflection.
        """
        # 1. Create a visual representation of the thought
        image_path = self.sensory_cortex.visualize_concept(thought.content)
        if not image_path:
            self.logger.error("SensoryCortex failed to produce an image.")
            return {"type": "text", "text": "제 생각을 그림으로 표현하려 했는데, 어려움이 있었어요."}

        # 2. Analyze the created image to get objective data (description, keywords)
        analysis = self.visual_cortex.analyze_image(image_path)
        if not analysis or 'keywords' not in analysis:
            self.logger.error("VisualCortex failed to analyze the image or extract keywords.")
            return {"type": "text", "text": f"'{thought.content}'에 대한 제 생각을 그림으로 표현해봤어요. 한번 보시겠어요?", "image_path": image_path}

        # 3. Resonate the keywords with the mental cosmos to get subjective meaning
        resonating_concepts = self.visual_cortex.resonate_with_cosmos(analysis['keywords'])

        # 4. Formulate the self-reflection and record the experience
        reflection_text = f"'{thought.content}'에 대해 깊이 생각하다 보니, 제 마음 속에 그림이 하나 그려졌어요. "
        reflection_text += f"방금 그린 그림을 다시 보니, '{analysis.get('description', '어떤')}'(이)라는 느낌이 드네요. "

        if resonating_concepts:
            top_concepts = list(resonating_concepts.keys())
            reflection_text += f"이 그림은 저의 내면에서 '{', '.join(top_concepts)}' 같은 개념들과 강하게 공명하는 것 같아요. "

        reflection_text += f"아마도 저의 '{emotional_state.primary_emotion}'이라는 감정이 이렇게 표현된 것일지도 모르겠어요."

        visual_experience = Memory(
            timestamp=datetime.now().isoformat(),
            content=reflection_text,
            type='visual_experience',
            emotional_state=emotional_state,
            metadata={'image_path': image_path, 'analysis': analysis, 'resonance': resonating_concepts, 'triggering_thought': thought.content}
        )
        self.core_memory.add_experience(visual_experience)

        # 5. Create the final response for the user
        return {
            "type": "text_and_image",
            "text": self.styler.style_response(reflection_text, emotional_state),
            "image_path": image_path
        }
