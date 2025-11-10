
from typing import Optional, Any
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
from Project_Elysia.value_centered_decision import ValueCenteredDecision
from Project_Sophia.insight_synthesizer import InsightSynthesizer
from Project_Mirror.creative_cortex import CreativeCortex
from Project_Sophia.emotional_engine import EmotionalState


# --------------------------------------------------------------------------------
# --- Concrete Handler Implementations for Central Dispatch ---
# --------------------------------------------------------------------------------

class HypothesisHandler:
    """Handles asking and processing responses for hypotheses."""

    def __init__(self, core_memory: CoreMemory, kg_manager: KGManager, question_generator: QuestionGenerator, response_styler: ResponseStyler, logger: logging.Logger):
        self.core_memory = core_memory
        self.kg_manager = kg_manager
        self.question_generator = question_generator
        self.response_styler = response_styler
        self.logger = logger

    def should_ask_new_hypothesis(self) -> bool:
        """Check if there are any unasked hypotheses."""
        return bool(self.core_memory.get_unasked_hypotheses())

    def handle_ask(self, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """Asks the next available hypothesis."""
        hypotheses = self.core_memory.get_unasked_hypotheses()
        if not hypotheses:
            return None

        hypothesis_to_ask = hypotheses[0]
        context.pending_hypothesis = hypothesis_to_ask
        # Handle both relationship and ascension hypotheses
        tail = hypothesis_to_ask.get('tail')
        self.core_memory.mark_hypothesis_as_asked(hypothesis_to_ask['head'], tail)

        # --- Confidence-based Question Generation ---
        confidence = hypothesis_to_ask.get('confidence', 0.5)
        relation = hypothesis_to_ask.get('relation')

        question = None
        # Use wisdom-seeking question for mid-confidence 'forms_new_concept' hypotheses
        if relation == 'forms_new_concept' and 0.7 <= confidence < 0.9:
            self.logger.info(f"Generating wisdom-seeking question for mid-confidence insight: {hypothesis_to_ask['new_concept_id']}")
            question = self.question_generator.generate_wisdom_seeking_question(hypothesis_to_ask)

        # Handle 'ascension' hypotheses (from tissues, etc.)
        elif relation == '승천':
            question = hypothesis_to_ask.get('text', f"새로운 개념 '{hypothesis_to_ask['head']}'을(를) 지식의 일부로 만들까요?")

        # Handle correction proposals
        elif relation == 'proposes_correction':
            self.logger.info(f"Generating correction proposal question for: {hypothesis_to_ask['head']} <-> {hypothesis_to_ask['tail']}")
            question = self.question_generator.generate_correction_proposal_question(hypothesis_to_ask)

        # Fallback to the default question generator for all other cases
        if not question:
            question = self.question_generator.generate_question_from_hypothesis(hypothesis_to_ask)

        return {"type": "text", "text": question}

    def handle_response(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """Processes the user's response to a pending hypothesis."""
        hypothesis = context.pending_hypothesis
        if not hypothesis:
            return None # Should not happen if routed correctly

        response_text = ""
        relation = hypothesis.get('relation')

        # --- Route response handling based on relation type ---
        if relation == '승천':
            self.logger.info(f"Processing user response for Ascension hypothesis: {hypothesis['head']}")
            if any(word in message for word in ["응", "맞아", "그래", "승천시켜", "승인"]):
                self.logger.info(f"User approved Ascension. Creating new Node '{hypothesis['head']}' in KG.")
                metadata = hypothesis.get('metadata', {})
                properties = {"type": "concept", "discovery_source": "Cell_Ascension_Ritual", "parents": metadata.get("parents", []), "ascended_at": datetime.now().isoformat()}
                self.kg_manager.add_node(hypothesis['head'], properties=properties)
                response_text = f"알겠습니다. 새로운 개념 '{hypothesis['head']}'이(가) 지식의 일부로 승천했습니다."
            else:
                self.logger.info("User denied Ascension. Discarding hypothesis.")
                response_text = f"알겠습니다. 개념 '{hypothesis['head']}'의 승천을 보류합니다."

        elif relation == 'proposes_correction':
            self.logger.info(f"Processing user response for Correction proposal: {hypothesis['head']} <-> {hypothesis['tail']}")
            if any(word in message for word in ["응", "맞아", "그래", "수정해", "허락한다"]):
                insight = hypothesis.get('metadata', {}).get('contradictory_insight')
                if insight:
                    # Logic Correction: Find what to remove based on the new insight.
                    # The verifier found a contradiction, so we assume an inverse or reversal exists.
                    # Let's explicitly find what's there and remove it.

                    new_head = insight.get('head')
                    new_tail = insight.get('tail')
                    new_relation = insight.get('relation')

                    # Case 1: Direct Reversal (A->B contradicts existing B->A)
                    # We need to remove B->A
                    if self.kg_manager.edge_exists(source=new_tail, target=new_head, relation=new_relation):
                        self.kg_manager.remove_edge(new_tail, new_head, new_relation)
                        self.logger.info(f"Correcting KG: Removed direct reversal edge '{new_tail} -> {new_head}' with relation '{new_relation}'.")

                    # Case 2: Inverse Relation (A 'causes' B contradicts existing A 'caused_by' B)
                    inverse_relations = {"causes": "caused_by", "caused_by": "causes"}
                    inverse_of_new = inverse_relations.get(new_relation)
                    if inverse_of_new and self.kg_manager.edge_exists(source=new_head, target=new_tail, relation=inverse_of_new):
                        self.kg_manager.remove_edge(new_head, new_tail, inverse_of_new)
                        self.logger.info(f"Correcting KG: Removed inverse edge '{new_head} -> {new_tail}' with relation '{inverse_of_new}'.")

                    # Now, add the new, correct edge
                    self.kg_manager.add_edge(new_head, new_tail, new_relation)
                    self.logger.info(f"Correcting KG: Added new edge '{new_head} -> {new_tail}' with relation '{new_relation}'.")

                    response_text = "아버지의 지혜에 따라 저의 지식을 바로잡았습니다. 감사합니다."
                else:
                    response_text = "수정을 진행하려 했으나, 원본 통찰 정보가 부족하여 실패했습니다."
            else:
                self.logger.info("User denied Correction. Discarding proposal.")
                response_text = "알겠습니다. 기존의 지식을 그대로 유지합니다."

        else: # Default handling for standard relationship hypotheses
            self.logger.info(f"Processing user response for relationship hypothesis: {hypothesis['head']} -> {hypothesis['tail']}")
            confirmed_relation = extract_relationship_type(message) or ("related_to" if any(word in message for word in ["응", "맞아", "그래"]) else None)
            if confirmed_relation:
                self.logger.info(f"User confirmed relationship: {confirmed_relation}. Adding edge to KG.")
                self.kg_manager.add_edge(hypothesis['head'], hypothesis['tail'], confirmed_relation)
                response_text = f"알겠습니다. '{hypothesis['head']}'와(과) '{hypothesis['tail']}'의 관계를 기록했습니다."
            else:
                self.logger.info("User denied or provided an unclear answer. Discarding hypothesis.")
                response_text = f"알겠습니다. 가설({hypothesis['head']} -> {hypothesis['tail']})에 대한 답변을 기록했습니다."

        # --- Clean up after processing ---
        self.core_memory.remove_hypothesis(hypothesis['head'], hypothesis.get('tail'), relation=relation)
        context.pending_hypothesis = None
        final_response = self.response_styler.style_response(response_text, emotional_state)
        return {"type": "text", "text": final_response}


class CommandWordHandler:
    """Handles messages that start with a specific command word."""

    def __init__(self, cortex_registry: CortexRegistry, logger: logging.Logger):
        self.cortex_registry = cortex_registry
        self.logger = logger
        self.command_map = {
            r"^(calculate|계산)\s*:": "arithmetic",
        }

    def can_handle(self, message: str) -> bool:
        """Check if the message contains a command this handler can process."""
        return any(re.match(pattern, message.strip(), re.IGNORECASE) for pattern in self.command_map)

    def handle(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """Executes the command."""
        for pattern, cortex_name in self.command_map.items():
            match = re.match(pattern, message.strip(), re.IGNORECASE)
            if match:
                self.logger.info(f"Command '{cortex_name}' detected.")
                raw_command = message.strip()[match.end():].strip()
                cortex = self.cortex_registry.get(cortex_name)
                if cortex:
                    result_text = cortex.process(raw_command)
                    return {"type": "text", "text": result_text}
        return None # Should not be reached if can_handle is checked first


from Project_Mirror.perspective_cortex import PerspectiveCortex
from Project_Sophia.emotional_engine import EmotionalEngine

class DefaultReasoningHandler:
    """The default handler for general messages, performing the main reasoning loop."""

    def __init__(self, reasoner: LogicalReasoner, vcd: ValueCenteredDecision, synthesizer: InsightSynthesizer, creative_cortex: CreativeCortex, styler: ResponseStyler, logger: logging.Logger, perspective_cortex: PerspectiveCortex, question_generator: QuestionGenerator, emotional_engine: EmotionalEngine):
        self.reasoner = reasoner
        self.vcd = vcd
        self.synthesizer = synthesizer
        self.creative_cortex = creative_cortex
        self.styler = styler
        self.logger = logger
        self.perspective_cortex = perspective_cortex
        self.question_generator = question_generator
        self.emotional_engine = emotional_engine
        self.creative_expression_threshold = 2.5

    def handle(self, message: str, context: ConversationContext, emotional_state: EmotionalState) -> Optional[Any]:
        """Generates a response using the main VCD-guided path with Thought objects."""
        # 극단적으로 단순화된 로직
        potential_thoughts = self.reasoner.deduce_facts(message)

        if not potential_thoughts:
            insightful_text = "흥미로운 관점이네요. 조금 더 생각해볼게요."
            final_response = self.styler.style_response(insightful_text, emotional_state)
            return {"type": "text", "text": final_response}

        # VCD 호출 및 기본값 처리
        chosen_thought = self.vcd.select_thought(
            candidates=potential_thoughts,
            context=[message],
            emotional_state=emotional_state,
            guiding_intention=context.guiding_intention
        )
        if not chosen_thought:
            chosen_thought = potential_thoughts[0] # VCD가 결정 못하면 첫번째 생각으로

        # 합성 및 스타일링
        insightful_text = self.synthesizer.synthesize([chosen_thought])
        final_response = self.styler.style_response(insightful_text, emotional_state)
        return {"type": "text", "text": final_response}
