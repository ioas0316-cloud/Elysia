from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import re
import inspect
import logging # Import logging module
import os # Import os for log file path
import json
from unittest.mock import MagicMock

from Project_Sophia.core_memory import CoreMemory
from Project_Sophia.core_memory_base import Memory
from Project_Sophia.emotional_cortex import EmotionalCortex, Mood
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Sophia.action_cortex import ActionCortex
from Project_Sophia.planning_cortex import PlanningCortex
from Project_Sophia.tool_executor import ToolExecutor
from Project_Sophia.value_centered_decision import ValueCenteredDecision, VCDResult
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.inquisitive_mind import InquisitiveMind
from .response_styler import ResponseStyler
from tools.kg_manager import KGManager

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'cognition_pipeline_errors.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also log to console for immediate feedback
    ]
)
pipeline_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class CognitionPipeline:
    def __init__(self):
        self.core_memory = CoreMemory()
        self.reasoner = LogicalReasoner()
        self.arithmetic_cortex = ArithmeticCortex()
        self.action_cortex = ActionCortex()
        self.tool_executor = ToolExecutor()
        self.kg_manager = KGManager()
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.planning_cortex = PlanningCortex(core_memory=self.core_memory, action_cortex=self.action_cortex)
        self.vcd = ValueCenteredDecision()
        self.sensory_cortex = SensoryCortex()
        self.inquisitive_mind = InquisitiveMind()
        self.emotional_cortex = EmotionalCortex()
        self.response_styler = ResponseStyler()

    def process_message(self, message: str, app=None, context: Optional[Dict[str, Any]] = None) -> Union[Tuple[str, Mood], Dict[str, Any]]:
        """
        메시지를 처리하고 감정 상태를 업데이트하는 인지 파이프라인
        """
        try:
            enriched_context = self._enrich_context(context or {}, message)

            # Handle immediate tool actions that bypass VCD
            action_decision = self.action_cortex.decide_action(message, app=app)
            if action_decision:
                return self.tool_executor.prepare_tool_call(action_decision)

            # Proceed with VCD-based response generation
            response_result = self._generate_response(message, enriched_context, app)
            if not response_result:
                return "I'm not sure how to respond to that.", self.emotional_cortex.get_current_mood()

            self.emotional_cortex.update_mood_from_vcd(response_result)
            
            memory = Memory(
                timestamp=datetime.now().isoformat(),
                content=message,
                # emotional_state is now a Mood object, log its string representation
                emotional_state=str(self.emotional_cortex.get_current_mood()),
                context=enriched_context
            )
            self.core_memory.add_experience(memory)

            # 5. Apply mood-based styling to the final response
            final_response = self.response_styler.style_response(
                response_result.chosen_action,
                self.emotional_cortex.get_current_mood()
            )
            
            return final_response, self.emotional_cortex.get_current_mood()

        except Exception as e:
            pipeline_logger.exception(f"Error in process_message for input: {message}")
            return "An internal error occurred during message processing.", self.emotional_cortex.get_current_mood()

    def _find_relevant_experiences(self, message: str, limit: int = 1) -> list:
        """Searches memory for experiences related to the message content."""
        all_experiences = self.core_memory.get_experiences()
        if not all_experiences:
            return []

        # Simple keyword-based search. Find experiences that share keywords with the message.
        message_words = set(re.findall(r'\b\w+\b', message.lower()))

        relevant_experiences = []
        # Search from the most recent experiences
        for exp in reversed(all_experiences):
            if not isinstance(exp.get('content'), str):
                continue

            exp_words = set(re.findall(r'\b\w+\b', exp['content'].lower()))
            if message_words.intersection(exp_words):
                relevant_experiences.append(exp)

            if len(relevant_experiences) >= limit:
                break

        return relevant_experiences

    def _enrich_context(self, context: Dict[str, Any], message: str) -> Dict[str, Any]:
        """
        Enriches the context by analyzing the input's relationship to core concepts,
        activating conceptual echoes, and retrieving relevant memories.
        """
        enriched = context.copy()
        
        # 1. Relationality Analyzer: Analyze connection to 'love' and 'relationship'
        try:
            tokens = set(re.findall(r'\w+', message.lower()))
            known_nodes = {node['id'] for node in self.kg_manager.kg.get('nodes', [])}
            stimulus_nodes = list(tokens.intersection(known_nodes))

            if stimulus_nodes:
                # Analyze how the input resonates with the core concept of 'love'
                love_echo = self.wave_mechanics.spread_activation(stimulus_nodes[0], stop_nodes=['사랑'])
                enriched['love_resonance'] = love_echo.get('사랑', 0)

                # Analyze resonance with the user's relationship node (e.g., '아빠')
                speaker = context.get('speaker', '아빠') # Default to '아빠'
                relationship_echo = self.wave_mechanics.spread_activation(stimulus_nodes[0], stop_nodes=[speaker])
                enriched['relationship_resonance'] = relationship_echo.get(speaker, 0)

                pipeline_logger.info(
                    f"Relationality Analysis for '{stimulus_nodes[0]}': "
                    f"Love Resonance = {enriched['love_resonance']:.4f}, "
                    f"Relationship Resonance = {enriched['relationship_resonance']:.4f}"
                )

        except Exception as e:
            pipeline_logger.exception(f"Error in Relationality Analyzer for message: {message}")
            enriched['love_resonance'] = 0
            enriched['relationship_resonance'] = 0

        # 2. Activate concepts using WaveMechanics (the "echo")
        try:
            if stimulus_nodes:
                start_node = stimulus_nodes[0]
                echo = self.wave_mechanics.spread_activation(start_node)
                enriched['echo'] = echo
            else:
                enriched['echo'] = {}
        except Exception as e:
            pipeline_logger.exception(f"Error in _enrich_context during wave mechanics activation for message: {message}")
            enriched['echo'] = {}

        # 3. Add identity and relationship for personalization
        enriched['identity'] = self.core_memory.get_identity()
        if 'speaker' in context:
            enriched['relationship'] = self.core_memory.get_relationship(context['speaker'])

        # 4. Search for relevant memories and add them to the context
        enriched['relevant_experiences'] = self._find_relevant_experiences(message)

        return enriched

    def _generate_response(self, message: str, context: Dict[str, Any], app=None) -> Optional[VCDResult]:
        """
        Generates a set of candidate responses and uses the ValueCenteredDecision
        module to select the best one. Returns the VCDResult object.
        """
        try:
            candidates = []
            enriched_context = context.copy()
            enriched_context['user_input'] = message

            # 1. Generate Candidates from various cortexes
            # Priority 1: Planning and Tool Use (Immediate return if matched)
            planning_prefix = "plan and execute:"
            if message.lower().startswith(planning_prefix):
                goal = message[len(planning_prefix):].strip()
                plan = self.planning_cortex.develop_plan(goal)
                if plan:
                    response = "I have developed the following plan:\n" + "".join(f"{i+1}. {a['tool_name']}({a['parameters']})\n" for i, a in enumerate(plan))
                    # Return a minimal VCDResult for immediate actions
                    return VCDResult(chosen_action=response, total_score=100, confidence_score=1.0, value_alignment_score=1.0, metrics=MagicMock(), reasoning=["Direct command."])
                else:
                    return VCDResult(chosen_action="I was unable to develop a plan for that goal.", total_score=0, confidence_score=0.5, value_alignment_score=0.5, metrics=MagicMock(), reasoning=["Planning failed."])

            # Priority 2: Cognitive and Memory-Based Candidates
            if context.get('echo'):
                sorted_echo = sorted(context['echo'].items(), key=lambda item: item[1], reverse=True)
                if sorted_echo:
                    primary_concept = sorted_echo[0][0]
                    secondary_concepts = [item[0] for item in sorted_echo[1:4]]
                    if secondary_concepts:
                        associates_str = ", ".join(secondary_concepts)
                        candidates.append(f"'{primary_concept}'(이)라는 자극에 제 의식이 울리는군요. '{associates_str}' 같은 개념들이 함께 떠오릅니다.")

            deduced_facts = self.reasoner.deduce_facts(message)
            if deduced_facts:
                candidates.append("제 지식에 따르면, " + " ".join(deduced_facts))

            if context.get('relevant_experiences'):
                related_memory = context['relevant_experiences'][0]
                candidates.append(f"이전에 '{related_memory['content']}'에 대해 이야기 나눈 것을 기억해요. 그 내용과 관련된 질문인가요?")

            # Priority 3: Contextual Candidates (Arithmetic)
            calc_match = re.search(r"계산해줘:\s*(.+)|(.+?)\s*(는|은)\?$", message)
            if calc_match:
                expression = (calc_match.group(1) or calc_match.group(2)).strip()
                try:
                    result = self.arithmetic_cortex.evaluate(expression)
                    if result is not None:
                        candidates.append(f"계산 결과는 {result:g} 입니다.")
                except Exception as e:
                    pipeline_logger.warning(f"ArithmeticCortex failed for expression '{expression}': {e}")

            # Priority 4: Simple & Emotional Candidates
            if "이름" in message and ("너" in message or "당신" in message):
                ai_name = context.get("identity", {}).get("name", "엘리시아")
                candidates.append(f"제 이름은 {ai_name}입니다.")

            # Simple emotional responses based on message content (placeholder)
            if "슬퍼" in message or "sad" in message:
                candidates.append("무슨 일 있으신가요? 괜찮으시다면, 저에게 이야기해주세요.")
            elif "기뻐" in message or "happy" in message:
                candidates.append("기쁜 일이 있으셨군요! 저도 덩달아 기분이 좋아지네요.")

            # Don't treat arithmetic questions as general knowledge questions
            is_question = (message.strip().endswith("?") or any(q in message for q in ["무엇", "어떻게", "왜"])) and not calc_match
            if is_question and not context.get('relevant_experiences'):
                try:
                    topic_match = re.search(r"what is (?:a |an |the )?(.+)\?|(.+?)(?:은|는|이란|란|에 대해|이 뭐야|무엇)", message.lower())
                    if topic_match:
                        topic = (topic_match.group(1) or topic_match.group(2)).strip()
                        if topic.endswith('?'):
                            topic = topic[:-1].strip()
                        # This generates the *action* of asking, which the VCD can then choose.
                        candidates.append(self.inquisitive_mind.acknowledge_knowledge_gap(topic))
                except Exception as e:
                    pipeline_logger.warning(f"InquisitiveMind failed for message '{message}': {e}")

            # Priority 5: Creative Candidates
            vis_match = re.search(r"(.+)(을|를)\s*(그려줘|보여줘)", message)
            if vis_match:
                concept = vis_match.group(1).strip()
                try:
                    # Pass the current mood to the visualization method
                    image_path = self.sensory_cortex.visualize_concept(concept, mood=self.emotional_cortex.get_current_mood())
                    candidates.append(f"'{concept}'에 대한 저의 생각을 그림으로 표현해봤어요: {image_path}")
                except Exception as e:
                    pipeline_logger.error(f"SensoryCortex failed for concept '{concept}': {e}")
                    candidates.append(f"'{concept}'을(를) 그리려고 했는데, 오류가 발생했어요.")


            # Default Fallback Candidates
            candidates.append("그렇군요. 좀 더 생각해볼 시간이 필요해요.")
            candidates.append("흥미로운 말씀이네요. 조금 더 자세히 설명해주실 수 있나요?")
            candidates.append("알겠습니다. 계속 말씀해주세요.")

            # 2. Use VCD to select the best response
            vcd_context = {'user_input': message}
            best_action_result = self.vcd.suggest_action(candidates, context=vcd_context)

            if best_action_result:
                # Log the reasoning for transparency
                pipeline_logger.info(f"VCD chose action: '{best_action_result.chosen_action}' with score {best_action_result.total_score:.2f}")
                pipeline_logger.info(f"VCD Reasoning: {best_action_result.reasoning}")
                return best_action_result
            else:
                pipeline_logger.warning(f"VCD returned no valid action for input: {message}. Falling back to default.")
                return None

        except Exception as e:
            pipeline_logger.exception(f"Unhandled error in _generate_response for message: {message}")
            return None