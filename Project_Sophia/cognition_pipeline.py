from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import re
import inspect
import logging # Import logging module
import os # Import os for log file path

from Project_Sophia.core_memory import CoreMemory
from Project_Sophia.core_memory_base import Memory
from Project_Sophia.emotional_state import EmotionalState
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Sophia.action_cortex import ActionCortex
from Project_Sophia.planning_cortex import PlanningCortex
from Project_Sophia.tool_executor import ToolExecutor
from Project_Sophia.value_centered_decision import ValueCenteredDecision, VCDResult
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.inquisitive_mind import InquisitiveMind
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
        self.current_emotional_state = EmotionalState(
            valence=0.0,
            arousal=0.0,
            dominance=0.0,
            primary_emotion="neutral",
            secondary_emotions=[]
        )

    def process_message(self, message: str, app=None, context: Optional[Dict[str, Any]] = None) -> Union[Tuple[str, EmotionalState], Dict[str, Any]]:
        """
        메시지를 처리하고 감정 상태를 업데이트하는 인지 파이프라인
        """
        try:
            # 1. 감정 분석
            emotional_state = self._analyze_emotions(message)
            
            # 2. 문맥 이해 (기억 검색 포함)
            enriched_context = self._enrich_context(context or {}, message)
            
            # 3. 경험 저장
            memory = Memory(
                timestamp=datetime.now().isoformat(),
                content=message,
                emotional_state=emotional_state,
                context=enriched_context
            )
            self.core_memory.add_experience(memory)
            
            # 4. 현재 감정 상태 업데이트
            self._update_emotional_state(emotional_state)
            
            return self._generate_response(message, emotional_state, enriched_context, app)
        except Exception as e:
            pipeline_logger.exception(f"Error in process_message for input: {message}")
            return "An internal error occurred during message processing.", self.current_emotional_state


    def _analyze_emotions(self, message: str) -> EmotionalState:
        """
        메시지의 감정을 분석하여 EmotionalState 반환
        """
        # TODO: 감정 분석 로직 구현
        # 현재는 기본값 반환
        return EmotionalState(
            valence=0.0,
            arousal=0.0,
            dominance=0.0,
            primary_emotion="neutral",
            secondary_emotions=[]
        )

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
        Uses WaveMechanics to generate a context based on the "echo" of an idea
        and searches for relevant past experiences.
        """
        enriched = context.copy()
        
        # 1. Activate concepts using WaveMechanics (the "echo")
        try:
            tokens = set(re.findall(r'\w+', message.lower()))
            # Assuming kg_manager.kg['nodes'] is accessible and structured
            known_nodes = [node['id'] for node in self.kg_manager.kg['nodes']] # This line might need error handling if kg is not loaded
            stimulus_nodes = tokens.intersection(known_nodes)

            if not stimulus_nodes:
                enriched['echo'] = {}
            else:
                start_node = list(stimulus_nodes)[0]
                echo = self.wave_mechanics.spread_activation(start_node) # Potential LLM interaction here
                enriched['echo'] = echo
        except Exception as e:
            pipeline_logger.exception(f"Error in _enrich_context during wave mechanics activation for message: {message}")
            enriched['echo'] = {} # Ensure echo is always present, even on error

        # 2. Add identity and relationship for personalization
        enriched['identity'] = self.core_memory.get_identity()
        if 'speaker' in context:
            enriched['relationship'] = self.core_memory.get_relationship(context['speaker'])

        # 3. Search for relevant memories and add them to the context
        enriched['relevant_experiences'] = self._find_relevant_experiences(message)

        return enriched

    def _update_emotional_state(self, new_state: EmotionalState):
        """
        현재 감정 상태를 새로운 상태로 부드럽게 전이
        """
        alpha = 0.3
        self.current_emotional_state = EmotionalState(
            valence=self.current_emotional_state.valence * (1-alpha) + new_state.valence * alpha,
            arousal=self.current_emotional_state.arousal * (1-alpha) + new_state.arousal * alpha,
            dominance=self.current_emotional_state.dominance * (1-alpha) + new_state.dominance * alpha,
            primary_emotion=new_state.primary_emotion,
            secondary_emotions=list(set(self.current_emotional_state.secondary_emotions + new_state.secondary_emotions))[-3:]
        )

    def _generate_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any], app=None) -> Union[Tuple[str, EmotionalState], Dict[str, Any]]:
        """
        Generates a set of candidate responses and uses the ValueCenteredDecision
        module to select the best one.
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
                    return response, self.current_emotional_state
                else:
                    return "I was unable to develop a plan for that goal.", self.current_emotional_state

            action_decision = self.action_cortex.decide_action(message, app=app)
            if action_decision:
                return self.tool_executor.prepare_tool_call(action_decision)

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

            if emotional_state.primary_emotion == "sad":
                candidates.append("무슨 일 있으신가요? 괜찮으시다면, 저에게 이야기해주세요.")
            elif emotional_state.primary_emotion == "happy":
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
                        candidates.append(self.inquisitive_mind.ask_external_llm(topic))
                except Exception as e:
                    pipeline_logger.warning(f"InquisitiveMind failed for message '{message}': {e}")

            # Priority 5: Creative Candidates
            vis_match = re.search(r"(.+)(을|를)\s*(그려줘|보여줘)", message)
            if vis_match:
                concept = vis_match.group(1).strip()
                # This doesn't generate the image, but proposes the action
                candidates.append(f"'{concept}'에 대한 저의 생각을 그림으로 표현해볼까요?")


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
                return best_action_result.chosen_action, self.current_emotional_state
            else:
                pipeline_logger.warning(f"VCD returned no valid action for input: {message}. Falling back to default.")
                return "죄송해요, 적절한 응답을 찾지 못했어요.", self.current_emotional_state

        except Exception as e:
            pipeline_logger.exception(f"Unhandled error in _generate_response for message: {message}")
            return "An unexpected error occurred while generating a response.", self.current_emotional_state