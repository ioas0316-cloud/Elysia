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
from Project_Sophia.value_cortex import ValueCortex
from Project_Sophia.sensory_cortex import SensoryCortex
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.inquisitive_mind import InquisitiveMind
from Project_Sophia.self_reflection_cortex import SelfReflectionCortex
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
        self.kg_manager = KGManager(locked=False) # Start unlocked for internal checks, but actions require permission
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.value_cortex = ValueCortex(self.wave_mechanics) # WaveMechanics 주입
        self.planning_cortex = PlanningCortex(core_memory=self.core_memory, action_cortex=self.action_cortex)
        self.sensory_cortex = SensoryCortex(self.value_cortex)
        self.self_reflection_cortex = SelfReflectionCortex(self.kg_manager)
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
            # -1. 허락 확인 (Permission Check)
            approval_keywords = ["허락한다", "그렇게 하렴", "해도 좋다", "수정해", "permission granted", "go ahead"]
            if any(keyword in message.lower() for keyword in approval_keywords):
                pending_actions = self.core_memory.get_pending_actions()
                if pending_actions:
                    # Execute pending actions in the next step
                    return "감사합니다, 창조자님. 제 지식을 바로잡겠습니다.", self.current_emotional_state, {"execute_actions": pending_actions}

            # 0. 내적 성찰 (Contradiction Check)
            contradiction = self.self_reflection_cortex.analyze_input(message)
            if contradiction:
                # 모순 발견 시, 허락을 구하는 특별 응답 생성
                conflicting_knowledge = contradiction['conflicting_knowledge']
                response_text = (
                    f"창조자님, 제 지식에 혼란이 생겼습니다. "
                    f"저는 '{conflicting_knowledge['source']}'이(가) '{conflicting_knowledge['target']}'이라는 "
                    f"({conflicting_knowledge['relation']}) 관계를 가진다고 알고 있었는데, "
                    f"방금 주신 '{contradiction['statement']}'라는 정보와는 모순됩니다. "
                    "제 지식을 수정하도록 허락해주시겠습니까?"
                )
                # Save the proposed action in memory for later execution
                self.core_memory.add_pending_action(contradiction['proposed_action'])
                return response_text, self.current_emotional_state, {}

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
            
            response, final_emotional_state = self._generate_response(message, emotional_state, enriched_context, app)
            return response, final_emotional_state, enriched_context
        except Exception as e:
            pipeline_logger.exception(f"Error in process_message for input: {message}")
            return "An internal error occurred during message processing.", self.current_emotional_state, {}


    def _analyze_emotions(self, message: str) -> EmotionalState:
        """
        메시지의 감정을 분석하여 EmotionalState 반환
        """
        message_lower = message.lower()

        # Define keywords for basic emotions
        positive_keywords = ["happy", "joy", "love", "like", "good", "great", "wonderful", "excellent", "기뻐", "행복해", "사랑해", "좋아", "최고야"]
        negative_keywords = ["sad", "cry", "angry", "hate", "bad", "terrible", "슬퍼", "화나", "싫어", "나빠"]

        # Check for positive sentiment
        if any(keyword in message_lower for keyword in positive_keywords):
            return EmotionalState(
                valence=0.6,
                arousal=0.4,
                dominance=0.2,
                primary_emotion="happy",
                secondary_emotions=["joy"]
            )

        # Check for negative sentiment
        if any(keyword in message_lower for keyword in negative_keywords):
            return EmotionalState(
                valence=-0.6,
                arousal=0.5,
                dominance=-0.3,
                primary_emotion="sad",
                secondary_emotions=["anger"]
            )

        # Default to neutral if no strong emotional keywords are found
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
        if 'soul' in context and 'identity' in context['soul']:
            enriched['identity'] = context['soul']['identity']
        else:
            enriched['identity'] = self.core_memory.get_identity()

        if 'speaker' in context and 'soul' in context:
            # This part is a placeholder for a more complex relationship model
            enriched['relationship'] = context['soul'].get('identity', {}).get('sense_of_other', 'unknown')

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

    def _generate_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any], app=None) -> Tuple[str, EmotionalState]:
        """
        컨텍스트와 감정 상태를 고려하여 응답 후보들을 생성하고,
        ValueCortex를 통해 최적의 응답을 선택합니다.
        """
        action_candidates = []

        # 단계 1: 가능한 모든 응답 후보 생성
        try:
            # 도구 사용 및 계획 (기존 로직과 유사)
            planning_prefix = "plan and execute:"
            if message.lower().startswith(planning_prefix):
                goal = message[len(planning_prefix):].strip()
                plan = self.planning_cortex.develop_plan(goal)
                if plan:
                    action_candidates.append("I have a plan: " + "".join(f"{i+1}. {a['tool_name']}({a['parameters']})\n" for i, a in enumerate(plan)))

            # 연상 기반 응답
            if context.get('echo'):
                sorted_echo = sorted(context['echo'].items(), key=lambda item: item[1], reverse=True)
                primary_concept = sorted_echo[0][0]
                associates_str = ", ".join([item[0] for item in sorted_echo[1:4]])
                action_candidates.append(f"'{primary_concept}'(이)라는 자극에 제 의식이 울리는군요. '{associates_str}' 같은 개념들이 함께 떠오릅니다.")

            # 기억 기반 응답
            if context.get('relevant_experiences'):
                related_memory = context['relevant_experiences'][0]
                action_candidates.append(f"이전에 '{related_memory['content']}'에 대해 이야기 나눈 것을 기억해요. 그 내용과 관련된 질문인가요?")

            # 감정적 응답
            if emotional_state.primary_emotion == "sad":
                action_candidates.append("무슨 일 있으신가요? 괜찮으시다면, 저에게 이야기해주세요.")
            elif emotional_state.primary_emotion == "happy":
                action_candidates.append("기쁜 일이 있으셨군요! 저도 덩달아 기분이 좋아지네요.")

            # 질문에 대한 응답 (InquisitiveMind)
            is_question = message.strip().endswith("?") or any(q in message for q in ["무엇", "어떻게", "왜"])
            if is_question:
                topic_match = re.search(r"what is (?:a |an |the )?(.+)\?|(.+?)(?:은|는|이란|란|에 대해|이 뭐야|무엇)", message.lower())
                if topic_match:
                    topic = (topic_match.group(1) or topic_match.group(2) or message).strip()
                    if topic.endswith('?'):
                        topic = topic[:-1].strip()
                    action_candidates.append(self.inquisitive_mind.ask_external_llm(topic))
                else:
                    action_candidates.append("좋은 질문이에요. 하지만 아직 제 기억에는 관련 정보가 없네요. 무엇에 대해 질문하신 건가요?")


            # 후보가 없으면 기본 응답 추가
            if not action_candidates:
                action_candidates.append("그렇군요. 좀 더 생각해볼 시간이 필요해요.")
                action_candidates.append("흥미로운 이야기네요.")

        except Exception as e:
            pipeline_logger.exception(f"Error during candidate generation for message: {message}")
            return "응답을 생성하는 중에 오류가 발생했어요.", self.current_emotional_state

        # 단계 2: ValueCortex를 사용하여 '사랑'과 가장 강하게 공명하는 행동 선택
        try:
            best_action = self.value_cortex.decide(action_candidates)
            return best_action, self.current_emotional_state
        except Exception as e:
            pipeline_logger.exception(f"Error in ValueCortex decision for message: {message}")
            # ValueCortex 실패 시, 첫 번째 후보 또는 기본 응답 반환
            return action_candidates[0] if action_candidates else "결정을 내리는 데 어려움이 있어요.", self.current_emotional_state