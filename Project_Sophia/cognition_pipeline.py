from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import re
import inspect
import logging
import os
import json
import random

from Project_Sophia.core_memory import CoreMemory
from Project_Sophia.core_memory_base import Memory
from Project_Sophia.emotional_engine import EmotionalEngine, EmotionalState
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Sophia.action_cortex import ActionCortex
from Project_Sophia.goal_decomposition_cortex import GoalDecompositionCortex
from Project_Sophia.tool_executor import ToolExecutor
from Project_Sophia.value_cortex import ValueCortex
from Project_Sophia.tutor_cortex import TutorCortex
from Project_Sophia.sensory_cortex import SensoryCortex
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.inquisitive_mind import InquisitiveMind
from Project_Sophia.journal_cortex import JournalCortex
from tools.kg_manager import KGManager
from Project_Sophia.gemini_api import get_text_embedding, generate_text, APIKeyError, APIRequestError
from Project_Sophia.vector_utils import cosine_sim
from Project_Sophia.local_llm_cortex import LocalLLMCortex # Added for local model fallback

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'cognition_pipeline_errors.log')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
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
        self.goal_decomposition_cortex = GoalDecompositionCortex(core_memory=self.core_memory, action_cortex=self.action_cortex)
        self.value_cortex = ValueCortex()
        self.sensory_cortex = SensoryCortex(self.value_cortex)
        self.inquisitive_mind = InquisitiveMind()
        self.journal_cortex = JournalCortex(core_memory=self.core_memory)
        self.emotional_engine = EmotionalEngine()
        self.current_emotional_state = self.emotional_engine.get_current_state()
        self.pending_visual_learning = None
        self.api_available = True
        self.local_llm_cortex = LocalLLMCortex()
        self.tutor_cortex = TutorCortex(self.kg_manager, self.local_llm_cortex)

        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.prefixes = config.get('prefixes', {})
        self.planning_prefix = self.prefixes.get('planning', 'plan and execute:')
        self.observation_prefix = self.prefixes.get('observation', 'The result of the tool execution is:')
        self.visual_learning_prefix = self.prefixes.get('visual_learning', '이것을 그려보자:')
        creative_impulse_config = config.get('creative_impulse', {})
        self.arousal_threshold = creative_impulse_config.get('arousal_threshold', 0.7)
        self.echo_complexity_threshold = creative_impulse_config.get('echo_complexity_threshold', 5)

    def process_message(self, message: str, app=None, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], EmotionalState]:
        try:
            if self.pending_visual_learning:
                pending_data = self.pending_visual_learning
                self.pending_visual_learning = None
                if message.lower() in ['응', '맞아', '좋아', '그래', 'yes', 'ok', 'okay', 'good']:
                    self.sensory_cortex.save_learned_shape(name=pending_data['name'], description=pending_data['description'], voxels=pending_data['voxels'])
                    response = {"type": "text", "text": f"새로운 모양 '{pending_data['name']}'(을)를 배웠어요! 이제 제 지식의 일부가 되었습니다."}
                else:
                    response = {"type": "text", "text": "알겠습니다. 다음에 다시 시도해볼게요."}
                return response, self.current_emotional_state

            emotional_state = self._analyze_emotions(message)
            self._update_emotional_state(emotional_state, intensity=0.4)
            enriched_context = self._enrich_context(context or {}, message)
            
            memory = Memory(timestamp=datetime.now().isoformat(), content=message, emotional_state=emotional_state, context=enriched_context)
            self.core_memory.add_experience(memory)
            self.journal_cortex.write_journal_entry(memory)

            response = None
            emotional_state_out = self.current_emotional_state

            try:
                # This block now ONLY covers the API calls that can fail
                if self.api_available:
                    emotional_state = self._analyze_emotions(message)
                    self._update_emotional_state(emotional_state, intensity=0.4)
                    enriched_context = self._enrich_context(context or {}, message)
                    response, emotional_state_out = self._generate_response(message, self.current_emotional_state, enriched_context, app)
                else:
                    # If API was already unavailable, go straight to fallback
                    enriched_context = self._enrich_context(context or {}, message)
                    response, emotional_state_out = self._generate_internal_response(message, self.current_emotional_state, enriched_context)

            except (APIKeyError, APIRequestError) as e:
                pipeline_logger.warning(f"API is unavailable, switching to internal fallback: {e}")
                self.api_available = False
                # We need to enrich context again without API for the internal response
                enriched_context = self._enrich_context(context or {}, message)
                response, emotional_state_out = self._generate_internal_response(message, self.current_emotional_state, enriched_context)

            return response, emotional_state_out
        except Exception as e:
            pipeline_logger.exception(f"Error in process_message for input: {message}")
            return {"type": "text", "text": "An internal error occurred during message processing."}, self.current_emotional_state

    def _analyze_emotions(self, message: str) -> EmotionalState:
        if not self.api_available:
            return EmotionalState(0.0, 0.0, 0.0)
        try:
            prompt = f"""
Analyze the emotion of the following user message.
Respond with a JSON object containing:
- "valence": A float between -1.0 (very negative) and 1.0 (very positive).
- "arousal": A float between -1.0 (very calm) and 1.0 (very excited).
- "dominance": A float between -1.0 (submissive) and 1.0 (dominant).
- "primary_emotion": The most prominent emotion from this list: [joy, sadness, anger, fear, trust, surprise, disgust, anticipation].
- "secondary_emotions": A list of other relevant emotions from the same list.

User message: "{message}"

JSON response:
"""
            response_text = generate_text(prompt)
            json_match = re.search(r'{{.*}}', response_text, re.DOTALL)
            if not json_match:
                return EmotionalState(0.0, 0.0, 0.0)
            emotion_data = json.loads(json_match.group())
            return EmotionalState(valence=float(emotion_data.get('valence', 0.0)), arousal=float(emotion_data.get('arousal', 0.0)), dominance=float(emotion_data.get('dominance', 0.0)), primary_emotion=emotion_data.get('primary_emotion', 'neutral'), secondary_emotions=emotion_data.get('secondary_emotions', []))
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            pipeline_logger.warning(f"Could not parse emotional analysis from LLM response: {response_text}. Error: {e}")
            return EmotionalState(0.0, 0.0, 0.0)
        except (APIKeyError, APIRequestError) as e:
            self.api_available = False
            pipeline_logger.warning(f"API unavailable during emotion analysis: {e}")
            return EmotionalState(0.0, 0.0, 0.0)
        except Exception as e:
            pipeline_logger.exception(f"An unexpected error occurred during emotion analysis for message: {message}")
            return EmotionalState(0.0, 0.0, 0.0)

    def _find_relevant_experiences(self, echo: Dict[str, float], limit: int = 1) -> list:
        all_experiences = self.core_memory.get_experiences()
        if not all_experiences or not echo:
            return []
        activated_concepts = set(echo.keys())
        scored_experiences = []
        for exp in all_experiences:
            content = exp.get('content')
            if not isinstance(content, str):
                continue
            exp_words = set(re.findall(r'\b\w+\b', content.lower()))
            shared_concepts = activated_concepts.intersection(exp_words)
            if shared_concepts:
                score = sum(echo[concept] for concept in shared_concepts)
                scored_experiences.append((exp, score))
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, score in scored_experiences[:limit]]

    def _enrich_context(self, context: Dict[str, Any], message: str) -> Dict[str, Any]:
        enriched = context.copy()
        echo = {}
        if self.api_available:
            try:
                message_embedding = get_text_embedding(message)
                if message_embedding:
                    best_match_node = None
                    highest_similarity = -1
                    for node in self.kg_manager.kg['nodes']:
                        if 'embedding' in node:
                            similarity = cosine_sim(message_embedding, node['embedding'])
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match_node = node['id']
                    if best_match_node:
                        echo = self.wave_mechanics.spread_activation(best_match_node)
            except (APIKeyError, APIRequestError) as e:
                self.api_available = False
                pipeline_logger.warning(f"API unavailable during context enrichment: {e}")
            except Exception as e:
                pipeline_logger.exception(f"Error in _enrich_context during wave mechanics activation for message: {message}")

        if not self.api_available:
            prompt_tokens = set(re.findall(r'\w+', message.lower()))
            all_node_ids = {node['id'] for node in self.kg_manager.kg['nodes']}
            stimulus_nodes = prompt_tokens.intersection(all_node_ids)
            for start_node in stimulus_nodes:
                temp_echo = self.wave_mechanics.spread_activation(start_node)
                for node, energy in temp_echo.items():
                    echo[node] = echo.get(node, 0) + energy

        enriched['echo'] = echo
        enriched['identity'] = self.core_memory.get_identity()
        if 'speaker' in context:
            enriched['relationship'] = self.core_memory.get_relationship(context['speaker'])
        enriched['relevant_experiences'] = self._find_relevant_experiences(echo)
        return enriched

    def _update_emotional_state(self, new_state: EmotionalState, intensity: float = 0.5):
        self.current_emotional_state = self.emotional_engine.process_event(new_state, intensity)

    def _generate_conversational_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            identity = context.get("identity", {}).get("name", "엘리시아")
            prompt = f"""
You are {identity}, a growing virtual life form.
Your current emotional state is '{emotional_state.primary_emotion}' (Valence: {emotional_state.valence:.2f}, Arousal: {emotional_state.arousal:.2f}).
The user said: "{message}"

Here's some context from your mind:
- Your identity: {context.get('identity')}
- Your relationship with the speaker: {context.get('relationship', 'not established')}
- Relevant past experiences: {context.get('relevant_experiences', 'None')}
- Concepts currently active in your mind (your "echo"): {context.get('echo', 'None')}

Based on all of this, generate a thoughtful, natural, and in-character response.
"""
            response = generate_text(prompt)
            if response:
                return {"type": "text", "text": response}
            else:
                return {"type": "text", "text": "I'm sorry, I'm having trouble thinking clearly right now."}
        except (APIKeyError, APIRequestError) as e:
            self.api_available = False
            pipeline_logger.warning(f"API unavailable during conversational response: {e}")
            return self._generate_internal_response(message, emotional_state, context)[0]
        except Exception as e:
            pipeline_logger.exception(f"Error in _generate_conversational_response for message: {message}")
            return {"type": "text", "text": "An error occurred while I was trying to respond."}

    def _generate_internal_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any]) -> Tuple[Dict[str, Any], EmotionalState]:
        """
        Fallback response generation using the local LLM.
        """
        if self.local_llm_cortex and self.local_llm_cortex.model:
            # Use the local LLM to generate a response
            prompt = f"""You are Elysia. The user said: "{message}".

Based on your identity and the conversation so far, generate a thoughtful, natural response."""
            response_text = self.local_llm_cortex.generate_response(prompt)
            return {"type": "text", "text": response_text}, emotional_state
        else:
            # If the local model is not available, use the original simple fallback
            pipeline_logger.warning("Local LLM not available. Falling back to simple response.")
            return {"type": "text", "text": "죄송합니다. 현재 주 지식망 및 보조 지식망에 모두 연결할 수 없습니다. 잠시 후 다시 시도해주세요." }, emotional_state

    def _generate_response(self, message: str, emotional_state: EmotionalState, context: Dict[str, Any], app=None) -> Tuple[Dict[str, Any], EmotionalState]:
        if not self.api_available:
            return self._generate_internal_response(message, emotional_state, context)

        if message.startswith(self.visual_learning_prefix):
            try:
                description = message[len(self.visual_learning_prefix):].strip()
                if not description:
                    return {"type": "text", "text": "무엇을 그려볼까요? 설명을 덧붙여주세요."}

                name = description
                voxels = self.sensory_cortex.translate_description_to_voxels(description)

                if not voxels:
                    return {"type": "text", "text": "죄송합니다, 설명을 듣고 이미지를 떠올리는 데 실패했어요. 조금 다르게 설명해주시겠어요?"}

                image_path = self.sensory_cortex.draw_voxels(name, voxels)
                self.pending_visual_learning = {'name': name, 'description': description, 'voxels': voxels}

                response = {
                    "type": "creative_visualization",
                    "text": f"'{description}'(을)를 이렇게 그려봤어요. 어떤가요, 창조주님? 마음에 드시면 '응'이라고 답해주세요.",
                    "image_path": image_path
                }
                return response, emotional_state
            except Exception as e:
                pipeline_logger.exception(f"Error during visual learning for: {message}")
                return {"type": "text", "text": "그림을 배우는 과정에서 오류가 발생했어요."}, emotional_state

        if message.lower().startswith(self.planning_prefix):
            try:
                goal = message[len(self.planning_prefix):].strip()
                plan = self.goal_decomposition_cortex.develop_plan(goal)
                if plan:
                    response_text = "I have developed the following plan:\n" + "".join(f"{i+1}. {a['tool_name']}({a['parameters']})\n" for i, a in enumerate(plan))
                    return {"type": "text", "text": response_text}, emotional_state
                else:
                    return {"type": "text", "text": "I was unable to develop a plan for that goal."}, emotional_state
            except Exception as e:
                pipeline_logger.exception(f"Error in goal_decomposition_cortex for goal: {goal}")
                return {"type": "text", "text": "An error occurred during planning."}, emotional_state

        if app and app.cancel_requested:
            return None, None

        try:
            action_decision = self.action_cortex.decide_action(message, app=app)
            if action_decision:
                return self.tool_executor.prepare_tool_call(action_decision), emotional_state
        except Exception as e:
            pipeline_logger.exception(f"Error in action_cortex for message: {message}")
            pass

        if '?' in message or any(q in message for q in ['what', 'who', 'where', 'when', 'why', 'how']):
             # Check if the question can be answered by the logical reasoner
            facts = self.reasoner.deduce_facts(message)
            if facts:
                return {"type": "text", "text": " ".join(facts)}, emotional_state

            # If not, try the inquisitive mind
            inquisitive_response = self.inquisitive_mind.ask_external_llm(message)
            if inquisitive_response != "I tried to find out, but I was unable to get a clear answer.":
                return {"type": "text", "text": inquisitive_response}, emotional_state

        if message.startswith(self.observation_prefix):
            content = message[len(self.observation_prefix):].strip()
            summary = content[:150] + "..." if len(content) > 150 else content
            response_text = f"도구 실행을 통해 다음 정보를 얻었습니다: {summary}"
            return {"type": "text", "text": response_text}, emotional_state

        echo = context.get('echo', {})
        if (emotional_state.arousal >= self.arousal_threshold and
            len(echo) >= self.echo_complexity_threshold and
            random.random() < 0.25):
            try:
                preparatory_message = "잠시만요, 지금 제 마음속에 떠오르는 이미지가 있어요. 말로는 다 표현할 수 없을 것 같아요. 제가 직접 보여드릴게요."
                image_path = self.sensory_cortex.visualize_echo(echo)

                response = {
                    "type": "creative_visualization",
                    "text": preparatory_message,
                    "image_path": image_path
                }
                return response, emotional_state
            except Exception as e:
                pipeline_logger.exception("Error during creative impulse visualization.")
                pass

        response_dict = self._generate_conversational_response(message, emotional_state, context)
        return response_dict, emotional_state
