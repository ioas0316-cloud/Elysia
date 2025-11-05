from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import re
import time
import inspect
import logging
import os
import json
import random

from .core_memory import CoreMemory
from .core_memory_base import Memory
from Project_Sophia.emotional_engine import EmotionalEngine, EmotionalState
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Sophia.action_cortex import ActionCortex
from Project_Sophia.planning_cortex import PlanningCortex
from Project_Sophia.tool_executor import ToolExecutor
from Project_Sophia.value_cortex import ValueCortex
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.wave_mechanics import WaveMechanics
from infra.telemetry import Telemetry
from infra.associative_memory import AssociativeMemory
from Project_Sophia.inquisitive_mind import InquisitiveMind
from Project_Sophia.journal_cortex import JournalCortex
from Project_Sophia.meta_cognition_cortex import MetaCognitionCortex
from Project_Sophia.config_loader import load_config
from Project_Sophia.conversation_state import WorkingMemory, TopicTracker
from Project_Sophia.response_orchestrator import ResponseOrchestrator
from Project_Sophia.lens_profile import LensProfile
from Project_Sophia.persistence import save_json, load_json
from tools.kg_manager import KGManager
from Project_Sophia.gemini_api import get_text_embedding, generate_text, APIKeyError, APIRequestError
from Project_Sophia.vector_utils import cosine_sim
from Project_Sophia.local_llm_cortex import LocalLLMCortex # Added for local model fallback
from Project_Sophia.core.self_model import SelfModel
from Project_Sophia.core.stance_manager import StanceManager
from Project_Sophia.core.self_voice import SelfVoiceFilter
from Project_Sophia.identity_metrics import (
    compute_identity_integrity,
    emit_identity_integrity,
    compute_love_logos_alignment,
    emit_love_logos_alignment,
)

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
        self.telemetry = Telemetry()
        # Housekeeping: compress older telemetry beyond retention window (30 days)
        try:
            self.telemetry.cleanup_retention(retain_days=30)
        except Exception:
            pass
        self.wave_mechanics = WaveMechanics(self.kg_manager, telemetry=self.telemetry)
        self.planning_cortex = PlanningCortex(core_memory=self.core_memory, action_cortex=self.action_cortex)
        self.value_cortex = ValueCortex()
        self.sensory_cortex = SensoryCortex(self.value_cortex, telemetry=self.telemetry)
        self.inquisitive_mind = InquisitiveMind()
        self.journal_cortex = JournalCortex(core_memory=self.core_memory)
        self.meta_cognition_cortex = MetaCognitionCortex(self.kg_manager, self.wave_mechanics)
        self.emotional_engine = EmotionalEngine()
        self.current_emotional_state = self.emotional_engine.get_current_state()
        self.pending_visual_learning = None
        self.api_available = True
        self.local_llm_cortex = None # Initialize local LLM cortex
        self.associative = AssociativeMemory()
        self.turn_counter = 0
        self.last_output_summary = None
        # Runtime attention parameters
        self.lens_alpha = 0.35
        self.lens_anchors = None  # e.g., ['love','logos']

        # Correctly locate config.json relative to this file's location
        # __file__ -> /app/Project_Elysia/cognition_pipeline.py
        # os.path.dirname(__file__) -> /app/Project_Elysia
        # os.path.join(..., '..', '..') -> /app/
        # final path -> /app/config.json
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: config.json not found at {config_path}. Using default settings.")
            config = {} # Use an empty config if the file is missing
        except Exception as e:
            print(f"ERROR: Failed to load or parse config.json from {config_path}: {e}")
            config = {}

        try:
            # This custom loader might have more complex logic, we keep it as a fallback
            loaded_config = load_config()
            if loaded_config:
                config = loaded_config
        except Exception:
            pass
        self.prefixes = config.get('prefixes', {})
        self.planning_prefix = self.prefixes.get('planning', 'plan and execute:')
        self.observation_prefix = self.prefixes.get('observation', 'The result of the tool execution is:')
        self.visual_learning_prefix = self.prefixes.get('visual_learning', '이것을 그려보자:')
        creative_impulse_config = config.get('creative_impulse', {})
        # Fix potentially garbled visual learning prefix
        try:
            if (not isinstance(self.visual_learning_prefix, str)) or ('\ufffd' in self.visual_learning_prefix) or ('?' in self.visual_learning_prefix):
                self.visual_learning_prefix = '이것을 그려보자:'
        except Exception:
            self.visual_learning_prefix = '이것을 그려보자:'
        self.arousal_threshold = creative_impulse_config.get('arousal_threshold', 0.7)
        self.echo_complexity_threshold = creative_impulse_config.get('echo_complexity_threshold', 5)

        # Initialize persistent conversation state and lens
        self.working_memory = WorkingMemory(size=10)
        self.topic_tracker = TopicTracker()
        self.lens = LensProfile()
        self.orchestrator = ResponseOrchestrator()
        # Identity-centered stance/voice
        try:
            self.self_model = SelfModel()
            self.stance_manager = StanceManager(self.self_model)
            self.self_voice = SelfVoiceFilter()
        except Exception:
            self.self_model = None
            self.stance_manager = None
            self.self_voice = None
        
        # --- helper to emit route telemetry arcs ---
        def _emit_route_arc(from_mod: str, to_mod: str, t0: float, outcome: str = 'ok', extra: Optional[Dict[str, Any]] = None):
            try:
                latency_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
                payload = {
                    'from_mod': from_mod,
                    'to_mod': to_mod,
                    'latency_ms': float(round(latency_ms, 3)),
                    'outcome': outcome,
                }
                if extra:
                    payload.update(extra)
                self.telemetry.emit('route.arc', payload)
            except Exception:
                pass
        # bind as instance method
        self._emit_route_arc = _emit_route_arc
        try:
            state = load_json('conversation_state.json')
            if isinstance(state, dict):
                self.working_memory.restore_from(state.get('working_memory', {}))
                self.topic_tracker.restore_from(state.get('topic_tracker', {}))
            self.lens.restore_from(load_json('lens_profile.json'))
        except Exception:
            pass

        # Offline-first defaults can be overridden by config
        try:
            llm_cfg = config.get('llm', {})
            self.api_available = bool(llm_cfg.get('use_external_api', False))
        except Exception:
            self.api_available = False

    def process_message(self, message: str, app=None, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], EmotionalState]:
        try:
            # --- Specialized Cortex Routing ---
            # Route to ArithmeticCortex if the message is a calculation request.
            if message.lower().startswith("calculate:") or message.lower().startswith("계산:"):
                try:
                    # The ArithmeticCortex's process method now handles everything.
                    result_text = self.arithmetic_cortex.process(message)
                    response = {"type": "text", "text": result_text}
                    return response, self.current_emotional_state
                except Exception as e:
                    pipeline_logger.warning(f"ArithmeticCortex failed for '{message}', falling through. Error: {e}")
                    # Fall through to the general pipeline if the cortex fails.

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
            # Update working memory and topic tracker per turn
            try:
                self.working_memory.add('user', message)
                self.topic_tracker.step()
                # Attention-guided pruning; compress pruned items into associative gists
                try:
                    topics = self.topic_tracker.snapshot()
                    stats = self.working_memory.prune(topics, keep=6, protect_recent=2, return_items=True)
                    try:
                        self.telemetry.emit('wm_pruned', {'kept': stats['kept'], 'pruned': stats['pruned']})
                    except Exception:
                        pass
                    pruned_items = stats.get('pruned_items', []) or []
                    if pruned_items:
                        # Build keywords union and a short gist
                        kw_counts = {}
                        for it in pruned_items:
                            for k in it.get('keywords', []):
                                kw_counts[k] = kw_counts.get(k, 0) + 1
                        top_keywords = [k for k, _ in sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
                        summary = ' '.join(top_keywords)
                        gid = self.associative.add_gist(top_keywords, summary, context={'topics': topics})
                        try:
                            self.telemetry.emit('associative_gist_saved', {'id': gid, 'keywords': top_keywords})
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            
            memory = Memory(timestamp=datetime.now().isoformat(), content=message, emotional_state=emotional_state, context=enriched_context)
            self.core_memory.add_experience(memory)
            self.journal_cortex.write_journal_entry(memory)

            # --- Self-Reflection Step (MetaCognition) ---
            try:
                # For now, we use the message itself as a concept. Future work can extract key nouns.
                concept_to_reflect = message.strip()
                if len(concept_to_reflect) > 3: # Reflect on reasonably complex ideas, allowing for core concepts like 'love'.
                    self.telemetry.emit('reflection_triggered', {'concept': concept_to_reflect})
                    reflection_result = self.meta_cognition_cortex.reflect_on_concept(
                        concept_id=concept_to_reflect,
                        context="User interaction"
                    )
                    self.telemetry.emit('reflection_completed', {
                        'concept': concept_to_reflect,
                        'reflection': reflection_result.get('reflection', '')[:200] # Log a snippet
                    })
            except Exception as e:
                pipeline_logger.error(f"MetaCognitionCortex failed during reflection: {e}")
                self.telemetry.emit('reflection_failed', {'concept': message.strip(), 'error': str(e)})
            # --- End Self-Reflection Step ---

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

            # Loop back assistant output into working/long-term memory
            try:
                out_text = None
                out_type = None
                out_image = None
                if isinstance(response, dict):
                    out_type = response.get('type')
                    out_text = response.get('text') if isinstance(response.get('text'), str) else None
                    out_image = response.get('image_path') or response.get('image')
                if not out_text:
                    out_text = str(response)[:500]

                # Identity voice filter and metric
                try:
                    stance = None
                    if self.stance_manager:
                        try:
                            stance = self.stance_manager.decide(message, context=enriched_context)
                        except Exception:
                            stance = {'name': 'companion'}
                    filtered = out_text
                    integrity = None
                    if self.self_voice and self.self_model:
                        try:
                            filtered, integrity = self.self_voice.filter_text(out_text, stance or {'name': 'companion'}, self.self_model)
                            if filtered:
                                out_text = filtered
                                if isinstance(response, dict):
                                    response['text'] = out_text
                        except Exception:
                            pass
                    # Emit identity integrity metric
                    try:
                        if integrity is None and self.self_model:
                            integrity = compute_identity_integrity(out_text, self.self_model.values)
                        if integrity is not None:
                            emit_identity_integrity(self.telemetry, integrity, (stance or {}).get('name'))
                        # Emit love/logos alignment (sacrificial love)
                        try:
                            ll = compute_love_logos_alignment(out_text)
                            emit_love_logos_alignment(self.telemetry, ll)
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
                # Working memory: mark as assistant output
                try:
                    self.working_memory.add('assistant', out_text)
                except Exception:
                    pass
                # Long-term memory: store as experience
                try:
                    # Simple value alignment (keyword-based)
                    val_kw = ['love', 'growth', 'creation', 'truth', 'care']
                    lower = out_text.lower()
                    matches = sum(1 for k in val_kw if k in lower)
                    align = min(1.0, matches / max(1, len(val_kw)))
                    mem = Memory(timestamp=datetime.now().isoformat(), content=out_text, emotional_state=emotional_state_out, context={'type': out_type, 'image_path': out_image}, value_alignment=align)
                    self.core_memory.add_experience(mem)
                except Exception:
                    pass
                # Telemetry: assistant output event
                try:
                    self.telemetry.emit('assistant_output', {'type': out_type or 'unknown', 'text_len': len(out_text or ''), 'has_image': bool(out_image)})
                    self.telemetry.emit('value_alignment', {'score': float(align)})
                except Exception:
                    pass
                # Keep a short UI summary
                self.last_output_summary = out_text[:120]
                try:
                    self._value_align_sum += float(align)
                    self._value_align_n += 1
                except Exception:
                    pass
                # Conversation quality (language/social markers)
                try:
                    text_low = (out_text or '').lower()
                    markers = {
                        'questions': text_low.count('?'),
                        'gratitude': sum(1 for k in ['thank', '고마', '감사'] if k in text_low),
                        'apology': sum(1 for k in ['sorry', '미안', '죄송'] if k in text_low),
                        'empathy': sum(1 for k in ['이해해', '공감', '들렸', '느껴'] if k in text_low),
                        'consent': sum(1 for k in ['괜찮', '좋을까요', '허락', '해도 되'] if k in text_low)
                    }
                    self.telemetry.emit('conversation_quality', markers)
                except Exception:
                    pass
            except Exception:
                pass

            # Persist state before returning
            try:
                save_json('conversation_state.json', {
                    'working_memory': self.working_memory.serialize(),
                    'topic_tracker': self.topic_tracker.serialize()
                })
                save_json('lens_profile.json', self.lens.serialize())
            except Exception:
                pass

            # Episode summary + snapshot every N turns
            try:
                self.turn_counter += 1
                if (self.turn_counter % 8) == 0:
                    top_topics = self.topic_tracker.snapshot()
                    try:
                        self.journal_cortex.write_episode_summary(top_topics, emotional_state_out)
                        self.telemetry.emit('episode_summary_saved', {'turn': self.turn_counter, 'top_topics': list(top_topics.keys())[:3]})
                        self._episodes_count += 1
                    except Exception:
                        pass
                    # Save a run snapshot manifest
                    try:
                        from tools.snapshot import snapshot as save_snapshot
                        save_snapshot()
                    except Exception:
                        pass
                    # Evaluate maturity and update guardian if needed
                    try:
                        # topic coherence with previous snapshot
                        cur = list(top_topics.keys())
                        if self._prev_topics:
                            inter = len(set(cur).intersection(set(self._prev_topics)))
                            union = max(1, len(set(cur).union(set(self._prev_topics))))
                            tc = inter / union
                        else:
                            tc = 0.5
                        self._prev_topics = cur
                        rr = min(1.0, self._episodes_count / max(1, self.turn_counter/8))
                        va = (self._value_align_sum / self._value_align_n) if self._value_align_n else 0.5
                        metrics = MaturityMetrics(echo_entropy=self._last_entropy, topic_coherence=tc, reflection_rate=rr, value_alignment=va)
                        score = self._maturity_eval.score(metrics)
                        self.telemetry.emit('maturity_evaluated', {'score': float(score), 'level': self._maturity_eval.level(score)})
                        guardian = SafetyGuardian()
                        level_name = self._maturity_eval.level(score)
                        target_level = MaturityLevel[level_name]
                        if guardian.current_maturity != target_level:
                            guardian.current_maturity = target_level
                            guardian.save_config()
                            self.telemetry.emit('maturity_updated', {'level': level_name})
                    except Exception:
                        pass
            except Exception:
                pass

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

    def _find_relevant_experiences(self, message: str, top_k=3) -> list[Memory]:
        """Finds relevant past experiences from core memory."""
        if not self.api_available:
            return []
        try:
            message_embedding = get_text_embedding(message)
            if message_embedding:
                return self.core_memory.find_relevant_experiences(message_embedding, top_k)
        except (APIKeyError, APIRequestError) as e:
            self.api_available = False
            pipeline_logger.warning(f"API unavailable during experience retrieval: {e}")
        except Exception as e:
            pipeline_logger.exception(f"Error finding relevant experiences for: {message}")
        return []

    def _enrich_context(self, context: Dict[str, Any], message: str) -> Dict[str, Any]:
        enriched = context.copy()
        echo = {}
        if self.api_available:
            # First, find relevant experiences from memory if the API is available.
            relevant_experiences = self._find_relevant_experiences(message)
            if relevant_experiences:
                enriched['relevant_experiences'] = [exp.content for exp in relevant_experiences]

            # Then, proceed with KG-based context enrichment.
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
                        node_w = self.lens.build_node_weights(self.topic_tracker.snapshot())
                        # Spatial gravity lens (3D distance to anchors)
                        spatial_w = self.lens.build_spatial_lens(self.kg_manager, centers=self.lens_anchors, alpha=self.lens_alpha)
                        if spatial_w:
                            for k, v in spatial_w.items():
                                if k in node_w:
                                    node_w[k] *= float(v)
                                else:
                                    node_w[k] = float(v)
                        gain = self.lens.emotion_gain(self.current_emotional_state.valence, self.current_emotional_state.arousal)
                        echo = self.wave_mechanics.spread_activation(best_match_node, lens_weights=node_w, emotion_gain=gain)
            except (APIKeyError, APIRequestError) as e:
                self.api_available = False
                pipeline_logger.warning(f"API unavailable during context enrichment: {e}")
            except Exception as e:
                pipeline_logger.exception(f"Error in _enrich_context during wave mechanics activation for message: {message}")

        if not self.api_available:
            lower_message = message.lower()
            all_node_ids = {node['id'] for node in self.kg_manager.kg.get('nodes', [])}
            stimulus_nodes = {node_id for node_id in all_node_ids if node_id in lower_message}
            node_w = self.lens.build_node_weights(self.topic_tracker.snapshot())
            spatial_w = self.lens.build_spatial_lens(self.kg_manager, centers=self.lens_anchors, alpha=self.lens_alpha)
            if spatial_w:
                for k, v in spatial_w.items():
                    if k in node_w:
                        node_w[k] *= float(v)
                    else:
                        node_w[k] = float(v)
            gain = self.lens.emotion_gain(self.current_emotional_state.valence, self.current_emotional_state.arousal)
            for start_node in stimulus_nodes:
                temp_echo = self.wave_mechanics.spread_activation(start_node, lens_weights=node_w, emotion_gain=gain)
                for node, energy in temp_echo.items():
                    echo[node] = echo.get(node, 0) + energy

        # Emit echo summary
        try:
            if echo:
                total = sum(echo.values())
                n = len(echo)
                probs = [v / total for v in echo.values()] if total > 0 else []
                import math
                entropy = -sum(p * math.log(p + 1e-12) for p in probs) if probs else 0.0
                topk = sorted(echo.items(), key=lambda x: x[1], reverse=True)[:5]
                self.telemetry.emit('echo_updated', {
                    'size': n,
                    'total_energy': float(total),
                    'entropy': float(entropy),
                    'top_nodes': [{'id': k, 'e': float(v)} for k, v in topk],
                })
        except Exception:
            pass

        # Drift the lens based on echo "stability" (simple complexity heuristic)
        try:
            stable = False
            if echo:
                # Option A: size-based threshold
                stable = len(echo) >= getattr(self, 'echo_complexity_threshold', 5)
            before = self.lens.serialize()
            self.lens.drift(arousal=self.current_emotional_state.arousal, stable=stable)
            after = self.lens.serialize()
            self.telemetry.emit('lens_drifted', {
                'stable': bool(stable),
                'before': before,
                'after': after,
                'arousal': float(self.current_emotional_state.arousal),
            })
        except Exception:
            pass

        try:
            self.topic_tracker.reinforce_from_echo(echo)
        except Exception:
            pass
        enriched['echo'] = echo
        # Echo spatial stats (center of mass & avg distance)
        try:
            if echo:
                import math
                total = sum(echo.values())
                if total > 0:
                    # positions map
                    pos = {n['id']: n.get('position', {'x': 0, 'y': 0, 'z': 0}) for n in self.kg_manager.kg.get('nodes', [])}
                    cx = sum(pos[k]['x'] * (echo[k]/total) for k in echo if k in pos)
                    cy = sum(pos[k]['y'] * (echo[k]/total) for k in echo if k in pos)
                    cz = sum(pos[k]['z'] * (echo[k]/total) for k in echo if k in pos)
                    center = {'x': cx, 'y': cy, 'z': cz}
                    dists = []
                    for k, e in echo.items():
                        if k in pos:
                            p = pos[k]
                            d = math.sqrt((p['x']-cx)**2 + (p['y']-cy)**2 + (p['z']-cz)**2)
                            dists.append(d)
                    avg_dist = sum(dists)/len(dists) if dists else 0.0
                    enriched['echo_spatial'] = {'center': center, 'avg_dist': avg_dist, 'count': len(echo)}
                    try:
                        self.telemetry.emit('echo_spatial_stats', {'center': center, 'avg_dist': float(avg_dist), 'count': len(echo)})
                    except Exception:
                        pass
        except Exception:
            pass
        enriched['identity'] = self.core_memory.get_identity()
        if 'speaker' in context:
            enriched['relationship'] = self.core_memory.get_relationship(context['speaker'])
        # Recall associative gists by echo-top keywords
        try:
            if echo:
                top_keys = [k for k, _ in sorted(echo.items(), key=lambda x: x[1], reverse=True)[:6]]
                gists = self.associative.search(top_keys, top_k=5)
                enriched['associative_gists'] = gists
        except Exception:
            pass
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
        Fallback response generation when external APIs are not available.
        This is the "Writing Room" for Elysia's inner voice.
        """
        facts = self.reasoner.deduce_facts(message)
        if facts:
            response_text = " ".join(facts)
        else:
            response_text = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."

        return {"type": "text", "text": response_text}, emotional_state

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
                _t0 = time.perf_counter()
                plan = self.planning_cortex.develop_plan(goal)
                try:
                    self._emit_route_arc('cognition_pipeline', 'planning_cortex', _t0, outcome='ok' if plan else 'empty', extra={'goal_len': len(goal)})
                except Exception:
                    pass
                if plan:
                    response_text = "I have developed the following plan:\n" + "".join(f"{i+1}. {a['tool_name']}({a['parameters']})\n" for i, a in enumerate(plan))
                    return {"type": "text", "text": response_text}, emotional_state
                else:
                    return {"type": "text", "text": "I was unable to develop a plan for that goal."}, emotional_state
            except Exception as e:
                pipeline_logger.exception(f"Error in planning_cortex for goal: {goal}")
                return {"type": "text", "text": "An error occurred during planning."}, emotional_state

        if app and app.cancel_requested:
            return None, None

        try:
            _t0 = time.perf_counter()
            action_decision = self.action_cortex.decide_action(message, app=app)
            try:
                self._emit_route_arc('cognition_pipeline', 'action_cortex', _t0, outcome='ok' if action_decision else 'empty')
            except Exception:
                pass
            if action_decision:
                return self.tool_executor.prepare_tool_call(action_decision), emotional_state
        except Exception as e:
            pipeline_logger.exception(f"Error in action_cortex for message: {message}")
            pass

        if '?' in message or any(q in message for q in ['what', 'who', 'where', 'when', 'why', 'how']):
             # Check if the question can be answered by the logical reasoner
            _t0 = time.perf_counter()
            facts = self.reasoner.deduce_facts(message)
            try:
                self._emit_route_arc('cognition_pipeline', 'logical_reasoner', _t0, outcome='ok' if facts else 'empty')
            except Exception:
                pass
            if facts:
                return {"type": "text", "text": " ".join(facts)}, emotional_state

            # If not, try the inquisitive mind (only if external API is available)
            if self.api_available:
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
