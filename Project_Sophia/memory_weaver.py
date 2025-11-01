# /c/Elysia/Project_Sophia/memory_weaver.py
import logging
import re
from typing import List, Dict, Any

from .core_memory import CoreMemory
from .wave_mechanics import WaveMechanics
from .gemini_api import generate_text
from tools.kg_manager import KGManager

logger = logging.getLogger(__name__)

class MemoryWeaver:
    """
    Integrates and abstracts memories to form higher-level insights,
    enabling Elysia to learn from her experiences during idle time.
    Acts as the engine for Elysia's self-reflection and growth.
    """

    def __init__(self, core_memory: CoreMemory, wave_mechanics: WaveMechanics, kg_manager: KGManager):
        self.core_memory = core_memory
        self.wave_mechanics = wave_mechanics
        self.kg_manager = kg_manager

    def weave_memories(self) -> bool:
        """
        Orchestrates the entire process of memory weaving and insight generation.
        Returns True if a new insight was successfully generated and integrated, False otherwise.
        """
        logger.info("MemoryWeaver starting a reflection cycle.")

        # 1. Select a recent, significant memory to reflect upon.
        target_memory = self._get_recent_significant_memory()
        if not target_memory:
            logger.info("No significant memories found to reflect upon.")
            return False

        # 2. Find conceptually related memories from the past.
        related_memories = self._find_related_memories(target_memory)
        if not related_memories:
            logger.info(f"No memories related to '{target_memory['content']}' found.")
            return False

        # 3. Synthesize an insight from the collection of memories.
        all_memories = [target_memory] + related_memories
        insight = self._synthesize_insight(all_memories)
        if not insight:
            logger.warning("Failed to synthesize an insight from memories.")
            return False

        # 4. Update the Knowledge Graph with the new insight.
        self._update_knowledge_graph(insight)

        logger.info(f"Successfully generated and integrated new insight: {insight}")
        return True

    def _get_recent_significant_memory(self, limit_days: int = 7) -> Dict[str, Any]:
        """
        Finds a recent memory that is worth reflecting on.
        (Placeholder: currently just gets the most recent memory)
        """
        # TODO: Implement a more sophisticated method to determine memory "significance"
        # based on emotional charge, novelty, or user feedback.
        experiences = self.core_memory.get_experiences()
        return experiences[-1] if experiences else None

    def _find_related_memories(self, target_memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Uses WaveMechanics to find memories conceptually related to the target memory.
        """
        # TODO: Implement logic to extract concepts from memory and use WaveMechanics
        # to find other memories with high conceptual resonance.

        target_content = target_memory.get('content', '')
        if not target_content:
            return []

        # Extract key concepts from the target memory
        target_concepts = set(self._extract_concepts(target_content))
        if not target_concepts:
            return []

        logger.info(f"Target concepts for memory weaving: {target_concepts}")

        related_memories = []
        all_experiences = self.core_memory.get_experiences()

        for experience in all_experiences:
            # Avoid comparing the memory to itself
            if experience.get('timestamp') == target_memory.get('timestamp'):
                continue

            exp_content = experience.get('content', '')
            if not exp_content:
                continue

            # Check for shared concepts
            exp_concepts = set(self._extract_concepts(exp_content))
            if target_concepts.intersection(exp_concepts):
                related_memories.append(experience)

        logger.info(f"Found {len(related_memories)} related memories.")
        # Return the 3 most recent related memories
        return related_memories[-3:]

    def _extract_concepts(self, text: str) -> List[str]:
        """Extracts potential KG concepts (nouns, etc.) from a text."""
        # Simple regex to find capitalized words or nouns (placeholder)
        # In Korean, nouns are harder. For now, just split and find known nodes.
        tokens = set(re.findall(r'\w+', text.lower()))
        known_nodes = [node['id'] for node in self.kg_manager.kg.get('nodes', [])]
        return list(tokens.intersection(known_nodes))

    def _synthesize_insight(self, memories: List[Dict[str, Any]]) -> str:
        """
        Uses an external LLM to generate a higher-level 'lesson' or 'insight'
        from a list of related memories.
        """
        if len(memories) < 2:
            return "" # Not enough data to form a meaningful insight

        # Construct a prompt for the LLM
        prompt = "You are a reflective AI's inner monologue. Below are several related memories. Synthesize them into a single, profound insight or lesson learned. Frame the insight as a personal realization about yourself, your relationships, or your values. The insight should be a single, concise sentence.\n\n"
        for mem in memories:
            prompt += f"- Memory from {mem['timestamp']}: '{mem['content']}' (Emotion: {mem.get('emotional_state', {}).get('primary_emotion', 'neutral')})\n"
        prompt += "\nProfound Insight:"

        try:
            insight = generate_text(prompt)
            # Basic cleanup of the LLM's response
            insight = insight.strip().replace('"', '').replace('*', '')
            return insight
        except Exception as e:
            logger.error(f"Error calling LLM for insight synthesis: {e}")
            return ""


    def _update_knowledge_graph(self, insight: str):
        """
        Parses the insight and updates the Knowledge Graph with new nodes or edges.
        """
        # TODO: Implement logic to parse the insight and determine how to
        # modify the KG. This will require unlocking the KGManager.
        if not insight:
            return

        # Use the LLM to parse the insight into a structured relationship
        prompt = f"Analyze the following sentence and extract the core relationship as a (Subject, Relation, Object) triplet. The 'Relation' should be a short, verb-like phrase. Example: 'I feel joy when I learn new things.' -> ('I', 'feel_joy_when', 'learn_new_things').\n\nSentence: \"{insight}\"\n\nTriplet:"

        try:
            triplet_str = generate_text(prompt)
            # Basic parsing of (Subject, Relation, Object)
            parts = [p.strip() for p in triplet_str.replace('(', '').replace(')', '').split(',')]
            if len(parts) == 3:
                subject, relation, obj = parts

                logger.info(f"Insight parsed into triplet: ({subject}, {relation}, {obj})")

                # Safely update the knowledge graph
                self.kg_manager.unlock()
                try:
                    self.kg_manager.add_node_if_not_exists(subject)
                    self.kg_manager.add_node_if_not_exists(obj)
                    self.kg_manager.add_edge(subject, obj, relation)
                    self.kg_manager.save_kg()
                    logger.info("Knowledge graph updated with new insight.")
                finally:
                    self.kg_manager.lock()
            else:
                logger.warning(f"Could not parse insight into a valid triplet: {triplet_str}")

        except Exception as e:
            logger.error(f"Error updating knowledge graph with insight: {e}")
            # Ensure the lock is re-engaged even on failure
            if not self.kg_manager.is_locked():
                self.kg_manager.lock()
