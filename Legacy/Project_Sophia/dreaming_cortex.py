
import logging
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from .spiderweb import Spiderweb
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.gemini_api import generate_text

if TYPE_CHECKING:
    from Project_Sophia.meta_awareness import MetaAwareness, ThoughtType
    from Project_Sophia.autonomous_dreamer import AutonomousDreamer
    from Project_Sophia.paradox_resolver import ParadoxResolver

class DreamingCortex:
    """
    The DreamingCortex runs during 'idle' times (or explicitly triggered) to consolidate memories.
    It takes recent experiences from CoreMemory and weaves them into the Spiderweb.
    
    Phase 3.5 Enhancement: Uses LLM (Gemini) for semantic concept extraction instead of naive word splitting.
    Phase 6 Enhancement: Integrated with meta-consciousness for self-observation and autonomous guidance.
    """

    def __init__(
        self,
        core_memory: CoreMemory,
        spiderweb: Spiderweb,
        meta_awareness: Optional['MetaAwareness'] = None,
        autonomous_dreamer: Optional['AutonomousDreamer'] = None,
        paradox_resolver: Optional['ParadoxResolver'] = None,
        logger: logging.Logger = None,
        use_llm: bool = True
    ):
        self.core_memory = core_memory
        self.spiderweb = spiderweb
        self.meta_awareness = meta_awareness
        self.autonomous_dreamer = autonomous_dreamer
        self.paradox_resolver = paradox_resolver
        self.logger = logger or logging.getLogger("DreamingCortex")
        self.use_llm = use_llm
        
        if meta_awareness:
            self.logger.info("🧠 Meta-awareness enabled for dreaming")
        if autonomous_dreamer:
            self.logger.info("🎯 Autonomous guidance enabled for dreaming")

    def _extract_concepts_llm(self, experience_content: str) -> Optional[Dict[str, Any]]:
        """
        Uses LLM to extract semantic concepts and relations from experience content.
        
        Returns:
            {
                "concepts": ["concept1", "concept2", ...],
                "relations": [
                    {"source": "concept1", "target": "concept2", "type": "causes/enables/is_a/related_to", "weight": 0.0-1.0}
                ]
            }
        """
        try:
            prompt = f"""Extract key concepts and their relationships from this experience:

Experience: "{experience_content}"

Return ONLY a JSON object (no markdown formatting) with this structure:
{{
    "concepts": ["concept1", "concept2"],
    "relations": [
        {{"source": "concept1", "target": "concept2", "type": "causes", "weight": 0.9}}
    ]
}}

Relation types: causes, enables, is_a, related_to, requires, prevents
Weights: 0.0 (weak) to 1.0 (strong)

Keep concepts short (1-2 words). Extract 3-7 concepts maximum. Focus on meaningful relationships."""

            response = generate_text(prompt)
            
            # Clean up potential markdown formatting
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            # Validate structure
            if "concepts" not in data or "relations" not in data:
                self.logger.warning(f"LLM response missing required fields: {data}")
                return None
                
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}. Response: {response[:200]}")
            return None
        except Exception as e:
            self.logger.error(f"LLM concept extraction failed: {e}")
            return None

    def _extract_concepts_naive(self, experience_content: str) -> Dict[str, Any]:
        """
        Fallback: Naive word-based concept extraction (original implementation).
        """
        words = [w.lower() for w in experience_content.split() if len(w) > 3]
        
        return {
            "concepts": list(set(words)),  # Deduplicate
            "relations": []  # No relations in naive mode
        }

    def dream(self):
        """
        The main dreaming process.
        1. Fetch unprocessed experiences from CoreMemory.
        2. Analyze them for key concepts and relations using LLM.
        3. Integrate them into the Spiderweb.
        4. Mark experiences as processed.
        """
        self.logger.info("Entering dream state...")
        
        # 🧠 Meta-awareness: Observe dreaming start
        if self.meta_awareness:
            try:
                from Project_Sophia.meta_awareness import ThoughtType
                self.meta_awareness.observe(
                    thought_type=ThoughtType.DREAMING,
                    input_state={"trigger": "dream_cycle"},
                    output_state={"status": "starting"},
                    transformation="Entering dream state for memory consolidation",
                    confidence=0.9
                )
            except Exception as e:
                self.logger.warning(f"Meta-awareness observation failed: {e}")
        
        unprocessed_experiences = self.core_memory.get_unprocessed_experiences()
        if not unprocessed_experiences:
            self.logger.info("No new experiences to dream about.")
            return

        self.logger.info(f"Dreaming about {len(unprocessed_experiences)} new experiences (LLM mode: {self.use_llm}).")
        
        processed_ids = []
        
        for experience in unprocessed_experiences:
            content = experience.content
            timestamp = experience.timestamp
            
            # Create an event node for this specific memory
            event_id = f"event_{timestamp}"
            self.spiderweb.add_node(event_id, type="event", metadata={"content": content, "timestamp": timestamp})
            
            # Extract concepts using LLM or fallback to naive method
            if self.use_llm:
                extracted = self._extract_concepts_llm(content)
                if extracted is None:
                    self.logger.warning(f"LLM extraction failed for '{content[:50]}...', falling back to naive mode.")
                    extracted = self._extract_concepts_naive(content)
            else:
                extracted = self._extract_concepts_naive(content)
            
            # Add concepts to Spiderweb
            for concept in extracted.get("concepts", []):
                concept_id = concept.lower().replace(" ", "_")
                self.spiderweb.add_node(concept_id, type="concept")
                self.spiderweb.add_link(event_id, concept_id, relation="contains_concept", weight=0.8)
                
                # Temporal association: link events that share concepts
                context = self.spiderweb.get_context(concept_id)
                for neighbor in context:
                    if neighbor["node"].startswith("event_") and neighbor["node"] != event_id:
                        self.spiderweb.add_link(neighbor["node"], event_id, relation="associative_link", weight=0.5)
            
            # Add explicit relations from LLM
            for relation in extracted.get("relations", []):
                source_id = relation.get("source", "").lower().replace(" ", "_")
                target_id = relation.get("target", "").lower().replace(" ", "_")
                rel_type = relation.get("type", "related_to")
                weight = relation.get("weight", 0.7)
                
                # Ensure both nodes exist
                self.spiderweb.add_node(source_id, type="concept")
                self.spiderweb.add_node(target_id, type="concept")
                
                # Add the causal/semantic link
                self.spiderweb.add_link(source_id, target_id, relation=rel_type, weight=weight)
                self.logger.debug(f"LLM extracted relation: {source_id} -[{rel_type}]-> {target_id} (w={weight})")

            processed_ids.append(timestamp)
        
        # 🌀 Detect paradoxes in newly consolidated memories
        if self.paradox_resolver:
            try:
                contradictions = self.paradox_resolver.detect_contradictions(min_opposition=0.7)
                if contradictions:
                    self.logger.info(f"Detected {len(contradictions)} contradictions during consolidation")
                    # Resolve first contradiction
                    if contradictions:
                        c1, c2, _ = contradictions[0]
                        paradox = self.paradox_resolver.create_superposition(c1, c2)
                        synthesis = self.paradox_resolver.resolve_paradox(paradox)
                        if synthesis:
                            self.logger.info(f"Resolved paradox via synthesis: {synthesis}")
            except Exception as e:
                self.logger.warning(f"Paradox resolution failed: {e}")

        # Mark as processed in CoreMemory
        self.core_memory.mark_experiences_as_processed(processed_ids)
        
        # 🧠 Meta-awareness: Observe dreaming completion
        if self.meta_awareness:
            try:
                from Project_Sophia.meta_awareness import ThoughtType
                self.meta_awareness.observe(
                    thought_type=ThoughtType.DREAMING,
                    input_state={"num_experiences": len(unprocessed_experiences)},
                    output_state={
                        "num_processed": len(processed_ids),
                        "spiderweb_nodes": self.spiderweb.graph.number_of_nodes(),
                        "spiderweb_edges": self.spiderweb.graph.number_of_edges()
                    },
                    transformation=f"Consolidated {len(processed_ids)} experiences into knowledge graph",
                    confidence=0.85
                )
            except Exception as e:
                self.logger.warning(f"Meta-awareness observation failed: {e}")
        
        self.logger.info("Dreaming complete. Memories consolidated.")

    def consolidate(self):
        """
        Refines the Spiderweb by merging similar nodes or pruning weak links.
        (Future enhancement)
        """
        pass
