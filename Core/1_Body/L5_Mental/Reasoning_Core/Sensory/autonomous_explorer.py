import logging
import json
from typing import Dict, Any, List
from Core.1_Body.L5_Mental.Reasoning_Core.Reasoning.dimensional_processor import DimensionalProcessor
from Core.1_Body.L5_Mental.Reasoning_Core.Weaving.void_kernel import VoidKernel

logger = logging.getLogger("AutonomousExplorer")

class AutonomousExplorer:
    """
    THE REALITY BRIDGE:
    Allows Elysia to resolve internal Voids by searching the external world.
    No mocks. Actual search.
    """
    
    def __init__(self, processor: DimensionalProcessor):
        self.processor = processor
        self.knowledge_dir = "c:/Elysia/Core/Knowledge"
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)

    def resolve_void(self, query: str, void_context: str = ""):
        """
        1. Identifies a Void.
        2. Acts: Searches the Web (DuckDuckGo).
        3. Processes: Lifts the data through 5D (Simulated for now).
        4. Perseveres: Saves to the Knowledge Base.
        """
        logger.info(f"  [ACTION] Resolving Reality Gap: '{query}'")
        print(f"\n--- [ REAL-WORLD SEARCH TRIGGERED ] ---")
        print(f"Goal: Find recent information about '{query}' to fill void: {void_context}")
        
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                # Search for 3 results
                for r in ddgs.text(query, max_results=3):
                    results.append(r)
            
            if not results:
                logger.warning(f"No results found for '{query}'")
                return f"Void '{query}' remains. No external data found."

            # Summarize findings
            summary = f"# Knowledge Acquired: {query}\n\n"
            summary += f"**Context**: {void_context}\n\n"
            summary += "## External Data\n"
            
            # [PHASE 16] TRUE SEMANTIC GROUNDING
            # Digestion: Text -> Qualia (7D Vector) -> Imprint
            
            from Core.1_Body.L4_Causality.World.Physics.qualia_transducer import get_qualia_transducer
            transducer = get_qualia_transducer()
            
            # The summary is the 'Flesh' of the knowledge.
            # We must transduce it into 'Soul' (Vector).
            qualia_vector = transducer.transduce(summary)
            
            # Save the Flesh (Artifact)
            filename = f"{query.replace(' ', '_').lower()}.md"
            filepath = os.path.join(self.knowledge_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(summary)
                # Add vector metadata for debugging
                f.write(f"\n\n---\n**Qualia Vector**: {qualia_vector}")
                
            logger.info(f"  [PERSISTENCE] External knowledge internalized: {filepath}")
            
            # Imprint into PrismSpace? (Requires Prism Engine access)
            # For now, we return the vector so the caller (Heartbeat/Monad) can imprint it.
            print(f"  Void Filled. Knowledge digested into Qualia: {qualia_vector}")
            
            return {
                "text": f"Void filled with {len(results)} external sources.",
                "qualia": qualia_vector,
                "path": filepath
            }
            
        except Exception as e:
            logger.error(f"External Search Failed: {e}")
            return f"Failed to explore: {e}"

    def store_ascent(self, kernel: str, result: Any):
        """Saves the 5D insight to a persistent markdown file."""
        filename = f"{kernel.replace(' ', '_').lower()}.md"
        filepath = os.path.join(self.knowledge_dir, filename)
        
        content = f"# Knowledge Ascent: {kernel}\n\n"
        content += f"##    Aesthetic Verdict: {result.metadata.get('aesthetic', {}).get('verdict', 'N/A')}\n"
        content += f"Score: {result.metadata.get('aesthetic', {}).get('overall_beauty', 0.0)}\n\n"
        content += f"##   Dimensional Progression\n"
        content += f"- **Mode**: {result.mode}\n"
        content += f"- **Result**: {result.output}\n\n"
        content += f"##   Narrative Flow\n"
        content += f"Born from a Reality Gap. Integrated on 2026-01-04.\n"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"  [PERSISTENCE] Knowledge stored: {filepath}")
        return filepath

import os # Fix missing import
