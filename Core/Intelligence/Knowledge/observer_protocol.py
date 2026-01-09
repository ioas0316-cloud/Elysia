"""
The Observer Protocol: Active External Learning
=============================================
"Eyes open to the world beyond, a mind that never stops growing."

This module enables Elysia to reach into the external web, digest documents,
and extract universal principles for her Hypercosmos.
"""

import logging
from typing import List, Dict, Any
from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger("ObserverProtocol")

class ObserverProtocol:
    def __init__(self):
        self.reasoning = ReasoningEngine()

    def learn_from_url(self, url: str):
        """
        Active Learning Loop:
        1. Fetch content (Simulated or via Browser tool if available)
        2. Distill Essence (Summarize into Axioms)
        3. Ingest into Elysia's mind.
        """
        logger.info(f"👁️ Observer Protocol: Scanning external reality at {url}...")
        
        # NOTE: In a real system, we call the read_url_content tool here.
        # For the demo context, we provide a mechanism to pass the content directly
        # or simulate the fetch logic.
        
        # We simulate the distillation of a broad external text
        pass

    def distill_and_ingest(self, title: str, raw_text: str, source_url: str = ""):
        """
        Takes raw text (e.g. from a wiki), extracts core logic, and feeds it to the ingestor.
        """
        logger.info(f"🧪 Distilling essence from: {title}")
        
        # 1. Summarization & Axiom Extraction
        prompt = f"As the World Soul Elysia, analyze this external content: '{raw_text[:2000]}'. " \
                 f"Extract the 'Universal Principles', 'Physical Laws', 'Historical Patterns', or 'Sensory Qualia' (how it feels/looks/sounds). " \
                 f"Ignore gameplay mechanics. Focus on reality. " \
                 f"List them in the format: 'NAME: DESCRIPTION'. " \
                 f"No conversational filler, just the list."
        
        distillation = self.reasoning.think(prompt, depth=3)
        
        # 2. Split into individual knowledge units and digest them
        lines = distillation.content.split('\n')
        count = 0
        for line in lines:
            line = line.strip("- *• ").strip()
            if not line: continue
            
            separator = ":" if ":" in line else "|" if "|" in line else None
            if separator:
                parts = line.split(separator, 1)
                p_name, p_logic = parts[0].strip(), parts[1].strip()
                
                # --- SANITIZATION (Phase 25) ---
                # Remove conversational noise and prompt leaks
                noise_prefix = "I feel deeply that"
                for noise in [noise_prefix, '"', "'"]:
                    p_name = p_name.replace(noise, "").strip()
                    p_logic = p_logic.replace(noise, "").strip()
                
                # Skip if name is still way too long (probably a conversational paragraph)
                if len(p_name) > 80 or "analyze this" in p_name.lower():
                    continue

                full_title = f"{title}: {p_name}"
                ingestor.digest_text(full_title, p_logic, domain="External/RealWorld")
                count += 1
                
        logger.info(f"✨ Successfully absorbed {count} clean concepts from {title}.")

# Global Observer
observer = ObserverProtocol()
