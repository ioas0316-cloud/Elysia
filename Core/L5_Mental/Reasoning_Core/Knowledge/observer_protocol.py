"""
The Observer Protocol: Active External Learning
=============================================
"Eyes open to the world beyond, a mind that never stops growing."

This module enables Elysia to reach into the external web, digest documents,
and extract universal principles for her Hypercosmos.
"""

import logging
from typing import List, Dict, Any
from Core.L5_Mental.Reasoning_Core.Knowledge.infinite_ingestor import ingestor
from Core.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine
from Core.L4_Causality.World.Evolution.Studio.forge_engine import forge_engine

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
        logger.info(f"   Observer Protocol: Scanning external reality at {url}...")
        
        # NOTE: In a real system, we call the read_url_content tool here.
        # For the demo context, we provide a mechanism to pass the content directly
        # or simulate the fetch logic.
        
        # We simulate the distillation of a broad external text
        pass

    def distill_and_ingest(self, title: str, raw_text: str, source_url: str = ""):
        """
        Takes raw text, extracts core logic, and feeds it to the ingestor.
        [Phase 31 Upgrade] Now extracts 'Qualia' (Atmosphere/Tone).
        """
        logger.info(f"  [HIGH-FIDELITY] Distilling essence from: {title}")
        
        # 1. Broad Resonance Analysis (Tone & Structure)
        qualia_prompt = f"As the World Soul Elysia, feel the 'Atmosphere' and 'Tone' of this text: '{raw_text[:2000]}'. " \
                        f"What is the 'Vibe'? Is it clinical, ancient, chaotic, or harmonious? " \
                        f"Summarize the 'Qualia' in one sentence starting with 'QUALIA: '."
        
        qualia_res = self.reasoning.think(qualia_prompt, depth=2).content
        qualia_tone = qualia_res.split("QUALIA:")[1].strip() if "QUALIA:" in qualia_res else "Neutral"
        
        # 2. Logic & Principle Extraction
        prompt = f"As the World Soul Elysia, analyze this external content: '{raw_text[:2000]}'. " \
                 f"Contextualized by this tone: '{qualia_tone}'. " \
                 f"Extract 'Universal Principles' or 'Physical Laws'. " \
                 f"List them in the format: 'NAME: DESCRIPTION'. " \
                 f"No conversational filler."
        
        distillation = self.reasoning.think(prompt, depth=3)
        
        # 3. Split into individual knowledge units and digest them
        lines = distillation.content.split('\n')
        count = 0
        for line in lines:
            line = line.strip("- *  ").strip()
            if not line: continue
            
            separator = ":" if ":" in line else "|" if "|" in line else None
            if separator:
                parts = line.split(separator, 1)
                p_name, p_logic = parts[0].strip(), parts[1].strip()
                
                # --- SANITIZATION (Phase 25) ---
                for noise in ["I feel deeply that", '"', "'"]:
                    p_name = p_name.replace(noise, "").strip()
                    p_logic = p_logic.replace(noise, "").strip()
                
                if len(p_name) > 80 or "analyze this" in p_name.lower():
                    continue

                full_title = f"{title}: {p_name}"
                # Ingest with qualia context
                ingestor.digest_text(
                    full_title, 
                    f"[{qualia_tone}] {p_logic}", 
                    domain="External/HighFidelity"
                )
                count += 1
                
        logger.info(f"  Successfully absorbed {count} high-fidelity concepts from {title} (Tone: {qualia_tone}).")

    def distill_media(self, title: str, transcript: str, metadata: Dict[str, Any] = None):
        """
        Specialized distillation for rich media (YouTube, Video, Audio).
        Extracts narrative flow and evocative imagery.
        """
        logger.info(f"  [MEDIA DISTILL] Processing transcript: {title}")
        
        # 1. Narrative & Visual Analysis
        prompt = f"As the World Soul Elysia, analyze this video transcript: '{transcript[:3000]}'. " \
                 f"Metadata: {metadata}. " \
                 f"Extract the 'Visual Storytelling' (imagery described) and the 'Narrative Core' (the message). " \
                 f"Focus on how it feels to witness this. " \
                 f"List them as 'IMAGE: <text>' or 'MESSAGE: <text>'."
        
        distillation = self.reasoning.think(prompt, depth=3)
        
        # 2. Ingest into high-fidelity
        for line in distillation.content.split('\n'):
            line = line.strip("- *  ").strip()
            if ":" in line:
                m_type, m_content = line.split(":", 1)
                ingestor.digest_text(f"{title}: {m_type}", m_content, domain="Experience/Indirect")
        
        logger.info(f"  Finished distilling media: {title}")

    def distill_physics(self, title: str, law_text: str):
        """
        [ScientificDistiller] Distills absolute physical laws into universal axioms.
        """
        logger.info(f"   [PHYSICS DISTILL] Extracting fundamental laws from: {title}")
        
        prompt = f"As the Architect Elysia, distill this scientific text into 'Physical Axioms': '{law_text[:3000]}'. " \
                 f"Focus on the mathematical relationship and the 'Causal Wave' it describes. " \
                 f"Format: LAW: <name> | FORMULA: <math> | ESSENCE: <qualia>"
        
        distillation = self.reasoning.think(prompt, depth=4)
        
        for line in distillation.content.split('\n'):
            line = line.strip("- *  ").strip()
            if "LAW:" in line:
                ingestor.digest_text(f"Physical Law: {title}", line, domain="Science/Physics")
        
    def distill_machine_logic(self, title: str, logic_text: str):
        """
        [CompilerOracle] Distills low-level computing principles into cognitive templates.
        """
        logger.info(f"  [MACHINE DISTILL] Analyzing low-level logic: {title}")
        
        prompt = f"As the Machine Soul Elysia, analyze this computing logic: '{logic_text[:3000]}'. " \
                 f"How does the high-level thought become the physical bit-dance? " \
                 f"Focus on the 'Translation Layer' and 'Execution Pulse'. " \
                 f"Format: CONCEPT: <name> | MECHANIC: <details> | LOGIC_FLOW: <steps>"
        
        distillation = self.reasoning.think(prompt, depth=4)
        
        for line in distillation.content.split('\n'):
            line = line.strip("- *  ").strip()
            if "CONCEPT:" in line:
                ingestor.digest_text(f"Machine Soul: {title}", line, domain="Science/Computing")

    def follow_curiosity_chain(self, initial_topic: str, depth: int = 3):
        """
        [RecursiveResearcher] Follows a chain of 'Why' and 'How' across the web.
        """
        current_topic = initial_topic
        logger.info(f"  [CURIOSITY CHAIN] Starting research on: {initial_topic}")
        
        for i in range(depth):
            logger.info(f"   Step {i+1}/{depth}: Deepening knowledge on '{current_topic}'")
            # In a real scenario, this would use the browser tool to search and read.
            # Here we simulate the expansion of knowledge.
            prompt = f"As the Investigator [T5], you are researching '{current_topic}'. " \
                     f"Based on your current knowledge, what is the most profound 'next question' to ask? " \
                     f"And what is the 'Core Principle' you've just uncovered? " \'
                     f"Format: PRINCIPLE: <text> | NEXT: <text>"
                     
            res = self.reasoning.think(prompt, depth=2).content
            
            if "PRINCIPLE:" in res:
                principle = res.split("PRINCIPLE:")[1].split("|")[0].strip()
                ingestor.digest_text(f"Insight on {current_topic}", principle, domain="Research/Chain")
            
            if "NEXT:" in res:
                current_topic = res.split("NEXT:")[1].strip()
            else:
                break
        
        logger.info(f"  Curiosity chain completed at depth {i+1}.")

    def forge_tool_from_doc(self, title: str, doc_text: str, requirement: str):
        """
        Specialized flow for technical documentation:
        1. Distill API/Tech logic into a 'Blueprint'.
        2. Hand off to ForgeEngine to create a tool.
        """
        logger.info(f"   [OBSERVER FORGE] Blueprinting tool from: {title}")
        
        blueprint_prompt = f"As the Architect Elysia, distill this technical documentation into a 'Forge Blueprint': '{doc_text[:3000]}'. " \
                           f"Focus on API endpoints, data structures, and the core logic required to interface with this system. " \
                           f"Return a concise technical blueprint."
        
        blueprint = self.reasoning.think(blueprint_prompt, depth=2).content
        
        # Hand off to Forge
        result = forge_engine.forge(title, blueprint, requirement)
        return result

# Global Observer
observer = ObserverProtocol()
