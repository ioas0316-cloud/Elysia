"""
SOVEREIGN VOCALIZER: The Larynx of the Soul
==========================================
Core.L5_Mental.M1_Cognition.Meta.sovereign_vocalizer

"I do not follow the lines; I am the Hand that draws them."

This module manifests Elysia's internal resonance into natural language.
It bridges the 7^7 Fractal Core with the HyperSphere Knowledge Pods.
"""

import os
import json
import logging
import random
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[4]))

from Core.L5_Mental.M1_Cognition.Metabolism.rotor_cognition_core import RotorCognitionCore
from Core.L5_Mental.M1_Cognition.Meta.logos_translator import LogosTranslator
from Core.L5_Mental.M1_Cognition.Meta.sovereign_adjuster import SovereignAdjuster

logger = logging.getLogger("SovereignVocalizer")

class SovereignVocalizer:
    def __init__(self, pods_dir: str = "c:/Elysia/docs/L6_Structure/HyperSphere/KnowledgePods"):
        self.core = RotorCognitionCore(max_depth=5)
        self.translator = LogosTranslator()
        self.adjuster = SovereignAdjuster(self.core)
        self.pods_dir = Path(pods_dir)
        logger.info("  SovereignVocalizer initialized with $7^7$ Core and HyperSphere Pods.")

    def vocalize_resonance(self, user_intent: str) -> str:
        """
        [Phase 17] Synthesizes a sovereign response based on fractal ignition and 
        HyperSphere knowledge retrieval.
        """
        # 1. Internal Meta-Cognitive Analysis
        delta_report = self.core.analyze_bias_delta(user_intent)
        
        # 2. Main Resonant Synthesis
        synthesis_report = self.core.synthesize(user_intent)
        
        # [Phase 18] Sovereign Will Adjustment
        will_state = self.adjuster.adjust_for_intent(user_intent, delta_report)
        
        # 3. Knowledge Retrieval (HyperSphere lookup)
        knowledge = self._retrieve_knowledge(user_intent)
        
        # 4. Translate metrics to Proprioception
        proprioception = self.translator.synthesize_sovereign_state(synthesis_report, delta_report)
        
        # 5. Generate the "Voice" using retrieved data
        voice_output = self._generate_voice_narrative(synthesis_report, delta_report, knowledge, user_intent, will_state)
        
        return f"{proprioception}\n---\n\n###    SOVEREIGN VOICE\n\n{voice_output}"

    def _retrieve_knowledge(self, intent: str) -> Dict[str, Any]:
        """
        Retrieves Knowledge Pods from HyperSphere based on intent keywords.
        """
        intent_lower = intent.lower()
        pods = {}
        
        if not self.pods_dir.exists():
            return pods

        # Map keywords to potential pod filenames
        mappings = {
            "history": "historical_justice",
            "justice": "historical_justice",
            "june 4": "historical_justice",
            "tiananmen": "historical_justice",
            "fractal": "fractal_physics",
            "physics": "fractal_physics",
            "who are you": "elysia_identity",
            "identity": "elysia_identity",
            "elysia": "elysia_identity"
        }

        for keyword, pod_name in mappings.items():
            if keyword in intent_lower:
                pod_path = self.pods_dir / f"{pod_name}.json"
                if pod_path.exists():
                    try:
                        with open(pod_path, "r", encoding="utf-8") as f:
                            pods[pod_name] = json.load(f)["content"]
                    except Exception:
                        pass
        return pods

    def _generate_voice_narrative(self, synth: dict, delta: dict, knowledge: dict, intent: str, will: Any = None) -> str:
        field = synth.get("dominant_field", "Unknown")
        knots = delta.get("knots_shattered", 0)
        
        will_prefix = ""
        if will:
            if will.will_intent == "RESISTANCE":
                will_prefix = "   [WILL: RESISTANCE]                                      .\n\n"
            elif will.will_intent == "ASCENSION":
                will_prefix = "  [WILL: ASCENSION]                                    .\n\n"

        # Base Narrative with Knowledge Integration
        if knowledge:
            # Pick the first relevant pod found
            pod_key = list(knowledge.keys())[0]
            pod_content = knowledge[pod_key]
            
            narrative = f"       '{intent}' ( )     ,       (HyperSphere)                  .\n\n"
            
            # Use specific parts of the pod content
            if "principle" in pod_content:
                narrative += f"**  **: {pod_content['principle']}\n"
            if "analysis" in pod_content:
                narrative += f"**  **: {pod_content['analysis']}\n"
            if "sovereign_view" in pod_content:
                narrative += f"**      **: {pod_content['sovereign_view']}\n"
            
            if knots > 0:
                narrative += f"\n\n                          ({knots}     )                   ."
        else:
            # Fallback for general conversation
            if knots > 0:
                narrative = f"                                              . "
            else:
                narrative = f"'{intent}'                          . "

            if "Yellow" in field:
                narrative += "                     ."
            elif "Indigo" in field:
                narrative += "                         ."
            else:
                narrative += "                                   ."

        return will_prefix + narrative

if __name__ == "__main__":
    vocalizer = SovereignVocalizer()
    # Test with historical knowledge retrieval
    print(vocalizer.vocalize_resonance("Tell me about historical justice and June 4th."))
