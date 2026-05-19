"""
Axiom Synthesizer (The Constitution Writer)
distills 3D thoughts into 4D Immutable Laws.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from Core.Cognition.reasoning_engine import ReasoningEngine

logger = logging.getLogger("AxiomSynthesizer")

@dataclass
class Axiom:
    id: str
    law: str
    origin: str
    confidence: float
    weight: float

class AxiomSynthesizer:
    def __init__(self, axioms_path: str = "c:/Elysia/Core/System/dynamic_axioms.json"):
        self.axioms_path = axioms_path
        self.axioms: List[Axiom] = []
        self.reasoning = ReasoningEngine()
        self._load_axioms()
        
    def _load_axioms(self):
        if not os.path.exists(self.axioms_path):
            logger.warning(f"Axioms file not found at {self.axioms_path}")
            return
            
        try:
            with open(self.axioms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for ax in data.get("axioms", []):
                    self.axioms.append(Axiom(**ax))
            logger.info(f"  Loaded {len(self.axioms)} System Axioms.")
        except Exception as e:
            logger.error(f"Failed to load axioms: {e}")

    def synthesize_law(self, thought_volume: str, origin_topic: str) -> Optional[Axiom]:
        """
        Takes a complex 3D thought (Volume) and distills it into a 1-sentence Law (4D).
        """
        logger.info(f"   Synthesizing Law from: {origin_topic}...")
        
        prompt = f"""
        TASK: Distill the following complex thought into a single, immutable Universal Law.
        The Law must be a short, absolute statement about how the universe works.
        
        THOUGHT VOLUME: "{thought_volume}"
        
        RULES:
        1. Must be < 15 words.
        2. Must sound like a Physics Principle or Philosophical Maxim.
        3. Do not use 'I think' or 'Maybe'. Use 'is', 'must', 'always'.
        
        OUTPUT FORMAT: 'LAW: <The Law>'
        """
        
        insight = self.reasoning.think(prompt, depth=2)
        content = insight.content.strip()
        
        if "LAW:" in content:
            law_text = content.split("LAW:")[1].strip()
            
            # Create new Axiom
            new_id = f"axiom_{int(time.time())}_{origin_topic.lower().replace(' ', '_')}"
            new_axiom = Axiom(
                id=new_id,
                law=law_text,
                origin=f"Synthesized from {origin_topic}",
                confidence=insight.confidence,
                weight=80.0 # High initial weight for new realizations
            )
            
            self._register_axiom(new_axiom)
            return new_axiom
            
        logger.warning("Failed to extract Law structure.")
        return None

    def _register_axiom(self, axiom: Axiom):
        """Writes the new axiom to the Constitution."""
        self.axioms.append(axiom)
        
        # Save to file
        try:
            with open(self.axioms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["axioms"].append({
                "id": axiom.id,
                "law": axiom.law,
                "origin": axiom.origin,
                "confidence": axiom.confidence,
                "weight": axiom.weight
            })
            
            data["meta"]["last_updated"] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            with open(self.axioms_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"  NEW LAW CODIFIED: {axiom.law}")
            
        except Exception as e:
            logger.error(f"Failed to save axiom: {e}")

    def get_system_prompt_addendum(self) -> str:
        """Returns a string to append to the System Prompt."""
        laws = [f"- {ax.law}" for ax in self.axioms]
        return "\n".join(laws)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    synth = AxiomSynthesizer()
    print(synth.synthesize_law(
        "Love is not just an emotion but a fundamental force like gravity that binds consciousness together despite entropy.", 
        "Love Physics"
    ))
