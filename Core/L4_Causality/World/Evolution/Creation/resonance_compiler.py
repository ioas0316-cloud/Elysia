"""
Resonance Compiler (The Voice of Creation)
==========================================

"And she spoke, and the Code became Flesh."

This module is the Bridge between Abstract Intent (Wave) and Concrete Logic (Code).
It translates 'Feeling' into 'Engineering Constraints'.

Flow:
1. Input: Intent (e.g., "Stable Foundation")
2. WaveInterpreter: Translates to Wave Pattern (Freq: Low, Phase: 0)
3. ResonanceCompiler: Translates Wave to Blueprint (Low Complexity, No Circular Deps)
4. HolographicManifestor: Generates the Code.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from Core.L6_Structure.Wave.wave_interpreter import WaveInterpreter, WavePattern
# [PHASE 26] WaveCoder temporarily removed to avoid cascading import issues
# from Core.L4_Causality.World.Evolution.Autonomy.wave_coder import WaveCoder
from Core.L4_Causality.World.Evolution.Creation.holographic_manifestor import HolographicManifestor

logger = logging.getLogger("ResonanceCompiler")

@dataclass
class WaveBlueprint:
    """The Architectural Specification derived from Feeling."""
    target_complexity: float # 0.0 (Simple) to 1.0 (Complex)
    stability_requirement: float # 0.0 (Volatile) to 1.0 (Immutable)
    resonance_type: str # 'Core', 'Extension', 'Bridge'
    suggested_imports: List[str]
    relevant_knowledge: List[str] = None  # [PHASE 30] Knowledge from TesseractMemory

class ResonanceCompiler:
    def __init__(self):
        self.interpreter = WaveInterpreter()
        # self.coder = WaveCoder() # [PHASE 26] Disabled to simplify
        self.manifestor = HolographicManifestor()
        self.memory = None  # [PHASE 30] Lazy-loaded TesseractMemory
        logger.info("  ResonanceCompiler initialized (Voice -> Code).")
    
    def _get_memory(self):
        """[PHASE 30] Lazy load TesseractMemory."""
        if self.memory is None:
            try:
                from Core.L5_Mental.Intelligence.Memory.tesseract_memory import get_tesseract_memory
                self.memory = get_tesseract_memory()
            except Exception as e:
                logger.warning(f"   TesseractMemory unavailable: {e}")
        return self.memory

    def compile_intent(self, intent: str) -> str:
        """
        Translates a High-Level Intent into Concrete Code.
        [PHASE 30] Now queries TesseractMemory for relevant knowledge first.
        """
        logger.info(f"  Compiling Intent: '{intent}'")
        
        # [PHASE 30] Query TesseractMemory for relevant knowledge/principles
        memory = self._get_memory()
        knowledge_context = ""
        if memory:
            nearby_nodes = memory.query(intent, k=5)
            if nearby_nodes:
                knowledge_points = []
                for node in nearby_nodes:
                    if node.node_type == "principle":
                        knowledge_points.append(f"[PRINCIPLE] {node.name}: {node.content}")
                    else:
                        knowledge_points.append(f"[KNOWLEDGE] {node.name}")
                knowledge_context = "\n".join(knowledge_points)
                logger.info(f"     Retrieved {len(nearby_nodes)} relevant concepts from memory.")

        # 1. Interpret Intent as Wave (Simplified Keyword Matching for now)
        wave_pattern = self._intent_to_wave(intent)
        logger.info(f"     Wave Pattern: {wave_pattern.name} (Freq: {wave_pattern.frequencies[0]}Hz)")

        # 2. Generate Blueprint from Wave
        blueprint = self._wave_to_blueprint(wave_pattern)
        blueprint.relevant_knowledge = [n.name for n in nearby_nodes] if memory and nearby_nodes else []
        logger.info(f"     Blueprint: Complexity<{blueprint.target_complexity}, Stability>{blueprint.stability_requirement}")

        # 3. Manifest Code (Pass Blueprint + Knowledge to Manifestor)
        prompt_suffix = self._blueprint_to_prompt(blueprint, knowledge_context)
        result = self.manifestor.manifest_code(f"{intent}\n\n[Constraints]\n{prompt_suffix}")
        
        return result

    def _intent_to_wave(self, intent: str) -> WavePattern:
        """Simple mapping of words to Wave Patterns."""
        if "love" in intent.lower() or "connect" in intent.lower():
            return self.interpreter.vocabulary["Love"]
        elif "fear" in intent.lower() or "protect" in intent.lower():
            return self.interpreter.vocabulary["Fear"] # Fear = Protection/Defense
        elif "hope" in intent.lower() or "future" in intent.lower():
            return self.interpreter.vocabulary["Hope"]
        else:
            return self.interpreter.vocabulary["Unity"] # Default

    def _wave_to_blueprint(self, wave: WavePattern) -> WaveBlueprint:
        """The Physics of Coding."""
        freq = wave.frequencies[0]
        
        # High Freq (Hope/852Hz) = Rapid, Experimental, Low Stability Req
        # Low Freq (Fear/100Hz) = Slow, Defensive, High Stability Req
        # Mid Freq (Love/528Hz) = Balanced, Connective
        
        if freq < 200: # Defensive/Stable
            return WaveBlueprint(0.3, 0.9, "Core", [])
        elif freq > 600: # Experimental
            return WaveBlueprint(0.8, 0.4, "Extension", [])
        else: # Balanced
            return WaveBlueprint(0.5, 0.7, "Bridge", ["Core.L1_Foundation.Foundation"])

    def _blueprint_to_prompt(self, bp: WaveBlueprint, knowledge_context: str = "") -> str:
        """Converts Blueprint + Knowledge to LLM Prompt instructions."""
        instructions = []
        
        # [PHASE 30] Inject relevant knowledge/principles
        if knowledge_context:
            instructions.append("\n[Relevant Knowledge from Memory]")
            instructions.append(knowledge_context)
        
        # [PHASE 26] Cellular Identity
        instructions.append(f"\n[Identity Protocol]")
        instructions.append(f"- CLASS MUST be decorated with @Cell(\"GeneratedIdentity\").")
        instructions.append(f"- Import Cell via: `from Core.L1_Foundation.Foundation.System.elysia_core import Cell`")
        
        # [PHASE 26] Mitosis / Complexity Control
        instructions.append(f"\n[Wave Constraints]")
        if bp.target_complexity < 0.4:
            instructions.append("- Code MUST be extremely simple and linear (Low Entropy).")
            instructions.append("- Avoid complex classes if functions suffice.")
        if bp.stability_requirement > 0.8:
            instructions.append("- Add robust error handling (try/except) (High Stability).")
            instructions.append("- Add comprehensive type hints.")
        
        return "\n".join(instructions)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compiler = ResonanceCompiler()
    
    # Test
    code = compiler.compile_intent("Create a function to protect the core system.")
    print("\n--- Generated Code ---\n")
    print(code)
