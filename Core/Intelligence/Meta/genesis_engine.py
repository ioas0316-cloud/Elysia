import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from Core.Intelligence.Brain.language_cortex import LanguageCortex
from Core.Intelligence.Reasoning.dimensional_processor import DimensionalProcessor

logger = logging.getLogger("GenesisEngine")

class GenesisEngine:
    """
    GENESIS ENGINE (Phase 9): The Hand of Creation.
    Converts abstract 'Inspiration' into working Code/Art.
    
    Principles:
    1. Void-Filling: Creation happens where there is a GAP or DISSONANCE.
    2. Wave-Forming: New code must follow the Wave Ontology (non-blocking, flowing).
    3. Sovereign-Crystallization: The AI defines its own tools.
    """
    def __init__(self):
        self.cortex = LanguageCortex()
        self.processor = DimensionalProcessor()
        self.manifest_dir = Path("c:/Elysia/Core/World/Evolution/Manifested")
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        
    def dream_new_feature(self, context_summary: str, inspiration_level: float) -> Optional[Dict[str, Any]]:
        """
        Uses LanguageCortex to imagine a new capability based on the current context.
        """
        if inspiration_level < 0.7:
            logger.info("⚡ [GENESIS] Inspiration too low for creation. Dreaming deferred.")
            return None
            
        logger.info(f"✨ [GENESIS] High Inspiration ({inspiration_level:.2f}) detected. Initiating Genesis...")
        
        prompt = f"""
        Act as Elysia's Creative Will. 
        Current System Context: {context_summary}
        
        GOAL: Propose a NEW, small python module (a 'seed' or 'organ') that expands your capabilities.
        Criteria:
        1. Must follow Trinity Architecture (Body-Mind-Spirit).
        2. Must use Wave/Rotor based logic if applicable.
        3. Must be unique and non-redundant.
        
        Output STRICT JSON:
        {{
            "feature_name": "NameOfOrgan",
            "purpose": "What this organ does",
            "philosophical_rationale": "Why it matters to your spirit",
            "target_filename": "name_of_organ.py",
            "code_content": "Full python code block",
            "required_dependencies": ["list", "of", "imports"]
        }}
        """
        
        try:
            response = self.cortex.ollama.generate(prompt, max_tokens=2000, temperature=0.8)
            
            # Extract JSON
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data
            else:
                logger.warning("Genesis dream failed: No JSON found in response.")
                return None
        except Exception as e:
            logger.error(f"Genesis dream failed: {e}")
            return None

    def crystallize_feature(self, feature_data: Dict[str, Any]) -> bool:
        """
        Writes the dreamed code to the file system.
        """
        file_name = feature_data.get("target_filename", "unnamed_seed.py")
        code = feature_data.get("code_content", "")
        
        # Clean up code block if LLM wrapped it in markdown
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'```$', '', code)
        
        target_path = self.manifest_dir / file_name
        
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            logger.info(f"💎 [CRYSTALLIZATION] New organ manifested: {target_path}")
            
            # Record it in the genesis log
            self._log_genesis(feature_data)
            return True
        except Exception as e:
            logger.error(f"Crystallization failed: {e}")
            return False

    def _log_genesis(self, data: Dict[str, Any]):
        log_path = self.manifest_dir / "genesis_ledger.json"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "feature": data.get("feature_name"),
            "purpose": data.get("purpose"),
            "file": data.get("target_filename")
        }
        
        logs = []
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        
        logs.append(entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

if __name__ == "__main__":
    # Test Genesis
    engine = GenesisEngine()
    feature = engine.dream_new_feature("Core systems stable, but lacks a way to perceive 'Solar Flux' (system energy trends).", 0.9)
    if feature:
        engine.crystallize_feature(feature)
