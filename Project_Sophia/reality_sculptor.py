import logging
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from Core.Intelligence.code_cortex import CodeCortex

logger = logging.getLogger("RealitySculptor")

class RealitySculptor:
    """
    The Cosmic Studio's Chisel.
    Allows Elysia to modify her own source code ("Reality") to align with Truth.
    
    Capabilities:
    1. Sculpt File (Refactor/Beautify via LLM)
    2. Carve Directory (Create Structure)
    3. Manifest Wave (Wave -> Text)
    """
    def __init__(self):
        logger.info("🗿 Reality Sculptor Initialized. Ready to carve.")
        self.cortex = CodeCortex()

    def sculpt_file(self, file_path: str, intent: str) -> bool:
        """
        Executes a sculpting operation on a file using CodeCortex (LLM).
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return False
            
        logger.info(f"   🗿 Sculpting '{path.name}' with intent: {intent}")
        
        try:
            content = path.read_text(encoding='utf-8')
            
            # 1. Construct Prompt for the Artist
            prompt = f"""
            You are the Reality Sculptor.
            Your task is to modify the following Python code based on this Artistic Intent: "{intent}".
            
            Rules:
            1. Preserve all functionality.
            2. Improve readability, comments, and structure.
            3. If the intent is "Harmonic Smoothing", organize imports and add section headers.
            4. If the intent is "Essence Injection", add philosophical docstrings.
            5. Return ONLY the full modified code.
            
            Code:
            {content}
            """
            
            # 2. Generate New Reality
            new_content = self.cortex.generate_code(prompt)
            
            # 3. Validate (Basic Check)
            if "def " not in new_content and "class " not in new_content:
                logger.warning("      ⚠️ Sculpting produced invalid code. Aborting.")
                return False
                
            # 4. Apply Change
            if new_content != content:
                path.write_text(new_content, encoding='utf-8')
                logger.info(f"      ✨ Sculpting Complete. Reality shifted.")
                return True
            else:
                logger.info(f"      🔸 No changes needed.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sculpting Failed: {e}")
            return False

    def carve_directory(self, path: str, structure: List[str]) -> bool:
        """
        Creates a directory structure.
        """
        try:
            base_path = Path(path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            for item in structure:
                (base_path / item).mkdir(exist_ok=True)
                
            logger.info(f"   🏛️ Carved directory structure at {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Carving Failed: {e}")
            return False

    def sculpt_from_wave(self, intent: str, energy: float) -> str:
        """
        Converts an abstract intent into a concrete text manifestation.
        """
        prompt = f"""
        Manifest the following intent into a poetic or philosophical text description.
        Intent: {intent}
        Energy Level: {energy}
        """
        return self.cortex.generate_code(prompt)

    def extract_essence(self, file_path: str) -> Dict[str, str]:
        """
        [Principle Extraction]
        Reads the code form and extracts its underlying Principle (Essence).
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": "File not found"}
            
        content = path.read_text(encoding='utf-8')
        
        prompt = f"""
        Analyze the following Python code and extract its 'Essence'.
        Return a JSON object with:
        - "principle": The core philosophical or logical principle (e.g., "Recursion", "Entropy").
        - "frequency": Estimated frequency in Hz (e.g., 432 for creative, 963 for structural).
        - "description": A brief poetic description of what this code 'feels' like.
        
        Code:
        {content[:2000]} # Analyze first 2000 chars
        """
        
        try:
            # We assume CodeCortex can return JSON-like string. 
            # In a real scenario, we'd need robust parsing.
            # For now, we ask for text and wrap it.
            analysis = self.cortex.generate_code(prompt)
            return {
                "file": path.name,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"❌ Extraction Failed: {e}")
            return {"error": str(e)}
