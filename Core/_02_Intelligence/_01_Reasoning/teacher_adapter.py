
import logging
from typing import Optional, Dict, Any
from Core._01_Foundation._04_Governance.Foundation.ollama_bridge import get_ollama_bridge

logger = logging.getLogger("TeacherAdapter")

class TeacherAdapter:
    """
    The Bridge to the Teacher (LLM).
    Used by LogicScout to ask 'Why did you do that?'
    """
    
    def __init__(self):
        self.bridge = get_ollama_bridge()
        logger.info("🎓 TeacherAdapter initialized - Ready to ask questions.")

    def explain_reasoning(self, input_text: str, output_text: str) -> str:
        """
        Asks the Teacher to explain the logic connecting Input to Output.
        Returns the raw explanation.
        """
        prompt = f"""
        Analyze this interaction:
        Input: "{input_text}"
        Output: "{output_text}"
        
        Extract the underlying logical rule that connects the Input to the Output.
        Format your answer as a Logic Chain.
        Do not explain the context, just give the Abstract Logic.
        """
        
        try:
            response = self.bridge.generate_response(prompt, context=[])
            if response and "Error" not in response:
                return response
        except Exception as e:
            logger.warning(f"TeacherAdapter: Ollama failed ({e}). Using MOCK logic.")
            
        # [MOCK MODE]
        # Use simple heuristic if LLM is offline
        return f"Logic inferred from '{input_text}': IF condition MET THEN execute Action."

_adapter = None
def get_teacher_adapter():
    global _adapter
    if not _adapter:
        _adapter = TeacherAdapter()
    return _adapter
