from typing import List, Dict, Any, Tuple
import re

class LogosParser:
    """
    [The Digestive Enzyme]
    Translates the 'Logos' (Language) of the LLM into the 'Physics' (WaveDNA) of the System.
    It extracts Intent, Target, and Phenomenon from the raw text stream.
    """
    def __init__(self):
        self.cmd_pattern = re.compile(r"\[ACT:(\w+):(.+?)\]")

    def digest(self, llm_output: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Separates the 'Voice' (Charisma) from the 'Will' (Command).
        Returns: (spoken_text, list_of_commands)
        """
        commands = []
        
        # 1. Extract Commands
        matches = self.cmd_pattern.findall(llm_output)
        for action, payload in matches:
            details = payload.split('|')
            target = details[0].strip() if len(details) > 0 else "Self"
            param = details[1].strip() if len(details) > 1 else "None"
            
            commands.append({
                "action": action.upper(), # e.g. CREATE, SPIN, IGNITE
                "target": target,         # e.g. STAR, TIME, MEMORY
                "param": param            # e.g. RED, +100, HAPPINESS
            })
            
        # 2. Clean Text (Remove commands from speech)
        clean_text = self.cmd_pattern.sub("", llm_output).strip()
        
        return clean_text, commands

    def encode_wave_dna(self, concept: str) -> List[float]:
        """
        Converts a concept string into a 7D Vector (Placeholder).
        In the future, this connects to the Embedding Model.
        """
        # Simulated DNA Encoding
        # R, G, B, Theta, Phi, Psi, Spin
        return [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0]
