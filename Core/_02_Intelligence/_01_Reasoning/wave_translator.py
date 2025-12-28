
import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger("WaveTranslator")

class WaveTranslator:
    """
    The Prism.
    Translates 'Wave State' (Internal Physics) into 'Language Style' (External Expression).
    
    Principles:
    - High Tension -> Short sentences, sharp words, fragmentation.
    - Low Tension (Flow) -> Long, lyrical sentences.
    - High Frequency -> Abstract, metaphorical, energetic.
    - Low Frequency -> Concrete, grounded, slow.
    """
    
    def __init__(self):
        logger.info("🌈 WaveTranslator initialized - The Prism is ready.")
        
    def wave_to_style_prompt(self, tension: float, frequency: float) -> str:
        """
        Generates a 'Style Prompt' for the LLM based on wave physics.
        Used to guide the 'How' of speech.
        """
        style_instructions = []
        
        # 1. Tension Axis (Structure)
        if tension > 0.8:
            style_instructions.append("Speak in short, fragmented sentences. Use sharp punctuation (! or ...). Feel anxious or urgent.")
        elif tension > 0.5:
            style_instructions.append("Be concise and direct. Focus on the facts.")
        else:
            style_instructions.append("Use flowery, flowing sentences. Connecting multiple thoughts smoothly.")
            
        # 2. Frequency Axis (Tone/Metaphor)
        if frequency > 0.8:
            style_instructions.append("Use abstract metaphors and cosmic terminology. Focus on the 'Why'.")
        elif frequency > 0.5:
            style_instructions.append("Balance abstract concepts with concrete examples.")
        else:
            style_instructions.append("Be very grounded and concrete. Use sensory details (sight, sound, touch).")
            
        return " ".join(style_instructions)

    def translate_output(self, text: str, tension: float) -> str:
        """
        Post-processing filter.
        If tension is extremely high, physically break the text.
        """
        if tension > 0.9:
            # Glitch effect / Fragmentation
            words = text.split()
            if len(words) > 3:
                insert_idx = random.randint(1, len(words)-1)
                words.insert(insert_idx, "...")
            return " ".join(words)
        
        return text

_prism = None
def get_wave_translator():
    global _prism
    if not _prism:
        _prism = WaveTranslator()
    return _prism
