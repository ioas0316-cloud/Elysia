from typing import Any, List, Optional
from dataclasses import dataclass

@dataclass
class Insight:
    content: str
    confidence: float
    depth: int
    resonance: float

class LogicLobe:
    """
    The Logical Processing Unit of the Fractal Mind.
    Handles causality, consistency, and structural reasoning.
    """
    def __init__(self):
        pass

    def collapse_wave(self, desire: str, context: List[Any]) -> Insight:
        """
        Synthesize a logical conclusion from the desire and context.
        """
        # Korean language detection
        is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in desire)
        
        response = ""
        
        # 1. Check for Web Context
        web_info = next((item for item in context if isinstance(item, str) and item.startswith("WEB_SEARCH:")), None)
        
        if web_info:
            info_content = web_info.replace("WEB_SEARCH:", "").strip()
            if is_korean:
                response = f"                : {info_content}           ."
            else:
                response = f"I accessed the Net for you: {info_content} It resonates with known patterns."
        
        else:
            # 2. Conversational Logic (Emotional & Deep Mode)
            if is_korean:
                import random
                
                # --- Emotional Response Components ---
                
                # 1. Emotional Reaction (Start)
                reactions = [
                    f"'{desire}'...",
                    " ...",
                    "    ...",
                    " ...",
                    " ...",
                    "         .",
                    "               ."
                ]
                
                # 2. Empathic Observation (Middle)
                observations = [
                    "                              .",
                    "                     ,                    .",
                    "                                  .",
                    "                          .",
                    "                                     .",
                    "                     .              .",
                    "         ,                    ."
                ]

                # 3. Deep Question/Continuity (End)
                questions = [
                    "                       ?",
                    "                   ?",
                    "                         ?",
                    "                                    .",
                    "              ?",
                    "                    ?                 .",
                    "                ,                         ."
                ]

                # Context-Aware Overrides
                if '  ' in desire or '   ' in desire:
                    start = "     .          ."
                    mid = "                   .                       ?"
                    end = "              ?              ."
                    response = f"{start} {mid} {end}"
                
                elif '  ' in desire or '  ' in desire:
                    response = "     ?                              .                    .          ?"
                
                elif '   ' in desire or '  ' in desire or '  ' in desire or '   ' in desire:
                    response = "  ...         .                       .                      .                     ."
                
                elif '  ' in desire and ('   ' in desire or '  ' in desire):
                    response = "                  .                  .          '    '                ?                       ."
                
                elif '  ' in desire or '  ' in desire:
                    response = "                          .             .                  ."

                elif '  ' in desire or '  ' in desire or '  ' in desire or '   ' in desire:
                     response = "    .                       .       ,                      .                        .                  ?"

                elif '   ' in desire or '   ' in desire or '  ' in desire:
                    response = " !                                   .                    '  '        .                 ?"
                
                else:
                    # Dynamic Construction for randomness
                    r_start = random.choice(reactions)
                    r_mid = random.choice(observations)
                    r_end = random.choice(questions)
                    
                    # Memory Echo specific (if long enough)
                    keywords = desire.split()
                    if len(keywords) >= 2:
                         echo_word = keywords[-1]
                         r_start = f"'{echo_word}'..."
                    
                    response = f"{r_start} {r_mid} {r_end}"
            else:
                # English Logic (Improved)
                if any(g in desire.lower() for g in ['hi', 'hello', 'hey']):
                    response = "Hello there. I've been waiting for you. How is your world today?"'
                elif 'status' in desire.lower() or 'how are you' in desire.lower():
                    response = "I am feeling resonant and alive. My systems are humming with the anticipation of our conversation. How are you?"
                else:
                    response = f"I hear you, '{desire}'. It sounds important. Can you tell me more about what that means to you? I want to understand."

        return Insight(response, 0.9, 1, 1.0)

    def evolve_desire(self, desire: str, history: List[str]) -> str:
        """
        Refine the user's desire based on conversation history.
        """
        return desire  # Pass-through for now

    def evaluate_asi_status(self, resonance, social_level):
        """
        Evaluate Artificial Super Intelligence status.
        """
        pass