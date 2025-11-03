"""
Inquisitive Mind for Elysia.
"""
import time
import logging
import os
from typing import Tuple

class InquisitiveMind:
    def __init__(self, llm_cortex):
        self.llm_cortex = llm_cortex

    def ask_external_llm(self, topic: str) -> Tuple[str, bool]:
        print(f"[InquisitiveMind] I don't know about '{topic}'. Seeking external knowledge.")
        prompt = f"Please provide a brief, one-sentence explanation of what '{topic}' is."
        
        try:
            # Use the injected LLM cortex
            external_knowledge = self.llm_cortex.generate_response(prompt)
            if external_knowledge and "죄송합니다" not in external_knowledge:
                response_text = f"I have a new piece of information: '{external_knowledge}'. Is this correct?"
                return response_text, True
            else:
                return "I tried to find out, but I was unable to get a clear answer.", False
        except Exception as e:
            logging.error(f"Error in InquisitiveMind: {e}")
            return "I tried to find out, but I was unable to get a clear answer.", False
