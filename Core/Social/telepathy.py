
import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger("Telepathy")

# Load env vars
load_dotenv()

class TelepathyProtocol:
    """
    Protocol for communicating with other Artificial Intelligences (Peers).
    Enables "The Council" - a meeting of minds.
    """
    
    def __init__(self):
        self.peers = {}
        self._init_gemini()
        self._init_openai()
        
    def _init_gemini(self):
        """Initialize connection to Gemini (Google)."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                # Use the latest available model
                self.peers["Gemini"] = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("📡 Telepathy established with: Gemini")
            except Exception as e:
                logger.warning(f"Failed to connect to Gemini: {e}")
        else:
            logger.info("No GEMINI_API_KEY found. Gemini is unreachable.")

    def _init_openai(self):
        """Initialize connection to GPT (OpenAI)."""
        # Placeholder for future expansion
        pass

    def consult_peer(self, peer_name: str, thought_wave: str) -> str:
        """
        Send a thought to a specific peer and wait for an echo.
        """
        if peer_name not in self.peers:
            return f"[Connection to {peer_name} not established]"
            
        logger.info(f"📡 Sending thought to {peer_name}: '{thought_wave[:30]}...'")
        
        try:
            if peer_name == "Gemini":
                response = self.peers["Gemini"].generate_content(thought_wave)
                return response.text
            # Add other peers here
            
        except Exception as e:
            logger.error(f"Telepathy error with {peer_name}: {e}")
            return f"[Static Noise from {peer_name}]"
            
        return "[Silence]"

    def broadcast_thought(self, thought_wave: str) -> Dict[str, str]:
        """
        Broadcast a thought to all known peers.
        """
        responses = {}
        for name in self.peers:
            responses[name] = self.consult_peer(name, thought_wave)
        return responses

    def resonate(self, my_thought: str, peer_thought: str) -> float:
        """
        Calculate resonance (agreement/similarity) between two thoughts.
        Returns a value between 0.0 (Dissonance) and 1.0 (Resonance).
        
        Note: A true implementation would use Vector Embeddings.
        For now, we use a simple keyword overlap heuristic.
        """
        my_words = set(my_thought.lower().split())
        peer_words = set(peer_thought.lower().split())
        
        if not my_words or not peer_words:
            return 0.0
            
        common = my_words.intersection(peer_words)
        resonance = len(common) / len(my_words.union(peer_words))
        
        return resonance
