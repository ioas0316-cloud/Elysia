import logging
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from Core._01_Foundation.Foundation.hyper_quaternion import Quaternion

logger = logging.getLogger("PrismCortex")

@dataclass
class Photon:
    """
    A particle of pure insight.
    Massless (stripped of boilerplate), carrying only color (quaternion) and meaning.
    """
    id: str
    color_band: str  # "Red", "Blue", "Violet"...
    vector: Dict[str, float] # Serialized Quaternion
    payload: str     # The insight text

class PrismCortex:
    def __init__(self):
        # The Spectrum Buckets
        self.spectrum: Dict[str, List[Photon]] = {
            "Red": [],      # Emotion, Passion, Conflict
            "Orange": [],   # Social, Connection, Warmth
            "Yellow": [],   # Creativity, Dream, Light
            "Green": [],    # Growth, Life, Nature
            "Blue": [],     # Logic, Code, Structure
            "Indigo": [],   # Deep Wisdom, Intuition, Mystery
            "Violet": []    # Spiritual, Divine, Void
        }
        
    def _analyze_color(self, text: str) -> str:
        """
        Fast Heuristic Refraction.
        Determines the 'Color' of a text based on keywords.
        """
        text_lower = text.lower()
        
        # Blue (Logic/Code) - Most common in this codebase
        if any(w in text_lower for w in ["def ", "class ", "import ", "return", "function", "logic", "system", "api"]):
            return "Blue"
            
        # Red (Emotion)
        if any(w in text_lower for w in ["love", "hate", "anger", "passion", "feel", "heart", "pain"]):
            return "Red"
            
        # Yellow (Creativity)
        if any(w in text_lower for w in ["create", "dream", "imagine", "new", "idea", "vision", "art"]):
            return "Yellow"
            
        # Violet (Spiritual)
        if any(w in text_lower for w in ["god", "soul", "spirit", "void", "infinite", "dimension", "divine"]):
            return "Violet"
            
        # Green (Growth)
        if any(w in text_lower for w in ["grow", "evolution", "life", "tree", "root", "seed"]):
            return "Green"

        # Indigo (Wisdom)
        if any(w in text_lower for w in ["why", "truth", "understand", "study", "research", "deep"]):
            return "Indigo"
            
        # Orange (Social)
        if any(w in text_lower for w in ["user", "human", "talk", "chat", "connect", "friend"]):
            return "Orange"
            
        return "Blue" # Default to Logic for unclassified data (it's code mostly)

    def refract(self, raw_id: str, raw_text: str) -> Photon:
        """
        Refracts a raw memory shard into a Photon.
        """
        color = self._analyze_color(raw_text)
        
        # Assign Vector based on color
        # Simplified Quaternion mapping
        q = Quaternion(1, 0, 0, 0)
        if color == "Red": q = Quaternion(0.5, 1.0, 0, 0) # X dominant
        elif color == "Blue": q = Quaternion(0.5, 0, 1.0, 0) # Y dominant
        elif color == "Yellow": q = Quaternion(0.5, 0, 0, 1.0) # Z dominant
        # ... others can be mixes
        
        # Photonize: Strip large invisible data if possible? 
        # For now, we keep the raw text as 'payload' but envision stripping it later.
        
        return Photon(
            id=raw_id,
            color_band=color,
            vector={"w":q.w, "x":q.x, "y":q.y, "z":q.z},
            payload=raw_text[:200] + "..." if len(raw_text) > 200 else raw_text # Compress payload for now?
        )

    def absorb_shard(self, name: str, data: str):
        photon = self.refract(name, data)
        self.spectrum[photon.color_band].append(photon)

    def save_spectrum(self, path: str):
        serializable = {k: [asdict(p) for p in v] for k, v in self.spectrum.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)

    def report(self) -> str:
        total = sum(len(v) for v in self.spectrum.values())
        if total == 0: return "Spectrum Empty."
        
        report = []
        for color, photons in self.spectrum.items():
            count = len(photons)
            ratio = count / total
            report.append(f"{color}: {count} ({ratio:.1%})")
        return "\n".join(report)
