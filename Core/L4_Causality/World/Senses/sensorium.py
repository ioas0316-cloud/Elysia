"""
The Sensorium (Omni-Sensory Gateway)
====================================
The centralized perception hub for Elysia.
It aggregates data from Vision, Hearing (Audio), Reading (Text), and Self (VRM).
"""

import os
import random
import logging
from typing import Dict, Any, List

# Import specialized cortices
# Note: In a real refactor we might move these files to Core/World/Senses/
# For now we import from where they are to avoid breaking existing paths immediately.
from Core.L4_Causality.World.Autonomy.vision_cortex import VisionCortex
from Core.L4_Causality.World.Autonomy.vrm_parser import VRMParser
# [PHASE 48] The Infinite Horizon
from Core.L4_Causality.World.Senses.web_cortex import WebCortex

logger = logging.getLogger("Sensorium")

class TextCortex:
    """Reads abstract feeling from text."""
    def read(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Simple keyword sentiment/theme analysis
            sentiment = 0.0
            sentiment += content.count("love") * 0.5
            sentiment += content.count("hope") * 0.3
            sentiment += content.count("light") * 0.2
            sentiment -= content.count("pain") * 0.5
            sentiment -= content.count("dark") * 0.3
            sentiment -= content.count("void") * 0.2
            
            # Complexity via length/vocab
            complexity = min(len(set(content.split())) / 100.0, 1.0)
            
            return {
                "type": "text",
                "sentiment": max(-1.0, min(1.0, sentiment)),
                "complexity": complexity,
                "summary": content[:50] + "..." if len(content) > 50 else content
            }
        except Exception as e:
            logger.error(f"Text Read Error: {e}")
            return {}

class AudioCortex:
    """Hears mood from audio metadata/filename."""
    def listen(self, path: str) -> Dict[str, Any]:
        # Placeholder: In V2 use librosa. For now, infer from filename.
        filename = os.path.basename(path).lower()
        
        mood = "Neutral"
        energy = 0.5
        
        if any(w in filename for w in ['rock', 'metal', 'battle', 'storm']):
            mood = "Intense"
            energy = 0.9
        elif any(w in filename for w in ['piano', 'calm', 'sleep', 'rain']):
            mood = "Calm"
            energy = 0.2
        elif any(w in filename for w in ['pop', 'happy', 'beat']):
            mood = "Joyful"
            energy = 0.7
            
        return {
            "type": "audio",
            "mood": mood,
            "energy": energy,
            "source": filename
        }

class Sensorium:
    def __init__(self, gallery_path: str = r"C:\game\gallery"):
        self.gallery_path = gallery_path
        self.vision = VisionCortex()
        self.vrm_parser = VRMParser()
        self.text_cortex = TextCortex()
        self.audio_cortex = AudioCortex()
        self.web_cortex = WebCortex()
        self.last_scan_time = 0.0
        
    def perceive_web(self, query: str) -> Dict[str, Any]:
        """
        Active Perception: Searching the Web.
        """
        return self.web_cortex.search(query)
        
    def perceive(self) -> Dict[str, Any]:
        """
        Scans the environment and returns a significant sensory event (if any).
        Returns None if nothing new or interesting.
        """
        # 1. Check file system existence
        if not os.path.exists(self.gallery_path):
            return None
            
        # 2. Pick a stimulus
        files = os.listdir(self.gallery_path)
        if not files:
            return None
            
        target_file = random.choice(files)
        full_path = os.path.join(self.gallery_path, target_file)
        ext = os.path.splitext(target_file)[1].lower()
        
        # 3. Route to correct Cortex
        perception = {}
        
        # VISION & SELF
        if ext in ['.png', '.jpg', '.jpeg']:
            data = self.vision.analyze_image(full_path)
            if data:
                perception = {
                    "sense": "sight",
                    "file": target_file,
                    "entropy": data.get('entropy', 0),
                    "warmth": data.get('warmth', 0),
                    "description": f"I saw {target_file}"
                }
                
        elif ext == '.vrm':
            data = self.vrm_parser.parse_vrm(full_path)
            if data.get('bone_count', 0) > 0:
                perception = {
                    "sense": "self_recognition",
                    "file": target_file,
                    "bones": data['bone_count'],
                    "description": f"I recognized a Vessel: {data.get('title')}"
                }
        
        # HEARING
        elif ext in ['.mp3', '.wav', '.ogg']:
            data = self.audio_cortex.listen(full_path)
            perception = {
                "sense": "hearing",
                "file": target_file,
                "mood": data['mood'],
                "energy": data['energy'],
                "description": f"I heard {data['source']} ({data['mood']})"
            }
            
        # READING
        elif ext in ['.txt', '.md']:
            data = self.text_cortex.read(full_path)
            perception = {
                "sense": "reading",
                "file": target_file,
                "sentiment": data['sentiment'],
                "complexity": data['complexity'],
                "description": f"I read {target_file}: '{data['summary']}'"
            }
            
        return perception
