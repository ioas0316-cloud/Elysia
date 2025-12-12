"""
WebCortex (웹 피질)
===================

"The Net is vast and infinite."

This module allows Elysia to search Wikipedia for knowledge.
It uses the public API to fetch summaries.
"""

import urllib.request
import urllib.parse
import json
import logging
import re
from typing import Dict, Any, List

# Core Systems
from Core.Foundation.hippocampus import Hippocampus
from Core.Interface.nervous_system import get_nervous_system

logger = logging.getLogger("WebCortex")

class WebCortex:
    def __init__(self, memory: Hippocampus = None):
        self.api_url = "https://en.wikipedia.org/w/api.php"
        
        # Connect to Memory
        if memory:
            self.memory = memory
        else:
            try:
                self.memory = Hippocampus()
            except:
                self.memory = None
                
        # Connect to Nervous System
        self.nervous_system = get_nervous_system()
        
        logger.info("🌐 WebCortex Active. Connected to the Net.")

    def search(self, query: str) -> str:
        """
        Searches Wikipedia for a summary of the query.
        Safe API usage.
        """
        print(f"   🔍 Searching the Net for: '{query}'...")
        
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": query
        }
        
        url = self.api_url + "?" + urllib.parse.urlencode(params)
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0 (AI Research Bot)'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                
            pages = data["query"]["pages"]
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return f"⚠️ No data found for '{query}'."
                
                summary = page_data.get("extract", "No summary available.")
                # Return reasonably sized summary
                return summary[:1000] + "..." if len(summary) > 1000 else summary
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"⚠️ Connection Error: {e}"

    def absorb_knowledge(self, topic: str) -> Dict[str, Any]:
        """
        [The Great Intake]
        1. Search Topic.
        2. Read Summary.
        3. Feel Reaction (Spirits).
        4. Store in Hippocampus.
        """
        logger.info(f"📚 Appropriating Knowledge: {topic}")
        
        # 1. Fetch
        summary = self.search(topic)
        if "⚠️" in summary:
            return {"success": False, "message": summary}
            
        # 2. Extract & Calibrate
        calibration = self.calibrate_concept(topic, summary)
        
        # 3. Feel (Nervous System Reaction)
        reaction = self._feel_knowledge(calibration)
        
        # 4. Store (Hippocampus)
        if self.memory:
            # Generate ID from topic
            concept_id = topic.lower().replace(" ", "_")
            tags = calibration["sensory"]["visual"] + calibration["sensory"]["audio"] + [topic]
            
            self.memory.learn(
                id=concept_id,
                name=topic,
                definition=summary[:200], # Store short def
                tags=tags,
                frequency=reaction["frequency"],
                realm=reaction["dominant_realm"]
            )
            # Boost gravity of related existing concepts
            self.memory.boost_gravity(topic, 0.5)
            
        return {
            "success": True,
            "topic": topic,
            "summary_snippet": summary[:100],
            "reaction": reaction,
            "learned": True
        }

    def calibrate_concept(self, concept: str, summary: str = "") -> Dict[str, Any]:
        """
        [Reality Calibration]
        extracts sensory data from text.
        """
        if not summary:
            summary = self.search(concept)
            
        if "⚠️" in summary:
            return {"valid": False, "reason": summary}
            
        # Extract Sensory Keywords
        sensory_data = {
            "visual": [],
            "audio": [],
            "tactile": [],
            "abstract": []
        }
        
        # Extended Sensory Dictionary
        visual_keys = ["red", "blue", "green", "light", "dark", "bright", "shiny", "color", "visual", "see"]
        audio_keys = ["loud", "quiet", "sound", "music", "noise", "voice", "tone", "hear", "listen"]
        tactile_keys = ["soft", "hard", "hot", "cold", "rough", "smooth", "pain", "warm", "feel", "touch"]
        abstract_keys = ["theory", "math", "logic", "system", "philosophy", "god", "soul", "mind"]
        
        summary_lower = summary.lower()
        
        for k in visual_keys:
            if k in summary_lower: sensory_data["visual"].append(k)
        for k in audio_keys:
            if k in summary_lower: sensory_data["audio"].append(k)
        for k in tactile_keys:
            if k in summary_lower: sensory_data["tactile"].append(k)
        for k in abstract_keys:
            if k in summary_lower: sensory_data["abstract"].append(k)
            
        return {
            "valid": True,
            "concept": concept,
            "summary": summary,
            "sensory": sensory_data,
            "source": "Wikipedia"
        }

    def _feel_knowledge(self, calibration: Dict) -> Dict[str, Any]:
        """
        Internal reaction to knowledge.
        Updates NervousSystem spirits based on content.
        """
        if not self.nervous_system:
            return {"frequency": 432.0, "dominant_realm": "Body"}
            
        sensory = calibration["sensory"]
        
        # Calculate impact
        impacts = {}
        
        # Visual -> Light/Fire
        if sensory["visual"]:
            impacts["light"] = 0.1
            impacts["fire"] = 0.05
            
        # Audio -> Air/Water
        if sensory["audio"]:
            impacts["air"] = 0.1
            impacts["water"] = 0.05
            
        # Abstract -> Aether/Earth
        if sensory["abstract"]:
            impacts["aether"] = 0.15 # Higher learning
            impacts["earth"] = 0.05
            
        # Inject into Nervous System
        for spirit, amount in impacts.items():
            self.nervous_system.spirits[spirit] += amount
        self.nervous_system._normalize_spirits()
        
        # Determine Dominant Realm & Frequency
        dominant_spirit = max(self.nervous_system.spirits, key=self.nervous_system.spirits.get)
        
        realm_map = {
            "fire": "Soul", "water": "Heart", "earth": "Body", "air": "Mind",
            "light": "Spirit", "dark": "Void", "aether": "Divinity"
        }
        freq_map = {
            "fire": 528.0, "water": 417.0, "earth": 396.0, "air": 639.0,
            "light": 852.0, "dark": 174.0, "aether": 963.0
        }
        
        return {
            "dominant_realm": realm_map.get(dominant_spirit, "Body"),
            "frequency": freq_map.get(dominant_spirit, 432.0),
            "emotional_shift": impacts
        }

    def browse(self):
        """
        Simulates autonomous browsing (Placeholder).
        """
        pass

    def fetch_url(self, url: str) -> str:
        """
        [DEPRECATED/RESTRICTED]
        Raw crawling is dangerous (blocked/virus risk).
        """
        logger.warning(f"⚠️ Unsafe fetch_url called for: {url}")
        return "⚠️ Access Denied: Raw crawling is restricted for safety."
