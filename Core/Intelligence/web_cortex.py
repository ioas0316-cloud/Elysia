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
from typing import Dict, Any, List

logger = logging.getLogger("WebCortex")

class WebCortex:
    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"
        logger.info("🌐 WebCortex Active. Connected to the Net.")

    def search(self, query: str) -> str:
        """
        Searches Wikipedia for a summary of the query.
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
                # Limit summary length
                return summary[:500] + "..." if len(summary) > 500 else summary
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"⚠️ Connection Error: {e}"

    def calibrate_concept(self, concept: str) -> Dict[str, Any]:
        """
        [Reality Calibration]
        Fetches 'Common Sense' from the Net to ground a concept.
        Returns Sensory Data and Semantic Context.
        """
        summary = self.search(concept)
        if "⚠️" in summary:
            return {"valid": False, "reason": summary}
            
        # Extract Sensory Keywords (Simple Heuristic)
        sensory_data = {
            "visual": [],
            "audio": [],
            "tactile": []
        }
        
        # Basic Sensory Dictionary (To be expanded or learned)
        visual_keys = ["red", "blue", "green", "light", "dark", "bright", "shiny", "color"]
        audio_keys = ["loud", "quiet", "sound", "music", "noise", "voice", "tone"]
        tactile_keys = ["soft", "hard", "hot", "cold", "rough", "smooth", "pain", "warm"]
        
        summary_lower = summary.lower()
        
        for k in visual_keys:
            if k in summary_lower: sensory_data["visual"].append(k)
        for k in audio_keys:
            if k in summary_lower: sensory_data["audio"].append(k)
        for k in tactile_keys:
            if k in summary_lower: sensory_data["tactile"].append(k)
            
        return {
            "valid": True,
            "concept": concept,
            "summary": summary,
            "sensory": sensory_data,
            "source": "Wikipedia"
        }

    def browse(self):
        """
        Simulates autonomous browsing (Placeholder).
        """
        pass

    def fetch_url(self, url: str) -> str:
        """
        Fetches the content of a specific URL.
        """
        logger.info(f"🌐 Fetching URL: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0 (AI Research Bot)'})
            with urllib.request.urlopen(req) as response:
                # Simple decoding, might need more robust charset handling
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return f"⚠️ Fetch Error: {e}"
