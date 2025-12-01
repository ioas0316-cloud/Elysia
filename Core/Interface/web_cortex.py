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
            with urllib.request.urlopen(url) as response:
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

    def browse(self):
        """
        Simulates autonomous browsing (Placeholder).
        """
        pass
