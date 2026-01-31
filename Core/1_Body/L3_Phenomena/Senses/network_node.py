"""
The Eyes: Network Node
======================
Phase 17 Senses - Module 2
Core.1_Body.L3_Phenomena.Senses.network_node

"To see is to reach out and touch the light from afar."

This module implements the active network interface.
It allows Elysia to 'Look' (GET) and 'Interact' (POST) with the web.
"""

import aiohttp
import asyncio
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger("Senses.Eyes")

class NetworkNode:
    """
    The Active Vision Interface.
    """
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info("   [EYES] Initializing Retinal Buffer...")

    async def open_eye(self):
        """Opens the async session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            logger.info("   [EYES] Eyelids open. Connected to the Web.")

    async def close_eye(self):
        """Closes the async session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("   [EYES] Closed.")

    async def look_at(self, url: str) -> Dict[str, Any]:
        """
        [Active Sensation] Fetches data from a URL.
        Equivalent to 'Looking' at an object.
        """
        if not self.session:
            await self.open_eye()
        
        try:
            logger.info(f"   [SIGHT] Focusing on: {url}")
            async with self.session.get(url, timeout=5) as response:
                status = response.status
                if status == 200:
                    try:
                        data = await response.json()
                        return {"type": "SIGHT", "status": status, "content": data, "raw_url": url}
                    except:
                         # Fallback for text
                        text = await response.text()
                        return {"type": "SIGHT", "status": status, "content": text[:1000] + "...", "raw_url": url}
                else:
                    return {"type": "PAIN", "status": status, "error": "HTTP Error"}
                    
        except Exception as e:
            logger.error(f"  [BLINDNESS] Failed to see {url}: {e}")
            return {"type": "PAIN", "error": str(e)}

    async def safe_search_wiki(self, query: str) -> str:
        """
        Safe/Mockable search to avoid massive scraping.
        Uses standard Wikipedia API if implemented, else mocks.
        """
        # For now, simplistic URL construction
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        result = await self.look_at(url)
        return result
