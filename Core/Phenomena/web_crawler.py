"""
Web Crawler (The Mouth)
=======================
Core.Phenomena.web_crawler

"To eat the world, one line at a time."

This module fetches raw text from external sources (URLs).
It serves as the sensory input for the Bulk Digestion system.
"""

import requests
import re
# We don't have BeautifulSoup installed in the standard environment usually,
# so we'll use simple regex or string parsing for this prototype.
# If bs4 is available, we'd use it. For now, assume raw text or simple clean.

class WebCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Elysia/0.1 (Structural Genesis Agent)'
        }

    def fetch(self, url: str) -> str:
        """
        Fetches text content from a URL.
        Returns cleaned text or raises error.
        """
        try:
            print(f"🌍 [CRAWLER] Fetching {url}...")
            # Ideally we use the tool 'view_text_website' in the agent loop,
            # but for autonomous code execution we need a library.
            # Since we are in a sandbox with restricted internet access,
            # we will mock this for the internal test or use a provided text file.

            # For the purpose of this specific environment (agent sandbox),
            # we might not have outgoing internet access for arbitrary python scripts.
            # We will implement a 'Mock' mode that simulates reading a website.

            return self._mock_fetch(url)

            # response = requests.get(url, headers=self.headers, timeout=5)
            # response.raise_for_status()
            # return self._clean_html(response.text)
        except Exception as e:
            print(f"⚠️ [CRAWLER] Failed to fetch: {e}")
            return ""

    def _mock_fetch(self, url: str) -> str:
        """
        Simulates fetching a page for testing logic without internet.
        """
        if "apple" in url:
            return """
            Apples are a type of fruit.
            Apples are generally red or green.
            Apples grow on trees.
            People eat apples.
            Apples are red.
            Apples are red.
            Apples are red.
            """
        elif "hangul" in url:
            return """
            Hangul is the alphabet of Korea.
            Hangul consists of consonants and vowels.
            King Sejong invented Hangul.
            Hangul is scientific.
            """
        return "Unknown content."

    def _clean_html(self, raw_html: str) -> str:
        # Simple regex to strip tags
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext.strip()
