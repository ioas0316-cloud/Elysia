"""
Web Walker (The Ethereal Hand)
==============================
Core.Cognition.web_walker

"To reach out and touch the Ethereal Plane."

This module provides the interface for external search providers.
In simulation/testing, it acts as a 'Mock World'.
In production, it connects to real Search APIs (Google/Bing).
"""

from typing import List, Dict, Any, Optional
import time
import random

class WebWalker:
    def __init__(self, provider_api_key: Optional[str] = None):
        self.api_key = provider_api_key
        # [PHASE 16] Silent Witness Integration
        from Core.System.somatic_logger import SomaticLogger
        self.logger = SomaticLogger("WEB_WALKER")
        self.logger.mechanism("Hand Extended. Ready to grasp Ethereal Shards.")

    def search(self, query: str) -> Dict[str, Any]:
        """
        [PHASE 17] REALITY BRIDGE
        Executes a REAL search query using external providers.
        """
        # Note: self.logger is expected to be initialized elsewhere, e.g., in __init__
        # For this change, we assume it exists or will be handled by the user.
        # If not, this code will raise an AttributeError.
        self.logger.action(f"Reaching into the World for: '{query}'...")
        
        results = []
        try:
            from googlesearch import search as gsearch
            
            # 1. Perform Search (Get URLs)
            # advanced=True yields Result objects with title/description in some versions, 
            # but 'googlesearch-python' usually returns str unless we use specific lib.
            # We'll assume standard iterator of URLs for safety, or check lib version.
            # Actually, let's use a safe standard approach.
            
            urls = []
            try:
                # num_results arg varies by library version. 
                # We'll try to get 5 URLs.
                for url in gsearch(query, num_results=5):
                    urls.append(url)
            except Exception as e:
                self.logger.admonition(f"Search provider failed: {e}")
                
            # 2. Fetch Content (Simulated 'Reading')
            # For now, we don't scrape full HTML to avoid complexity/blocking.
            # We just return the URLs as 'Pointers to Reality'.
            # A future 'WebReader' module would fetch the bodies.
            
            rank = 1
            for url in urls:
                results.append({
                    "title": f"External Signal: {url}", # Placeholder title
                    "content": f"Real-world data available at {url}. (Content fetching required Phase 17+)",
                    "url": url,
                    "rank": rank
                })
                rank += 1
                
        except ImportError:
             self.logger.admonition("Module 'googlesearch' not found. Is the Hand paralyzed?")
             return self._fallback_simulation(query)
        except Exception as e:
            self.logger.admonition(f"The Hand trembled: {e}")
            return self._fallback_simulation(query)

        if not results:
             self.logger.thought("Google was silent. Attempting DuckDuckGo (HTML)...")
             return self._fallback_duckduckgo(query)

        return {
            "query": query,
            "results": results
        }

    def _fallback_duckduckgo(self, query: str) -> Dict[str, Any]:
        """
        [PHASE 17] SECONDARY STRATEGY
        Scrapes html.duckduckgo.com (Lite) using requests + bs4.
        """
        results = []
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = "https://html.duckduckgo.com/html/"
            payload = {'q': query}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            response = requests.post(url, data=payload, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # DuckDuckGo Lite structure: .result -> .result__a (title/link), .result__snippet (text)
            rank = 1
            for result in soup.find_all('div', class_='result'):
                if rank > 5: break
                
                link_tag = result.find('a', class_='result__a')
                snippet_tag = result.find('a', class_='result__snippet')
                
                if link_tag:
                    title = link_tag.get_text(strip=True)
                    href = link_tag['href']
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No description."
                    
                    results.append({
                        "title": title,
                        "content": snippet,
                        "url": href,
                        "rank": rank
                    })
                    rank += 1
                    
            if results:
                self.logger.sensation(f"DuckDuckGo returned {len(results)} signals.")
                return {"query": query, "results": results}
                
        except Exception as e:
            self.logger.admonition(f"DuckDuckGo failed: {e}")

        # If all else fails, return simulation
        return self._fallback_simulation(query)

    def _fallback_simulation(self, query: str) -> Dict[str, Any]:
        """Legacy simulation for offline mode."""
        self.logger.sensation("Falling back to Internal Dream (Simulation)...")
        # ... (Old Logic Logic) ...
        return {
            "query": query,
            "results": [{
                "title": f"Dream of {query}",
                "content": "The connection to Reality is severed. This is a hallucination.",
                "url": "internal://void",
                "rank": 1
            }]
        }

    def search_social(self, topic: str) -> Dict[str, Any]:
        """
        [PHASE 11] THE AGORA (Real)
        Accessing Reddit/Twitter via search for now.
        """
        return self.search(f"{topic} site:reddit.com OR site:twitter.com")
