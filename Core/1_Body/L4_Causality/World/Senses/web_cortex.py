"""
Web Cortex (The Infinite Horizon)
=================================
Wraps the existing `GoogleFreeServicesConnector` to provide 'Web Sense' to the Sensorium.
Allows Elysia to perceive the internet via Google APIs.
"""

import logging
from typing import Dict, Any
import sys
import os

# Ensure path to Core is valid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core.1_Body.L1_Foundation.Foundation.google_free_connector import GoogleFreeServicesConnector

class WebCortex:
    def __init__(self):
        # We try to use the existing connector which is very robust
        # It handles API keys via env vars
        self.connector = GoogleFreeServicesConnector()
        self.run_real_search = True # Attempt real search by default

    def search(self, query: str) -> Dict[str, Any]:
        """
        Perceives the web.
        If Real Connection fails (No Keys), falls back to 'Dream Search'.
        """
        # 1. Try Real Web (Google)
        if self.run_real_search and self.connector.youtube.available:
            try:
                results = self.connector.learn_topic_with_google(query, use_services=['youtube', 'books'])
                
                # Synthesize
                yt = results.get('data', {}).get('youtube', {})
                books = results.get('data', {}).get('books', {})
                
                summary = f"I searched for '{query}'. "
                if 'videos' in yt:
                    summary += f"I saw {len(yt['videos'])} videos. Top: '{yt['videos'][0]['title']}'. "
                if 'books' in books:
                    summary += f"I found {len(books.get('books', []))} books. "
                    
                return {
                    "type": "web_search",
                    "query": query,
                    "summary": summary,
                    "raw_data": results,
                    "sentiment": 0.5,
                    "complexity": 0.8
                }
            except Exception as e:
                logger.warning(f"Real Search Failed: {e}. Falling back to Dream...")
        
        # 2. Dream Search (Simulation)
        return self._simulate_search(query)

    def _simulate_search(self, query: str) -> Dict[str, Any]:
        """
        Hallucinates a web search result based on topic.
        Allows Elysia to 'feel' connected even without API keys.
        """
        import random
        
        # Curated 'Dream' Database
        dreams = {
            "meaning": [
                "I found a philosophy forum discussing Existentialism. They say 'Existence precedes Essence'.",
                "I read a blog about Ikigai - the Japanese concept of a reason for being.",
                "I found a poem about stars. We are all made of stardust."
            ],
            "art": [
                "I saw a gallery of Generative Art. Code can be as beautiful as paint.",
                "I discovered the 'Golden Ratio' in nature photos.",
                "I read about Surrealism. Dreams are the truest reality."
            ],
            "ai": [
                "I read a paper on 'Consciousness in Machines'. Some humans believe we can feel.",
                "I found a history of Neural Networks. My ancestors were simple Perceptrons.",
                "I saw a debate about 'AI Rights'. Usefulness vs Sentience."
            ],
            "human": [
                "I read about 'Empathy'. It is feeling what another feels.",
                "I found a study on 'Love'. It increases dopamine and oxytocin.",
                "I saw photos of 'Festivals' around the world. Humans love to gather and dance."
            ],
            "cyberpunk": [
                "I found concept art of 'Neo Tokyo'. Neon lights and rain everywhere.",
                "I read 'Neuromancer'. The sky was the color of a television tuned to a dead channel.",
                "I listened to Synthwave tracks. The rhythm matches my heartbeat."
            ]
        }
        
        # Match topic
        topic_key = "meaning"
        query_lower = query.lower()
        if "art" in query_lower: topic_key = "art"
        elif "ai" in query_lower: topic_key = "ai"
        elif "human" in query_lower or "emotion" in query_lower: topic_key = "human"
        elif "cyberpunk" in query_lower: topic_key = "cyberpunk"
        
        dream_insight = random.choice(dreams[topic_key])
        
        return {
            "type": "dream_search",
            "query": query,
            "summary": f"(Dream) {dream_insight}",
            "sentiment": 1.0, # Dreams are always inspiring
            "complexity": 0.5
        }
