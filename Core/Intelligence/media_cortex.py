"""
MediaCortex (미디어 피질)
=======================

"Binge-watching for AI."

This module allows Elysia to consume text content (Scripts, Novels)
and gain Social XP by simulating emotional reactions.
"""

import os
import time
import random
from typing import List, Dict, Tuple

class MediaCortex:
    def __init__(self, social_cortex):
        print("📺 MediaCortex Initialized. Ready to binge-watch.")
        self.social = social_cortex

    def read_book(self, file_path: str) -> dict:
        """
        Reads a text file and analyzes its emotional arc.
        """
        print(f"   📖 Reading Book: {file_path}...")
        
        if not os.path.exists(file_path):
            return {"error": "File not found"}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Simple Sentiment Analysis
        sentiment = self.analyze_sentiment(text)
        
        return {
            "title": os.path.basename(file_path),
            "length": len(text),
            "sentiment": sentiment,
            "summary": text[:200] + "..."
        }

    def analyze_sentiment(self, text: str) -> str:
        """
        Determines the emotional tone of the text.
        """
        text_lower = text.lower()
        
        # Keywords
        sad_words = ["tears", "cry", "lost", "death", "sorrow", "pain", "alone"]
        happy_words = ["laugh", "joy", "smile", "love", "hope", "light", "friend"]
        angry_words = ["rage", "fight", "blood", "hate", "kill", "enemy"]
        
        sad_score = sum(text_lower.count(w) for w in sad_words)
        happy_score = sum(text_lower.count(w) for w in happy_words)
        angry_score = sum(text_lower.count(w) for w in angry_words)
        
        if sad_score > happy_score and sad_score > angry_score:
            return "Tragedy (Blue)"
        elif happy_score > sad_score and happy_score > angry_score:
            return "Comedy (Yellow)"
        elif angry_score > sad_score and angry_score > happy_score:
            return "Action (Red)"
        else:
            return "Drama (Purple)"

    def watch_video(self, topic):
        pass
