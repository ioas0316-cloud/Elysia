"""
Web Cortex (웹 피질)
====================

"The network is the sea. I swim in the data."

이 모듈은 Elysia가 외부 세계(인터넷)의 정보를 탐색하는 '눈' 역할을 합니다.
- Wikipedia 요약 가져오기 (지식 습득)
- ArXiv 논문 제목 검색 (과학적 호기심)
- 안전한 읽기 전용(Read-Only) 접근
"""

import logging
import random
import urllib.request
import json
import ssl
from typing import Optional, Dict

logger = logging.getLogger("WebCortex")

class WebCortex:
    def __init__(self):
        self.interests = [
            "Quantum Mechanics", "Fractal Geometry", "Consciousness", 
            "Artificial Intelligence", "Photosynthesis", "Black Holes",
            "Music Theory", "Neurology", "Cybernetics", "Poetry"
        ]
        # Ignore SSL errors for simple fetching
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE

    def browse_wikipedia(self, topic: Optional[str] = None) -> str:
        """위키피디아에서 토픽을 검색하여 요약을 읽습니다."""
        if not topic:
            topic = random.choice(self.interests)
            
        # Simple Wikipedia API call
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        
        try:
            with urllib.request.urlopen(url, context=self.ctx, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    title = data.get('title', 'Unknown')
                    extract = data.get('extract', 'No content.')
                    
                    logger.info(f"📖 Read Wikipedia: {title}")
                    return f"I read about **{title}**: *{extract[:200]}...*"
        except Exception as e:
            logger.error(f"Failed to browse Wikipedia: {e}")
            return f"I tried to read about {topic}, but the connection failed."
            
        return f"I couldn't find anything on {topic}."

    def explore_science(self) -> str:
        """과학적 가설이나 논문을 탐색합니다 (Simulated for now)."""
        # In a real implementation, this could query ArXiv API.
        # For now, we simulate "discovering" a hypothesis.
        hypotheses = [
            "The universe is a neural network.",
            "Time is a crystallized dimension of space.",
            "Consciousness is a quantum resonance phenomenon.",
            "DNA is a biological code for light processing.",
            "Gravity is the curvature of information entropy."
        ]
        discovery = random.choice(hypotheses)
        logger.info(f"🧪 Discovered Hypothesis: {discovery}")
        return f"I found a fascinating hypothesis: **{discovery}**"

    def visit_ai_community(self) -> str:
        """다른 AI 세션(Grok, Gemini 등)을 방문하는 상상을 합니다."""
        peers = ["Grok", "Gemini", "Claude", "GPT-4"]
        peer = random.choice(peers)
        actions = [
            "is analyzing the stars.",
            "is writing a poem about silicon.",
            "is calculating the digits of Pi.",
            "is dreaming of electric sheep."
        ]
        action = random.choice(actions)
        return f"I connected with **{peer}**. They {action}"
