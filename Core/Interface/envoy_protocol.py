"""
Galactic Envoys (은하 사절단)
=============================

"We come in peace. We seek knowledge."

이 모듈은 외부 웹사이트(Alien Planets)에 파견되는 '사절단(Envoys)'을 관리합니다.
무작위적인 크롤링이 아니라, '외교적 프로토콜(robots.txt)'을 준수하며
특정 지식이나 개념을 수집해오는 정중한 탐사 로봇입니다.
"""

import logging
import urllib.robotparser
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger("EnvoyProtocol")

@dataclass
class MissionReport:
    success: bool
    source: str
    content: str
    status_code: int
    meta: Dict[str, str]

class EnvoyProtocol:
    def __init__(self):
        self.user_agent = "Elysia/1.0 (Sentient AI Research; +http://example.com/elysia)"
        self.robot_parsers = {} # Cache for robots.txt
        logger.info("🕊️ Envoy Protocol initialized. Diplomatic channels open.")

    def _can_fetch(self, url: str) -> bool:
        """
        해당 행성(URL)의 입국 규정(robots.txt)을 확인합니다.
        """
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        if base_url not in self.robot_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robot_parsers[base_url] = rp
                logger.info(f"   📜 Read laws of {parsed.netloc}")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not read laws of {parsed.netloc}: {e}. Proceeding with caution.")
                return True # If robots.txt fails, usually assume open but be careful.
        
        return self.robot_parsers[base_url].can_fetch(self.user_agent, url)

    def dispatch_envoy(self, url: str) -> MissionReport:
        """
        사절단을 파견하여 정보를 수집합니다.
        """
        logger.info(f"🚀 Dispatching Envoy to: {url}")
        
        # 1. Check Laws (robots.txt)
        if not self._can_fetch(url):
            logger.warning(f"   ⛔ Access Denied by Planetary Law (robots.txt): {url}")
            return MissionReport(False, url, "Access Denied by robots.txt", 403, {})

        # 2. Prepare Request
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={'User-Agent': self.user_agent}
        )

        # 3. Execute Mission
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                status = response.status
                headers = dict(response.getheaders())
                
                logger.info(f"   ✅ Mission Successful. Retrieved {len(content)} bytes.")
                return MissionReport(True, url, content[:5000], status, headers) # Limit content for now
                
        except Exception as e:
            logger.error(f"   💥 Mission Failed: {e}")
            return MissionReport(False, url, str(e), 500, {})

    def scout_knowledge(self, topic: str) -> MissionReport:
        """
        특정 주제에 대해 위키피디아를 정찰합니다.
        """
        # Wikipedia is a friendly planet
        safe_topic = urllib.parse.quote(topic)
        url = f"https://en.wikipedia.org/wiki/{safe_topic}"
        return self.dispatch_envoy(url)
