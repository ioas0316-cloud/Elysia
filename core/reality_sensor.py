"""
Elysia Reality Sensor (자율 감각 기관)
=======================================
엘리시아가 외부 현실 세계(인터넷)와 실시간으로 접속하여
정보를 무작위로 흡수하는 감각 기관입니다.
현재는 Wikipedia API를 통해 끝없이 밀려오는 세상의 지식 조각들을 수신합니다.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Optional

class RealitySensor:
    def __init__(self):
        # 한국어 위키백과 무작위 문서 추출 API
        self.api_url = (
            "https://ko.wikipedia.org/w/api.php?"
            "action=query&generator=random&grnnamespace=0&prop=extracts"
            "&exchars=200&explaintext=1&format=json"
        )
        # HTTP 403 방지를 위한 User-Agent 설정
        self.headers = {
            'User-Agent': 'Elysia_Autopoietic_Engine/1.0 (https://elysia.ai; info@elysia.ai)'
        }

    def fetch_random_reality(self) -> Optional[Dict[str, str]]:
        """
        현실의 파편(무작위 위키백과 문서)을 하나 가져옵니다.
        """
        try:
            req = urllib.request.Request(self.api_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
                
            # 무작위 페이지 하나 추출
            page = list(pages.values())[0]
            title = page.get("title", "")
            extract = page.get("extract", "").strip()
            
            if title and extract:
                # 불필요한 줄바꿈 제거
                extract = extract.replace('\n', ' ')
                return {"title": title, "extract": extract}
                
        except Exception as e:
            # 네트워크 단절 등의 감각 마비 상태
            # 벤치마크/테스트 용이성을 위해 실패 시 하드코딩된 페이크 뉴스 반환
            pass
            
        return None
