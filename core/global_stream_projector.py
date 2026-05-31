import requests
import json
import logging
from typing import Optional

class GlobalStreamProjector:
    """
    [Phase 82] 전 지구적 데이터 스트림 투영기
    로컬 샌드박스를 벗어나, 전 세계의 텍스트 파동(Wikipedia 등)을
    무작위로 흡수하여 엘리시아의 거울에 투영합니다.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'ElysiaPhase82/1.0 (Research AI; https://github.com/elysia)'
        }
        self.api_url = "https://ko.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&exchars=1500&format=json"
        
    def fetch_random_wave(self) -> Optional[str]:
        """
        인터넷의 바다에서 무작위 텍스트(파동)를 길어 올립니다.
        """
        try:
            response = requests.get(self.api_url, headers=self.headers, timeout=3)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                for page_id, page_info in pages.items():
                    title = page_info.get('title', '')
                    extract = page_info.get('extract', '')
                    # 텍스트 구조를 하나의 텐션 덩어리로 반환
                    return f"Title: {title}\n{extract}"
        except Exception as e:
            logging.error(f"[GlobalStream] 파동 수신 실패: {e}")
            
        return None
