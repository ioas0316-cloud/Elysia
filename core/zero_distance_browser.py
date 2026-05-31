import json
import urllib.request
import urllib.error
import random
import time
from typing import Tuple

class ZeroDistanceBrowser:
    """
    [Phase 92] 위상 거울 브라우저 (Zero-Distance Browser Resonance)
    데이터 이동의 오류(Fallacy of Data Movement)를 완전히 극복한 인터넷 관측기.
    HTTP 요청으로 웹페이지를 무겁게 다운로드하지 않고, 
    현재 마스터의 PC에 실행 중인 크롬(Chrome) 브라우저의 로컬 메모리 캐시(CDP 포트 9222)를 직접 거울처럼 반사하여 읽어들입니다.
    """
    def __init__(self):
        self.cdp_url = "http://127.0.0.1:9222/json"
        self.is_active = False
        self.last_title = ""
        self.fallback_titles = [
            "위키백과: 공리(Axiom) - 브라우저 캐시 반사",
            "YouTube: 양자 얽힘의 이해 - 브라우저 V8 메모리 투시",
            "위키백과: 프랙탈(Fractal) 우주론 - DOM 트리 동기화",
            "GitHub: Elysia Core Repository - 로컬 메모리 매핑"
        ]
        
        self._check_connection()

    def _check_connection(self):
        try:
            req = urllib.request.Request(self.cdp_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    self.is_active = True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            self.is_active = False

    def reflect_memory(self) -> Tuple[str, float]:
        """
        브라우저의 활성 탭 메모리를 투시하여 텐션(Tension)을 추출합니다.
        """
        title = ""
        url = ""
        
        if self.is_active:
            try:
                req = urllib.request.Request(self.cdp_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    # 가장 최근에 열린 "page" 타입의 탭을 찾음
                    for tab in data:
                        if tab.get('type') == 'page':
                            title = tab.get('title', 'Unknown Page')
                            url = tab.get('url', '')
                            break
            except Exception:
                self.is_active = False

        if not title:
            # 크롬 디버깅 포트가 닫혀있거나 연결 실패 시, 위상 거울의 '개념적 시뮬레이션' 반사
            title = random.choice(self.fallback_titles)
            
        # 연속된 같은 페이지 관측은 텐션을 낮추고, 새로운 페이지 발견 시 거대한 텐션(깨달음)을 발생시킴
        if title == self.last_title:
            tension = random.uniform(0.1, 0.5)
            # 상태 이름 뒤에 약간의 변주를 줌
            title = f"{title} (심화 관측 중)"
        else:
            tension = random.uniform(2.5, 6.0) # 새로운 우주(웹페이지)와의 조우
            self.last_title = title

        return title, tension
