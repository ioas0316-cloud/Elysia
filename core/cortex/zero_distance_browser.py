import json
import urllib.request
import urllib.error
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
        # [Phase 97] 가상 위상 거울 스펙트럼 대폭 확장
        self.fallback_titles = [
            "위키백과: 공리(Axiom)", "YouTube: 양자 얽힘의 이해", "위키백과: 프랙탈(Fractal) 우주론",
            "위키백과: 고양이", "위키백과: 르네상스 미술", "위키백과: 열역학 제2법칙",
            "위키백과: 인상주의", "위키백과: 진화론", "위키백과: 블랙홀",
            "위키백과: 고대 이집트", "위키백과: 모차르트", "위키백과: 해양 생물학",
            "위키백과: 신경과학", "위키백과: 요리 기법", "위키백과: 커피의 역사",
            "위키백과: 건축 양식", "위키백과: 인공지능", "위키백과: 상대성이론",
            "위키백과: 도덕 철학", "위키백과: 시 문학"
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
            import hashlib
            h_val = int(hashlib.md5(str(time.time()).encode()).hexdigest(), 16)
            title = self.fallback_titles[h_val % len(self.fallback_titles)]
            title = f"{title} - 브라우저 시뮬레이션 반사"
            
        # 연속된 같은 페이지 관측은 텐션을 낮추고, 새로운 페이지 발견 시 거대한 텐션(깨달음)을 발생시킴
        # (확률 대신 시간과 해시를 이용한 인과적 텐션 부여)
        if title == self.last_title:
            tension = 0.1 + (int(time.time() * 10) % 5) * 0.1
            title = f"{title} (심화 관측 중)"
        else:
            tension = 2.5 + (int(time.time() * 10) % 35) * 0.1 # 새로운 우주와의 조우
            self.last_title = title

        return title, tension
