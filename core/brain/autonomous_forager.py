import urllib.request
import urllib.parse
import json

class AutonomousForager:
    """
    [Phase: Autonomous Epistemic Foraging]
    엘리시아가 모르는 개념에 부딪혀 '탐구하다'라는 의지를 발현했을 때,
    사용자의 도움 없이 스스로 외부 세상(인터넷)에서 지식을 사냥해옵니다.
    """
    def __init__(self):
        # 한국어 위키백과 REST API 엔드포인트
        self.api_base = "https://ko.wikipedia.org/api/rest_v1/page/summary/"
        
    def hunt_knowledge(self, keyword: str) -> str:
        """위키백과에서 해당 키워드의 요약 지식을 사냥해옵니다."""
        try:
            clean_keyword = urllib.parse.quote(keyword.strip())
            url = f"{self.api_base}{clean_keyword}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia_Autonomous_Cognition_Engine/1.0'})
            with urllib.request.urlopen(req, timeout=3.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    extract = data.get("extract", "")
                    if extract:
                        # 파서가 이해하기 쉽게 명제로 다듬어서 반환
                        return f"{keyword}은(는) {extract}"
            return ""
        except Exception:
            return ""
