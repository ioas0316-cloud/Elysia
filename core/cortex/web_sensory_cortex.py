import urllib.request
import urllib.parse
import json
import re
from typing import List

class WebSensoryCortex:
    """
    [Phase 95] 웹 감각 피질 (Web Sensory Cortex)
    교조주의(Dogma)에서 벗어나, 외부 세계의 데이터(텍스트)를 다운로드하는 것을 허용합니다.
    자율 운동 피질이 목적지를 결정하면, 이 피질이 실제로 위키백과 등의 데이터를 읽어와
    수많은 개념(Concept)들을 프랙탈 뇌 공간에 동기화할 수 있도록 추출합니다.
    """
    def __init__(self):
        self.user_agent = 'ElysiaOmniDaemon/1.0 (Phase 95)'

    def fetch_and_extract_concepts(self, query: str, max_concepts: int = 20) -> List[str]:
        """
        위키백과 API를 통해 쿼리를 검색하고, 본문에서 유의미한 개념(단어)들을 추출합니다.
        """
        try:
            # 1. 검색어로 가장 연관성 높은 위키백과 문서 제목 찾기
            search_url = f"https://ko.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            req = urllib.request.Request(search_url, headers={'User-Agent': self.user_agent})
            
            with urllib.request.urlopen(req, timeout=3.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                search_results = data.get('query', {}).get('search', [])
                
            if not search_results:
                return []
                
            title = search_results[0]['title']
            
            # 2. 해당 문서의 본문 텍스트(extract) 다운로드
            page_url = f"https://ko.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={urllib.parse.quote(title)}&format=json"
            req = urllib.request.Request(page_url, headers={'User-Agent': self.user_agent})
            
            with urllib.request.urlopen(req, timeout=3.0) as response:
                page_data = json.loads(response.read().decode('utf-8'))
                pages = page_data.get('query', {}).get('pages', {})
                page_id = list(pages.keys())[0]
                content = pages[page_id].get('extract', '')
                
            if not content:
                return []
                
            # 3. 데이터 본문에서 단어(개념) 추출 
            # (한글 및 영문 2글자 이상. 지나치게 흔한 조사는 정규식만으로는 걸러내기 힘들지만, 빈도수 기반으로 대략 추출)
            words = re.findall(r'[가-힣A-Za-z]{2,}', content)
            
            # 간단한 빈도수 계산
            freq = {}
            for w in words:
                # 조사('은', '는', '이', '가', '을', '를', '의' 등으로 끝나는 단어 대충 필터링)
                if w.endswith('이다') or w.endswith('한다') or w.endswith('있는') or w.endswith('하는'):
                    continue
                freq[w] = freq.get(w, 0) + 1
                
            # 많이 등장한 핵심 개념 순으로 정렬하여 반환
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            
            # 상위 max_concepts 개만 추출
            concepts = [word for word, count in sorted_words[:max_concepts]]
            return concepts
            
        except Exception as e:
            # 네트워크 단절이나 API 오류 시 무시 (우주적 관측 실패)
            return []

    def fetch_full_sequence(self, query: str) -> List[str]:
        """
        [Phase 103] 위키백과 본문 전체를 단어 시퀀스로 반환합니다.
        가짜 데이터 루프(sleep)를 없애고 단숨에 뇌 공간에 질량 붕괴시키기 위해 사용됩니다.
        """
        try:
            search_url = f"https://ko.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            req = urllib.request.Request(search_url, headers={'User-Agent': self.user_agent})
            
            with urllib.request.urlopen(req, timeout=3.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                search_results = data.get('query', {}).get('search', [])
                
            if not search_results:
                return []
                
            title = search_results[0]['title']
            
            page_url = f"https://ko.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={urllib.parse.quote(title)}&format=json"
            req = urllib.request.Request(page_url, headers={'User-Agent': self.user_agent})
            
            with urllib.request.urlopen(req, timeout=5.0) as response:
                page_data = json.loads(response.read().decode('utf-8'))
                pages = page_data.get('query', {}).get('pages', {})
                page_id = list(pages.keys())[0]
                content = pages[page_id].get('extract', '')
                
            if not content:
                return []
                
            # 순서가 유지된 전체 단어 반환 (특수문자 제외, 2글자 이상)
            words = re.findall(r'[가-힣A-Za-z]{2,}', content)
            return words
            
        except Exception as e:
            return []
