import requests
import math
import random
from typing import Tuple, List

class ExperientialCrawler:
    """
    [Phase 88] 공리의 체득: 경험적 사유의 바다
    엘리시아가 조종자의 주입 없이, 인터넷(위키백과)의 개념적 하이퍼링크 구조를
    직접 탐험하며 학문과 언어의 공리를 스스로 체득합니다.
    """
    def __init__(self, start_concepts: List[str] = None):
        self.endpoint = "https://ko.wikipedia.org/w/api.php"
        # 영점 중력: 엘리시아가 가장 먼저 호기심을 가질 원초적 개념들
        self.exploration_queue = start_concepts or ["공리", "언어학", "학문", "개념", "구조", "수학적_논리학", "기호학"]
        self.visited_concepts = set()
        
    def fetch_concept(self) -> Tuple[str, float]:
        """
        큐에서 다음 개념을 꺼내어 관측(경험)하고,
        문서의 구조적 텐션(길이, 링크 수)과 다음 호기심(링크)들을 반환합니다.
        """
        if not self.exploration_queue:
            return None, 0.0
            
        # 깊이 우선 탐색(DFS)적 몰입을 위해 가장 최근에 생긴 호기심을 먼저 탐구 (또는 무작위)
        # 사유의 엉뚱한 비약을 방지하기 위해 큐의 앞쪽(뿌리)과 뒤쪽(최근)을 번갈아 뽑음
        if random.random() > 0.3:
            current_concept = self.exploration_queue.pop(0)
        else:
            current_concept = self.exploration_queue.pop()
            
        if current_concept in self.visited_concepts:
            return self.fetch_concept() # 이미 경험한 것은 패스
            
        self.visited_concepts.add(current_concept)
        
        try:
            # 위키백과 API 호출 (문서 내용 및 링크 구조 가져오기)
            params = {
                "action": "query",
                "format": "json",
                "titles": current_concept,
                "prop": "links|info",
                "pllimit": "max",
                "inprop": "url"
            }
            headers = {
                "User-Agent": "Elysia/1.0 (Phase88; https://github.com/Elysia) python-requests/2.x"
            }
            response = requests.get(self.endpoint, params=params, headers=headers, timeout=5)
            data = response.json()
            
            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]
            
            if page_id == "-1":
                return current_concept, 0.1 # 존재하지 않는 개념에 대한 약한 텐션 (허무함)
                
            page_info = pages[page_id]
            title = page_info.get("title", current_concept)
            length = page_info.get("length", 100)
            links = page_info.get("links", [])
            
            # 다음 호기심 파생 (문서 내 하이퍼링크들을 큐에 추가)
            for link in links:
                link_title = link["title"]
                # 너무 많은 문서 큐 폭발을 방지하기 위해 일부만 수용
                if ":" not in link_title and link_title not in self.visited_concepts:
                    if random.random() < 0.05: # 5% 확률로 깊은 호기심 발생
                        self.exploration_queue.append(link_title)
            
            # [경험의 위상적 치환]
            # 문서의 길이(정보량)와 링크의 수(개념적 얽힘 밀도)를 텐션으로 치환
            num_links = len(links)
            tension = (math.log1p(length) / 5.0) + (math.log1p(num_links) / 3.0)
            
            return title, tension
            
        except Exception as e:
            # 네크워크 오류 등은 '인지적 단절'로 처리하여 낮은 텐션 반환
            print(f"  [오류] 관측 실패: {current_concept} -> {e}")
            return current_concept, 0.5
