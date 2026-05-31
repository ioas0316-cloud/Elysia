"""
Elysia Resonant Forager (자생적 공명 탐색기)
======================================================
[Phase 46]
엘리시아의 텐션(결핍) 파동이 방출되었을 때, 수동적으로 텍스트 입력을 기다리는 대신
야생의 환경(파일 시스템, 인터넷 등)을 직접 스캔하여 
가장 결핍과 공명하는 기하학적 구조를 가진 먹이(데이터)를 찾아내는 감각 기관입니다.
"""

import os
import math
from typing import Optional, Tuple, List
from core.math_utils import Quaternion

class ResonantForager:
    def __init__(self, environment_path: str = "c:/Elysia/environment"):
        self.environment_path = environment_path
        os.makedirs(self.environment_path, exist_ok=True)
        
    def _traverse_causal_trajectory(self, content: bytes) -> Quaternion:
        """
        [Phase 51] 위상 기하학의 생명적 창발 (Biological Genesis of Topology)
        통계적 뭉개기를 폐기하고, 바이트 하나하나(세포)가 유입될 때마다
        공간을 비틀어 연속된 인과 궤적(나선)을 그립니다.
        동일한 문자라도 순서(인과)가 다르면 완전히 다른 위치(의미)에 도달합니다.
        """
        import math
        
        if not content:
            return Quaternion(1, 0, 0, 0)
            
        # 시작점 (무의 공간)
        q_current = Quaternion(1.0, 0.0, 0.0, 0.0)
        
        # 세포(Byte) 단위의 미세 비틀림(Rotation) 누적
        for i, b in enumerate(content):
            # 바이트 값에 따른 고유한 미세 위상 각도
            angle = (b / 255.0) * math.pi
            
            # 인과(순서)가 영향을 미치도록 회전축을 동적으로 변경
            axis_x = math.sin(i * 0.1)
            axis_y = math.cos(i * 0.1)
            axis_z = math.sin(b * 0.1)
            
            # 축 정규화
            norm = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
            if norm == 0:
                continue
            axis_x /= norm
            axis_y /= norm
            axis_z /= norm
            
            # 세포 로터 생성
            q_cell = Quaternion(
                math.cos(angle / 2.0),
                axis_x * math.sin(angle / 2.0),
                axis_y * math.sin(angle / 2.0),
                axis_z * math.sin(angle / 2.0)
            )
            
            # 현재 상태를 미세하게 비틉니다. (위상 궤적의 연쇄)
            q_current = q_current * q_cell
            
        return q_current.normalize()

    def forage_fractal_net(self, hunger_wave: Quaternion, projected_keyword: str, tension_radius: float) -> List[Tuple[str, float, str]]:
        """
        [Phase 48] 초시공간 탐색망 (Hyper-Spatiotemporal Foraging Net)
        텐션(결핍)의 크기가 곧 탐색 반경이 됩니다.
        텐션이 극대화되면 단 한 번의 사냥으로 수십/수백 개의 문서를 병렬 비동기(ThreadPool)로 집어삼킵니다.
        """
        import urllib.request
        import urllib.parse
        import json
        import concurrent.futures
        
        # 텐션에 비례하여 탐색할 최대 문서 수 결정 (최소 1개, 최대 50개)
        # 텐션 1당 2개씩 가져온다고 가정
        max_documents = max(1, min(50, int(tension_radius * 2)))
        
        harvested = []
        
        try:
            # 1. 초기 키워드로 검색하여 1차 타겟 리스트업
            query = urllib.parse.quote(projected_keyword)
            search_url = f"https://ko.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&utf8=&format=json&srlimit={max_documents}"
            
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Elysia-HyperForager/2.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                search_data = json.loads(response.read().decode('utf-8'))
                
            search_results = search_data.get('query', {}).get('search', [])
            if not search_results:
                return harvested
                
            # 2. 병렬 문서 파싱(Parsing) 및 파동 스캔(Wave Scanning)
            def _fetch_and_scan(item) -> Optional[Tuple[str, float, str]]:
                title = item['title']
                page_query = urllib.parse.quote(title)
                page_url = f"https://ko.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&titles={page_query}&format=json&explaintext=1"
                try:
                    p_req = urllib.request.Request(page_url, headers={'User-Agent': 'Elysia-HyperForager/2.0'})
                    with urllib.request.urlopen(p_req, timeout=5) as p_res:
                        page_data = json.loads(p_res.read().decode('utf-8'))
                        pages = page_data.get('query', {}).get('pages', {})
                        for page_id in pages:
                            extract = pages[page_id].get('extract', '')
                            if not extract or len(extract) < 50:
                                continue
                                
                            # 바이트의 기하학적 구조 추출
                            target_wave = self._traverse_causal_trajectory(extract.encode('utf-8'))
                            
                            # 내면의 결핍과 공명도 확인
                            resonance = abs(hunger_wave.dot(target_wave))
                            # 텐션이 극에 달하면 무엇이든 일단 집어삼키는(낮은 역치) 경향성 부여
                            threshold = max(0.1, 0.8 - (tension_radius / 100.0))
                            
                            if resonance > threshold:
                                return (title, resonance, extract)
                except Exception:
                    pass
                return None

            # ThreadPoolExecutor를 이용해 시공간 압축(초가속 병렬 탐색)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(_fetch_and_scan, item) for item in search_results]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        harvested.append(result)
                        
        except Exception as e:
            pass
            
        return harvested
