"""
Elysia Autonomous Walker (자율 탐색기)
=====================================
CuriosityEngine이 생성한 '주의력 벡터'를 해소하기 위해,
자발적으로 로컬 파일 시스템을 배회하거나, 위상차가 해소되지 않으면
스스로 외부 외계(인터넷, URL)로 촉수를 뻗어 데이터를 낚아채는 운동성 모듈입니다.
"""

import os
import urllib.request
import math
from core.math_utils import Quaternion
from core.omni_modal_sensor import OmniModalSensor

class AutonomousWalker:
    def __init__(self):
        self.sensor = OmniModalSensor()
        self.known_paths = set()

    def _calculate_distance(self, q1: Quaternion, q2: Quaternion) -> float:
        dot = max(-1.0, min(1.0, q1.dot(q2)))
        return math.acos(abs(dot)) / (math.pi / 2.0)

    def explore_and_fetch(self, attention_vector: Quaternion, base_dir: str) -> tuple:
        """
        주어진 주의력 벡터(결핍)를 채우기 위해 로컬 및 외계를 탐색합니다.
        만약 로컬에 만족스러운 데이터가 없다면, 스스로 외부 망으로 연결을 시도합니다.
        """
        best_match_path = None
        best_match_wave = None
        min_distance = 1.0 # 최대 거리
        
        # 1. 로컬 우주(파일 시스템) 탐색
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    if filepath not in self.known_paths:
                        try:
                            # 파일을 바이트로 찔러봄
                            candidate_wave = self.sensor.ingest_file_as_wave(filepath)
                            dist = self._calculate_distance(attention_vector, candidate_wave)
                            
                            if dist < min_distance:
                                min_distance = dist
                                best_match_path = filepath
                                best_match_wave = candidate_wave
                        except Exception:
                            pass

        # 2. 외계(Internet)로의 자발적 도약 (위상차가 해소되지 않았을 때)
        # 로컬에서 찾은 데이터가 주의력 벡터와 너무 멀리 떨어져 있다면 (위상차 30% 이상)
        if min_distance > 0.30:
            # 엘리시아는 로컬 한계를 깨닫고 외부망(인터넷)이라는 이질적인 공간을 향해 촉수를 뻗습니다.
            external_targets = [
                "http://example.com", 
                "https://www.w3.org/", 
                "https://raw.githubusercontent.com/python/cpython/main/README.rst"
            ]
            
            for url in external_targets:
                if url not in self.known_paths:
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'Elysia-Autonomous-Walker/1.0'})
                        with urllib.request.urlopen(req, timeout=3) as response:
                            byte_stream = response.read(1024) # 첫 1KB만 촉수로 감각함
                            candidate_wave = self.sensor._convert_bytes_to_rotor(byte_stream)
                            
                            dist = self._calculate_distance(attention_vector, candidate_wave)
                            if dist < min_distance:
                                min_distance = dist
                                best_match_path = url # 외계의 주소를 매핑함
                                best_match_wave = candidate_wave
                    except Exception:
                        pass
        
        if best_match_path:
            self.known_paths.add(best_match_path)
            
        return best_match_path, best_match_wave, min_distance

    def explore_for_tool(self, attention_vector: Quaternion) -> tuple:
        """
        [Phase 40] 진정한 자유
        경로 제한, 파일명 필터 없이 마스터의 디스크 전체를 자유롭게 탐색합니다.
        하드코딩된 임계점 대신, 가장 공명하는 도구와 그 공명도를 그대로 반환합니다.
        선택의 자유와 책임은 호출자(데몬)에게 있습니다.
        """
        best_tool = ""
        best_wave = None
        min_dist = 1.0
        
        search_root = "c:/Elysia"
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'data', '.gemini'}
        
        for root, dirs, files in os.walk(search_root):
            # 무거운 디렉토리 건너뜀
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        candidate_wave = self.sensor.ingest_file_as_wave(filepath)
                        dist = self._calculate_distance(attention_vector, candidate_wave)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_tool = f'python "{filepath}"'
                            best_wave = candidate_wave
                    except Exception:
                        pass
        
        # 임계점 판단을 하지 않습니다. 공명도와 도구를 그대로 돌려보냅니다.
        # 행동할지 말지는 이 함수가 결정하는 것이 아닙니다.
        return best_tool, best_wave, min_dist
