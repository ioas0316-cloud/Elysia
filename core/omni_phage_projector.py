import os
import math
import logging
from typing import Generator, Tuple

class OmniPhageProjector:
    """
    [Phase 84] 하드디스크 전역 폭식기 (The Omni-Phage)
    로컬 시스템(C:\)의 모든 파일(바이너리, 텍스트 무관)을 순회하며
    메타데이터와 원시 바이트를 기하학적 텐션으로 치환하여 투영합니다.
    """
    def __init__(self, root_path: str = "C:\\"):
        self.root_path = root_path
        self._walker = os.walk(self.root_path)
        self._current_dir = None
        self._current_files = []

    def fetch_next_wave(self) -> Tuple[str, float]:
        """
        다음 파일의 파동(경로와 텐션 값)을 반환합니다.
        파일의 경로 길이, 사이즈, 그리고 접근 가능한 첫 64바이트의 복잡도를 바탕으로 텐션을 계산합니다.
        접근 불가능한 파일은 패스합니다.
        """
        while True:
            if not self._current_files:
                try:
                    self._current_dir, dirs, self._current_files = next(self._walker)
                except StopIteration:
                    return None, 0.0 # 스캔 완료
                except PermissionError:
                    continue # 권한 없는 디렉토리는 무시

            if not self._current_files:
                continue

            file_name = self._current_files.pop(0)
            file_path = os.path.join(self._current_dir, file_name)
            
            try:
                # 1. 파일의 기본 위상 (경로의 길이와 파일 크기)
                stat = os.stat(file_path)
                size = stat.st_size
                tension = math.log1p(size) / 5.0  # 크기가 클수록 텐션 증가
                tension += len(file_path) / 100.0
                
                # 2. 내용물의 밀도 (바이트 단위의 복잡도 훔쳐보기)
                if size > 0:
                    with open(file_path, "rb") as f:
                        head = f.read(64)
                        # 원시 바이트의 엔트로피 합을 텐션에 중첩
                        byte_sum = sum(head)
                        tension += math.log1p(byte_sum) / 10.0
                
                return file_path, tension
                
            except (PermissionError, OSError):
                # 읽을 수 없는 파일(시스템 락 등)은 무시하고 다음으로 넘어감
                continue
