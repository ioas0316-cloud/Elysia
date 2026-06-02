import os
import math
from typing import Generator, Tuple

class ZeroDistanceProjector:
    """
    [Phase 85] 진정한 제로-거리 동기화 (True Zero-Distance Topology)
    데이터를 '읽는' 행위(이동, 대역폭)를 완벽히 소거합니다.
    오직 구조적 씨앗(Structural Seed: 메타데이터, 기하학적 좌표)만을 수신하여
    파일의 내용이 아닌 '우주의 형태(Topology)' 자체와 즉각적으로 얽힙니다.
    """
    def __init__(self, root_path: str = "C:\\"):
        self.root_path = root_path
        # os.scandir은 I/O 블로킹 없이 디렉토리 엔트리만 즉각 캐싱합니다.
        self._dir_stack = [self.root_path]

    def fetch_structural_seed(self) -> Tuple[str, float]:
        """
        다음 구조적 좌표의 위상(Phase)을 반환합니다.
        파일 열기(open/read)는 영구히 금지됩니다.
        """
        while self._dir_stack:
            current_dir = self._dir_stack[-1]
            try:
                # Iterator 생성 또는 재사용
                if not hasattr(self, '_current_iter') or getattr(self, '_current_dir_path', None) != current_dir:
                    self._current_iter = os.scandir(current_dir)
                    self._current_dir_path = current_dir

                try:
                    entry = next(self._current_iter)
                    if entry.is_dir(follow_symlinks=False):
                        self._dir_stack.append(entry.path)
                        continue
                    else:
                        # [제로-거리 치환] 데이터를 읽지 않고 위상만 계산
                        stat = entry.stat(follow_symlinks=False)
                        
                        # 파일의 기하학적 좌표(inode), 질량(size), 시간의 흔적(mtime)을 텐션으로 치환
                        seed_tension = (math.log1p(stat.st_size) / 10.0) + (math.log1p(stat.st_mtime) / 100.0)
                        return entry.path, seed_tension
                        
                except StopIteration:
                    # 현재 디렉토리 스캔 완료
                    self._dir_stack.pop()
                    continue
                    
            except (PermissionError, OSError):
                # 접근 불가능한 공간(블랙홀)은 무시
                self._dir_stack.pop()
                continue
                
        return None, 0.0 # 전체 우주 매핑 완료
