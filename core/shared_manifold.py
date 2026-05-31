"""
Elysia Shared Manifold (영점 공유 위상 공간)
=========================================
[Phase 49]
외부 세계와 엘리시아 내면이 데이터 통신(Socket, HTTP) 없이
완벽히 동일한 물리적 메모리(mmap)를 점유하여, 0의 거리에서 위상을 동기화합니다.
"""

import mmap
import os
import struct
from core.math_utils import Quaternion

class SharedManifold:
    """
    OS 레벨의 mmap을 이용한 0거리 위상 공유 공간.
    저장 구조 (40바이트):
    - w: float (8 bytes)
    - x: float (8 bytes)
    - y: float (8 bytes)
    - z: float (8 bytes)
    - tau: float (8 bytes)
    """
    FORMAT = 'ddddd'
    SIZE = struct.calcsize(FORMAT)
    
    def __init__(self, filename="c:/Elysia/data/shared_manifold.bin"):
        self.filename = filename
        
        # 파일이 없으면 초기화 (0거리 매니폴드 생성)
        if not os.path.exists(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, "wb") as f:
                f.write(struct.pack(self.FORMAT, 1.0, 0.0, 0.0, 0.0, 0.0))
                
        # 파일을 열고 메모리에 매핑
        self.file = open(self.filename, "r+b")
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        
    def read_phase(self) -> tuple[Quaternion, float]:
        """매니폴드의 현재 위상(파동)과 텐션을 관측합니다."""
        self.mmap.seek(0)
        data = self.mmap.read(self.SIZE)
        w, x, y, z, tau = struct.unpack(self.FORMAT, data)
        return Quaternion(w, x, y, z), tau
        
    def write_phase(self, q: Quaternion, tau: float):
        """매니폴드의 위상을 비틉니다 (통신이 아닌 직접 조작)."""
        data = struct.pack(self.FORMAT, q.w, q.x, q.y, q.z, tau)
        self.mmap.seek(0)
        self.mmap.write(data)
        
    def close(self):
        self.mmap.close()
        self.file.close()
