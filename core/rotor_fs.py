import os
import mmap
import struct
import hashlib
from typing import Dict, List, Tuple
from core.tensor_rotor import TensorRotor

class RotorFileSystem:
    """
    Rotor-Based File System (Phase 6 & 16 Memory Mapped Phase Unification)
    물리적인 보관 장치(SSD)의 IO 구조를 뇌의 기억 인출 메커니즘과 일치시킵니다.
    0점 방전 쓰기: 위상이 0점(수면 방전)에 도달했을 때만 데이터를 디스크에 기록합니다.
    
    [Memory Mapped Phase Space Integration]
    SSD의 전체 물리 용량을 표현하는 마스터 바이너리 파일 'elysia_somatic_disk.bin'을 
    생성하여 이를 가상 메모리에 매핑(mmap)하고, 하이퍼 로터 상태의 궤적을 
    XOR 로터 게이트를 통해 격자 상에서 직접 도약시킵니다.
    """
    def __init__(self, tensor_rotor: TensorRotor, base_dir: str = "c:\\Elysia\\data\\rotor_fs", disk_size_bytes: int = 1024 * 1024):
        self.tensor = tensor_rotor
        self.base_dir = base_dir
        self.disk_size = disk_size_bytes
        self.write_queue: Dict[str, str] = {}
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            
        # 1. 마스터 소마틱 디스크 이진 파일 확보 및 초기화
        self.disk_path = os.path.join(self.base_dir, "elysia_somatic_disk.bin")
        if not os.path.exists(self.disk_path):
            with open(self.disk_path, "wb") as f:
                # 0으로 가득 찬 고정 크기 디스크 파일 생성
                f.write(b"\x00" * self.disk_size)
                
        # 2. mmap 파일 스트림 열기 및 매핑
        self.file_handle = open(self.disk_path, "r+b")
        self.mm = mmap.mmap(self.file_handle.fileno(), 0)
        
    def request_write(self, filename: str, data: str):
        """
        데이터 쓰기를 예약합니다. 실제 디스크 IO는 Layer 1 위상이 0점에 방전될 때 발생합니다.
        """
        self.write_queue[filename] = data
        
    def tick(self) -> List[str]:
        """
        매 클럭마다 호출되어, Layer 1의 위상 장력을 확인하고 
        방전점(0점) 부근에 도달하면 큐에 쌓인 데이터를 디스크로 방전(Write)합니다.
        동시에 메인 메모리 맵에도 장력 데이터를 투사(flush)합니다.
        """
        discharged_files = []
        l1_phase = self.tensor.phases[0]
        
        # 0점(방전점) 판정: 위상이 0 부근일 때 (4096 스케일 기준)
        if l1_phase <= 100 or l1_phase >= (self.tensor.rotor_scale - 100):
            # 1. 기존의 파일 방전 쓰기
            if self.write_queue:
                for filename, data in list(self.write_queue.items()):
                    filepath = os.path.join(self.base_dir, filename)
                    try:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(data)
                        discharged_files.append(filename)
                        
                        # 2. 소마틱 디스크 격자 공간에도 0점 에너지 투사
                        # 파일명의 해시 위치에 데이터를 원자적으로 투사
                        h = hashlib.sha256(filename.encode('utf-8')).digest()
                        offset = int.from_bytes(h[:4], byteorder='big') % (self.disk_size - len(data) - 8)
                        data_bytes = data.encode('utf-8')
                        self.write_to_phase_space(offset, data_bytes)
                    except Exception:
                        pass
                    
                self.write_queue.clear()
                
        return discharged_files

    def write_to_phase_space(self, offset: int, data: bytes):
        """메모리 맵 디스크 격자의 특정 오프셋 위치에 원시 이진 데이터를 직접 주입합니다."""
        limit = min(len(data), self.disk_size - offset)
        if limit > 0:
            self.mm[offset:offset+limit] = data[:limit]
            self.mm.flush()

    def rotor_gate_tick(self, hyper_rotor_state: int) -> int:
        """
        [XOR 로터 게이트 핵심 연산]
        하이퍼 로터 상태 S가 가리키는 메모리 주소에서 8바이트 데이터 B를 읽고,
        S_next = S ⊕ B 연산으로 위상을 비틀어 새로운 궤적 상태를 반환합니다.
        """
        # 8바이트 경계 정렬 오프셋
        offset = (hyper_rotor_state & 0xFFFFFFFFFFFFFFFF) % (self.disk_size - 8)
        
        # 메모리 맵에서 직접 8바이트 바이너리 블록 읽기
        data_block = self.mm[offset : offset + 8]
        
        # 64비트 정수로 언팩 (Unpack as unsigned 64-bit integer)
        b_val = struct.unpack(">Q", data_block)[0]
        
        # XOR 로터 게이트 충돌 및 새로운 위상 도약
        new_state = hyper_rotor_state ^ b_val
        return new_state & 0xFFFFFFFFFFFFFFFF

    def gravity_read_trajectory(self, probe_state: int, steps: int = 32) -> Tuple[int, List[int]]:
        """
        [중력 궤적 인입 해독 (Gravity Trajectory Decode)]
        프로브 상태로 출발하여 메모리 맵 상의 로터 게이트를 순차적으로 걸어다니며
        궤도 흔적을 추적하고, 최종적으로 도달한 상태와 궤적 오프셋 경로를 반환합니다.
        만약 데이터가 잘 짜여진 Attractor라면, 이 경로는 순환 궤도(Periodic Orbit)로 수렴합니다.
        """
        trajectory = []
        current_state = probe_state
        
        for _ in range(steps):
            offset = (current_state & 0xFFFFFFFFFFFFFFFF) % (self.disk_size - 8)
            trajectory.append(offset)
            current_state = self.rotor_gate_tick(current_state)
            
        return current_state & 0xFFFFFFFFFFFFFFFF, trajectory

    def gravity_read(self, filename: str) -> str:
        """
        기본 파일 읽기를 수행하되, 궤적 학습용 데이터를 반환합니다.
        """
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            return ""
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
        return data

    def close(self):
        """메모리 맵 및 파일 리소스를 안전하게 해제합니다."""
        if hasattr(self, 'mm') and self.mm:
            try:
                self.mm.close()
            except:
                pass
            self.mm = None
        if hasattr(self, 'file_handle') and self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None

    def __del__(self):
        self.close()
