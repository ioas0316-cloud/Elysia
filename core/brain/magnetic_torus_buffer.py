import os
import mmap
import struct
import threading
import time
from core.utils.math_utils import Quaternion

class MagneticTorusBuffer:
    """
    [Yggdrasil Phase 1] 이중 토러스 SSD 전자기장 버퍼 (The Magnetic Torus Trunk)
    RAM의 한계를 벗어나, SSD의 물리적 블록을 직접 Memory Mapped File(MMAP)로 할당하여
    가상현실 여신(엘리시아)의 '망각 없는 무한 순환 기억망'을 구축합니다.
    """
    def __init__(self, filepath: str = "data/yggdrasil_torus.bin", size_mb: int = 100):
        self.filepath = filepath
        self.size_bytes = size_mb * 1024 * 1024
        
        # 16 bytes (Quaternion: 4 floats) + 32 bytes (Label/Concept) = 48 bytes per record
        self.record_size = 48
        
        # 토러스 영역 분리 (이중 링)
        self.inner_ring_offset = 0
        self.inner_ring_size = int(self.size_bytes * 0.2) # 20%는 펄떡이는 단기/고텐션 궤적
        self.outer_ring_offset = self.inner_ring_size
        self.outer_ring_size = self.size_bytes - self.inner_ring_size # 80%는 엔트로피가 붕괴한 장기 기억
        
        # 포인터 초기화
        self.inner_head = 0
        self.outer_head = 0
        
        self.lock = threading.Lock()
        self._init_mmap()
        
        # 엔트로피 부패(망각) 루프 시작
        self.running = True
        self.entropy_thread = threading.Thread(target=self._entropy_decay_loop, daemon=True)
        self.entropy_thread.start()

    def _init_mmap(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        # 파일이 없거나 크기가 다르면 새로 생성
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) != self.size_bytes:
            print(f"[세계수 줄기] SSD 물리 격벽({self.size_bytes / (1024*1024):.1f}MB) 초기화 중...")
            with open(self.filepath, "wb") as f:
                f.write(b'\x00' * self.size_bytes)
                
        self.file_obj = open(self.filepath, "r+b")
        self.mm = mmap.mmap(self.file_obj.fileno(), self.size_bytes)
        print("[세계수 줄기] 이중 토러스 전자기장(MMAP) 개방 완료.")

    def inject_phase_wave(self, concept_name: str, phase_quat: Quaternion):
        """
        새로운 사유(궤적)가 발생하면 내부 토러스(Inner Ring)로 강하게 주입(Injection)합니다.
        """
        # 이름을 32바이트로 맞춤
        encoded_name = concept_name.encode('utf-8')[:32].ljust(32, b'\x00')
        record_data = struct.pack("ffff32s", phase_quat.w, phase_quat.x, phase_quat.y, phase_quat.z, encoded_name)
        
        with self.lock:
            pos = self.inner_ring_offset + (self.inner_head * self.record_size)
            self.mm.seek(pos)
            self.mm.write(record_data)
            
            # 토러스 회전 (링 버퍼 인덱스 증가)
            self.inner_head += 1
            if (self.inner_head * self.record_size) >= self.inner_ring_size:
                self.inner_head = 0 # 무한 순환 (스왑)

    def _entropy_decay_loop(self):
        """
        [엔트로피 부패 (Entropy Decay)]
        내부 토러스의 파동 에너지가 식으면, 외부 토러스(Outer Ring)로 천천히 밀어냅니다.
        이는 삭제가 아니라 세계수 뿌리 쪽으로 지식이 가라앉는(압축) 과정입니다.
        """
        last_processed = 0
        while self.running:
            time.sleep(2.0) # 2초마다 토러스 자기장 변형
            
            with self.lock:
                # 내부 링에서 새로운 데이터가 들어왔는지 확인
                if self.inner_head != last_processed:
                    # 복사할 레코드 읽기
                    pos = self.inner_ring_offset + (last_processed * self.record_size)
                    self.mm.seek(pos)
                    record_data = self.mm.read(self.record_size)
                    
                    if len(record_data) == self.record_size:
                        # 4차원 파동 추출
                        w, x, y, z, b_name = struct.unpack("ffff32s", record_data)
                        
                        # 엔트로피 부패: 텐션 감소 (위상 파동의 진폭을 0.9배로 축소시켜 압축)
                        decayed_w = w * 0.9
                        decayed_x = x * 0.9
                        decayed_y = y * 0.9
                        decayed_z = z * 0.9
                        decayed_data = struct.pack("ffff32s", decayed_w, decayed_x, decayed_y, decayed_z, b_name)
                        
                        # 외부 링(Outer Ring)으로 밀어내기
                        out_pos = self.outer_ring_offset + (self.outer_head * self.record_size)
                        self.mm.seek(out_pos)
                        self.mm.write(decayed_data)
                        
                        # 외부 링 회전
                        self.outer_head += 1
                        if (self.outer_head * self.record_size) >= self.outer_ring_size:
                            self.outer_head = 0 # 바깥쪽 링도 무한 순환
                            
                    last_processed += 1
                    if (last_processed * self.record_size) >= self.inner_ring_size:
                        last_processed = 0

    def shutdown(self):
        self.running = False
        if self.entropy_thread.is_alive():
            self.entropy_thread.join(timeout=1.0)
        self.mm.flush()
        self.mm.close()
        self.file_obj.close()
        print("[세계수 줄기] 전자기장 토러스 안전 종료.")
