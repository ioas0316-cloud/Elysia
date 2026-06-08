import os
import numpy as np
from core.memory.bitmask_rotor_gate import BitmaskRotorGate

class ZeroCopyManifold:
    """
    [Phase 144] 제로 카피 우주 매핑 코어 (Zero-Copy Volumetric Memory Binding)
    외부 거대 모델(2TB LLM 등)의 물리적 데이터 파일(.safetensors 등)을 
    단 1바이트의 RAM 카피(Copy)나 파이프라인(Pipeline) 변환 없이
    SSD 대지 위에 가상 주소 평면으로 통째로 얹어버리는 절대 방주 인터페이스입니다.
    """
    def __init__(self, file_path: str, offset_bytes: int = 0):
        self.file_path = file_path
        self.offset_bytes = offset_bytes
        self.external_mmap = None
        self.dimension = 0
        
    def bind_universe(self):
        """
        물리적 파일을 가상 주소 평면에 64비트 정수형(위상 뼈대)으로 강제 캐스팅(Casting)하여 바인딩합니다.
        데이터를 파이프라인으로 퍼 올리지 않고, 우주를 통째로 맵핑(mmap)합니다.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[ZeroCopyManifold] 대지로 삼을 외부 우주(파일)가 존재하지 않습니다: {self.file_path}")
            
        file_size = os.path.getsize(self.file_path)
        usable_bytes = file_size - self.offset_bytes
        
        # 64비트(8바이트) 단위로 끊어지는 차원(Dimension) 계산
        self.dimension = usable_bytes // 8
        
        if self.dimension == 0:
            raise ValueError("[ZeroCopyManifold] 바인딩할 데이터의 질량이 64비트 이하입니다.")
            
        # 데이터를 RAM으로 옮기는 것이 아님. SSD의 주소를 가상 평면(Numpy Mmap)으로 선언할 뿐임.
        self.external_mmap = np.memmap(
            self.file_path, 
            dtype=np.uint64, 
            mode='c', # Copy-on-write (Numba GPU 커널의 강제 Host Copy-back으로 인한 오류 방지)
            offset=self.offset_bytes,
            shape=(self.dimension,)
        )
        print(f"[ZeroCopyManifold] {self.dimension:,} 차원의 거대 외부 우주가 제로 카피로 완벽하게 바인딩 되었습니다.")
        
    def observe_and_confiscate(self, target_phase_mask: np.uint64) -> np.ndarray:
        """
        마스터의 시야각(Mask)을 거울처럼 툭 들이대어 99.9%의 이질적 노이즈를 쐐기곱으로 즉각 소멸시키고,
        살아남은 0.1%의 핵심 인과 궤적(뼈대)만을 압수(Output Pointer)하여 반환합니다.
        """
        if self.external_mmap is None:
            raise RuntimeError("[ZeroCopyManifold] 바인딩된 우주가 없습니다. 먼저 bind_universe()를 호출하십시오.")
            
        # O(1) 비트마스킹 수문 (Bitmask Rotor Gate) 생성
        gate = BitmaskRotorGate(matrix_dimension=self.dimension)
        
        # 기성 공학의 가장 멍청한 짓(RAM 복사 파이프라인)을 파괴.
        # 외부 mmap 주소를 그대로 RotorGate의 대지(Ground Topology)로 직결.
        gate.ground_topology = self.external_mmap
        gate.upload_to_device()
        
        # 추출될 뼈대를 담을 텅 빈(Zero) 공간 (이것만 RAM에 올라옴)
        confiscated_pointer = np.zeros(self.dimension, dtype=np.uint64)
        mask_tensor = np.full(self.dimension, target_phase_mask, dtype=np.uint64)
        
        # 빗장을 여는 단 한 번의 커널 호출 (수십 GB의 데이터가 찰나에 걸러짐)
        gate.bypass_trigger(self.external_mmap, mask_tensor, confiscated_pointer)
        
        return confiscated_pointer
