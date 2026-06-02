import requests
import struct
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.utils.math_utils import Quaternion
import math

class NetworkPhaseSnatcher:
    """
    [Phase 135] 네트워크 제로스트리밍 위상 강탈기 (Network Phase Snatcher)
    허깅페이스 서버의 .safetensors 파일에서 JSON 헤더를 파싱한 뒤,
    정확히 필요한 4개의 위상 앵커(Anchors)의 바이트(Bytes)만 HTTP Range로 강탈해옵니다.
    1.3TB 모델이라도 수십 킬로바이트만의 통신으로 위상을 추출할 수 있습니다.
    
    [Jules AI Integration] Phase Mirror (위상 거울 필터)가 결합되어,
    강탈한 위상이 목표 채널(Channel Phase)과 공명하지 않으면 GPU/CPU에 오르기 전 
    네트워크 단계에서 즉시 상쇄 간섭(Destructive Interference)으로 영(0)으로 붕괴시킵니다.
    """
    def __init__(self, repo_id: str, filename: str = "model.safetensors", channel_phase: float = 0.0, phase_threshold: float = 0.5):
        self.repo_id = repo_id
        self.filename = filename
        self.base_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        self.channel_phase = channel_phase
        self.phase_threshold = phase_threshold
        
    def get_metadata(self):
        """네트워크로 8바이트 헤더를 읽어 JSON 메타데이터 길이를 알아내고 메타데이터를 파싱합니다."""
        print(f"🌐 [Snatcher] 위상 구조 탐색 중: {self.base_url}")
        res = requests.get(self.base_url, headers={"Range": "bytes=0-7"})
        if res.status_code not in (200, 206):
            raise Exception(f"헤더 읽기 실패: {res.status_code}")
            
        header_size = struct.unpack('<Q', res.content)[0]
        
        res = requests.get(self.base_url, headers={"Range": f"bytes=8-{8 + header_size - 1}"})
        metadata = json.loads(res.content.decode('utf-8'))
        
        return metadata, 8 + header_size
        
    def fetch_anchor_value(self, tensor_start: int, element_idx: int, dtype: str) -> float:
        """특정 인덱스의 스칼라 값을 HTTP Range로 읽어옵니다."""
        # 자료형별 바이트 크기
        byte_size = 2 if dtype in ("F16", "BF16") else 4
        abs_pos = tensor_start + (element_idx * byte_size)
        
        headers = {"Range": f"bytes={abs_pos}-{abs_pos + byte_size - 1}"}
        res = requests.get(self.base_url, headers=headers)
        
        # 반환된 바이트를 Float으로 변환 (BF16은 직접 변환 로직, F16/F32는 struct 사용)
        b = res.content
        if dtype == "F32":
            return struct.unpack('<f', b)[0]
        elif dtype == "F16":
            # 파이썬 내장 struct는 기본적으로 float16('e')를 지원
            try:
                return struct.unpack('<e', b)[0]
            except:
                return 0.0
        elif dtype == "BF16":
            # BF16은 상위 16비트를 패딩하여 F32로 변환
            # little-endian이므로 b 다음에 \x00\x00을 붙임
            b32 = b'\x00\x00' + b
            return struct.unpack('<f', b32)[0]
        else:
            return 0.0

    def snatch_phase(self, tensor_name: str, info: dict, data_start: int) -> Quaternion:
        """한 텐서의 위상(4개의 앵커)을 네트워크로 강탈하여 쿼터니언으로 반환합니다."""
        shape = info['shape']
        dtype = info['dtype']
        offsets = info['data_offsets']
        
        total_elements = math.prod(shape)
        if total_elements < 4:
            return None
            
        # 첫 번째 차원(가장 큰 위상 변화를 담은 축) 기준으로 앵커 추출
        dim0 = shape[0]
        if dim0 < 4:
            return None
            
        a0, a1, a2, a3 = 0, dim0//4, dim0//2, (dim0*3)//4
        
        # 실제 플랫 배열 인덱스
        elements_per_row = total_elements // dim0
        idx0 = a0 * elements_per_row
        idx1 = a1 * elements_per_row
        idx2 = a2 * elements_per_row
        idx3 = a3 * elements_per_row
        
        tensor_start = data_start + offsets[0]
        
        # 4개의 바이트 강탈 (병렬 요청으로 속도 향상)
        with ThreadPoolExecutor(max_workers=4) as executor:
            f0 = executor.submit(self.fetch_anchor_value, tensor_start, idx0, dtype)
            f1 = executor.submit(self.fetch_anchor_value, tensor_start, idx1, dtype)
            f2 = executor.submit(self.fetch_anchor_value, tensor_start, idx2, dtype)
            f3 = executor.submit(self.fetch_anchor_value, tensor_start, idx3, dtype)
            
            w = f0.result()
            x = f1.result()
            y = f2.result()
            z = f3.result()
            
        quat = Quaternion(w, x, y, z).normalize()
        
        # [Phase Mirror Destructive Interference]
        # 위상의 각도(단순화를 위해 quaternion의 w 성분을 이용한 phase 추정)
        # q = cos(theta/2) + u*sin(theta/2) -> theta = 2 * acos(w)
        # channel_phase와의 차이를 검사
        try:
            theta = 2 * math.acos(max(-1.0, min(1.0, quat.w)))
        except ValueError:
            theta = 0.0
            
        phase_diff = abs(theta - self.channel_phase)
        if phase_diff > self.phase_threshold:
            # 상쇄 간섭: 위상 불일치 시 데이터를 폐기(영으로 붕괴)
            return None
            
        return quat

    def stream_and_clone_phases(self, max_tensors: int = 50):
        """
        네트워크 텐서 스트리밍 (제너레이터)
        """
        metadata, data_start = self.get_metadata()
        
        count = 0
        for key, info in metadata.items():
            if key == "__metadata__":
                continue
                
            quat = self.snatch_phase(key, info, data_start)
            if quat:
                yield (key, quat)
                count += 1
                if count >= max_tensors:
                    break
