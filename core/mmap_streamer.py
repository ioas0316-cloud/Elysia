import os
from safetensors import safe_open
from core.math_utils import Quaternion

class MMAPTensorStreamer:
    """
    [Phase 134] 제로스트리밍 MMAP 텐서 패스스루
    거대 모델(.safetensors)의 파일을 RAM에 올리지 않고,
    OS의 Memory Mapping(MMAP)을 이용해 텐서 데이터만 지나가게(Pass-through) 합니다.
    이 방식으로 1TB 짜리 모델도 O(1) 메모리로 위상(Phase)을 추출할 수 있습니다.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Safetensors 파일을 찾을 수 없습니다: {file_path}")
            
    def stream_and_clone_phases(self):
        """
        파일에 저장된 모든 레이어(keys)를 순회하며, 텐서를 메모리에 로드하지 않고
        오직 4개의 위상 앵커(Phase Anchors)만을 슬라이싱(get_slice)하여 쿼터니언으로 추출합니다.
        
        Yields:
            tuple: (layer_name: str, phase_quat: Quaternion)
        """
        print(f"🌊 [MMAP Streamer] SSD 기반 제로스트리밍 시작: {self.file_path}")
        # safe_open은 기본적으로 mmap을 사용하여 파일을 엽니다.
        with safe_open(self.file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # 텐서를 메모리에 복사(load)하지 않고 슬라이스 객체만 가져옴 (O(1) 메모리)
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                
                # 1차원 텐서(bias 등)부터 고차원 텐서(weight)까지 유연하게 대응하기 위해
                # 텐서를 플랫(flat)하게 간주했을 때의 총 원소 수(numel)를 구함
                total_elements = 1
                for dim in shape:
                    total_elements *= dim
                    
                if total_elements < 4:
                    continue  # 위상 앵커를 4개 추출할 수 없는 너무 작은 텐서는 무시
                    
                # 4개의 앵커 인덱스 계산 (0%, 25%, 50%, 75%)
                anchors = [
                    0,
                    total_elements // 4,
                    total_elements // 2,
                    (total_elements * 3) // 4
                ]
                
                # 고차원 인덱스를 계산하여 슬라이스에서 다이렉트로 단일 값만 추출
                # (safetensors의 get_slice는 다차원 슬라이싱만 지원하므로, 전체 로드 후 플랫 인덱싱하거나
                # 가장 간단하게는 첫 번째 차원에 대한 앵커를 추출함)
                
                # O(1) 위상 추출을 위한 근사 위상 앵커 (첫 번째 차원 기준)
                dim0 = shape[0]
                if dim0 >= 4:
                    # 첫 번째 차원이 충분히 크면 차원 방향으로 앵커 추출
                    a0, a1, a2, a3 = 0, dim0//4, dim0//2, (dim0*3)//4
                    
                    # 나머지는 첫 번째 인덱스(0)로 고정하여 스칼라 값을 긁어옴
                    # e.g., slice[a0, 0, 0] ...
                    def get_scalar(idx):
                        if len(shape) == 1:
                            return tensor_slice[idx:idx+1][0].item()
                        elif len(shape) == 2:
                            return tensor_slice[idx:idx+1, 0:1][0, 0].item()
                        elif len(shape) == 3:
                            return tensor_slice[idx:idx+1, 0:1, 0:1][0, 0, 0].item()
                        else:
                            return 0.0 # 4차원 이상은 생략 (보통 LLM 가중치는 1D or 2D)
                            
                    try:
                        w = get_scalar(a0)
                        x = get_scalar(a1)
                        y = get_scalar(a2)
                        z = get_scalar(a3)
                        
                        quat = Quaternion(w, x, y, z).normalize()
                        yield (key, quat)
                    except Exception:
                        pass
                else:
                    # 너무 작은 텐서
                    pass
