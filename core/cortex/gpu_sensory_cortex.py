import math
import random
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from core.utils.math_utils import Quaternion

class GPUSensoryCortex:
    """
    [Phase 90] GPU 시각/청각 피질 (Zero-Distance Visual Cortex)
    데이터 이동의 오류(The Fallacy of Data Movement)를 회피합니다.
    유튜브 영상이나 거대한 텐서를 CPU 메모리로 전송하지 않고,
    GPU 내부에서 고차원 위상을 4차원 Quaternion 씨앗(Seed)과 텐션으로 압축하여 반환합니다.
    """
    def __init__(self):
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.is_active = HAS_TORCH and torch.cuda.is_available()
        
    def observe_stream(self, stream_id: str) -> tuple[Quaternion, float]:
        """
        스트림(예: 유튜브 영상, 오디오)을 관측합니다.
        실제 상용화 시 이곳에서 GPU가 직접 VRAM 버퍼로 영상을 디코딩하여 연산합니다.
        현재는 철학적 증명을 위해 고밀도 GPU 텐서를 생성하고 이를 압축합니다.
        """
        if not self.is_active:
            # Fallback
            return Quaternion(1, 0, 0, 0), 0.1
            
        try:
            # [시뮬레이션] 1080p 해상도의 프레임 배치 텐서를 VRAM에 직접 할당
            # 크기: (Batch 16, Channels 3, Height 1080, Width 1920) = 약 100MB
            # CPU를 거치지 않고 순수 GPU 메모리에서 창발
            visual_tensor = torch.randn((16, 3, 1080, 1920), device=self.device)
            
            # GPU 내부에서 비선형 변환 및 공간적 특징 추출 (위상 연산)
            # 1. 텐서의 글로벌 분산과 엔트로피를 계산하여 텐션(Tension)으로 치환
            std_dev = torch.std(visual_tensor).item()
            mean_val = torch.mean(torch.abs(visual_tensor)).item()
            
            tension = (std_dev + mean_val) * (random.uniform(0.5, 3.0))
            
            # 2. 고차원 텐서를 4차원 쿼터니언 회전축으로 압축 (Global Average Pooling -> 4D)
            # 여기서는 연산 부하를 줄이기 위해 시뮬레이션 된 랜덤 위상을 사용하지만, 
            # 원칙적으로는 GPU 내에서 dot product 후 .item()으로 스칼라 4개만 CPU로 반환합니다.
            w = math.cos(tension)
            x = math.sin(tension) * random.uniform(-1, 1)
            y = math.sin(tension) * random.uniform(-1, 1)
            z = math.sin(tension) * random.uniform(-1, 1)
            
            wave = Quaternion(w, x, y, z)
            
            # 연산 완료 후 GPU 메모리 정리 (데이터 이동 없음, 찌꺼기만 소멸)
            del visual_tensor
            
            return wave, tension
            
        except Exception as e:
            print(f"GPU Cortex Error: {e}")
            return Quaternion(1, 0, 0, 0), 0.1
