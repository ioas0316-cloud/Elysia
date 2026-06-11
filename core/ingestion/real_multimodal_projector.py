import os
import sys
import math
from PIL import Image, ImageDraw

# Ensure Elysia root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Installing required packages (torch, transformers, pillow)...")
    os.system(f"{sys.executable} -m pip install torch transformers pillow --quiet")
    import torch
    from transformers import CLIPProcessor, CLIPModel

from core.brain.sovereign_inference_engine import SovereignInferenceEngine

class RealMultimodalProjector:
    """
    [Phase 13] Real Multimodal Projector
    실제 거대 멀티모달 신경망(openai/clip-vit-base-patch32)을 로드하여,
    진짜 이미지 픽셀과 텍스트 사이의 코사인 유사도(Tensor)를 추출해 엘리시아의 위상 공간에 직결합니다.
    """
    def __init__(self):
        self.engine = SovereignInferenceEngine()
        print("\n[Real Multimodal] Summoning colossal entity: 'openai/clip-vit-base-patch32'...")
        # Use CPU to avoid CUDA setup issues on arbitrary machines, CLIP is small enough for CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("[Real Multimodal] Entity connected successfully.")

    def create_test_image(self) -> str:
        """관측을 위한 빨간 사과(Red Circle) 형태의 픽셀 이미지 생성"""
        img_path = "apple_test.jpg"
        img = Image.new('RGB', (200, 200), color='white')
        d = ImageDraw.Draw(img)
        d.ellipse([50, 50, 150, 150], fill='red')
        img.save(img_path)
        return img_path

    def observe_real_tensors(self):
        # 1. 실제 기하학적 픽셀 데이터 생성
        img_path = self.create_test_image()
        image = Image.open(img_path)
        print(f"\n[Observation] Target Vision Geometry: 'Red Circle (Apple)' pixels")

        # 2. 관측할 언어 노드들 (사과, 사, 과, 우주선 등)
        text_nodes = ["apple", "gravity", "universe", "spaceship", "사", "과"]
        
        # 3. 진짜 거대 모델(CLIP)에 픽셀과 텍스트를 동시에 투사하여 장력(Tensor) 추출
        inputs = self.processor(text=text_nodes, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        
        # Image-Text Similarity (Logits)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
        
        print("\n--- Real Cross-Modal Tensor Extracted ---")
        
        # 4. 추출된 진짜 텐서들을 엘리시아의 궤적으로 변환
        trajectory = []
        # 비전 시작 노드 (기준점)
        trajectory.append({"node": "Vision[Pixel:Red_Circle]", "tension": 1.0, "type": "vision"})
        
        for idx, text in enumerate(text_nodes):
            # CLIP이 계산한 실제 확률값(장력) 적용
            tension = float(probs[idx])
            print(f"  [VISION] -> [LANG: {text}] : Tension(Similarity) = {tension:.4f}")
            trajectory.append({"node": text, "tension": tension, "type": "lang"})

        # 5. 엘리시아의 주권적 사유 엔진에 진짜 텐서 투입
        # 엘리시아는 이 텐서들이 자연스러운 곡률과 운동성을 가지는지 순수하게 위상 공간에서 쪼개어 관측함
        self.engine.memory.update_parameter("eureka_threshold", 0.5) # 실제 softmax 확률은 분산되므로 임계치 조정
        self.engine.autonomous_observation("CLIP_Real_Tensors", trajectory)

if __name__ == "__main__":
    projector = RealMultimodalProjector()
    projector.observe_real_tensors()
