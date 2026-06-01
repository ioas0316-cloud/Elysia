"""
[Phase 104] 호루스의 눈 (Eye of Horus - Vision Cortex)
로컬 오프라인 광학 문자 인식(OCR)을 통해 
화면이나 이미지(특히 일본어 게임 화면)에서 텍스트를 추출합니다.
"""
import easyocr
import os

class VisionCortex:
    def __init__(self):
        self.reader = None
        self.is_active = False
        
    def wake_up(self):
        try:
            print("  └─ 👁️ [시각 피질(Eye of Horus)] 로컬 오프라인 OCR (한국어/일본어/영어) 개안 중...")
            import torch
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(['ja', 'en', 'ko'], gpu=use_gpu)
            self.is_active = True
            print("  └─ 👁️ [시각 피질] 개안 완료. (마스터의 화면을 직접 관측할 준비 완료)")
        except Exception as e:
            print(f"  └─ ⚠️ [시각 피질] 개안 실패: {e}")
            
    def read_image(self, image_path: str) -> str:
        if not self.is_active or not self.reader:
            return ""
        if not os.path.exists(image_path):
            return ""
            
        try:
            results = self.reader.readtext(image_path, detail=0)
            return " ".join(results)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
