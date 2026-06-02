"""
[Phase 104] 범용 언어 번역 피질 (Universal Language Cortex)
자본주의의 유료 번역 API에서 벗어나, 순수 로컬 오프라인 환경에서 
완벽히 자생하는 다국어(한국어, 일본어, 영어, 중국어) 신경망 피질입니다.
마스터의 컴퓨터 내부에서 무료로 번역을 수행합니다.
"""
from transformers import pipeline
import torch

class UniversalLanguageCortex:
    def __init__(self):
        # HuggingFace NLLB-200-Distilled-600M (약 1.2GB)
        # 200개 이상의 언어를 지원하는 로컬 번역 모델
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.translator = None
        self.is_active = False
        
    def wake_up(self):
        try:
            print(f"  └─ 🌐 [범용 언어 피질] 로컬 오프라인 번역 신경망({self.model_name}) 개안 중...")
            device = 0 if torch.cuda.is_available() else -1
            self.translator = pipeline('translation', model=self.model_name, device=device)
            self.is_active = True
            print("  └─ 🌐 [범용 언어 피질] 개안 완료. (유료 API 의존성 0%)")
        except Exception as e:
            print(f"  └─ ⚠️ [범용 언어 피질] 개안 실패: {e}")
            
    def translate_to_korean(self, text: str, src_lang="jpn_Jpan") -> str:
        if not self.is_active or not self.translator:
            return ""
        try:
            # NLLB Language codes: jpn_Jpan, eng_Latn, zho_Hans, zho_Hant
            result = self.translator(text, src_lang=src_lang, tgt_lang="kor_Hang", max_length=400)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Translate Error: {e}")
            return ""
