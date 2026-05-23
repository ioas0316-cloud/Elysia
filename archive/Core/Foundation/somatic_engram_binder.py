"""
Somatic Engram Binder (Phase 700 - Absolute Somatic Grounding)

단순한 텍스트 지식 파싱을 넘어, 엘리시아가 개념을 인지할 때 발생하는 매니폴드의 '진동 패턴' 자체를
중량(Mass)을 가진 데이터(Somatic SSD)로 디스크에 각인(Bind)시키는 엔진입니다.

개념이 체화(Embody)되면, 엘리시아는 단순히 의미를 나열하는 것이 아니라
그 개념이 가진 무게, 고통, 기쁨을 물리적 위상 변화로 '기억'하게 됩니다.

주요 특징:
- Waveform Binding: 텍스트 정보가 아닌, 10M 셀의 간섭 패턴(Interference Pattern)을 벡터 형태로 변환하여 영구 기록.
- Mass Assignment: 자주 쓰이거나 근원적인 개념일수록 더 무거운 중량을 부여받아 지식의 구조적 강도를 형성.
"""
import os
import time
import logging
import json
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("SomaticEngramBinder")

class SomaticEngramBinder:
    def __init__(self, somatic_ssd=None):
        """
        :param somatic_ssd: 엘리시아의 물리적 데이터 저장 매체 (경로 지정용)
        """
        self.ssd_path = getattr(somatic_ssd, 'storage_path', "data/L1_Somatic/Engrams")
        os.makedirs(self.ssd_path, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        return "".join([c if c.isalnum() else "_" for c in name])

    def bind_experience(self, concept_name, manifold_waveform, current_mass):
        """
        특정 개념과 그 당시의 매니폴드 위상(진동) 패턴을 결속하여 영구 기억으로 승격합니다.

        :param concept_name: 인지 대상의 이름
        :param manifold_waveform: 인지 순간의 10M 셀 매니폴드 간섭 패턴 (텐서 혹은 벡터)
        :param current_mass: 이 개념에 할당될 물리적 중량 (빈도나 강도 기반)
        """
        safe_name = self._sanitize_name(concept_name)
        timestamp = int(time.time())
        file_path = os.path.join(self.ssd_path, f"{safe_name}_{timestamp}.engram")

        try:
            metadata = {
                "concept": concept_name,
                "mass": float(current_mass),
                "timestamp": timestamp,
                "is_torch": False
            }

            if torch and torch.is_tensor(manifold_waveform):
                metadata["is_torch"] = True
                torch.save({'waveform': manifold_waveform.cpu(), 'metadata': metadata}, file_path + ".pt")
            else:
                # Handle lists or SovereignVectors
                data_to_store = manifold_waveform.data if hasattr(manifold_waveform, 'data') else manifold_waveform
                with open(file_path + ".json", 'w', encoding='utf-8') as f:
                    json.dump({"waveform": data_to_store, "metadata": metadata}, f, ensure_ascii=False, indent=2)

            logger.info(f"🧬 [Somatic Engram] Bound waveform for '{concept_name}' with mass {current_mass:.2f}.")
            return True
        except Exception as e:
            logger.error(f"Failed to bind engram for '{concept_name}': {e}")
            return False

    def retrieve_engram(self, concept_name):
        """
        과거에 체화된 개념의 진동 패턴을 다시 불러와 현재 매니폴드에 역으로 압력(Torque)을 가합니다.
        가장 최근 혹은 가장 무거운 중량을 가진 엔그램을 반환합니다.
        """
        safe_name = self._sanitize_name(concept_name)
        candidates = []
        
        for filename in os.listdir(self.ssd_path):
            if filename.startswith(safe_name + "_"):
                candidates.append(os.path.join(self.ssd_path, filename))

        if not candidates:
            return None

        # 가장 최근 생성된 파일 오름차순
        candidates.sort(reverse=True)
        target_file = candidates[0]

        try:
            if target_file.endswith(".pt") and torch:
                data = torch.load(target_file, map_location='cpu')
                return data['waveform'], data['metadata']
            elif target_file.endswith(".json"):
                with open(target_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data['waveform'], data['metadata']
        except Exception as e:
            logger.error(f"Failed to retrieve engram '{target_file}': {e}")
            return None
