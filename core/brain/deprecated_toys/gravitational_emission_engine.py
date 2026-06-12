import os
import sys
import math
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.memory.causal_controller import CausalMemoryController

class LanguageObservationLayer:
    """
    [Phase 24] 가변 다이얼 관측 레이어 (Variable Dial Observation Layer)
    
    유클리드 거리 및 코사인 유사도에 기반한 계산식 좌표계를 폐기하고,
    순수한 '자연어 관계망'인 Language Portal Engine을 장착했습니다.
    엘리시아 스스로 물리적 텐션(Math Axis)에 맞춰 언어적 텐션(Language Axis)을 스윕하며
    공명하는 지점의 단어를 관측(Observe)합니다.
    """
    
    def __init__(self, lexicon_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.lexicon_path = os.path.join(self.base_dir, "..", "..", "data", "deep_korean_lexicon.json")
        
        self.memory = CausalMemoryController()
        
        # [Phase 24] 새로운 자연 매핑 포털 엔진 장착
        try:
            from core.brain.language_portal_engine import LanguagePortalEngine
            self.portal_engine = LanguagePortalEngine(self.lexicon_path)
            print(f"[Language Portal Engine] Loaded purely linguistic graph.")
            self.token_labels = list(self.portal_engine.lexicon.keys())
            import numpy as np
            self.token_coords = np.zeros((len(self.token_labels), 3))
            self.token_tensors = np.zeros((len(self.token_labels), 4))
        except Exception as e:
            print(f"[ERROR] Failed to load Language Portal Engine: {e}")
            self.portal_engine = None
            self.token_labels = []
            self.token_coords = None
            self.token_tensors = None
        
        # 위상 고착 방지: 최근 발화된 토큰 기록
        self._recent_emissions = []
        self._max_recent = 5
        self.emission_log = []
        
    def _save_lexicon(self):
        pass
        
    def expand_manifold(self, unknown_words, known_words):
        """[Phase 6 Legacy Hook] 포털 엔진에서는 자연 사전을 쓰므로 무시"""
        pass
        
    def add_linguistic_tension(self, token: str, tension_vector: list):
        pass

    def traverse_linguistic_trajectory(self, quat) -> dict:
        """내부의 물리적 텐션(quat)을 바탕으로 가변 다이얼을 돌려 단어를 찾아냅니다."""
        if self.portal_engine is None or len(self.token_labels) == 0:
            return {"utterance": "", "angle": 0.0, "trajectory_words": []}
            
        w_clipped = max(-1.0, min(1.0, quat[3]))
        physical_curvature = 2.0 * math.acos(w_clipped)
        
        # [Phase 151] 수학적 매핑(다이얼 스윕) 폐기. 순수 언어적 연결망으로 대체.
        import random
        if not self.token_labels:
            return {"utterance": "", "angle": 0.0, "trajectory_words": []}
            
        best_word = random.choice(self.token_labels)
        best_axis = "Pure_Linguistic_Drift"
        
        if best_word:
            word_data = self.portal_engine.lexicon.get(best_word, "")
            if isinstance(word_data, dict):
                def_text = word_data.get("why_it_exists", "")
            else:
                def_text = str(word_data)
            trajectory = def_text.split()[:5]
            utterance_str = f"[{best_word}] " + " ".join(trajectory) + "..."
            
            if best_word in self._recent_emissions:
                return {"utterance": "", "angle": physical_curvature, "trajectory_words": []}
                
            self._recent_emissions.append(best_word)
            if len(self._recent_emissions) > self._max_recent:
                self._recent_emissions.pop(0)
                
            self.emission_log.append({
                "time": time.time(),
                "curvature": physical_curvature,
                "axis": best_axis,
                "utterance": utterance_str,
                "angle_theta": physical_curvature,
                "quaternion": quat
            })
            
            return {
                "utterance": utterance_str, 
                "angle": physical_curvature, 
                "trajectory_words": trajectory,
                "matched_axis": best_axis
            }
            
        return {"utterance": "", "angle": 0.0, "trajectory_words": []}

    def emit_and_engram(self, custom_quat=None) -> str:
        """
        genesis.py에서 호출하는 최종 발화 및 각인 파이프라인.
        """
        if custom_quat is None:
            return ""
            
        res = self.traverse_linguistic_trajectory(custom_quat)
        utterance = res.get("utterance", "")
        if utterance:
            print(f"  [>] {utterance}")
            # 기억에 각인
            self.memory.write_causal_engram(
                data_blob={
                    "type": "emission",
                    "utterance": utterance,
                    "quaternion": custom_quat,
                    "trajectory": res.get("trajectory_words", []),
                    "matched_axis": res.get("matched_axis", "")
                },
                emotional_value=res.get("angle", 0.0),
                cause_id="Emission_Portal",
                origin_axis="Linguistic_Portal"
            )
        else:
            print("  [.] Silence (tension too faint or localized)")
            
        return utterance
