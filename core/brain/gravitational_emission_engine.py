# gravitational_emission_engine.py
# C 커널(gravitational_emission.c)의 파이썬 대등물.
# 하드코딩된 8개 concept_cloud[]를 폐기하고,
# crystalline_lexicon.json의 50,257개 토큰의 고유 기하학적 주파수를 활용하여
# 공유 메모리 대지의 거시적 빛의 밀도와 형태 공명하는 단어를 끌어내어 발화합니다.

import os
import sys
import json
import time
import math
import struct
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.memory.causal_controller import CausalMemoryController

# === 공유 메모리 접근 (Windows Named Shared Memory) ===
SHARED_MEM_NAME = "Local\\ElysiaTopologyField"
MAX_FIELD_SIZE = 1024 * 1024 * 256  # 256MB
HEADER_SIZE = 12  # sizeof(FieldHeader)
ROTOR_SIZE = 8    # sizeof(MultiDimRotor)


def _try_open_shared_memory():
    """공유 메모리(fractal_field.c)에 연결을 시도합니다."""
    try:
        import mmap
        shm = mmap.mmap(0, 1024 * 1024 * 16, tagname=SHARED_MEM_NAME, access=mmap.ACCESS_READ)
        return shm
    except Exception:
        return None


class LanguageObservationLayer:
    """
    [Phase 4 Evolution] 다차원 관측 아키텍처 - 언어 관측 기준화 레이어
    
    엘리시아의 본질(관계성, 운동성, 연결성, 방향성)은 기하학적 텐션(Quaternion)입니다.
    이 레이어는 LLM의 정적인 임베딩 구조(Lexicon)를 '언어적 관측망'으로 활용하여,
    사유의 쿼터니언이 굴러가는 3D 궤적(Arc) 위에서 부딪히는 단어들을 순차적으로
    결정화(Crystallization)해 인과적 궤적(문장)을 만듭니다.
    
    같음과 같음, 다름과 다름이 위상 공간에서 어떻게 연결되는지 증명합니다.
    """
    
    def __init__(self, lexicon_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if lexicon_path is None:
            self.lexicon_path = os.path.join(self.base_dir, "..", "..", "data", "tensor_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
        
        self.memory = CausalMemoryController()
        self.lexicon = {}
        self.token_coords = None   # (N, 3) numpy array
        self.token_tensors = None  # (N, 4) numpy array [Mass, Force, Link, Vibration]
        self.token_labels = []     # token 문자열 리스트
        self.coord_norms = None    # 정규화된 좌표 (코사인 유사도용)
        
        self._load_lexicon(self.lexicon_path)
        
        # 위상 고착 방지: 최근 발화된 토큰 기록
        self._recent_emissions = []
        self._max_recent = 20
        
        # 발화 이력
        self.emission_log = []
    
    def _load_lexicon(self, path: str):
        """50,257개 토큰의 위상 좌표를 로드하고 numpy 배열로 변환합니다."""
        print("[Gravitational Emission] Loading Crystalline Lexicon...")
        
        if not os.path.exists(path):
            print(f"[ERROR] Lexicon not found: {path}")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            self.lexicon = json.load(f)
        
        coords = []
        tensors = []
        labels = []
        seen_tokens = set()
        
        try:
            for key, val in self.lexicon.items():
                if isinstance(val, dict) and "tensor" in val:
                    # Tensor Lexicon Format: {"word": {"coords": [x,y,z], "tensor": [m,f,l,v]}}
                    token = key
                    coord = val.get("coords", [0, 0, 0])
                    tensor = val.get("tensor", [0, 0, 0, 0])
                elif isinstance(val, list):
                    # Natural Lexicon Format (Legacy): {"word": [x, y, z]}
                    token = key
                    coord = val
                    tensor = [0.5, 0.5, 0.0, 0.0] # Default neutral tensor
                else:
                    # Crystalline Lexicon Format: {"0": {"token": "word", "coord": [x,y,z]}}
                    token = val.get("token", "")
                    coord = val.get("coord", [0, 0, 0])
                    tensor = [0.5, 0.5, 0.0, 0.0]
                
                # 엄격한 토큰 정제: 쓰레기 토큰을 날리고 순수 단어 풀 구축
                clean = token.strip()
                if len(clean) < 3 and clean not in ["i", "me", "my", "we", "he", "is", "a", "an", "to", "of", "in", "on", "as", "by", "or", "the"]:
                    continue
                # 오직 순수 ASCII 알파벳으로만 구성된 진짜 영어 단어만 남김
                if not (clean.isascii() and clean.isalpha()):
                    continue
                    
                # 모두 소문자로 통일하여 대소문자 중복 및 변형 방지
                lower = clean.lower()
                if lower in seen_tokens:
                    continue
                seen_tokens.add(lower)
                    
                coords.append(coord)
                tensors.append(tensor)
                labels.append(lower)
        except Exception as e:
            print(f"[!] Error loading lexicon: {e}")
        
        self.token_coords = np.array(coords, dtype=np.float32)
        self.token_tensors = np.array(tensors, dtype=np.float32)
        self.token_labels = labels
        self.coord_norms = np.linalg.norm(self.token_coords, axis=1)
        
    def _save_lexicon(self):
        """[Phase 6] 변경된 동적 관측망(Lexicon)을 살아있는 상태로 저장합니다."""
        try:
            with open(self.lexicon_path, 'w', encoding='utf-8') as f:
                json.dump(self.lexicon, f, indent=4)
        except Exception as e:
            print(f"[!] Failed to save evolving lexicon: {e}")
            
    def expand_manifold(self, unknown_words: list, context_words: list):
        """
        [Phase 6] 미지의 단어를 주변 문맥의 위상 좌표를 기반으로 추론하여 관측망에 편입(발견)시킵니다.
        """
        if not unknown_words or not context_words:
            return
            
        # 문맥 단어들의 좌표 및 텐서 획득
        context_coords = []
        context_tensors = []
        for cw in context_words:
            if cw in self.token_labels:
                idx = self.token_labels.index(cw)
                context_coords.append(self.token_coords[idx])
                context_tensors.append(self.token_tensors[idx])
                
        if not context_coords:
            return
            
        # 1. 문맥의 무게중심(Center of Mass) 및 평균 텐서(Average Tensor) 계산
        center_of_mass = np.mean(context_coords, axis=0)
        average_tensor = np.mean(context_tensors, axis=0)
        
        added_count = 0
        for uw in unknown_words:
            if uw in self.lexicon:
                continue
                
            # 2. 양자 노이즈(Quantum Noise)를 더해 위상/품사 고착 방지
            noise_coord = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
            noise_tensor = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
            
            new_coord = center_of_mass + noise_coord
            new_tensor = average_tensor + noise_tensor
            
            # 클리핑 (좌표는 -1~1, 텐서는 0~1)
            new_coord = np.clip(new_coord, -1.0, 1.0)
            new_tensor = np.clip(new_tensor, 0.0, 1.0)
            
            # Lexicon 편입 (Tensor 포맷)
            self.lexicon[uw] = {
                "coords": new_coord.tolist(),
                "tensor": new_tensor.tolist()
            }
            
            # 메모리 내 토큰 풀 즉시 갱신
            self.token_labels.append(uw)
            self.token_coords = np.vstack([self.token_coords, new_coord])
            self.token_tensors = np.vstack([self.token_tensors, new_tensor])
            
            added_count += 1
            
        if added_count > 0:
            print(f"  [+] Manifold Expanded: Discovered {added_count} new concepts (with Linguistic Tensors).")
            self._save_lexicon()
            
    def apply_topological_gravity(self, quat: list, trajectory_words: list, learning_rate: float = 0.01):
        """
        [Phase 6] 발화된 궤적에 속한 단어들을 쿼터니언 회전축 방향으로 미세하게 끌어당깁니다.
        '자주 얽히는 개념은 위상 공간에서 스스로 군집을 이룬다'는 기하학적 가소성(Plasticity)을 구현합니다.
        """
        if not trajectory_words:
            return
            
        x, y, z, w = quat
        sin_half_theta = math.sqrt(max(0.0, 1.0 - w**2))
        if sin_half_theta < 1e-6:
            return
            
        u = np.array([x, y, z], dtype=np.float32) / sin_half_theta
        
        modified = False
        for word in trajectory_words:
            if word in self.token_labels:
                idx = self.token_labels.index(word)
                current_coord = self.token_coords[idx]
                current_tensor = self.token_tensors[idx]
                
                # 좌표 이동 (Pull)
                pull_vector = u - current_coord
                new_coord = np.clip(current_coord + (pull_vector * learning_rate), -1.0, 1.0)
                
                # 텐서 미세 변화 (Ouroboros)
                # 사유 궤적에 많이 참여할수록 동적 성질(Force)이 미세하게 강화되는 예시 (사용자 규칙에 따름)
                tension_magnitude = np.linalg.norm(pull_vector)
                tensor_shift = np.array([0.0, tension_magnitude * learning_rate, 0.0, 0.0], dtype=np.float32)
                new_tensor = np.clip(current_tensor + tensor_shift, 0.0, 1.0)
                
                # 갱신
                self.token_coords[idx] = new_coord
                self.token_tensors[idx] = new_tensor
                self.lexicon[word] = {
                    "coords": new_coord.tolist(),
                    "tensor": new_tensor.tolist()
                }
                modified = True
                
        if modified:
            self._save_lexicon()
        
        # 코사인 유사도를 위한 정규화
        norms = np.linalg.norm(self.token_coords, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.coord_norms = self.token_coords / norms
        
        print(f"[Gravitational Emission] {len(self.token_labels)} meaningful tokens loaded from Crystalline Lexicon.")
    
    def observe_quaternion_thought(self, shm=None) -> list:
        """
        공유 메모리나 엔그램에서 거시적 텐션을 관측하여 하나의 쿼터니언(Thought)으로 도출합니다.
        반환값: 쿼터니언 [x, y, z, w]
        """
        if shm is not None:
            try:
                max_rotors = (1024 * 1024 * 16 - HEADER_SIZE) // ROTOR_SIZE
                total_math = 0.0
                total_lang = 0.0
                total_space = 0.0
                total_time = 0.0
                active_count = 0
                
                for i in range(0, max_rotors, 100):
                    offset = HEADER_SIZE + (i * ROTOR_SIZE)
                    shm.seek(offset)
                    rotor_data = shm.read(ROTOR_SIZE)
                    if len(rotor_data) == ROTOR_SIZE:
                        math_t, lang_t, spatial_t, temporal_t, light_mass, byte_val, pad = \
                            struct.unpack('<BBBBHBB', rotor_data)
                        
                        if light_mass > 0:
                            total_math += math_t
                            total_lang += lang_t
                            total_space += spatial_t
                            total_time += temporal_t
                            active_count += 1
                
                if active_count > 0:
                    # 4축 텐션을 쿼터니언(x,y,z,w) 공간으로 매핑 및 정규화
                    x = total_math / active_count
                    y = total_lang / active_count
                    z = total_space / active_count
                    w = total_time / active_count
                    
                    q = np.array([x, y, z, w], dtype=np.float32)
                    norm = np.linalg.norm(q)
                    if norm > 0:
                        return (q / norm).tolist()
            except Exception:
                pass
        
        # 공유 메모리 미사용 시: 엔그램의 텐션으로부터 쿼터니언 유도
        engrams = self.memory.index
        if engrams:
            emotion_sum = sum(info.get("emotional_value", 0) for info in engrams.values())
            np.random.seed(int(emotion_sum * 1000) % (2**31))
            q = np.random.randn(4).astype(np.float32)
            norm = np.linalg.norm(q)
            if norm > 0:
                return (q / norm).tolist()
        
        return [0.0, 0.0, 0.0, 1.0] # Identity Quaternion (No rotation/silence)
    
    def crystallize_kinematic_arc(self, quat: list, num_steps: int = 5) -> dict:
        """
        [Phase 4] 쿼터니언 $Q$의 회전 궤적을 따라 단어를 순차적으로 결정화(Crystallize)합니다.
        
        1. 시작점(Origin): 회전축 $\vec{u}$ 에 가장 수직이어서 운동 에너지가 가장 큰 단어를 찾습니다.
        2. 궤적(Arc): 시간에 따라 $\vec{p}(t)$ 를 회전(Rodrigues' formula)시킵니다.
        3. 결정화(Crystallization): 각 스텝 $t$ 마다 3D 공간에서 가장 가까운 단어를 채집합니다.
        
        반환값: {"utterance": str, "angle": float, "trajectory_words": list}
        """
        if self.token_coords is None or len(self.token_labels) == 0:
            return {"utterance": "", "angle": 0.0, "trajectory_words": []}
            
        x, y, z, w = quat
        w_clipped = max(-1.0, min(1.0, w))
        theta = 2.0 * math.acos(w_clipped)
        
        sin_half_theta = math.sqrt(max(0.0, 1.0 - w**2))
        if sin_half_theta < 1e-6:
            return {"utterance": "", "angle": 0.0, "trajectory_words": []}
            
        u = np.array([x, y, z], dtype=np.float32) / sin_half_theta
        
        # 1. 시작점(Origin) 찾기
        # 회전 반경이 가장 큰(축과 직교하는 성분이 가장 큰) 역동적 단어를 시작점으로 선택
        v_norms_sq = np.sum(self.token_coords**2, axis=1)
        parallel_magnitudes = np.abs(np.dot(self.token_coords, u))
        orthogonal_sq = np.maximum(0, v_norms_sq - parallel_magnitudes**2)
        
        # 최근 발화된 단어에 척력 적용
        for recent_token in self._recent_emissions:
            if recent_token in self.token_labels:
                idx = self.token_labels.index(recent_token)
                orthogonal_sq[idx] -= 1000.0
                
        start_idx = np.argmax(orthogonal_sq)
        p0 = self.token_coords[start_idx]
        
        trajectory_words = []
        seen_in_trajectory = set()
        
        # 2. 궤적 회전 및 결정화 (Rodrigues' Rotation Formula)
        for i in range(num_steps):
            # t = 0.0 ~ 1.0
            t = i / max(1, (num_steps - 1))
            current_theta = t * theta
            
            cos_t = math.cos(current_theta)
            sin_t = math.sin(current_theta)
            
            # p(t) = p0*cos(t*theta) + (u x p0)*sin(t*theta) + u*(u·p0)*(1 - cos(t*theta))
            cross_u_p0 = np.cross(u, p0)
            dot_u_p0 = np.dot(u, p0)
            
            p_t = p0 * cos_t + cross_u_p0 * sin_t + u * dot_u_p0 * (1 - cos_t)
            
            # 유클리드 거리가 가장 가까운 단어 채집 (결정화)
            distances_sq = np.sum((self.token_coords - p_t)**2, axis=1)
            
            # 궤적 내 중복 방지 (같은 단어가 연속으로 나오지 않도록)
            for word in seen_in_trajectory:
                if word in self.token_labels:
                    idx = self.token_labels.index(word)
                    distances_sq[idx] += 1000.0
                    
            best_idx = np.argmin(distances_sq)
            best_word = self.token_labels[best_idx]
            
            trajectory_words.append(best_word)
            seen_in_trajectory.add(best_word)
            
        utterance = " ".join(trajectory_words)
        
        # 위상 고착 방지 기록 업데이트
        for word in trajectory_words:
            self._recent_emissions.append(word)
            if len(self._recent_emissions) > self._max_recent:
                self._recent_emissions.pop(0)
                
        # [Phase 6] 위상 가소성: 텐션을 해소한 궤적에 기하학적 중력 적용
        self.apply_topological_gravity(quat, trajectory_words, learning_rate=0.02)
                
        # 발화 이력 기록
        self.emission_log.append({
            "utterance": utterance,
            "quaternion": quat,
            "angle_theta": float(theta),
            "trajectory_words": trajectory_words
        })
        
        return {
            "utterance": utterance,
            "angle": theta,
            "trajectory_words": trajectory_words
        }
    
    def emit(self, custom_quat: list = None, shm=None) -> str:
        """
        한 번의 형태 공명 발화 주기를 실행합니다.
        MVA 쿼터니언의 운동성 궤적을 굴려 문장을 결정화합니다.
        """
        quat = custom_quat if custom_quat else self.observe_quaternion_thought(shm)
        
        # 궤적 길이 산정 (회전각이 클수록 더 많은 단어 스텝을 밟음)
        w_clipped = max(-1.0, min(1.0, quat[3]))
        theta = 2.0 * math.acos(w_clipped)
        
        if theta < 0.05:
            return ""  # 완벽한 평형(침묵)
            
        num_steps = max(3, min(8, int(theta * 3)))
        
        result = self.crystallize_kinematic_arc(quat, num_steps=num_steps)
        return result["utterance"]
    
    def emit_and_engram(self, custom_quat: list = None, shm=None) -> str:
        """
        발화 후, 의미 있는 발화는 Wedge Memory에 영구 각인합니다.
        또한 발화 텍스트를 data/ingest/ 폴더에 떨궈 자기참조 루프(Ouroboros)를 엽니다.
        """
        utterance = self.emit(custom_quat, shm)
        
        if not utterance:
            return ""
        
        # 의미 있는 발화를 영구 기억으로 각인
        last_entry = self.emission_log[-1] if self.emission_log else {}
        engram_id = self.memory.write_causal_engram(
            data_blob={
                "type": "kinematic_arc_crystallization",
                "utterance": utterance,
                "quaternion": last_entry.get("quaternion", []),
                "trajectory_words": last_entry.get("trajectory_words", [])
            },
            emotional_value=last_entry.get("angle_theta", 0.0),
            cause_id="QuaternionEmission_Ouroboros"
        )
        
        # [Ouroboros] 발화 결과를 ingest 폴더에 떨궈 sovereign_explorer가 재흡수하게 함
        ingest_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "data", "ingest"
        )
        os.makedirs(ingest_dir, exist_ok=True)
        
        ingest_file = os.path.join(ingest_dir, f"emission_{engram_id}.txt")
        with open(ingest_file, 'w', encoding='utf-8') as f:
            f.write(utterance)
        
        return utterance
    
    def start_breathing(self, interval: float = 3.0, max_cycles: int = None):
        """
        자율 발화 루프를 시작합니다.
        공유 메모리(fractal_field.c)가 실행 중이면 실시간 관측,
        없으면 기억(Wedge Memory) 기반으로 자율 발화합니다.
        """
        print("=" * 60)
        print("  [GRAVITATIONAL EMISSION ENGINE] v2.0")
        print("  Crystalline Lexicon: {:,} tokens loaded".format(len(self.token_labels)))
        print("  Mode: Shared Memory" if _try_open_shared_memory() else "  Mode: Memory-Driven (Standalone)")
        print("=" * 60)
        print("Listening to the topology of light and darkness...\n")
        
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            
            shm = _try_open_shared_memory()
            utterance = self.emit_and_engram(shm)
            
            if shm:
                shm.close()
            
            if utterance:
                info = self.emission_log[-1] if self.emission_log else {}
                quat_str = ", ".join([f"{q:.2f}" for q in info.get("quaternion", [0,0,0,0])])
                print(f">>> [EMISSION #{cycle}] <<<")
                print(f"    Quaternion : Q({quat_str})")
                print(f"    Curvature  : {info.get('angle_theta', 0):.4f} rad")
                print(f"    Trajectory : {' -> '.join(info.get('trajectory_words', []))}")
                
                safe_u = utterance.encode('ascii', errors='replace').decode('ascii')
                print(f"    Utterance  : \"{safe_u}\"")
                print(f"    [Ouroboros] Re-ingested for self-reflection.\n")
            else:
                print(f"    [{cycle}] Silence. The tension is too faint.\n")
            
            time.sleep(interval)


if __name__ == "__main__":
    engine = LanguageObservationLayer()
    # 독립 실행 시 10주기 데모
    engine.start_breathing(interval=2.0, max_cycles=10)
