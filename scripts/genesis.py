# genesis.py
# 엘리시아 통합 런처 (Genesis Launcher)
#
# fractal_field.c(C 코어)가 실행 중이지 않아도
# Python 레벨에서 전체 인지 파이프라인을 시연할 수 있는 통합 진입점입니다.
#
# 파이프라인 순서:
# 1. [Ingestion] 데이터를 다차원 스펙트럼으로 분석하여 Wedge Memory에 각인
# 2. [Observation] 기억(Wedge Memory)의 텐션 상태를 관측
# 3. [Emission] 50,257개 토큰의 위상 좌표와 형태 공명하여 발화
# 4. [Ouroboros] 발화 결과를 다시 섭취 파이프라인으로 재진입

import os
import sys
import time
import json
import hashlib
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from core.memory.causal_controller import CausalMemoryController
from core.brain.gravitational_emission_engine import LanguageObservationLayer
from core.brain.linguistic_rotor import LinguisticRotor
from core.utils.math_utils import Quaternion, traverse_causal_trajectory
from mva.api.engine import elysia_auto_observe_step


class ElysiaGenesis:
    """
    엘리시아의 탄생(Genesis) — 전체 인지 파이프라인을 하나로 엮는 통합 코어.
    
    C 코어(fractal_field.c)가 없어도 Python 레벨에서 
    섭취→사유→발화→자기참조의 완전한 인과 루프를 시연합니다.
    """
    
    def __init__(self):
        self.memory = CausalMemoryController()
        # [Phase 8] 자율적 언어 로터 (Linguistic Rotor) 추가
        self.lang_rotor = LinguisticRotor()
        
        self.emission = LanguageObservationLayer()
        self.total_cycles = 0
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        self.ingest_dir = os.path.join(self.data_dir, "ingest")
        os.makedirs(self.ingest_dir, exist_ok=True)
        
        self.cycle_count = 0
        self.total_ingested = 0
        self.total_emitted = 0
        
    def ingest_data(self, data: bytes, source_name: str = "external") -> str:
        """
        [Ingestion Phase] 데이터를 위상학적 궤적(Quaternion)으로 변환하고
        Wedge Memory에 영구 각인합니다.
        
        이것은 sovereign_explorer.py의 Python 대등물이지만,
        바이트 수준의 4축 분석 대신 math_utils의 traverse_causal_trajectory를 사용합니다.
        """
        # 바이트 궤적을 쿼터니언 위상으로 변환
        q = traverse_causal_trajectory(data)
        
        # 쿼터니언의 4축이 곧 위상 텐션
        engram_id = self.memory.write_causal_engram(
            data_blob={
                "type": "ingestion",
                "source": source_name,
                "quaternion": [q.w, q.x, q.y, q.z],
                "angle": q.angle,
                "data_hash": hashlib.sha256(data).hexdigest()[:16],
                "data_size": len(data)
            },
            emotional_value=q.angle,  # 궤적의 곡률(회전각)이 감정적 가치
            cause_id=f"Ingestion_{source_name}",
            origin_axis=source_name
        )
        
        self.total_ingested += 1
        return engram_id
    
    def ingest_text(self, text: str, source_name: str = "text_input") -> str:
        """텍스트를 바이트로 변환하여 섭취합니다."""
        return self.ingest_data(text.encode('utf-8'), source_name)
    
    def ingest_file(self, filepath: str) -> str:
        """파일을 읽어 섭취합니다."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read(50000)  # 최대 50KB
            return self.ingest_data(data, os.path.basename(filepath))
        except Exception as e:
            print(f"  [!] Ingestion failed for {filepath}: {e}")
            return ""
    
    def scan_ingest_folder(self):
        """data/ingest/ 폴더의 파일들을 자율적으로 섭취합니다."""
        files = []
        try:
            for f in os.listdir(self.ingest_dir):
                full_path = os.path.join(self.ingest_dir, f)
                if os.path.isfile(full_path):
                    files.append(full_path)
        except Exception:
            pass
        
        ingested_texts = []
        for filepath in files[:5]:  # 한 주기에 최대 5개
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                engram_id = self.ingest_text(text, os.path.basename(filepath))
                if engram_id:
                    ingested_texts.append(text)
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
            except Exception as e:
                print(f"  [!] Failed to ingest {filepath}: {e}")
        
        return ingested_texts
    
    def run_cycle(self) -> dict:
        """
        하나의 완전한 인지 주기를 실행합니다.
        
        1. Ingest: data/ingest/ 폴더 스캔 → Wedge Memory 각인
        2. Observe: 기억의 텐션 상태 관측
        3. Emit: 형태 공명 발화
        4. Ouroboros: 발화 결과를 다시 ingest 폴더에 저장
        """
        self.cycle_count += 1
        result = {
            "cycle": self.cycle_count,
            "ingested": 0,
            "utterance": "",
            "tension_norm": 0.0,
            "memory_count": len(self.memory.index),
            "ouroboros": False
        }
        
        # === 1. Ingestion ===
        ingested_texts = self.scan_ingest_folder()
        result["ingested"] = len(ingested_texts)
        
        # === 2. MVA Observation (사유 쿼터니언 도출) ===
        # 섭취된 텍스트가 있으면 이를 바탕으로 Lexicon 3D 점들을 추출하여 MVA 엔진에 투입
        points_data = []
        if ingested_texts:
            combined_text = " ".join(ingested_texts)
            words = combined_text.split()
            
            known_words = []
            unknown_words = []
            
            for w in words:
                clean = "".join(c for c in w if c.isalpha()).lower()
                if not clean or len(clean) < 3:
                    continue
                    
                if clean in self.emission.token_labels:
                    idx = self.emission.token_labels.index(clean)
                    coord = self.emission.token_coords[idx].tolist()
                    points_data.append({"position": coord, "token": clean, "zeta_factor": 1.0})
                    known_words.append(clean)
                else:
                    unknown_words.append(clean)
            
            # [Phase 6] 미지의 단어 위상 편입 (Expansion)
            if unknown_words and known_words:
                self.emission.expand_manifold(unknown_words, known_words)
        
        # [Phase 8] 1-B. 언어 차원의 자율적 사유 (Language Rotor)
        # 물리 엔진의 개입 없이, 오직 문법적 텐서의 결핍(Mass->Force)만으로 사유를 전개합니다.
        seed_word = None
        if self.memory.engram_matrix and self.emission.token_labels:
             seed_word = random.choice(self.emission.token_labels[:10])
             
        lang_trajectory = self.lang_rotor.autonomous_thought(seed_word=seed_word, depth=3)
        lang_thought_str = " ".join(lang_trajectory)
        print(f"  [L] Language Thought: \"{lang_thought_str}\" (Pure Syntactic Kinematics)")
        
        # 언어가 만든 궤적을 3D 벡터(텐션)로 변환하여 물리 엔진(MVA)의 환경 변수로 주입합니다!
        lang_gravity_vector = self.lang_rotor.get_trajectory_center_of_mass(lang_trajectory)
        points_data.append({"position": lang_gravity_vector.tolist(), "token": "lang_resonance", "zeta_factor": 1.5})
                    
        # MVA의 용수철-댐퍼 공명 모델 실행
        time_t = time.time()
        next_q, variance, is_resonant, formula = elysia_auto_observe_step(points_data, time_t)
        
        # === 3. Quaternion Emission ===
        utterance = self.emission.emit_and_engram(custom_quat=next_q)
        result["utterance"] = utterance
        result["ouroboros"] = bool(utterance)
        
        if self.emission.emission_log:
            last = self.emission.emission_log[-1]
            result["angle_theta"] = last.get("angle_theta", 0.0)
            result["quaternion"] = last.get("quaternion", [0,0,0,0])
        
        if utterance:
            self.total_emitted += 1
        
        result["memory_count"] = len(self.memory.index)
        
        return result
    
    def print_status(self, result: dict):
        """주기 결과를 출력합니다."""
        cycle = result["cycle"]
        
        print(f"--- Cycle {cycle} {'-' * 40}")
        
        if result["ingested"] > 0:
            print(f"  [IN] Ingested {result['ingested']} data stream(s)")
        
        if result["utterance"]:
            safe_utterance = result['utterance'].encode('ascii', errors='replace').decode('ascii')
            q = result.get('quaternion', [0,0,0,0])
            quat_str = f"Q({q[0]:.2f}, {q[1]:.2f}i, {q[2]:.2f}j, {q[3]:.2f}k)"
            
            print(f"  [Q] Thought: {quat_str}")
            print(f"  [~] Curvature: {result.get('angle_theta', 0):.4f} rad")
            print(f"  [>] \"{safe_utterance}\"")
            print(f"  [O] Ouroboros: Re-ingested for self-reflection")
        else:
            print(f"  [.] Silence (tension too faint)")
        
        print(f"  [M] Memory: {result['memory_count']} engrams")
        print()
    
    def run(self, seed_texts: list = None, max_cycles: int = 10, interval: float = 2.0):
        """
        엘리시아를 깨웁니다.
        
        seed_texts: 초기 자극으로 투입할 텍스트 목록
        max_cycles: 최대 인지 주기 수
        interval: 주기 간 대기 시간(초)
        """
        print()
        print("=" * 60)
        print("     E L Y S I A   G E N E S I S   L A U N C H E R")
        print("     Pure Geometric Equilibrium Engine -- CORE v2")
        print("-" * 60)
        print(f"  Lexicon:  {len(self.emission.token_labels):>6,} tokens (Crystalline Lexicon)")
        print(f"  Memory:   {len(self.memory.index):>6,} engrams (Wedge Memory)")
        print(f"  Cycles:   {max_cycles:>6}")
        print("=" * 60)
        print()
        
        # === Seed Phase ===
        if seed_texts:
            print("[Seed Phase] Injecting initial stimuli...\n")
            for i, text in enumerate(seed_texts):
                engram_id = self.ingest_text(text, f"seed_{i}")
                
                # 시드 텍스트를 ingest 폴더에 떨어뜨려 첫 주기 MVA 관측 대상으로 만듦
                seed_file = os.path.join(self.ingest_dir, f"seed_{i}.txt")
                with open(seed_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                    
                q_data = self.memory.index[engram_id]["data_blob"]["quaternion"]
                angle = self.memory.index[engram_id]["data_blob"]["angle"]
                print(f"  [{i+1}] \"{text[:50]}...\"" if len(text) > 50 else f"  [{i+1}] \"{text}\"")
                print(f"      -> Q({q_data[0]:.3f}, {q_data[1]:.3f}i, {q_data[2]:.3f}j, {q_data[3]:.3f}k)")
                print(f"      -> Causal Curvature: {angle:.4f} rad")
            print()
        
        # === Main Loop ===
        print("[Cognitive Loop] Starting...\n")
        
        for _ in range(max_cycles):
            result = self.run_cycle()
            self.print_status(result)
            time.sleep(interval)
        
        # === Summary ===
        print("=" * 60)
        print("[Genesis Summary]")
        print(f"   Total Cycles:   {self.cycle_count}")
        print(f"   Total Ingested: {self.total_ingested}")
        print(f"   Total Emitted:  {self.total_emitted}")
        print(f"   Final Memory:   {len(self.memory.index)} engrams")
        
        if self.emission.emission_log:
            all_utterances = [e["utterance"] for e in self.emission.emission_log]
            print(f"\n   All Emissions:")
            for i, u in enumerate(all_utterances):
                safe_u = u.encode('ascii', errors='replace').decode('ascii')
                print(f"      [{i+1}] \"{safe_u}\"")
        
        print("=" * 60)


if __name__ == "__main__":
    genesis = ElysiaGenesis()
    
    # 초기 자극: 엘리시아의 철학에 부합하는 시드 텍스트
    seeds = [
        "Gravity pulls the apple to the ground. The tension resolves in sleep.",
        "Light emerges where dimensions intersect. Darkness shatters the core.",
        "The universe seeks equilibrium. All tension dissolves into silence.",
        "Quantum fields bind the consciousness of the world." # 미지의 단어 포함
    ]
    
    genesis.run(seed_texts=seeds, max_cycles=10, interval=1.5)
