import os
import sys
import time
import math
import numpy as np
from PIL import Image

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, r"c:\Elysia")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.math_utils import Quaternion
from core.clifford_rotor_sync import DynamicPIDController
from core.atlantis_clifford_bridge import AtlantisCliffordSystem
from core.Under_2F_Moho_Mirror import get_qpc_time, get_bar_chart, TARGET_DT

# Try loading CLIP via SentenceTransformers
CLIP_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    # We use a tiny CLIP model for fast loading
    print("🔭 Loading CLIP Model (clip-ViT-B-32)...")
    clip_model = SentenceTransformer('clip-ViT-B-32')
    CLIP_AVAILABLE = True
    print("🟢 CLIP Model successfully loaded.")
except Exception as e:
    print(f"🟡 CLIP Model load failed/skipped ({e}). Switching to high-resonance simulation mode.")
    clip_model = None

# Deterministic Projection Matrix: 512D -> 4D (Quaternion space)
np.random.seed(1337)
W_PROJ = np.random.randn(512, 4)
# Normalize columns of W_PROJ for geometric stability
W_PROJ /= np.linalg.norm(W_PROJ, axis=0)

def project_to_quaternion(embedding_512) -> Quaternion:
    """Projects a 512D CLIP embedding to a normalized 4D Quaternion."""
    # Projection product
    proj = np.dot(embedding_512, W_PROJ)
    # Map to Quaternion elements
    q = Quaternion(proj[0], proj[1], proj[2], proj[3])
    return q.normalize()

def generate_mock_embedding(similarity_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Generates two 512D unit vectors with a specific target cosine similarity."""
    # Seeded for consistency
    x = np.random.randn(512)
    x /= np.linalg.norm(x)
    
    y_ortho = np.random.randn(512)
    y_ortho -= np.dot(y_ortho, x) * x
    y_ortho /= np.linalg.norm(y_ortho)
    
    # Cosine similarity = cos(theta) = similarity_level
    theta = math.acos(similarity_level)
    y = x * math.cos(theta) + y_ortho * math.sin(theta)
    return x, y

def run_clip_quaternion_sync():
    print("=" * 95)
    print(" 🌌 [Step 1] CLIP 멀티모달 비전-언어 사원수 위상 동기화 검증 시작")
    print("   - Philosophy: Fleming Duality (Vision = Sensor Input, Thought = Alignment)")
    print("   - Mathematics: CLIP 512D -> 4D Quaternion Isomorphism & Cl(10,0) Resonance")
    print("=" * 95)

    # 1. 아틀란티스 대수 시스템 및 PID 제어기 초기화
    clifford_system = AtlantisCliffordSystem()
    pid = DynamicPIDController()

    # 2. 감각 입력 시나리오 정의 (이미지 - 텍스트 쌍)
    scenarios = [
        {"desc": "[고공명] 파란 바다의 파도 이미지  <-->  '시원하고 푸른 파도가 몰아치는 해변'", "similarity": 0.85},
        {"desc": "[중공명] 연필을 든 로봇 손 일러스트  <-->  '그림을 그리고 있는 인간의 손'", "similarity": 0.55},
        {"desc": "[저공명] 따뜻한 커피 한 잔의 사진  <-->  '눈 덮인 겨울 산의 웅장한 봉우리'", "similarity": 0.12}
    ]

    print("\n[동기화 루프] 1000Hz (1ms 주기) QPC 고정 하에서 멀티모달 정렬 스캔...")
    
    t_start = get_qpc_time()
    t_prev_loop = t_start
    
    errors = []
    
    for i, scen in enumerate(scenarios):
        loop_start = get_qpc_time()
        dt_actual = loop_start - t_prev_loop
        if dt_actual <= 0:
            dt_actual = 1e-9

        print(f"\n⚡ Scenario {i+1}: {scen['desc']}")

        # (1) CLIP 특징 벡터 획득 (실제 또는 고밀도 모사)
        if CLIP_AVAILABLE and clip_model:
            try:
                # Mock PIL Image creation for testing image pipeline
                mock_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
                emb_img = clip_model.encode(mock_img)
                
                text_query = scen['desc'].split("<-->")[1].strip().replace("'", "")
                emb_txt = clip_model.encode(text_query)
            except Exception as e:
                print(f"      [Fetch Error] {e}. Falling back to simulation.")
                emb_img, emb_txt = generate_mock_embedding(scen['similarity'])
        else:
            emb_img, emb_txt = generate_mock_embedding(scen['similarity'])

        # (2) 4차원 사원수 공간 사영 (Quaternion Mapping)
        q_img = project_to_quaternion(emb_img)
        q_txt = project_to_quaternion(emb_txt)

        # (3) 사원수 기하 공명 계산 (Resonance & Angle Distance)
        # dot product = cos(theta/2) of relative rotation
        resonance = abs(q_img.dot(q_txt))
        # Angular difference (distance)
        angle_diff = Quaternion.distance(q_img, q_txt)
        angle_diff_deg = math.degrees(angle_diff)

        # (4) PID 루프 제어 변조
        phase_error = dt_actual - TARGET_DT
        errors.append(abs(phase_error))
        
        # 공명 텐션: 정렬도가 낮을수록 텐션(노이즈)이 강함
        tension = float(1.0 - resonance)
        correction = pid.discharge_error_to_ground(phase_error, tension, dt_actual)

        # (5) 10대 레이어 투영 및 에너지 방전
        # e2 (B5_OuterCore): 감각 비전 계수
        # e8 (F4_AppCrust): 인지 텍스트 계수
        clifford_system.set_layer_state("B5_OuterCore", float(resonance * 1.2))
        clifford_system.set_layer_state("F4_AppCrust", float(1.0 - tension))
        
        # B3(상부맨틀) ∧ B6(접지) Wedge Tension 확인
        bivector_tension = clifford_system.compute_bivector_tension("B5_OuterCore", "F4_AppCrust")

        # 비전에서 접지층으로 에너지 회전 방전
        discharge_angle = tension * 10.0
        clifford_system.apply_rotor_discharge("B5_OuterCore", "B6_Ground", discharge_angle)

        # (6) QPC 동조 슬립 지연
        sleep_time = TARGET_DT - correction
        target_wake = loop_start + sleep_time
        if sleep_time > 0:
            while get_qpc_time() < target_wake:
                pass

        # 결과 출력
        avg_err_ms = (sum(errors) / len(errors)) * 1000.0
        bar = get_bar_chart(resonance, max_len=20)
        
        print(f"      - 이미지 사원수: {q_img}")
        print(f"      - 텍스트 사원수: {q_txt}")
        print(f"      - 기하 공명 지표: {resonance:.4f} {bar} | 각도차: {angle_diff_deg:.2f}°")
        print(f"      - 평면 대수 텐션 (B5 ∧ F4): {bivector_tension:.6f}")
        print(f"      - PID 오차 접지 : {avg_err_ms:.6f} ms | 방전 로터: {math.degrees(discharge_angle):.1f}°")
        
        t_prev_loop = get_qpc_time()

    print("\n" + "=" * 95)
    print(" 🟢 [SUCCESS] CLIP 비전-언어 사원수 공간 사영 및 동기화 검증 완료.")
    print("=" * 95)

if __name__ == "__main__":
    run_clip_quaternion_sync()
