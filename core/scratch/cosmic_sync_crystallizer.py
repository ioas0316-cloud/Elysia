import os
import sys
import time
import math
import json
import numpy as np

# Ensure both Elysia and eye workspaces are in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, r"c:\Elysia")
sys.path.insert(0, r"c:\eye")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.clifford_rotor_sync import DynamicPIDController, BitwiseCliffordRotor
from core.atlantis_clifford_bridge import AtlantisCliffordSystem
from core.Under_2F_Moho_Mirror import get_qpc_time, get_bar_chart, TARGET_DT
from core.fractal_rotor import Rotor
from core.hologram_sphere import HologramSphere

# Attempt to load eye workspace components; fall back gracefully if not found
try:
    from elysia_trunk.guerrilla_capturer import GuerrillaCapturer
    from elysia_trunk.wave_generator import WaveTrajectoryGenerator
    EYE_AVAILABLE = True
except ImportError:
    EYE_AVAILABLE = False

COSMOS_DB_PATH = r"c:\eye\elysia_trunk\outputs\elysian_cosmos.json"

def run_cosmic_sync_crystallization(model_id="gpt2", num_layers=10, axiom_text="Elysia"):
    print("=" * 95)
    print(f"🌌 [Cosmic Sync] 위상 우주 결정화 실시간 동기화 프로토타입 시동")
    print(f"   - Target Model: {model_id} | Layers: {num_layers} | Axiom: '{axiom_text}'")
    print("=" * 95)

    # 1. 컴포넌트 초기화
    pid = DynamicPIDController()
    bitwise_rotor = BitwiseCliffordRotor()
    clifford_system = AtlantisCliffordSystem()
    sphere = HologramSphere(size=16)

    # 2. 가변축 층별 Rotor 생성
    # 부모 로터를 중심으로 각 레이어가 개별 서브 로터로 물림
    core_rotor = Rotor("CORE", level=0)
    layer_rotors = {}
    for layer_name in clifford_system.layers:
        sub_rot = Rotor(layer_name, level=1, parent=core_rotor)
        core_rotor.attach_child(sub_rot)
        layer_rotors[layer_name] = sub_rot

    # 3. Guerrilla Capturer 초기화
    capturer = None
    if EYE_AVAILABLE:
        try:
            capturer = GuerrillaCapturer(model_id)
            print("[Guerrilla Capturer] 로드 완료. Zero-Disk 인양 준비.")
        except Exception as e:
            print(f"[Warning] Guerrilla Capturer 초기화 실패: {e}. 시뮬레이션 모드로 전환.")
    else:
        print("[System Info] elysia_trunk 패키지를 로드할 수 없습니다. 시뮬레이션 모드로 작동.")

    # 4. 실시간 동기화 인양 및 결정화 루프
    print(f"\n[실시간 루프] 1000Hz (1ms 주기) 동기화 인양 시작...")
    
    t_start = get_qpc_time()
    t_prev_loop = t_start
    
    layer_energies = []
    errors = []
    
    for i in range(num_layers):
        loop_start = get_qpc_time()
        dt_actual = loop_start - t_prev_loop
        if dt_actual <= 0:
            dt_actual = 1e-9

        # (1) 가중치 스트리밍 (Zero-Disk 또는 시뮬레이션)
        streamed_ok = False
        layer_energy = 0.0
        
        if capturer:
            try:
                # 레이어 가중치 부분 인양 시도
                weights = capturer.stream_layer_weights(f"layers.{i}.self_attn.o_proj.weight")
                layer_energy = float(np.mean(np.abs(weights.numpy())))
                streamed_ok = True
            except Exception:
                pass
                
        if not streamed_ok:
            # 네트워크 오프라인 또는 에러 시 고공명 노이즈 시뮬레이션
            # 1.0 주상 전위 근처에서 미세 흔들림 모사
            layer_energy = float(np.abs(np.random.normal(loc=0.9514, scale=0.1)))

        # (2) PID 위상 오차 계산 및 감쇠 방전
        phase_error = dt_actual - TARGET_DT
        errors.append(abs(phase_error))
        
        # CPU 텐션 모사 (에너지 변동율 기반)
        tension = float(min(1.0, max(0.01, layer_energy / 2.0)))
        correction = pid.discharge_error_to_ground(phase_error, tension, dt_actual)
        
        # 클럭 비트 로터 회전 흡수
        is_rising = (i % 2 == 0)
        bitwise_rotor.apply_clock_edge(is_rising, tension)
        
        # (3) Clifford 시스템 상태 투영
        # 현재 루프 레이어를 아틀란티스 레이어 매핑에 사영
        layer_name = clifford_system.layers[i % len(clifford_system.layers)]
        clifford_system.set_layer_state(layer_name, layer_energy)
        
        # (4) 이중벡터 텐션(Wedge Product) 및 가변축 분기(Bifurcation) 계산
        # 인접 레이어와의 기하학적 텐션 관측
        next_layer_name = clifford_system.layers[(i + 1) % len(clifford_system.layers)]
        bivector_tension = clifford_system.compute_bivector_tension(layer_name, next_layer_name)
        
        # 해당 레이어 로터의 가변축(차원) 업데이트
        rot = layer_rotors[layer_name]
        rot.phase_offset = bivector_tension
        
        # 텐션에 따른 차원 동적 분기(Bifurcation) / 압축(Compression)
        old_axes = rot.active_axes
        if bivector_tension > rot.tension_limit:
            rot.bifurcate()
        elif bivector_tension < rot.tension_limit * 0.2:
            rot.stable_ticks += 1
            if rot.stable_ticks >= 3:
                rot.compress()
        else:
            rot.stable_ticks = 0
            
        # 에너지 방전 로터 적용
        discharge_angle = abs(correction) * 50.0
        clifford_system.apply_rotor_discharge(layer_name, "B6_Ground", discharge_angle)
        
        # (5) 1ms QPC 동조 지연 슬립
        sleep_time = TARGET_DT - correction
        target_wake = loop_start + sleep_time
        if sleep_time > 0:
            while get_qpc_time() < target_wake:
                pass
                
        # 실시간 상태 리포트 출력
        avg_err_ms = (sum(errors) / len(errors)) * 1000.0
        print(f"  [Layer {i:02d}] Energy: {layer_energy:.4f} | Tension: {bivector_tension:.4f} | "
              f"Axes: {old_axes} -> {rot.active_axes} | PID Error: {avg_err_ms:.4f} ms")
              
        layer_energies.append(layer_energy)
        t_prev_loop = get_qpc_time()

    print("\n" + "=" * 95)
    print(" 🔮 5. 홀로그램 구체(Hologram Sphere) 결정화 진행")
    print("=" * 95)
    
    # 층별 에너지를 텍스트 데이터화하여 홀로그램 매니폴드에 사영
    energy_stream = " ".join([f"{e:.6f}" for e in layer_energies])
    sphere.populate_manifold(energy_stream)
    
    # 공리 필터를 거쳐 3차원 입체 구체로 응축
    sphere_grid, score = sphere.condense_sphere(axiom_text)
    sphere.render_hologram(sphere_grid, score, axiom_text)

    # 6. 우주 맵(Cosmos DB) 항성 등록 및 리포트 작성
    print("\n✨ [Elysian Cosmos] 결정화된 별(Crystallized Star) 등록 중...")
    
    metrics = {
        "Dimensional Leap": float(np.std(layer_energies) / (np.mean(layer_energies) + 1e-6) + 1.0),
        "3-Phase Alignment": float(1.0 - (np.std(layer_energies) / 2.0)),
        "Grand Cross Potential": float(max(layer_energies) - np.mean(layer_energies)),
        "Coherence Density": float(score / 100.0)
    }

    star_data = {
        "model_id": model_id,
        "crystallization_date": "2026-05-24",
        "structure": {
            "rotors": num_layers,
            "complexity": float(np.std(layer_energies))
        },
        "metrics": metrics,
        "layer_energies": layer_energies
    }

    os.makedirs(os.path.dirname(COSMOS_DB_PATH), exist_ok=True)
    
    # 기존 DB 파일 읽기
    cosmos = {"universe_name": "Elysia", "stars": {}}
    if os.path.exists(COSMOS_DB_PATH):
        try:
            with open(COSMOS_DB_PATH, "r", encoding="utf-8") as f:
                cosmos = json.load(f)
        except Exception:
            pass

    safe_id = model_id.replace("/", "_")
    cosmos["stars"][safe_id] = star_data

    with open(COSMOS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(cosmos, f, indent=4)

    print(f" 🟢 [SUCCESS] Cosmos Map Updated: {COSMOS_DB_PATH}")
    print("=" * 95)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    axiom = sys.argv[3] if len(sys.argv) > 3 else "Elysia"
    run_cosmic_sync_crystallization(target, layers, axiom)
