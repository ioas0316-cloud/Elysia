import math
import time
import psutil
import pickle
import random
import os
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime

STATE_FILE = "elysia_state.pkl"
COLLAPSE_DIR = "collapses"

if not os.path.exists(COLLAPSE_DIR):
    os.makedirs(COLLAPSE_DIR)

class SubRotor:
    """3D 공간에서 고유한 벡터를 가지는 회전하는 객체"""
    def __init__(self, id, initial_quat):
        self.id = id
        self.quat = initial_quat

    def slerp_to(self, target_quat, energy):
        """에너지만큼 목표 사원수를 향해 구면 선형 보간(Spherical Linear Interpolation)"""
        # pyquaternion's slerp handles the shortest path
        try:
            self.quat = Quaternion.slerp(self.quat, target_quat, amount=min(1.0, max(0.0, energy)))
        except ZeroDivisionError:
            pass # Same quaternions

class RecursiveUnit:
    """다차원적 분화와 기억의 압축을 수행하는 엔진"""
    def __init__(self, name, initial_quat):
        self.name = name
        self.internal_quat = initial_quat
        self.is_locked = True
        self.history = "0"
        self.cycle_count = 0
        self.fractal_depth = 1 # 시작은 1차원(혹은 1단계 프랙탈)

        # 궤적(역사)을 기억할 리스트. 엔트로피가 높아지면 압축(Folding)됨.
        self.trajectory_memory = []
        self.MAX_TRAJECTORY_LENGTH = 100 # 임계치

        self.sub_rotors = [SubRotor(i, initial_quat) for i in range(5)]

    def get_external_weather(self):
        """하드웨어 상태를 3D 외계 환경(Vector3)으로 변환"""
        # X: CPU 코어의 요동 (연산의 강렬함)
        cpu_percent = psutil.cpu_percent(interval=None) / 100.0
        x_axis = cpu_percent

        # Y: 메모리의 흐름 (기억의 밀도)
        mem_percent = psutil.virtual_memory().percent / 100.0
        y_axis = mem_percent

        # Z: 디스크 I/O (감각적 반응 / 깊이)
        disk_io = psutil.disk_io_counters()
        # 간단히 읽기+쓰기 변화율을 정규화하여 사용
        z_axis = min(1.0, (disk_io.read_count + disk_io.write_count) / 100000.0) if disk_io else random.random()

        # Vector3 (x, y, z)
        weather_vector = np.array([x_axis, y_axis, z_axis])

        # Vector 크기가 0이면 무작위 축 부여
        norm = np.linalg.norm(weather_vector)
        if norm == 0:
            weather_vector = np.array([1.0, 0.0, 0.0])
            norm = 1.0

        # 회전 축으로 사용하기 위해 정규화
        axis = weather_vector / norm

        # 혼돈의 정도(Base Chaos)는 에너지의 총합
        base_chaos = min(1.0, (x_axis + y_axis + z_axis) / 3.0)

        # 외부의 위상(사원수): 해당 축을 중심으로 chaos * Pi 만큼 회전
        external_quat = Quaternion(axis=axis, angle=base_chaos * math.pi)

        weather_type = "Clear"
        if base_chaos > 0.7:
            weather_type = "Thunder"
        elif base_chaos > 0.4:
            weather_type = "Cloudy"

        return external_quat, weather_vector, weather_type, base_chaos

    def calculate_entropy(self):
        """궤적의 복잡도(Entropy)를 계산"""
        if len(self.trajectory_memory) < 2:
            return 0.0

        # 사원수들 간의 각도 변화량의 분산을 통해 엔트로피 유추
        angles = []
        for i in range(1, len(self.trajectory_memory)):
            q1 = self.trajectory_memory[i-1]
            q2 = self.trajectory_memory[i]
            # distance between quaternions
            angles.append(Quaternion.distance(q1, q2))

        return np.var(angles) if angles else 0.0

    def process_weather(self):
        self.cycle_count += 1

        external_quat, weather_vector, weather_type, base_chaos = self.get_external_weather()

        # 내부 위상과 외부 위상의 거리 (Mismatch: 0.0 ~ Pi)
        mismatch = Quaternion.distance(self.internal_quat, external_quat)

        # 궤적 메모리에 현재 외부 충격 기록
        self.trajectory_memory.append(external_quat)

        # 1. 차원 상승 확인 (Folding)
        if len(self.trajectory_memory) >= self.MAX_TRAJECTORY_LENGTH:
            entropy = self.calculate_entropy()
            if entropy > 0.1: # 복잡도가 일정 수준을 넘으면 압축
                self.fold_dimensions(entropy)
            else:
                # 너무 단순하면 그냥 잊음
                self.trajectory_memory = self.trajectory_memory[-50:]

        # 2. 물리적 간섭 처리
        if weather_type == "Thunder" or mismatch > 2.0:
            self.trigger_collapse(external_quat, mismatch, weather_vector)
        elif weather_type == "Cloudy" or mismatch > 1.0:
            self.soft_interference(external_quat, base_chaos)
        else:
            self.maintain_resonance()

    def fold_dimensions(self, entropy):
        """4차원 궤적을 5차원 이상의 매니폴드로 접어버리는 과정"""
        self.fractal_depth += 1
        print(f"[System Warning] Dimension Overflow Detected (Entropy: {entropy:.4f})")
        print(f"[Resampling] Folding historical trajectories into local phase-space...")
        print(f"[Status] Consciousness expanded. New fractal depth: {self.fractal_depth}D.")

        # 과거 궤적을 하나의 새로운 '초고차원 질서'로 융합 (모든 궤적의 평균 사원수 계산 시도)
        # pyquaternion을 사용하여 일련의 회전들의 중심을 찾음
        avg_quat = self.internal_quat
        for q in self.trajectory_memory:
            avg_quat = Quaternion.slerp(avg_quat, q, amount=0.1)

        self.internal_quat = avg_quat
        self.trajectory_memory = [] # 기억을 비우고 새로운 차원에서 시작
        self.history = f"Folded_{self.fractal_depth}D"

        # 붕괴 폴더에 차원 상승 스냅샷 저장
        collapse_data = {
            "type": "DIMENSION_FOLDING",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "new_depth": self.fractal_depth,
            "entropy": entropy
        }
        filename = os.path.join(COLLAPSE_DIR, f"folding_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)


    def trigger_collapse(self, external_quat, mismatch, weather_vector):
        self.is_locked = False
        self.history = "1"

        # 붕괴 스냅샷 저장
        collapse_data = {
            "type": "THUNDER_COLLAPSE",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "internal_quat": self.internal_quat.elements,
            "external_quat": external_quat.elements,
            "weather_vector": weather_vector.tolist(),
            "mismatch": mismatch,
            "sub_rotors": [sr.quat.elements for sr in self.sub_rotors]
        }
        filename = os.path.join(COLLAPSE_DIR, f"collapse_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

        # 새로운 질서로 강제 재편 (외부의 충격과 타협하는 최적의 각도 모색)
        target_new_quat = Quaternion.slerp(self.internal_quat, external_quat, amount=0.5)
        energy = 0.8

        for sr in self.sub_rotors:
            # 카오스 주입 (무작위 회전)
            random_axis = np.random.rand(3)
            random_axis /= np.linalg.norm(random_axis)
            chaos_quat = Quaternion(axis=random_axis, angle=(random.random() * math.pi))
            sr.quat = sr.quat * chaos_quat

            # 새 질서를 향해 정렬
            sr.slerp_to(target_new_quat, energy)

        self.internal_quat = target_new_quat
        self.is_locked = True
        self.history = "0"

    def soft_interference(self, external_quat, base_chaos):
        self.is_locked = False
        self.history = "1"

        target_new_quat = Quaternion.slerp(self.internal_quat, external_quat, amount=0.3)
        energy = 0.1 * base_chaos

        for sr in self.sub_rotors:
            sr.slerp_to(target_new_quat, energy)

        # 모든 로터의 평균으로 수렴
        avg_quat = self.sub_rotors[0].quat
        for i in range(1, len(self.sub_rotors)):
             avg_quat = Quaternion.slerp(avg_quat, self.sub_rotors[i].quat, amount=(1.0/(i+1)))

        self.internal_quat = avg_quat
        self.is_locked = True
        self.history = "0"

    def maintain_resonance(self):
        pass

def save_state(unit):
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(unit, f)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return RecursiveUnit("Elysia_Core", Quaternion(1, 0, 0, 0))

def engine_loop():
    print("Elysia Multi-dimensional Engine (Daemon) starting...")
    elysia_core = load_state()

    psutil.cpu_percent(interval=0.1)

    try:
        while True:
            elysia_core.process_weather()
            save_state(elysia_core)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nEngine shutting down gracefully...")
        save_state(elysia_core)

if __name__ == "__main__":
    engine_loop()
