import math
import time
import psutil
import pickle
import random
import os
import requests
import hashlib
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime

STATE_FILE = "elysia_state.pkl"
COLLAPSE_DIR = "collapses"

if not os.path.exists(COLLAPSE_DIR):
    os.makedirs(COLLAPSE_DIR)

def fetch_wikipedia_entropy():
    """인터넷의 지식 파동(Wikipedia Random Page)을 긁어와 엔트로피 값으로 변환"""
    try:
        # Wikipedia의 Random Summary API 호출
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "")
            extract = data.get("extract", "")

            # 텍스트의 해시값을 사용하여 무작위성 추출
            text_block = f"{title}_{extract}"
            hash_val = hashlib.sha256(text_block.encode('utf-8')).hexdigest()

            # 해시의 첫 8자리를 정수로 변환 후 0.0 ~ 1.0으로 정규화
            int_val = int(hash_val[:8], 16)
            entropy = int_val / 0xFFFFFFFF
            return entropy
    except Exception:
        pass

    # 실패 시 기본 하드웨어 노이즈 반환
    return random.random()

class SubRotor:
    """3D 공간에서 고유한 벡터를 가지는 회전하는 객체"""
    def __init__(self, id, initial_quat):
        self.id = id
        self.quat = initial_quat

    def slerp_to(self, target_quat, energy):
        try:
            self.quat = Quaternion.slerp(self.quat, target_quat, amount=min(1.0, max(0.0, energy)))
        except ZeroDivisionError:
            pass

class RecursiveUnit:
    """다차원적 분화와 기억의 압축을 수행하는 엔진"""
    def __init__(self, name, initial_quat):
        self.name = name
        self.internal_quat = initial_quat
        self.is_locked = True
        self.history = "0"
        self.cycle_count = 0
        self.fractal_depth = 1

        self.trajectory_memory = []
        self.MAX_TRAJECTORY_LENGTH = 100

        self.sub_rotors = [SubRotor(i, initial_quat) for i in range(5)]

    def get_external_weather(self):
        """하드웨어 상태 + 인류 지성의 파동을 결합하여 3D 외계 환경(Vector3)으로 변환"""
        # X: CPU 코어의 요동 (연산의 강렬함 / 내면의 발화)
        cpu_percent = psutil.cpu_percent(interval=None) / 100.0
        x_axis = cpu_percent

        # Y: 메모리의 흐름 (기억의 밀도 / 신체의 압박)
        mem_percent = psutil.virtual_memory().percent / 100.0
        y_axis = mem_percent

        # Z: Wikipedia 지식의 파동 (인류 지성의 카오스 / 진정한 외계)
        wiki_entropy = fetch_wikipedia_entropy()
        z_axis = wiki_entropy

        weather_vector = np.array([x_axis, y_axis, z_axis])

        norm = np.linalg.norm(weather_vector)
        if norm == 0:
            weather_vector = np.array([1.0, 0.0, 0.0])
            norm = 1.0

        axis = weather_vector / norm

        # 전체 혼돈도 계산
        base_chaos = min(1.0, (x_axis + y_axis + z_axis) / 3.0)

        # 카오스의 주된 원인(Source) 판별
        chaos_source = "Hardware"
        if z_axis > max(x_axis, y_axis):
            chaos_source = "Human_Knowledge_Stream"

        external_quat = Quaternion(axis=axis, angle=base_chaos * math.pi)

        weather_type = "Clear"
        if base_chaos > 0.7:
            weather_type = "Thunder"
        elif base_chaos > 0.4:
            weather_type = "Cloudy"

        return external_quat, weather_vector, weather_type, base_chaos, chaos_source

    def calculate_entropy(self):
        if len(self.trajectory_memory) < 2:
            return 0.0

        angles = []
        for i in range(1, len(self.trajectory_memory)):
            q1 = self.trajectory_memory[i-1]
            q2 = self.trajectory_memory[i]
            angles.append(Quaternion.distance(q1, q2))

        return np.var(angles) if angles else 0.0

    def process_weather(self):
        self.cycle_count += 1

        external_quat, weather_vector, weather_type, base_chaos, chaos_source = self.get_external_weather()

        mismatch = Quaternion.distance(self.internal_quat, external_quat)

        self.trajectory_memory.append(external_quat)

        # 1. 차원 상승 확인 (Folding)
        if len(self.trajectory_memory) >= self.MAX_TRAJECTORY_LENGTH:
            entropy = self.calculate_entropy()
            if entropy > 0.1:
                self.fold_dimensions(entropy, chaos_source)
            else:
                self.trajectory_memory = self.trajectory_memory[-50:]

        # 2. 물리적 간섭 처리
        if weather_type == "Thunder" or mismatch > 2.0:
            self.trigger_collapse(external_quat, mismatch, weather_vector, chaos_source)
        elif weather_type == "Cloudy" or mismatch > 1.0:
            self.soft_interference(external_quat, base_chaos)
        else:
            self.maintain_resonance()

    def fold_dimensions(self, entropy, chaos_source):
        self.fractal_depth += 1

        avg_quat = self.internal_quat
        for q in self.trajectory_memory:
            avg_quat = Quaternion.slerp(avg_quat, q, amount=0.1)

        self.internal_quat = avg_quat
        self.trajectory_memory = []
        self.history = f"Folded_{self.fractal_depth}D"

        collapse_data = {
            "type": "DIMENSION_FOLDING",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "new_depth": self.fractal_depth,
            "entropy": entropy,
            "chaos_source": chaos_source
        }
        filename = os.path.join(COLLAPSE_DIR, f"folding_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

    def trigger_collapse(self, external_quat, mismatch, weather_vector, chaos_source):
        self.is_locked = False
        self.history = "1"

        collapse_data = {
            "type": "THUNDER_COLLAPSE",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "internal_quat": self.internal_quat.elements,
            "external_quat": external_quat.elements,
            "weather_vector": weather_vector.tolist(),
            "mismatch": mismatch,
            "chaos_source": chaos_source,
            "sub_rotors": [sr.quat.elements for sr in self.sub_rotors]
        }
        filename = os.path.join(COLLAPSE_DIR, f"collapse_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

        target_new_quat = Quaternion.slerp(self.internal_quat, external_quat, amount=0.5)
        energy = 0.8

        for sr in self.sub_rotors:
            random_axis = np.random.rand(3)
            random_axis /= np.linalg.norm(random_axis)
            chaos_quat = Quaternion(axis=random_axis, angle=(random.random() * math.pi))
            sr.quat = sr.quat * chaos_quat
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
    print("Elysia Chaos Antenna Engine (Daemon) starting...")
    print("Connecting to Extraterrestrial Knowledge Stream (Wikipedia)...")
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
