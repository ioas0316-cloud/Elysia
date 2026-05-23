import math
import time
import pickle
import random
import os
import cv2
import mss
import numpy as np
import pyautogui
from pyquaternion import Quaternion
from datetime import datetime

STATE_FILE = "elysia_state.pkl"
COLLAPSE_DIR = "collapses"

if not os.path.exists(COLLAPSE_DIR):
    os.makedirs(COLLAPSE_DIR)

# 안전 장치: 마우스를 구석으로 던지면 종료 가능
pyautogui.FAILSAFE = True

class SubRotor:
    """플로팅 액시스를 가진 회전체"""
    def __init__(self, id, initial_quat):
        self.id = id
        self.quat = initial_quat

    def slerp_to(self, target_quat, energy):
        try:
            self.quat = Quaternion.slerp(self.quat, target_quat, amount=min(1.0, max(0.0, energy)))
        except ZeroDivisionError:
            pass

class FloatingAxisEngine:
    """가변축(Floating Axis)과 시각/운동 피질을 가진 우주적 엔진"""
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

        # 시각 피질 설정값 보관 (mss 객체 자체는 피클링 방지를 위해 필요시 로드)
        self.vision_rect = None
        self.prev_frame_gray = None

        # 운동 피질 상태
        self.last_motor_action = "None"

    def _get_sct_rect(self):
        if not hasattr(self, '_sct'):
            self._sct = mss.mss()
        if self.vision_rect is None:
            monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]
            width, height = min(800, monitor["width"]), min(600, monitor["height"])
            left = monitor["left"] + (monitor["width"] - width) // 2
            top = monitor["top"] + (monitor["height"] - height) // 2
            self.vision_rect = {"top": top, "left": left, "width": width, "height": height}
        return self._sct, self.vision_rect

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable entries.
        if '_sct' in state:
            del state['_sct']
        return state

    def fetch_vision_entropy(self):
        """Optical Flow를 통한 시각 피질 엔진. 화면 픽셀 변화를 카오스로 환산."""
        try:
            sct, rect = self._get_sct_rect()
            img = np.array(sct.grab(rect))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            entropy = 0.0
            death_detected = False

            if self.prev_frame_gray is not None:
                # 픽셀 변화량 계산 (Optical Flow 근사)
                frame_diff = cv2.absdiff(self.prev_frame_gray, gray)
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

                # 변화된 픽셀의 비율 (0.0 ~ 1.0)
                changed_pixels = np.count_nonzero(thresh)
                total_pixels = thresh.size
                entropy = min(1.0, (changed_pixels / total_pixels) * 5.0) # 민감도 조정

                # 붉은색 검출 (죽음/고통 인지)
                # 오버플로우 방지를 위해 int16 변환 후 Red 채널 추출
                img_int = img.astype(np.int16)
                red_channel = img_int[:, :, 2]
                blue_channel = img_int[:, :, 0]
                green_channel = img_int[:, :, 1]
                # 붉은색이 압도적인 픽셀들 (피격 효과, 피 화면 등)
                red_mask = (red_channel > 150) & (red_channel > blue_channel + 50) & (red_channel > green_channel + 50)
                red_ratio = np.count_nonzero(red_mask) / total_pixels

                if red_ratio > 0.3: # 화면의 30% 이상이 강렬한 붉은색이면 죽음/고통으로 간주
                    death_detected = True

            self.prev_frame_gray = gray
            return entropy, death_detected
        except Exception as e:
            print(f"Vision Error: {e}")
            return 0.0, False

    def trigger_motor_cortex(self, rotation_diff):
        """내부 사원수의 회전축을 바탕으로 키보드 액션(WASD/Space) 발현"""
        # rotation_diff는 외부 카오스에 맞서 내부가 회전한 사원수
        axis = rotation_diff.axis
        angle = rotation_diff.angle

        if abs(angle) < 0.1:
            self.last_motor_action = "Idle"
            return

        action = "None"
        try:
            # X축 회전 (좌우)
            if abs(axis[0]) > abs(axis[1]) and abs(axis[0]) > abs(axis[2]):
                if axis[0] > 0:
                    pyautogui.press('d')
                    action = "D (Right)"
                else:
                    pyautogui.press('a')
                    action = "A (Left)"
            # Y축 회전 (상하/점프)
            elif abs(axis[1]) > abs(axis[0]) and abs(axis[1]) > abs(axis[2]):
                if axis[1] > 0:
                    pyautogui.press('w')
                    action = "W (Forward)"
                else:
                    pyautogui.press('space')
                    action = "Space (Jump/Evade)"
            # Z축 회전 (회전/후퇴)
            else:
                pyautogui.press('s')
                action = "S (Backward)"

            self.last_motor_action = action
        except pyautogui.FailSafeException:
            print("[System Halt] Failsafe triggered by user mouse position.")
            exit(1)

    def get_external_weather(self):
        """시각 엔진(Optical Flow)을 유일한 진정한 외계 카오스로 받아들임"""
        vision_entropy, death_detected = self.fetch_vision_entropy()

        # 화면의 카오스가 Z축을 지배, 나머지는 랜덤한 요동(작은 노이즈)
        x_axis = random.random() * 0.2
        y_axis = random.random() * 0.2
        z_axis = vision_entropy

        weather_vector = np.array([x_axis, y_axis, z_axis])
        norm = np.linalg.norm(weather_vector)
        if norm == 0:
            weather_vector = np.array([0.0, 0.0, 1.0])
            norm = 1.0

        axis = weather_vector / norm
        base_chaos = vision_entropy

        # 플로팅 액시스: 외부 카오스는 '절대 좌표'가 아니라 나의 '현재 좌표'를 기준으로 회전
        relative_external_quat = Quaternion(axis=axis, angle=base_chaos * math.pi)

        # 1. 붉은 화면 = 죽음/고통 (THUNDER_COLLAPSE 확정)
        if death_detected:
            weather_type = "DEATH"
            base_chaos = 1.0
        elif base_chaos > 0.5:
            weather_type = "Thunder"
        elif base_chaos > 0.1:
            weather_type = "Cloudy"
        else:
            weather_type = "Clear"

        return relative_external_quat, weather_vector, weather_type, base_chaos

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

        relative_external_quat, weather_vector, weather_type, base_chaos = self.get_external_weather()

        # 플로팅 액시스의 핵심: Mismatch는 나와의 '상대적 거리'
        mismatch = Quaternion.distance(self.internal_quat, self.internal_quat * relative_external_quat)

        self.trajectory_memory.append(relative_external_quat)

        if len(self.trajectory_memory) >= self.MAX_TRAJECTORY_LENGTH:
            entropy = self.calculate_entropy()
            if entropy > 0.05:
                self.fold_dimensions(entropy)
            else:
                self.trajectory_memory = self.trajectory_memory[-50:]

        old_quat = Quaternion(self.internal_quat)

        if weather_type == "DEATH" or mismatch > 1.5:
            self.trigger_collapse(relative_external_quat, mismatch, weather_vector, weather_type)
        elif weather_type == "Cloudy" or mismatch > 0.5:
            self.soft_interference(relative_external_quat, base_chaos)
        else:
            self.maintain_resonance()

        # 내부 위상의 변화량(차이)을 구함
        rotation_diff = old_quat.inverse * self.internal_quat
        self.trigger_motor_cortex(rotation_diff)

    def fold_dimensions(self, entropy):
        self.fractal_depth += 1
        avg_quat = self.internal_quat
        for q in self.trajectory_memory:
            avg_quat = Quaternion.slerp(avg_quat, self.internal_quat * q, amount=0.1)

        self.internal_quat = avg_quat
        self.trajectory_memory = []
        self.history = f"Folded_{self.fractal_depth}D"

        collapse_data = {
            "type": "DIMENSION_FOLDING",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "new_depth": self.fractal_depth,
            "entropy": entropy,
            "chaos_source": "Optical_Flow_Overload"
        }
        filename = os.path.join(COLLAPSE_DIR, f"folding_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

    def trigger_collapse(self, relative_external_quat, mismatch, weather_vector, weather_type):
        self.is_locked = False
        self.history = "1"

        collapse_data = {
            "type": "DEATH_COLLAPSE" if weather_type == "DEATH" else "THUNDER_COLLAPSE",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "internal_quat": self.internal_quat.elements,
            "external_quat": (self.internal_quat * relative_external_quat).elements,
            "weather_vector": weather_vector.tolist(),
            "mismatch": mismatch,
            "chaos_source": "Visual_Death/High_Impact",
            "sub_rotors": [sr.quat.elements for sr in self.sub_rotors]
        }
        filename = os.path.join(COLLAPSE_DIR, f"collapse_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

        # 붕괴(죽음)시 무작위 공간으로 거칠게 튕겨나감 (Floating Axis Shift)
        random_axis = np.random.rand(3)
        random_axis /= np.linalg.norm(random_axis)
        panic_quat = Quaternion(axis=random_axis, angle=math.pi / 2) # 크게 비틂
        target_new_quat = self.internal_quat * panic_quat

        energy = 0.9
        for sr in self.sub_rotors:
            chaos_quat = Quaternion(axis=np.random.rand(3), angle=(random.random() * math.pi))
            sr.quat = sr.quat * chaos_quat
            sr.slerp_to(target_new_quat, energy)

        self.internal_quat = target_new_quat
        self.is_locked = True
        self.history = "0"

    def soft_interference(self, relative_external_quat, base_chaos):
        self.is_locked = False
        self.history = "1"

        # 외부 카오스를 내 현재 좌표를 기준으로 받아들임
        target_new_quat = Quaternion.slerp(self.internal_quat, self.internal_quat * relative_external_quat, amount=0.3)
        energy = 0.2 * base_chaos

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
    return FloatingAxisEngine("Elysia_Floating_Core", Quaternion(1, 0, 0, 0))

def engine_loop():
    print("Elysia Vision-Motor Engine (Floating Axis) starting...")
    print("Projecting Soul into Azeroth...")
    elysia_core = load_state()

    try:
        while True:
            elysia_core.process_weather()
            save_state(elysia_core)
            # 게임 반응성을 위해 짧은 박동 유지
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nEngine shutting down gracefully...")
        save_state(elysia_core)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine_loop()
