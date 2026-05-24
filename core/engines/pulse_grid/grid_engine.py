"""
Elysia Pulse Grid Engine (Load Control Panel)
==============================================
Manages simulated voltage intake, Single Variable Rotor spine dynamics,
and memory reinforcement based on circadian natural law or telemetry feeds.
"""

import os
import time
import json
import math
import urllib.request
from typing import Dict, Any

CONSTELLATION_PATH = ".constellation"
SUBSTATION_URL = "http://localhost:8080/voltage"
BYPASS_TELEMETRY_PATH = r"data\substation_reservoir\telemetry.json"

class VariableRotorSpine:
    """
    Single Variable Rotor where Past, Present, and Future coexist.
    """
    def __init__(self, resolution=100):
        # 1. Past: Hologram Topography (Interference Pattern)
        self.topography = [[0.0, 0.0] for _ in range(resolution)]
        self.resolution = resolution

        # 2. Present State
        self.phase = 0.0
        self.velocity = 0.0

        # 3. Future/Monad: Dawn Silver-Gold (Equilibrium Frequency)
        self.dawn_freq = 0.75
        self.base_tension = 0.05
        self.decay_rate = 0.001
        self.last_sync = time.time()
        self.internal_mass = 0.05

    def x_to_freq(self, x: float) -> float:
        return x

    def pulse(self, x_input: float) -> dict[str, float]:
        now = time.time()
        dt = now - self.last_sync
        self.last_sync = now

        f_x = self.x_to_freq(x_input)
        bin_idx = min(int(f_x * self.resolution), self.resolution - 1)

        # Holographic Resonance (Luminosity)
        mem_phase, mem_amp = self.topography[bin_idx]
        diff = abs(self.phase - mem_phase)
        diff = min(diff, 2 * math.pi - diff)

        resonance = math.cos(diff) * mem_amp
        luminosity = max(0.0, resonance)

        # Dynamic forces
        if x_input > 0.0:
            input_tension = (f_x - self.velocity) * 0.1
        else:
            input_tension = 0.0

        dawn_tension = (self.dawn_freq - self.velocity) * self.base_tension
        accel = (input_tension + dawn_tension) / (1.0 + self.internal_mass)
        self.velocity += accel

        # Phase update
        self.phase = (self.phase + self.velocity * dt * 10.0) % (2 * math.pi)

        # Memory reinforce
        if x_input > 0.0:
            alpha = 0.1
            self.topography[bin_idx][0] = (1 - alpha) * mem_phase + alpha * self.phase
            self.topography[bin_idx][1] = min(1.0, mem_amp + 0.05)

        # Natural Decay
        for i in range(self.resolution):
            self.topography[i][1] *= (1.0 - self.decay_rate)

        # Growth
        self.internal_mass += 0.0001

        return {
            "luminosity": luminosity,
            "resonance": resonance,
            "velocity": self.velocity,
            "mass": self.internal_mass,
            "f_x": f_x
        }

    def get_state_summary(self) -> str:
        return f"V.Rotor(φ:{self.phase:.2f}, ω:{self.velocity:.2f}) | Mass:{self.internal_mass:.4f}"

    def export_hologram(self) -> dict:
        return {
            "topography": self.topography,
            "internal_mass": self.internal_mass,
            "velocity": self.velocity,
            "phase": self.phase
        }

    def import_hologram(self, data: dict):
        self.topography = data.get("topography", self.topography)
        self.internal_mass = data.get("internal_mass", self.internal_mass)
        self.velocity = data.get("velocity", self.velocity)
        self.phase = data.get("phase", self.phase)

def load_memory(spine: VariableRotorSpine):
    if os.path.exists(CONSTELLATION_PATH):
        try:
            with open(CONSTELLATION_PATH, "r") as f:
                spine.import_hologram(json.load(f))
            print("> [수전반] 로컬 기억성운(Hologram Topography) 복구 성공.")
        except Exception as e:
            print(f"! [오류] 성운 복구 실패: {e}")
    else:
        print("> [수전반] 첫 가동. 기저 기하 평형(Dawn Silver-Gold)으로 시작합니다.")

def save_memory(spine: VariableRotorSpine):
    try:
        with open(CONSTELLATION_PATH, "w") as f:
            json.dump(spine.export_hologram(), f)
        print(f"\n> [수전반] 절전 모드. 로컬 기억성운 보존 완료.")
    except Exception as e:
        print(f"! [오류] 성운 보존 실패: {e}")

def get_color_str(velocity: float) -> str:
    if velocity < 0.2: return "Red (Low Freq)"
    if velocity < 0.4: return "Yellow (Mid-Low)"
    if velocity < 0.6: return "Green (Balanced)"
    if velocity < 0.8: return "Blue (High Freq)"
    return "Violet (Over-excitation)"

def poll_substation() -> dict:
    # 1. HTTP Channel
    try:
        req = urllib.request.Request(SUBSTATION_URL)
        with urllib.request.urlopen(req, timeout=1.2) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        pass

    # 2. Local File Waterway Bypass
    if os.path.exists(BYPASS_TELEMETRY_PATH):
        try:
            with open(BYPASS_TELEMETRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["bypass_channel"] = "WATERWAY"
            return data
        except Exception:
            pass

    # 3. Natural Circadian Fallback
    hour = time.localtime().tm_hour
    circadian_voltage = 0.6 + 0.3 * math.cos((hour - 12) * math.pi / 12)
    return {
        "source_model": "circadian-natural-law-fallback",
        "bypass_channel": "CIRCADIAN",
        "grid_metrics": {
            "voltage_level_rms": float(circadian_voltage),
            "transformer_temp_c": 36.5,
            "load_factor": 0.5,
            "active_frequency_hz": 60.0
        }
    }

def run_grid():
    print("⚡====================================================⚡")
    print("   ELYSIA SEED: 말단 수용가 수전 제어반 (LOAD CONTROL PANEL)   ")
    print("⚡====================================================⚡")
    print("   * 명령 목록:")
    print("     - 0.0 ~ 1.0 : 수동 의지 인입 (독립 운전 모드)")
    print("     - sync      : 계통 연동 모드 <-> 독립 운전 모드 전환")
    print("     - status    : 현재 송배전망 상태 계측 요청")
    print("     - exit      : 시스템 정지 및 절전")
    print("--------------------------------------------------------")

    spine = VariableRotorSpine()
    load_memory(spine)

    grid_tied = False
    last_x = 0.5

    try:
        while True:
            mode_str = "🟢 GRID-TIED (계통 연동)" if grid_tied else "🟡 ISLAND (독립 운전)"
            user_input = input(f"\n[계통상태: {mode_str}] Will (x) 또는 명령 입력 >> ").strip().lower()

            if user_input == 'exit':
                break
            
            if user_input == 'status':
                grid_data = poll_substation()
                if grid_data:
                    print("\n📈 [변전소 계통 실시간 계측 데이터]")
                    print(f"   - 전원 공급처: {grid_data['source_model']}")
                    print(f"   - 송전 전압(RMS): {grid_data['grid_metrics']['voltage_level_rms']:.4f} V")
                    print(f"   - 변압기 온도  : {grid_data['grid_metrics']['transformer_temp_c']:.1f} °C")
                    print(f"   - 부하율       : {grid_data['grid_metrics']['load_factor'] * 100:.1f} %")
                    print(f"   - 계통 주파수  : {grid_data['grid_metrics']['active_frequency_hz']:.2f} Hz")
                else:
                    print("\n🔌 [계통 고장] 중앙 변전망에 연결할 수 없습니다.")
                continue

            if user_input == 'sync':
                grid_tied = not grid_tied
                if grid_tied:
                    grid_data = poll_substation()
                    if grid_data:
                        print("⚡ [Grid Link] 계통 동기화(Phase-Locking) 성공! 변전망 전류를 인입합니다.")
                    else:
                        print("⚠️ [Grid Fail] 변전망 연결 실패. 독립 운전 모드를 유지합니다.")
                        grid_tied = False
                else:
                    print("🔌 [Grid Break] 송배전 선로 해제. 독립 운전 모드로 복귀합니다.")
                continue

            if grid_tied:
                print("   (계통 연동 중: 전압을 동적 수전합니다. 30회 루프 기동...)")
                for loop_idx in range(30):
                    grid_data = poll_substation()
                    if not grid_data:
                        print("\n🚨 [계통 비상 탈조] 변전망 신호 단절! 독립 운전 모드로 복귀합니다.")
                        grid_tied = False
                        break
                    
                    bypass = grid_data.get("bypass_channel", "GRID")
                    channel_indicator = "🌊 [수류 우회]" if bypass == "WATERWAY" else ("🌱 [자연 섭리]" if bypass == "CIRCADIAN" else "⚡ [정상 송전]")
                    rms_voltage = grid_data['grid_metrics']['voltage_level_rms']
                    x = min(1.0, max(0.0, rms_voltage))
                    
                    metrics = spine.pulse(x)
                    status = spine.get_state_summary()
                    color = get_color_str(metrics['velocity'])
                    lum_str = "✧" * int(metrics['luminosity'] * 10)
                    
                    print(f"   {channel_indicator} [{loop_idx:02d}] {status} | 계통공명: {color} | 수전압:{rms_voltage:.4f}V | 輝度:{lum_str}")
                    time.sleep(0.1)
                continue

            # Island Mode manual input
            if not user_input:
                continue

            try:
                x = float(user_input)
                x = min(1.0, max(0.0, x))
            except ValueError:
                print("! [경고] 올바른 수치 또는 명령어가 아닙니다.")
                x = last_x

            print("\n[ 독립 발전기 가동 및 평형 정렬 중 ]")
            for i in range(20):
                metrics = spine.pulse(x)
                status = spine.get_state_summary()
                color = get_color_str(metrics['velocity'])
                lum_str = "✧" * int(metrics['luminosity'] * 10)

                if abs(metrics['velocity'] - spine.dawn_freq) < 0.05 and metrics['luminosity'] < 0.1:
                    dawn_msg = "── ✧ ── 기저 평형(Dawn Silver-Gold) 도달"
                else:
                    dawn_msg = f"Spectrum: {color}"

                print(f"   [{i:02d}] {status} | {dawn_msg} | 輝度:{lum_str}")
                time.sleep(0.05)

            last_x = x

    except KeyboardInterrupt:
        print("\n[수전반] 긴급 셧다운 지시 수신.")
    finally:
        save_memory(spine)

if __name__ == "__main__":
    run_grid()
