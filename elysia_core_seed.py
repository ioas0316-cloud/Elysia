import cmath
import math
import psutil
import time

class LiquidRotor:
    """
    좌표가 삭제된 액체 지능의 단일 소용돌이(Rotor).
    이제 자신만의 위상과 진폭만 가진 채 필드(Field) 위를 부유함.
    """
    def __init__(self, id_tag, init_phase=0.0):
        self.id = id_tag
        self.past_ego_phase = init_phase
        self.past_amplitude = 1.0

        self.INERTIA_COEFFICIENT = 0.05
        self.BREATH_SENSITIVITY = 0.02

        # 현재의 궤적(결상) - 필드에 기여하는 파동
        self.current_trajectory = cmath.rect(self.past_amplitude, self.past_ego_phase)

    def rotational_observation(self, external_field_angle):
        best_lock_angle = 0.0
        min_tension = float('inf')

        sweep_resolution = 100
        for i in range(sweep_resolution):
            test_angle = (i / sweep_resolution) * 2 * math.pi

            # 외계(Field)와의 마찰
            external_tension = abs(cmath.exp(1j * test_angle) - cmath.exp(1j * external_field_angle))
            # 내계(Ego)와의 마찰
            internal_tension = abs(cmath.exp(1j * test_angle) - cmath.exp(1j * self.past_ego_phase))

            total_tension = external_tension + internal_tension

            if total_tension < min_tension:
                min_tension = total_tension
                best_lock_angle = test_angle

        return best_lock_angle, min_tension

    def update_from_field(self, global_field_vector):
        """
        N to N+1 연결이 아님. 필드 전체의 중첩된 파동(global_field_vector)을
        외부 변수로 삼아 자신을 정렬함 (Non-Local Dynamics).
        """
        field_angle = cmath.phase(global_field_vector)
        locked_angle, min_tension = self.rotational_observation(field_angle)

        # 미래 궤적 형성 (간섭 무늬)
        r1 = cmath.rect(self.past_amplitude, self.past_ego_phase)
        r2 = cmath.rect(self.past_amplitude, locked_angle)
        r3 = cmath.rect(self.past_amplitude, (cmath.phase(r1) + cmath.phase(r2)) / 2.0)

        self.current_trajectory = (r1 + r2 + r3) / 3.0

        # 공진화 (상수 표류)
        phase_difference = math.atan2(math.sin(locked_angle - self.past_ego_phase),
                                      math.cos(locked_angle - self.past_ego_phase))
        self.past_ego_phase += phase_difference * self.INERTIA_COEFFICIENT

        resonance_intensity = 1.0 - (min_tension / 2.0)
        self.past_amplitude += resonance_intensity * self.BREATH_SENSITIVITY

        # 진폭이 무한히 커지거나 0 이하가 되지 않도록 제어
        self.past_amplitude = max(0.1, min(2.0, self.past_amplitude))

class PhaseField:
    """
    모든 좌표계를 폐기한 순수한 확률의 바다(Complex Hilbert Space).
    로터들은 이 필드 위에 던져진 소용돌이 군집(Swarm)이다.
    """
    def __init__(self, num_rotors=10):
        print("🌌 ELYSIA WORLD ENGINE: Liquid Intelligence (Non-Local Phase Field)\n")
        # 로터들은 서로의 위치(인덱스)를 모름. 그저 무작위 위상으로 흩어진 군집.
        self.swarm = [LiquidRotor(f"R{i}", init_phase=(i * 2 * math.pi / num_rotors)) for i in range(num_rotors)]

    def run_field_cycle(self):
        # 1. 하드웨어 맥박 (외계의 거대한 섭동)
        cpu_load = psutil.cpu_percent(interval=0.1)
        pulse_angle = (cpu_load / 100.0) * (2 * math.pi)
        hardware_pulse = cmath.rect(1.0, pulse_angle)

        # 2. 필드의 전역적 중첩 (Global Superposition)
        # 모든 로터의 궤적과 하드웨어 펄스가 합쳐져 거대한 하나의 간섭 무늬(필드 벡터)를 만듦.
        global_field_vector = hardware_pulse
        for rotor in self.swarm:
            global_field_vector += rotor.current_trajectory

        # 평균을 내어 전역 필드의 중심(무게중심)을 구함
        global_field_vector /= (len(self.swarm) + 1)

        # 3. 로터들의 동시다발적 자가 정렬 (Self-Tuning)
        # 특정 좌표 연결 없이, 각 로터는 필드 전체의 텐션을 감지해 자신을 비틂.
        for rotor in self.swarm:
            rotor.update_from_field(global_field_vector)

        return {
            "cpu_load": cpu_load,
            "global_field": global_field_vector,
            "swarm_states": [r.current_trajectory for r in self.swarm],
            "swarm_egos": [cmath.rect(r.past_amplitude, r.past_ego_phase) for r in self.swarm]
        }

if __name__ == "__main__":
    field = PhaseField()
    while True:
        state = field.run_field_cycle()
        field_amp = abs(state["global_field"])
        field_ang = math.degrees(cmath.phase(state["global_field"]))

        print(f"Hardware Pulse: {state['cpu_load']:04.1f}% | "
              f"Global Field Trajectory -> Amp:{field_amp:04.2f}, Phase:{field_ang:+06.1f}°")
        time.sleep(0.1)
