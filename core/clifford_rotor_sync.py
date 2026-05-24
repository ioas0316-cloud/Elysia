import math
import time

class DynamicPIDController:
    """
    PID Controller where gains (P, I, D) dynamically scale based on
    the system's tension and fractal scale.
    """
    def __init__(self, base_scale=1.0):
        self.integral = 0.0
        self.prev_error = 0.0
        self.base_scale = base_scale

    def get_dynamic_gains(self, tension: float):
        """
        Calculates dynamic PID gains based on current tension.
        Higher tension (chaos) -> stronger damping, smaller integral.
        """
        # Dynamic variable mapping, avoiding fixed hardcoded constants.
        kp = 0.5 * self.base_scale * (1.0 + tension)
        ki = 0.01 * self.base_scale / (1.0 + tension)
        kd = 0.1 * self.base_scale * (1.0 + tension * 2.0)

        return kp, ki, kd

    def discharge_error_to_ground(self, error: float, tension: float, dt: float) -> float:
        """
        방전(Discharge): Calculates the correction value to apply to the loop
        to phase-lock and sink error to 0.
        """
        if dt <= 0.0:
            dt = 1e-6

        kp, ki, kd = self.get_dynamic_gains(tension)

        # Proportional
        p_term = kp * error

        # Integral
        self.integral += error * dt
        # Anti-windup cap
        if self.integral > 1.0: self.integral = 1.0
        if self.integral < -1.0: self.integral = -1.0
        i_term = ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / dt
        d_term = kd * derivative

        self.prev_error = error

        # Total discharge (correction)
        correction = p_term + i_term + d_term
        return correction

class BitwiseCliffordRotor:
    """
    단일 레이어 로터 동기화: CPU 클럭의 0과 1에 위상 각도를 다이렉트로 매핑.
    비트 연산을 사용하여 가장 빠르고 로우레벨하게 구현.
    """
    def __init__(self):
        # 16-bit representation for phases (0 to 65535 mapped to 0 to 2PI)
        self.phase_state = 0x0001
        self.PHASE_MASK = 0xFFFF

    def apply_clock_edge(self, is_rising: bool, tension: float):
        """
        Maps a clock edge (0 or 1) directly to a geometric bitwise rotation.
        is_rising=True(1), False(0).
        """
        # 텐션 강도를 16비트 순환 범위 내(0~15)로 안전하게 제한
        base_step = int(abs(tension) * 10) % 16
        if base_step == 0:
            base_step = 1

        if is_rising:
            # 1: 양각 -> Left shift circular
            self.phase_state = ((self.phase_state << base_step) | (self.phase_state >> (16 - base_step))) & self.PHASE_MASK
        else:
            # 0: 음각 -> Right shift circular
            self.phase_state = ((self.phase_state >> base_step) | (self.phase_state << (16 - base_step))) & self.PHASE_MASK

    def get_phase_angle(self) -> float:
        """
        Converts the bitwise state to a geometric angle [0, 2PI).
        """
        return (self.phase_state / float(self.PHASE_MASK + 1)) * 2.0 * math.pi

    def get_rotor_state_str(self) -> str:
        """
        Returns a binary string representation of the rotor state.
        """
        return f"{self.phase_state:016b}"
