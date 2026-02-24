"""
PID Feedback Controller (L6 Structure)
======================================
"The Nunchi System (Tact)."

Proportional-Integral-Derivative Controller for Cognitive Temperature.
Regulates Emotional RPM to prevent overshoot (Manic) or undershoot (Depressive).

P: React to the moment.
I: Remember the context (Karma).
D: Predict the trend (Intuition).
"""

class PIDController:
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp # Proportional Gain
        self.ki = ki # Integral Gain
        self.kd = kd # Derivative Gain

        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, target_value: float, current_value: float, dt: float = 1.0) -> float:
        """
        Calculates the control variable (Adjustment) to reach the target state.
        """
        error = target_value - current_value

        # P Term
        p_out = self.kp * error

        # I Term
        self.integral += error * dt
        i_out = self.ki * self.integral

        # D Term
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative

        output = p_out + i_out + d_out

        self.prev_error = error

        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

# Global Instance for Emotional Regulation
emotional_pid = PIDController(kp=0.5, ki=0.05, kd=0.1)
