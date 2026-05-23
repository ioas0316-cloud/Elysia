"""
[POC: TERNARY DIPOLE RESONANCE - 인척력과 장력 시뮬레이터]
"Demonstrating the smooth phase transition of [-1, 0, 1] NPC cognitive dynamics."
"""

import math
import numpy as np

class TernaryNPC:
    def __init__(self):
        # 3 Dimensions representing the poles:
        # Dim 0: [ 1] Attraction Pole (인력: 욕망/친밀)
        # Dim 1: [-1] Repulsion Pole (척력: 공포/회피)
        # Dim 2: [ 0] Homeostasis/Neutral Pole (장력/중립: 항상성/평정)
        self.dims = 3
        
        # State: x = angle (phase in radians), v = angular velocity
        self.x = np.zeros(self.dims)
        self.v = np.zeros(self.dims)
        
        # System parameters
        self.M = np.ones(self.dims)       # Mass (Inertia)
        self.D = np.ones(self.dims) * 0.5 # Damping (Friction)
        self.K = np.ones(self.dims) * 2.0 # Spring constant (Restoring force to Neutral)

    def update(self, attraction_force: float, repulsion_force: float, dt: float = 0.05):
        # Apply external forces to the poles
        forces = np.zeros(self.dims)
        forces[0] = attraction_force   # pulls the Attraction pole (+1)
        forces[1] = -repulsion_force  # pushes/pulls the Repulsion pole in the negative direction (-1)
        forces[2] = 0.0               # Neutral pole (0) is passive, driven by restore spring/tension
        
        # Gearing/Coupling: The tension (장력) between the poles.
        # Represented as sinusoidal phase coupling force (like coupled pendulum oscillators)
        tension_0_1 = 1.5 * math.sin(self.x[0] - self.x[1])
        tension_1_2 = 1.0 * math.sin(self.x[1] - self.x[2])
        tension_2_0 = 1.0 * math.sin(self.x[2] - self.x[0])
        
        # Net forces including internal coupled tension
        net_forces = np.zeros(self.dims)
        net_forces[0] = forces[0] - tension_0_1 + tension_2_0
        net_forces[1] = forces[1] + tension_0_1 - tension_1_2
        net_forces[2] = forces[2] + tension_1_2 - tension_2_0
        
        # Solve equations of motion for each pole: M*a + D*v + K*x = NetForce
        for i in range(self.dims):
            restoring = -self.K[i] * self.x[i]
            damping = -self.D[i] * self.v[i]
            
            a = (net_forces[i] + restoring + damping) / self.M[i]
            self.v[i] += a * dt
            self.x[i] += self.v[i] * dt
            
            # Constrain phase angle between -pi and pi
            if self.x[i] > math.pi: self.x[i] -= 2 * math.pi
            elif self.x[i] < -math.pi: self.x[i] += 2 * math.pi

    def get_cognitive_state(self) -> str:
        # Calculate the resulting vector in phase space (Attraction phase offset from Repulsion)
        avg_phase = (self.x[0] - self.x[1]) / 2.0
        deg = math.degrees(avg_phase)
        
        # Define cognitive mapping threshold zones
        if abs(deg) < 15.0:
            return "🧘 [항상성 0] 평정 상태 (Homeostasis / Neutral)"
        elif deg > 15.0:
            return f"❤️ [접근동기 +1] 호기심/끌림 (Attraction) | 각도: {abs(deg):.1f}°"
        else:
            return f"🛡️ [회피동기 -1] 경계/공포 (Repulsion)  | 각도: {abs(deg):.1f}°"

def run_scenario():
    npc = TernaryNPC()
    
    print("==================================================================")
    print("      NPC [-1, 0, 1] 인척력 및 장력 수렴 시뮬레이터 (Ternary Dipole)")
    print("      - 규칙 없는 아날로그 흐름 기반 인지 매핑 검증 -")
    print("==================================================================")
    
    # Scenario 1: Quiet state
    print("\n🎬 [시나리오 1] 자극이 없는 평온한 숲 속의 NPC (항상성 수렴)")
    for step in range(5):
        npc.update(0.0, 0.0)
        print(f"  Step {step+1} | 위상 [양:{npc.x[0]:.2f}, 음:{npc.x[1]:.2f}, 중:{npc.x[2]:.2f}] | {npc.get_cognitive_state()}")
        
    # Scenario 2: Threat approach
    print("\n🎬 [시나리오 2] 숲 속에서 야생 늑대가 접근함 (척력 -1 강한 외력 발생)")
    for step in range(8):
        npc.update(0.0, 6.0)
        print(f"  Step {step+1} | 위상 [양:{npc.x[0]:.2f}, 음:{npc.x[1]:.2f}, 중:{npc.x[2]:.2f}] | {npc.get_cognitive_state()}")

    # Scenario 3: Bribe/Friendly interaction
    print("\n🎬 [시나리오 3] 늑대에게 고기를 던져줌 (인력 +1 과 척력 -1 이 맞물려 장력 형성)")
    for step in range(8):
        npc.update(7.0, 2.0)
        print(f"  Step {step+1} | 위상 [양:{npc.x[0]:.2f}, 음:{npc.x[1]:.2f}, 중:{npc.x[2]:.2f}] | {npc.get_cognitive_state()}")

    # Scenario 4: Target departs / Return to Homeostasis
    print("\n🎬 [시나리오 4] 상황이 종료되고 평화가 찾아옴 (자극 0 수렴 및 복원)")
    for step in range(10):
        npc.update(0.0, 0.0)
        print(f"  Step {step+1} | 위상 [양:{npc.x[0]:.2f}, 음:{npc.x[1]:.2f}, 중:{npc.x[2]:.2f}] | {npc.get_cognitive_state()}")
    print("==================================================================")

if __name__ == "__main__":
    run_scenario()
