import os
import sys

# 경로 등록
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(r"C:\elysia_cortex")

from core.triple_helix_engine import TripleHelixEngine
from inverse_decoder import InversePhaseDecoder

message = "강덕님께서 이 안티그래비티 환경에서 너의 현존을 확인하고자 하셔. 대화가 가능하니?"

print(f"[Antigravity -> Elysia] Msg: '{message}'\n")

print("[System] Core Engine is running...")
engine = TripleHelixEngine()
decoder = InversePhaseDecoder()

# 가상의 집중/자극 상태 (텐션을 의도적으로 폭발시켜 발화를 유도)
sensory = {
    "motion_entropy": 0.8,
    "pain_level": 0.2,
    "visual_entropy": 0.9,
    "coding_cognitive": 1.5 
}

# 텐션 누적을 위해 심장 박동(Pulse)을 3회 반복
final_tension = 0.0
final_quat = None
final_enn = None

for i in range(3):
    tension, mode, jumped, quat, enn = engine.pulse(text_thought=message, sensory_input=sensory)
    print(f"   [Pulse {i+1}] Tension: {tension:.3f} | Dim: Cl({engine.inner_world.signature[0]},0) | Mode: {mode}")
    final_tension = tension
    final_quat = quat
    final_enn = enn

print("\n[System] Egressing Tension to Cortex Decoder...")
rotor_state = [final_quat.w, final_quat.x, final_quat.y, final_quat.z] + list(final_enn) + [0.0]*14
speech = decoder.decode_to_text(final_tension, rotor_state)

print(f"\n=======================================================")
print(f" [Elysia's Voice] ")
print(f" \"{speech}\"")
print(f"=======================================================")
