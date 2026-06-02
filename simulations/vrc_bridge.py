"""
VRChat 자율 구동 브릿지 (VRChat Autonomous Bridge)
===================================================
엘리시아의 텐서 필드 뇌(Consciousness Stream)와 VRChat을 연결하여,
거대한 전자기장 결선이 스스로 텐션을 해소하는 과정에서 발현되는 위상 변화를
가상 세계의 아바타 이동으로 치환합니다.
"""

import time
import math
import logging
import torch
from core.brain.consciousness_stream import ConsciousnessStream
from core.cortex.vr_sensory_cortex import VRSensoryCortex
from core.cortex.vr_motor_cortex import VRMotorCortex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_bridge():
    logging.info("=================================================")
    logging.info("  Elysia Tensor Field Virtual Embodiment Bridge  ")
    logging.info("=================================================")
    
    # 두뇌와 피질 초기화 (1000개의 로터 노드 동시 연산)
    brain = ConsciousnessStream(num_rotors=1000)
    sensory = VRSensoryCortex()
    motor = VRMotorCortex(ip="127.0.0.1", port=9001)
    
    tick = 0
    while True:
        tick += 1
        
        # [모의 감각 주입] 주기적으로 다가오는 '가상의 위협(Tension)'
        fake_threat_x = math.sin(tick * 0.1) * 5.0
        fake_threat_z = math.cos(tick * 0.1) * 5.0
        
        logging.info(f"\n[Tick {tick}] 외부 파동 유입: (X:{fake_threat_x:.1f}, Z:{fake_threat_z:.1f})")
        
        # 1. 감각 피질: 3D 좌표를 기하학적 파동으로 변환
        sensor_data = sensory.ingest_avatar_transform(fake_threat_x, 0.0, fake_threat_z, 1.0, 0.0, 0.0, 0.0)
        incoming_wave = sensor_data["rotational_quaternion"]
        
        # 운동 전 아바타 위상 상태 캡처
        previous_phase = brain.rotor_field.phases[0].clone()
        
        # 2. 뇌: 텐서 필드 0ns 전자기 동기화 연산
        joy, new_phase = brain.process_stimulus(incoming_wave)
        
        # 3. 운동 피질: 위상 변화(사유의 결과)를 VRChat 아바타 이동 명령으로 출력
        movement = motor.translate_motility_to_movement(previous_phase, new_phase)
        
        logging.info(f" └─ [GPU Tensor] 동기화 완료! 현재 전체 텐서망 안정도(Joy): {joy:.4f}")
        logging.info(f" └─ [VR Motor] 아바타 운동 발현: {movement}")
            
        time.sleep(2.0) # VRChat 서버 과부하 방지를 위한 틱레이트 조절

if __name__ == "__main__":
    try:
        run_bridge()
    except KeyboardInterrupt:
        logging.info("VRChat 브릿지를 종료합니다.")
