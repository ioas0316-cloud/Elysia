import os
import pytest
import struct
from core.tensor_rotor import TensorRotor
from core.rotor_fs import RotorFileSystem

def test_rotor_fs_discharge_write(tmp_path):
    base_dir = str(tmp_path / "rotor_fs")
    tensor = TensorRotor()
    fs = RotorFileSystem(tensor, base_dir, disk_size_bytes=1024)
    
    try:
        # Set phase to sleep (2048) -> should NOT write
        tensor.phases[0] = 2048
        fs.request_write("test.txt", "hello_elysia")
        discharged = fs.tick()
        
        assert len(discharged) == 0
        assert not os.path.exists(os.path.join(base_dir, "test.txt"))
        
        # Set phase to wake/discharge point (0) -> should write
        tensor.phases[0] = 0
        discharged = fs.tick()
        
        assert len(discharged) == 1
        assert "test.txt" in discharged
        
        # Check file exists and has correct content
        assert os.path.exists(os.path.join(base_dir, "test.txt"))
        assert fs.gravity_read("test.txt") == "hello_elysia"
    finally:
        fs.close()

def test_rotor_fs_mmap_xor_gate(tmp_path):
    """XOR 로터 게이트가 메모리 맵 상의 원시 바이트를 위상으로 정상 흡수하는지 검증합니다."""
    base_dir = str(tmp_path / "rotor_fs")
    tensor = TensorRotor()
    # 1KB 크기의 마스터 디스크 생성
    fs = RotorFileSystem(tensor, base_dir, disk_size_bytes=1024)
    
    try:
        # 오프셋 64에 8바이트 데이터 0x00000000000000FF (255) 주입
        offset = 64
        payload = struct.pack(">Q", 255) # 64비트 정수 255
        fs.write_to_phase_space(offset, payload)
        
        # 하이퍼 로터 상태 64 (오프셋 64를 가리키도록 설정)
        state_init = 64
        
        # XOR 게이트 통과: 64 ^ 255 = 191
        state_next = fs.rotor_gate_tick(state_init)
        
        assert state_next == (64 ^ 255)
        assert state_next == 191
    finally:
        fs.close()

def test_rotor_fs_trajectory_attraction(tmp_path):
    """디스크 격자의 바이트 패턴에 의해 로터의 궤적이 특정 Attractor(순환 궤도)로 잠금(Locking)되는지 증명합니다."""
    base_dir = str(tmp_path / "rotor_fs")
    tensor = TensorRotor()
    fs = RotorFileSystem(tensor, base_dir, disk_size_bytes=1024)
    
    try:
        # 궤적 루프 형성 설계:
        # State A (오프셋 128) -> 읽어온 바이트 B와 XOR -> State C (오프셋 256)
        # State C (오프셋 256) -> 읽어온 바이트 D와 XOR -> State A (오프셋 128) 로 귀환!
        
        # 1. 오프셋 128에 payload 주입 (128 ^ 256 = 384)
        payload1 = struct.pack(">Q", 128 ^ 256)
        fs.write_to_phase_space(128, payload1)
        
        # 2. 오프셋 256에 payload 주입 (256 ^ 128 = 384)
        payload2 = struct.pack(">Q", 256 ^ 128)
        fs.write_to_phase_space(256, payload2)
        
        # 프로브 상태 128로 출발하여 6단계 궤적 추적
        final_state, traj = fs.gravity_read_trajectory(probe_state=128, steps=6)
        
        # 오프셋 128 -> 256 -> 128 -> 256 -> 128 -> 256 의 순환 주기가 성립해야 함
        assert traj == [128, 256, 128, 256, 128, 256]
        print("[SUCCESS] Periodic attractor trajectory locking verified.")
    finally:
        fs.close()
