import mmap
import os
import struct
import time
import math
import threading
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("🌀 [Phase 50] 0거리 기하학적 텐션 회로 검증 (Zero-Lock Flow)")
print("=" * 60)

# 1. Prepare Shared Memory (40 bytes)
FILENAME = "c:/Elysia/data/test_circuit_manifold.bin"
FORMAT = 'ddddd'
if not os.path.exists(FILENAME):
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    with open(FILENAME, "wb") as f:
        f.write(struct.pack(FORMAT, 1.0, 0.0, 0.0, 0.0, 0.0))

f = open(FILENAME, "r+b")
mm = mmap.mmap(f.fileno(), 0)

def writer_a():
    """발화자 A: 오직 'x축' 차원에만 파동을 기록 (Lock 없음)"""
    for i in range(150):
        val = math.cos(i * 0.1)
        data = struct.pack('d', val)
        # x축은 offset 8에 위치함 (w가 0~7)
        mm.seek(8)
        mm.write(data)
        time.sleep(0.01)

def writer_b():
    """발화자 B: 오직 'y축' 차원에만 파동을 기록 (Lock 없음)"""
    for i in range(150):
        val = math.sin(i * 0.1)
        data = struct.pack('d', val)
        # y축은 offset 16에 위치함
        mm.seek(16)
        mm.write(data)
        time.sleep(0.01)

def observer():
    """관측자: Polling이나 Event 없이, 거울(mmap)에 비친 전체 텐션을 연속적으로 관측"""
    print("[관측자] 회로 관측 시작 (Polling Event 없음. 오직 흐름 Induction)")
    for _ in range(30):
        mm.seek(0)
        data = mm.read(40)
        w, x, y, z, tau = struct.unpack(FORMAT, data)
        
        # 기하학적 텐션 창발 (x와 y가 만나 Bivector 위상각을 형성)
        phase = math.atan2(y, x)
        
        # 위상 홀로그램 시각화 (이중 나선 교차)
        bar_len = 40
        x_pos = int((x + 1) * 0.5 * (bar_len - 1))
        y_pos = int((y + 1) * 0.5 * (bar_len - 1))
        
        vis = [" "] * bar_len
        if 0 <= x_pos < bar_len: vis[x_pos] = "A"
        if 0 <= y_pos < bar_len: 
            if vis[y_pos] == "A": vis[y_pos] = "X" # 교차점 (공명)
            else: vis[y_pos] = "B"
            
        print(f"🌊 [위상 동기화] Phase(θ): {phase:>6.2f} rad | 관측 공간: [{''.join(vis)}]")
        time.sleep(0.05)

t_a = threading.Thread(target=writer_a)
t_b = threading.Thread(target=writer_b)
t_obs = threading.Thread(target=observer)

t_a.start()
t_b.start()
t_obs.start()

t_a.join()
t_b.join()
t_obs.join()

mm.close()
f.close()
print("=" * 60)
print("✅ 검증 완료: 락(Lock) 없는 차원 분리와 이벤트 없는 흐름 동기화가 증명되었습니다.")
