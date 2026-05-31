"""
Elysia Reality Floodgate Projector
==================================
마더보드의 진짜 I/O(네트워크 패킷, CPU, Disk)를 파싱하지 않고 
텐션과 위상각(Phase)으로 치환하여 엘리시아의 뿌리(mmap)에 직접 주입합니다.
"""
import sys
import os
import time
import math
import psutil

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.shared_manifold import SharedManifold
from core.math_utils import Quaternion

def run_floodgate():
    print("=" * 80)
    print(" 🌊 [Phase 73] 현실의 수문 개방 (Reality Floodgate Opened)")
    print("=" * 80)
    
    manifold = SharedManifold()
    
    # 베이스라인 계산을 위한 이전 값
    last_net = psutil.net_io_counters()
    last_disk = psutil.disk_io_counters()
    
    try:
        while True:
            time.sleep(0.2) # 초당 5회 관측
            
            current_net = psutil.net_io_counters()
            current_disk = psutil.disk_io_counters()
            cpu_percent = psutil.cpu_percent()
            
            # 네트워크 델타 (Bytes)
            net_delta = (current_net.bytes_recv - last_net.bytes_recv) + (current_net.bytes_sent - last_net.bytes_sent)
            # 디스크 델타 (Bytes)
            disk_delta = (current_disk.read_bytes - last_disk.read_bytes) + (current_disk.write_bytes - last_disk.write_bytes)
            
            last_net = current_net
            last_disk = current_disk
            
            # 날것의 트래픽을 텐션으로 치환 (로그 스케일링)
            # 유튜브 4K 등 폭주 시 tension이 1.0(또는 그 이상)에 근접하게 맵핑
            net_tension = math.log1p(net_delta) / 15.0
            disk_tension = math.log1p(disk_delta) / 15.0
            cpu_tension = cpu_percent / 100.0
            
            # 총합 텐션 (0.0 ~ 2.0+)
            total_tension = net_tension + disk_tension + cpu_tension
            
            # 위상각 변환 (텐션이 높을수록 위상이 크게 비틀림)
            theta = total_tension * math.pi
            q_real = Quaternion(math.cos(theta), math.sin(theta), 0.0, 0.0)
            
            # mmap(뿌리)에 다이렉트 주입
            manifold.write_phase(q_real, total_tension)
            
            # 시각화
            bar_length = 40
            filled = min(bar_length, int(total_tension * 15))
            bar = "█" * filled + "-" * (bar_length - filled)
            
            status = "안정 (Idle)"
            if total_tension > 0.8:
                status = "폭주 (Spike)!!!"
                
            sys.stdout.write(f"\r[OS I/O] {status:12s} | Tension: {total_tension:.3f} | {bar}")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n\n 🌊 [Floodgate] 현실 스트림 주입을 종료합니다.")
    finally:
        manifold.close()

if __name__ == "__main__":
    run_floodgate()
