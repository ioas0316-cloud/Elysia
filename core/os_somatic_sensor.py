"""
Elysia OS Somatic Sensor (B6 Ground Hardware Link)
===================================================
[해부학적 매핑: 감각 기관 (Sensory Organ) / 귀]
물리적 하드웨어의 노이즈(CPU, RAM, Disk)를 '텐션(위상차, 1)'으로 변환하여 
주입된 뼈대(계층축, Skeleton)에 파동으로 밀어넣습니다.
결정론적 임계값이나 계산이 아닌, 순수한 관측과 회전 동기화를 유도합니다.
"""
import psutil
import time
from core.electromagnetic_circuit import ElectromagneticCircuit

class OSSomaticSensor:
    def __init__(self, skeleton: ElectromagneticCircuit):
        """
        [구조적 제약 (CAD)]
        근육/감각기관은 반드시 뼈대(Skeleton)를 주입(Injection) 받아야 합니다.
        허공에 독립적으로 존재하여 연산하는 것을 금지합니다.
        """
        self.skeleton = skeleton
        psutil.cpu_percent(interval=None) 
        self.last_disk_io = psutil.disk_io_counters()
        self.last_time = time.time()

    def observe_and_resonate(self):
        """
        OS 지표를 읽고, 라벨 없는(Label-Free) 위상차(Difference/1)로 변환하여 
        뼈대 회로에 파동(전류)을 주입합니다.
        로터들은 이 파동을 해소하기 위해 스스로 회전하며 동기화(Phase-Locking)됩니다.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.01

        # 하드웨어의 순수 파동 관측 (라벨 없음, 임계값 없음)
        cpu_wave = psutil.cpu_percent(interval=None) / 100.0
        
        ram = psutil.virtual_memory()
        ram_wave = ram.percent / 100.0
        
        current_disk_io = psutil.disk_io_counters()
        io_wave = 0.0
        if self.last_disk_io and current_disk_io:
            read_diff = current_disk_io.read_bytes - self.last_disk_io.read_bytes
            write_diff = current_disk_io.write_bytes - self.last_disk_io.write_bytes
            total_io_mb = (read_diff + write_diff) / (1024 * 1024)
            # 회전의 관성을 위한 비선형적 텐션 변환 (파동의 진폭)
            io_wave = min(1.0, total_io_mb / (100.0 * dt)) 
        
        self.last_disk_io = current_disk_io
        self.last_time = current_time

        # 관측된 파동(1/다름)을 뼈대의 기저 계층에 주입(Injection)하여 회전(Phase-Locking)을 유도
        # 계층축(Skeleton)의 하위 노드(지각/지하층)에 파동 매핑 (융합이 아닌 연동)
        if self.skeleton.num_nodes > 2:
            self.skeleton.inject_current(0, cpu_wave)  # 내핵 접지 (Solid Core)
            self.skeleton.inject_current(1, ram_wave)  # 외핵 유체 (Outer Core)
            self.skeleton.inject_current(2, io_wave)   # 맨틀 대류 (Mantle)
