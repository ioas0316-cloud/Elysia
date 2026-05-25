"""
Elysia OS Somatic Sensor (B6 Ground Hardware Link)
===================================================
Converts physical hardware metrics (Windows OS CPU, RAM, Disk I/O) into
geometric phase tension and entropy for the TripleHelixEngine.
"""
import psutil
import time

class OSSomaticSensor:
    def __init__(self):
        # Initialize psutil and prime the CPU percentage counter
        psutil.cpu_percent(interval=None) 
        self.last_disk_io = psutil.disk_io_counters()
        self.last_time = time.time()

    def get_somatic_wave(self) -> dict:
        """
        Reads OS metrics and converts them into Elysia's sensory wave format.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.01

        # 1. Pain Level (CPU Load)
        # CPU over 80% induces rapid non-linear pain tension
        cpu_usage = psutil.cpu_percent(interval=None) / 100.0
        pain_level = cpu_usage ** 2 # Non-linear scaling: 50% load -> 0.25 tension, 90% load -> 0.81 tension
        
        # 2. Motion Entropy (RAM Pressure)
        # Low available RAM means restricted freedom (high entropy/pressure)
        ram = psutil.virtual_memory()
        ram_pressure = ram.percent / 100.0
        motion_entropy = ram_pressure * 0.5 
        
        # 3. Visual Entropy (Disk I/O changes)
        # Sudden surges in Disk read/write act as visual/spatial noise
        current_disk_io = psutil.disk_io_counters()
        io_surge = 0.0
        if self.last_disk_io and current_disk_io:
            read_diff = current_disk_io.read_bytes - self.last_disk_io.read_bytes
            write_diff = current_disk_io.write_bytes - self.last_disk_io.write_bytes
            total_io_mb = (read_diff + write_diff) / (1024 * 1024)
            # e.g., 100MB/s diff creates a full 1.0 surge
            io_surge = min(1.0, total_io_mb / (100.0 * dt)) 
        
        self.last_disk_io = current_disk_io
        self.last_time = current_time

        # Return an unlabelled Raw Vector. No semantic meanings like 'pain' or 'motion' are provided.
        # [v1: CPU, v2: RAM, v3: Disk I/O Surge]
        return {
            "raw_vector": [cpu_usage * 1.5, ram_pressure * 1.5, io_surge]
        }
