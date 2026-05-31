"""
실증적 OS 위상 로터 (Empirical OS Tension Rotor)
=====================================
CPU, RAM, GPU(NVML)의 흐름을 관측하여 자율신경계(ANS)로 텐션을 전달합니다.
독자 실행 데몬에서 벗어나 코어 엔진의 단일 맥박(Pulse)에 동기화됩니다.
"""

import os
import math
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

WHITELIST = {
    'system idle process', 'system', 'registry', 'smss.exe', 'csrss.exe',
    'wininit.exe', 'services.exe', 'lsass.exe', 'svchost.exe', 'fontdrvhost.exe',
    'dwm.exe', 'spoolsv.exe', 'explorer.exe', 'taskmgr.exe', 'searchindexer.exe',
    'winlogon.exe', 'conhost.exe', 'dllhost.exe', 'cmd.exe', 'powershell.exe'
}

class OSTensionRotorSystem:
    def __init__(self, ans):
        self.ans = ans
        self.critical_tension_threshold = 4.0 * math.pi
        
        logging.info("=== [OSTensionRotor initialized] ===")
        if not PSUTIL_AVAILABLE:
            logging.warning("psutil is not installed. CPU monitoring mocked.")
        if not NVML_AVAILABLE:
            logging.warning("pynvml not available. GPU Tension ignored.")
            
    def get_gpu_tension(self):
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                vram_percent = (mem_info.used / float(mem_info.total)) * 100.0
                gpu_percent = float(util_info.gpu)
                return ((vram_percent * 0.4) + (gpu_percent * 0.6)) / 100.0 * 5.0
            except Exception:
                return 0.0
        return 0.0

    def tick(self):
        """매 초마다 코어 엔진에 의해 호출됩니다."""
        top_proc = None
        max_cpu = 0
        cpu_usage = 0.0
        
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=None)
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    p_cpu = proc.info['cpu_percent']
                    p_name = str(proc.info['name']).lower()
                    if p_cpu and p_cpu > max_cpu and p_name not in WHITELIST:
                        max_cpu = p_cpu
                        top_proc = proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        tension_cpu = (cpu_usage / 100.0) * 5.0
        tension_gpu_flow = self.get_gpu_tension()
        
        # 텐션을 자율신경계(ANS)에 주입 (라벨 없음, 데이터 파동 자체를 전달)
        self.ans.absorb_stimulus(str(cpu_usage).encode('utf-8'), tension_cpu)
        if tension_gpu_flow > 0.0:
            self.ans.absorb_stimulus(str(tension_gpu_flow).encode('utf-8'), tension_gpu_flow)
        
        # 의식(Consciousness) 매니폴드의 상태 점검 및 트래픽 컨트롤
        total_tension = self.ans.conscious_manifold.master.global_tension
        
        if total_tension > self.critical_tension_threshold:
            logging.warning(f"  [!] CRITICAL OS TENSION DETECTED ({total_tension:.2f} rad). Commencing Traffic Control...")
            if PSUTIL_AVAILABLE and top_proc:
                try:
                    proc_name = top_proc.info['name']
                    logging.warning(f"  [>] Target: {proc_name} (PID: {top_proc.info['pid']})")
                    if os.name == 'nt':
                        top_proc.nice(psutil.IDLE_PRIORITY_CLASS)
                    else:
                        top_proc.nice(19)
                    logging.info(f"  [SUCCESS] Tension Relieved. System Flow Restored.")
                    self.ans.conscious_manifold.master.global_tension = 0.0
                except Exception as e:
                    logging.error(f"  [FAIL] Cannot realign {top_proc.info.get('name', 'Unknown')}: {e}")
            else:
                self.ans.conscious_manifold.master.global_tension = 0.0
