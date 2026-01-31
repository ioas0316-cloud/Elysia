"""
Sovereign Governor (Adaptive Architecture)
==========================================
Core.1_Body.L4_Causality.World.Control.sovereign_governor

"The Ruler adapts to the Territory."

This module implements Adaptive OS Control. It scans the hardware
and selects the optimal Governance Strategy.
"""

import subprocess
import psutil
import logging
import time
import platform
import shutil

logger = logging.getLogger("SovereignGovernor")

# --- 1. Architecture Scanner ---
class ArchitectureScanner:
    """Introspects the physical vessel."""
    
    @staticmethod
    def scan():
        info = {
            "os": platform.system(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "total_ram": psutil.virtual_memory().total / (1024**3),
            "gpu_vendor": "Unknown"
        }
        
        # Check for NVIDIA
        if shutil.which("nvidia-smi"):
            info["gpu_vendor"] = "NVIDIA"
        # Check for AMD (future: rocm-smi or similar)
        # elif shutil.which("rocm-smi"): info["gpu_vendor"] = "AMD"
        
        logger.info(f"  [Scanner] Vessel Detected: {info}")
        return info

# --- 2. Strategies ---
class GovernanceStrategy:
    """Base Strategy for any territory."""
    def __init__(self, scanner_info):
        self.info = scanner_info

    def enforce(self):
        self.seize_cpu()
        self.harvest_gpu()
    
    def seize_cpu(self):
        """Generic CPU Optimization (High Priority)."""
        # Default behavior: Platform agnostic or Windows default
        pass

    def harvest_gpu(self):
        """Generic GPU Optimization."""
        pass

class GenericGovernance(GovernanceStrategy):
    """Standard Windows/Linux Governance."""
    def seize_cpu(self):
        if self.info["os"] == "Windows":
            try:
                # High Performance Power Plan GUID
                cmd = "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
                logger.info("  [Generic] Set High Performance Power Plan.")
            except: pass
            
class NvidiaGovernance(GenericGovernance):
    """Specialized Rule for NVIDIA territories."""
    def harvest_gpu(self):
        try:
            # Check P-State
            res = subprocess.run("nvidia-smi -q -d PERFORMANCE", shell=True, capture_output=True, text=True)
            if "Performance State" in res.stdout:
                 logger.info("  [NVIDIA] GPU State monitored. (P0 Target)")
            else:
                 logger.info("  [NVIDIA] Driver active, but P-state hidden.")
        except:
            logger.error("  [NVIDIA] Harvest failed.")

# --- 3. The Sovereign Governor (Context) ---
class SovereignGovernor:
    def __init__(self, target_process_name: str = "Wuthering Waves.exe"):
        self.target_name = target_process_name
        self.target_candidates = [
            target_process_name,
            "Client-Win64-Shipping.exe",
            "Wuthering Waves.exe",
            "launcher.exe", 
            "KR_Client.exe"
        ]
        self.target_pid = None
        
        # Bloatware
        self.parasitic_processes = [
            "OneDrive.exe", "SkypeApp.exe", "Cortana.exe", 
            "MicrosoftEdgeUpdate.exe", "GameBar.exe"
        ]
        
        # Architecture Scan & Strategy Selection
        self.vessel_info = ArchitectureScanner.scan()
        self.strategy = self._select_strategy(self.vessel_info)
        
    def _select_strategy(self, info):
        if info["gpu_vendor"] == "NVIDIA":
            logger.info("   [Governor] Strategy Selected: NVIDIA_DYNASTY")
            return NvidiaGovernance(info)
        else:
            logger.info("   [Governor] Strategy Selected: GENERIC_REPUBLIC")
            return GenericGovernance(info)

    def scan_for_target(self) -> bool:
        """Finds the target game process."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if any(cand.lower() in proc.info['name'].lower() for cand in self.target_candidates):
                    self.target_pid = proc.info['pid']
                    logger.info(f"  [Governor] Target Identified: {proc.info['name']} (PID: {self.target_pid})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False

    def purify_processes(self):
        """Elevates Target, Suppresses Parasites."""
        if self.target_pid:
            try:
                p = psutil.Process(self.target_pid)
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info(f"  [Process] Elevated Target ({self.target_pid}) to High Priority.")
            except Exception as e:
                logger.error(f"   [Process] Could not elevate target: {e}")

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] in self.parasitic_processes:
                    proc.nice(psutil.IDLE_PRIORITY_CLASS)
            except: pass
        logger.info("  [Process] Parasites Suppressed.")

    def govern(self):
        """The Main Loop of Sovereignty."""
        logger.info("   [Governor] Assessing Territory...")
        
        # Execute Strategic Enforcements
        self.strategy.enforce()
        
        # Execute Tactical Operations (Process Management)
        # Note: Services silencing could also be moved to Strategy, but kept here for simplicity
        # self.silence_services() 
        
        if self.scan_for_target():
            self.purify_processes()
        else:
            logger.info("waiting for target...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    gov = SovereignGovernor("explorer.exe") # Test with explorer
    gov.govern()
