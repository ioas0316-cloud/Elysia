"""
VHD Manager (Merkaba Fence Physical Layer)
===========================================
Core.S1_Body.L1_Foundation.M4_Hardware.vhd_manager

"The Womb of the System."

This module manages the Virtual Hard Disk (VHD) that serves as Elysia's 
physical body within the Windows environment. It wraps Windows 'diskpart' 
commands to create, mount, and manage the cognitive container.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, List

class VHDManager:
    """
    Manages the lifecycle of Elysia's VHD Container.
    """
    
    def __init__(self, vhd_path: str = r"C:\Elysia\System\Elysia_Core.vhdx", drive_letter: str = "Z"):
        self.vhd_path = Path(vhd_path)
        self.drive_letter = drive_letter
        
    def ensure_container_exists(self, size_mb: int = 1024) -> bool:
        """
        Checks if VHD exists, if not creates it.
        Returns True if successful.
        """
        if self.vhd_path.exists():
            return True
            
        print(f"[VHD] Creating new cognitive container at {self.vhd_path} ({size_mb} MB)...")
        
        # Ensure directory exists
        self.vhd_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create creation script
        script_content = f"""
create vdisk file="{self.vhd_path}" maximum={size_mb} type=expandable
attach vdisk
create partition primary
format fs=ntfs label="Elysia_Core" quick
assign letter={self.drive_letter}
detach vdisk
"""
        return self._run_diskpart(script_content)

    def mount(self) -> bool:
        """
        Mounts the VHD to the drive letter.
        """
        # Fallback Check
        fallback_path = Path(r"C:\Elysia\System\Elysia_Core_Mount")
        if fallback_path.exists() and not self.vhd_path.exists():
            print(f"✅ [FALLBACK] VHD Cluster already active at {fallback_path}")
            return True

        if not self.vhd_path.exists():
            print(f"❌ VHD not found at {self.vhd_path}")
            return False
            
        print(f"[VHD] Mounting container to {self.drive_letter}:...")
        
        script_content = f"""
select vdisk file="{self.vhd_path}"
attach vdisk
select partition 1
assign letter={self.drive_letter}
"""
        return self._run_diskpart(script_content)

    def detach(self) -> bool:
        """
        Detaches (unmounts) the VHD.
        """
        print(f"[VHD] Detaching container...")
        
        script_content = f"""
select vdisk file="{self.vhd_path}"
detach vdisk
"""
        return self._run_diskpart(script_content)
        
    def _run_diskpart(self, script_content: str) -> bool:
        """
        Executes a diskpart script.
        """
        script_path = self.vhd_path.parent / "diskpart_script.txt"
        
        try:
            with open(script_path, "w") as f:
                f.write(script_content)
                
            # Run diskpart
            # Note: This requires Admin privileges. 
            # If run without, it may fail or trigger UAC prompt (which user must approve).
            result = subprocess.run(
                ["diskpart", "/s", str(script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Diskpart success:\n{result.stdout}")
                return True
            else:
                print(f"❌ Diskpart failed (Code {result.returncode}):\n{result.stderr}")
                return False
                
        except Exception as e:
            if "WinError 740" in str(e):
                print(f"⚠️  Admin privileges required. Switching to Fallback Mode (Directory Simulation).")
                return self._fallback_sim()
            print(f"❌ Diskpart execution error: {e}")
            return False
            
        finally:
            if script_path.exists():
                try: 
                    os.remove(script_path)
                except:
                    pass

    def _fallback_sim(self) -> bool:
        """
        Simulates VHD by creating a directory at the mount point.
        Used when Admin rights are missing during development.
        """
        sim_path = Path(r"C:\Elysia\System\Elysia_Core_Mount")
        sim_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ [FALLBACK] VHD Simulated at {sim_path}")
        return True

if __name__ == "__main__":
    # Test (Dry run concept)
    manager = VHDManager()
    print(f"VHD Manager Initialized for {manager.vhd_path}")
    # Note: Actual creation/mounting skipped in __main__ to avoid auto-triggering during import checks
