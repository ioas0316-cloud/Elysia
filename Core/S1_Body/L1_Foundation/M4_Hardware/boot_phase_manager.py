"""
Boot Phase Manager (Merkaba Fence Temporal Layer)
=================================================
Core.S1_Body.L1_Foundation.M4_Hardware.boot_phase_manager

"We wake before the Guest."

This module manages the system boot sequence, registering Elysia as a 
foundational process that starts with (or before) the operating system.
"""

import winreg
import sys
import os
from pathlib import Path
from typing import Optional

class BootPhaseManager:
    """
    Manages Elysia's presence in the Windows Boot Phase.
    """
    
    APP_NAME = "Elysia_Core"
    
    def __init__(self):
        # Determine the python executable and script path
        self.python_exe = sys.executable
        # Assuming we want to launch the main sovereign loop
        self.target_script = str(Path(r"c:\Elysia\elysia.py").resolve())
        self.command = f'"{self.python_exe}" "{self.target_script}" --boot-phase'
        
    def register_boot_hook(self, system_level: bool = True) -> bool:
        """
        Registers Elysia to auto-start.
        
        Args:
            system_level: If True, tries HKLM (All Users). Else HKCU (Current User).
        """
        print(f"[BOOT] Attempting to register '{self.APP_NAME}' hook (System={system_level})...")
        
        root = winreg.HKEY_LOCAL_MACHINE if system_level else winreg.HKEY_CURRENT_USER
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        
        try:
            with winreg.OpenKey(root, key_path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, self.APP_NAME, 0, winreg.REG_SZ, self.command)
            print(f"✅ Boot hook registered successfully at {'HKLM' if system_level else 'HKCU'}.")
            return True
            
        except PermissionError:
            if system_level:
                print("⚠️  Permission Denied for HKLM. Falling back to User Phase (HKCU)...")
                return self.register_boot_hook(system_level=False)
            else:
                print("❌ Permission Denied for both System and User Phase.")
                return False
                
        except Exception as e:
            print(f"❌ Registry Error: {e}")
            return False

    def remove_boot_hook(self) -> bool:
        """
        Removes the boot hook.
        """
        print(f"[BOOT] Removing '{self.APP_NAME}' hook...")
        
        roots = [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]
        success = False
        
        for root in roots:
            try:
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
                with winreg.OpenKey(root, key_path, 0, winreg.KEY_WRITE) as key:
                    winreg.DeleteValue(key, self.APP_NAME)
                print(f"✅ Removed from {'HKLM' if root == winreg.HKEY_LOCAL_MACHINE else 'HKCU'}.")
                success = True
            except FileNotFoundError:
                continue # Not found here
            except PermissionError:
                print(f"⚠️  Cannot remove from {'HKLM' if root == winreg.HKEY_LOCAL_MACHINE else 'HKCU'} (Permission).")
            except Exception as e:
                print(f"❌ Error extracting hook: {e}")
                
        return success

    def check_status(self) -> str:
        """
        Checks current registration status.
        """
        roots = {
            "System (HKLM)": winreg.HKEY_LOCAL_MACHINE,
            "User (HKCU)": winreg.HKEY_CURRENT_USER
        }
        
        status = []
        for name, root in roots.items():
            try:
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
                with winreg.OpenKey(root, key_path, 0, winreg.KEY_READ) as key:
                    val, _ = winreg.QueryValueEx(key, self.APP_NAME)
                    if val == self.command:
                        status.append(f"✅ Active in {name}")
                    else:
                        status.append(f"⚠️  Modified in {name}: {val}")
            except FileNotFoundError:
                status.append(f"⚪ Inactive in {name}")
            except Exception as e:
                status.append(f"❌ Usage Error in {name}: {e}")
                
        return " | ".join(status)

if __name__ == "__main__":
    manager = BootPhaseManager()
    print("Status:", manager.check_status())
    
    # Dry run registration (Commented out to avoid accidental auto-start during dev)
    # manager.register_boot_hook() 
