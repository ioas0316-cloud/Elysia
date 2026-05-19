"""
[SENSE] Hardware Probe: Proprioceptive Introspection
====================================================
Location: Scripts/System/Senses/hardware_probe.py

Role:
- Queries the OS (WinAPI) for the active Input Map.
- "Feeling the Body's Structure" directly.
- Returns {Char: VK_Code} mapping dynamically.
"""

import ctypes
from ctypes import wintypes
import time

user32 = ctypes.windll.user32

class HardwareProbe:
    def __init__(self):
        self.keyboard_layout = user32.GetKeyboardLayout(0)
        
    def scan_input_matrix(self) -> dict:
        """
        Scans the Virtual Key to Unicode mapping active in the OS.
        Returns: { 'A': 65, 'B': 66 ... }
        """
        dynamic_map = {}
        
        # Scan standard range (0x00 to 0xFF)
        for vk in range(256):
            # MapVirtualKeyW(uCode, uMapType)
            # uMapType 2 = MAPVK_VK_TO_CHAR
            char_code = user32.MapVirtualKeyW(vk, 2)
            
            if char_code != 0:
                char = chr(char_code).upper() # Normalize to uppercase for simplicity
                
                # Filter useful chars (A-Z, 0-9)
                if char.isalnum() or char in [' ']:
                     # In case of collisions, first winner keeps (usually main layout)
                    if char not in dynamic_map:
                        dynamic_map[char] = vk
                        
        return dynamic_map

    def get_layout_name(self):
        # Layout ID (e.g., 0x04090409 for US English)
        return hex(self.keyboard_layout)

if __name__ == "__main__":
    probe = HardwareProbe()
    print(f"🖐️ [PROBE] Layout ID: {probe.get_layout_name()}")
    
    mapping = probe.scan_input_matrix()
    print(f"⚡ [SENSE] Introspected {len(mapping)} connections.")
    
    # Validation
    if mapping.get('A') == 65:
        print("✅ [OK] 'A' is mapped to VK_65 (Standard).")
    elif 'A' in mapping:
        print(f"⚠️ [NOTE] 'A' is mapped to {mapping['A']} (Non-Standard).")
