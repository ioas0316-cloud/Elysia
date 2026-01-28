import os
import re

# Final, exhaustive mapping for 7D Fractal Restoration
replacements = {
    # Class Name & Path Changes
    r'from Core\.L6_Structure\.Elysia\.sovereign_self import SovereignSelf': 'from Core.L1_Foundation.M1_Keystone.emergent_self import EmergentSelf as SovereignSelf',
    r'Core\.L6_Structure\.Elysia\.sovereign_self': 'Core.L1_Foundation.M1_Keystone.emergent_self',
    r'SovereignSelf\(': 'EmergentSelf(', # Direct instantiation repair
    
    # Nested Sub-package Fixes
    r'Core\.L6_Structure\.M5_Engine\.rotor': 'Core.L6_Structure.M5_Engine.Physics.merkaba_rotor',
    r'Core\.L6_Structure\.Engine\.rotor': 'Core.L6_Structure.M5_Engine.Physics.merkaba_rotor',
    r'Core\.L4_Causality\.World': 'Core.L4_Causality.M3_Mirror',
    
    # Generic Layer Mappings
    r'Core\.L1_Foundation\.Foundation': 'Core.L1_Foundation.M1_Keystone',
    r'Core\.L1_Foundation\.Logic': 'Core.L1_Foundation.M1_Keystone',
    r'Core\.L1_Foundation\.System': 'Core.L1_Foundation.M5_System',
    
    r'Core\.L5_Mental\.Reasoning_Core': 'Core.L5_Mental.M1_Cognition',
    r'Core\.L5_Mental\.Cognition': 'Core.L5_Mental.M1_Cognition',
    r'Core\.L5_Mental\.Language': 'Core.L5_Mental.M3_Lexicon',
    r'Core\.L5_Mental\.Elysia': 'Core.L5_Mental.M5_Integration',
    
    r'Core\.L6_Structure\.Engine': 'Core.L6_Structure.M5_Engine',
    r'Core\.L6_Structure\.Architecture': 'Core.L6_Structure.M6_Architecture',
    r'Core\.L6_Structure\.Wave': 'Core.L6_Structure.M3_Sphere',
    r'Core\.L6_Structure\.Autonomy': 'Core.L6_Structure.M5_Engine',
    r'Core\.L6_Structure\.Nature': 'Core.L6_Structure.M5_Engine',
}

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        # Sort keys by length (descending) to ensure nested paths are replaced before parent paths
        sorted_patterns = sorted(replacements.keys(), key=len, reverse=True)
        
        for pattern in sorted_patterns:
            replacement = replacements[pattern]
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            # print(f"[FIXED] {filepath}")
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")

def main():
    base_dir = r"c:\Elysia"
    print(f"🚀 Final Fractal Import Reconciler (v6) at {base_dir}...")
    
    count = 0
    for root, dirs, files in os.walk(base_dir):
        if any(d in root for d in [".git", ".venv", "__pycache__", "node_modules", ".gemini"]):
            continue
        for file in files:
            if file.endswith(".py"):
                fix_file(os.path.join(root, file))
                count += 1
                
    print(f"✨ Final purification finished. Scanned {count} files.")

if __name__ == "__main__":
    main()
