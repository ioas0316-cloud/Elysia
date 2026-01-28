import os
import re

# Comprehensive mapping from horizontal/legacy to 7D Fractal depth
replacements = {
    # L1_Foundation mappings
    r'Core\.L1_Foundation\.Foundation': 'Core.L1_Foundation.M1_Keystone',
    r'Core\.L1_Foundation\.Logic': 'Core.L1_Foundation.M1_Keystone',
    r'Core\.L1_Foundation\.System': 'Core.L1_Foundation.M5_System',
    
    # L5_Mental mappings
    r'Core\.L5_Mental\.Reasoning_Core': 'Core.L5_Mental.M1_Cognition',
    r'Core\.L5_Mental\.Cognition': 'Core.L5_Mental.M1_Cognition',
    r'Core\.L5_Mental\.Language': 'Core.L5_Mental.M3_Lexicon',
    r'Core\.L5_Mental\.Elysia': 'Core.L5_Mental.M5_Integration',
    
    # L6_Structure mappings
    r'Core\.L6_Structure\.Engine': 'Core.L6_Structure.M5_Engine',
    r'Core\.L6_Structure\.Architecture': 'Core.L6_Structure.M6_Architecture',
    r'Core\.L6_Structure\.Wave': 'Core.L6_Structure.M3_Sphere',
    r'Core\.L6_Structure\.Autonomy': 'Core.L6_Structure.M5_Engine', # Often overlaps
    r'Core\.L6_Structure\.Nature': 'Core.L6_Structure.M5_Engine',
}

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for pattern, replacement in replacements.items():
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"[FIXED] {filepath}")
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")

def main():
    # Target Elysia root
    base_dir = r"c:\Elysia"
    print(f"🚀 Initializing Fractal Import Neutralizer (v4) at {base_dir}...")
    
    count = 0
    for root, dirs, files in os.walk(base_dir):
        # Skip infra/temp
        if any(d in root for d in [".git", ".venv", "__pycache__", "node_modules", ".gemini"]):
            continue
            
        for file in files:
            if file.endswith(".py"):
                fix_file(os.path.join(root, file))
                count += 1
                
    print(f"✨ Purification finished. Scanned {count} files.")

if __name__ == "__main__":
    main()
