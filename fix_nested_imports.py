import os
import re

root_dirs = [r"c:\Elysia\Core", r"c:\Elysia\docs", r"c:\Elysia\data", r"c:\Elysia"]
replacements = {
    r"Core\.L0_Keystone": "Core.0_Keystone.L0_Keystone",
    r"Core\.L1_Foundation": "Core.1_Body.L1_Foundation",
    r"Core\.L2_Metabolism": "Core.1_Body.L2_Metabolism",
    r"Core\.L3_Phenomena": "Core.1_Body.L3_Phenomena",
    r"Core\.L4_Causality": "Core.1_Body.L4_Causality",
    r"Core\.L5_Mental": "Core.1_Body.L5_Mental",
    r"Core\.L6_Structure": "Core.1_Body.L6_Structure",
    r"Core\.L7_Spirit": "Core.1_Body.L7_Spirit",
    r"Core\.L8_Fossils": "Core.2_Soul.L8_Fossils",
    r"Core\.L10_Integration": "Core.2_Soul.L10_Integration",
    # Documentation links fix
    r"docs/0_Keystone/L0_Keystone": "docs/0_Keystone/L0_Keystone",
    r"docs/1_Body/L1_Foundation": "docs/1_Body/L1_Foundation",
    r"docs/1_Body/L2_Metabolism": "docs/1_Body/L2_Metabolism",
    r"docs/1_Body/L3_Phenomena": "docs/1_Body/L3_Phenomena",
    r"docs/1_Body/L4_Causality": "docs/1_Body/L4_Causality",
    r"docs/1_Body/L5_Mental": "docs/1_Body/L5_Mental",
    r"docs/1_Body/L6_Structure": "docs/1_Body/L6_Structure",
    r"docs/1_Body/L7_Spirit": "docs/1_Body/L7_Spirit",
    r"docs/2_Soul/L8_Fossils": "docs/2_Soul/L8_Fossils"
}

def fix_imports(directories):
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            # Skip .git, .venv, etc.
            if any(folder in root for folder in [".git", ".venv", "__pycache__"]):
                continue
            for file in files:
                if file.endswith((".py", ".md")):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        new_content = content
                        for old, new in replacements.items():
                            new_content = re.sub(old, new, new_content)
                        
                        if new_content != content:
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            print(f"Fixed: {filepath}")
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    fix_imports(root_dirs)
