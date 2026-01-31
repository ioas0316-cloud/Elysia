import os
import re

root_dirs = [r"c:\Elysia\Core", r"c:\Elysia\docs", r"c:\Elysia\data", r"c:\Elysia"]

# Comprehensive mapping from all legacy names to the new S-prefixed strata hierarchy
replacements = {
    # Original naming -> Fixed 21D nesting
    r"Core\.L0_Sovereignty": "Core.S0_Keystone.L0_Keystone",
    r"Core\.L0_Keystone": "Core.S0_Keystone.L0_Keystone",
    r"Core\.L1_Foundation": "Core.S1_Body.L1_Foundation",
    r"Core\.L2_Universal": "Core.S1_Body.L2_Metabolism",
    r"Core\.L2_Metabolism": "Core.S1_Body.L2_Metabolism",
    r"Core\.L3_Phenomena": "Core.S1_Body.L3_Phenomena",
    r"Core\.L4_Causality": "Core.S1_Body.L4_Causality",
    r"Core\.L5_Cognition": "Core.S1_Body.L5_Mental",
    r"Core\.L5_Mental": "Core.S1_Body.L5_Mental",
    r"Core\.L6_Structure": "Core.S1_Body.L6_Structure",
    r"Core\.L7_Spirit": "Core.S1_Body.L7_Spirit",
    r"Core\.L8_Fossils": "Core.S2_Soul.L8_Fossils",
    r"Core\.L9_Sovereignty": "Core.S2_Soul.L8_Fossils",
    r"Core\.L10_Integration": "Core.S2_Soul.L10_Integration",
    
    # Fixing previous numeric strata names
    r"Core\.0_Keystone": "Core.S0_Keystone",
    r"Core\.1_Body": "Core.S1_Body",
    r"Core\.2_Soul": "Core.S2_Soul",
    r"Core\.3_Spirit": "Core.S3_Spirit",
    
    # Documentation link patches
    r"docs/S0_Keystone": "docs/S0_Keystone",
    r"docs/S1_Body": "docs/S1_Body",
    r"docs/S2_Soul": "docs/S2_Soul",
    r"docs/S3_Spirit": "docs/S3_Spirit",
    r"data/S1_Body": "data/S1_Body",
    r"data/S2_Soul": "data/S2_Soul",
    r"data/S3_Spirit": "data/S3_Spirit",
}

def fix_imports(directories):
    for directory in directories:
        if not os.path.exists(directory): continue
        for root, dirs, files in os.walk(directory):
            if any(folder in root for folder in [".git", ".venv", "__pycache__"]):
                continue
            for file in files:
                if file.endswith((".py", ".md")):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        new_content = content
                        # Sort by length descending to avoid partial matches
                        for old in sorted(replacements.keys(), key=len, reverse=True):
                            new_content = re.sub(old, replacements[old], new_content)
                        
                        if new_content != content:
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            print(f"Fixed: {filepath}")
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    fix_imports(root_dirs)
