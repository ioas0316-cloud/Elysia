import os
import re

root_dirs = [r"c:\Elysia\Core", r"c:\Elysia\docs", r"c:\Elysia\data", r"c:\Elysia"]
replacements = {
    # Fix previous numeric names to S-names
    r"Core\.0_Keystone": "Core.S0_Keystone",
    r"Core\.1_Body": "Core.S1_Body",
    r"Core\.2_Soul": "Core.S2_Soul",
    r"Core\.3_Spirit": "Core.S3_Spirit",
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
        for root, dirs, files in os.walk(directory):
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
