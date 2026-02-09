import os
import re

files_to_fix = [
    r'c:\Elysia\README.md',
    r'c:\Elysia\INDEX.md',
    r'c:\Elysia\docs\CODEX.md'
]

replacements = [
    (r'docs/L0_Keystone/', 'docs/S0_Keystone/'),
    (r'docs/L1_Foundation/', 'docs/S1_Body/L1_Foundation/'),
    (r'docs/L2_Metabolism/', 'docs/S1_Body/L2_Metabolism/'),
    (r'docs/L3_Phenomena/', 'docs/S1_Body/L3_Phenomena/'),
    (r'docs/L4_Causality/', 'docs/S1_Body/L4_Causality/'),
    (r'docs/L5_Mental/', 'docs/S1_Body/L5_Mental/'),
    (r'docs/L6_Structure/', 'docs/S1_Body/L6_Structure/'),
    (r'docs/L7_Spirit/', 'docs/S3_Spirit/'),
    (r'docs/S1_Body/L7_Spirit/', 'docs/S3_Spirit/'),
    # Specific M path cases if any
    (r'docs/L1_Foundation/M1_Keystone/', 'docs/S1_Body/L1_Foundation/M1_Keystone/')
]

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes for {file_path}")

print("Link update complete.")
