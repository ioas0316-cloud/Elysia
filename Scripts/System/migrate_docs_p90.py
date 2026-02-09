import os
import shutil

base_dir = r'c:\Elysia\docs'
mapping = {
    'L0_Keystone': 'S0_Keystone',
    'L1_Foundation': 'S1_Body/L1_Foundation',
    'L2_Metabolism': 'S1_Body/L2_Metabolism',
    'L3_Phenomena': 'S1_Body/L3_Phenomena',
    'L4_Causality': 'S1_Body/L4_Causality',
    'L5_Mental': 'S1_Body/L5_Mental',
    'L6_Structure': 'S1_Body/L6_Structure',
    'L7_Spirit': 'S3_Spirit'
}

for src_name, dst_rel in mapping.items():
    src_path = os.path.join(base_dir, src_name)
    dst_path = os.path.join(base_dir, dst_rel)
    
    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        print(f"Moving {src_path} -> {dst_path}")
        
        # If dst exists, move files individually or merge
        if os.path.exists(dst_path):
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dst_path, item)
                if os.path.exists(d):
                    if os.path.isdir(d):
                         shutil.rmtree(d)
                    else:
                         os.remove(d)
                shutil.move(s, d)
            os.rmdir(src_path)
        else:
            shutil.move(src_path, dst_path)
    else:
        print(f"Source not found: {src_path}")

print("Migration complete.")
