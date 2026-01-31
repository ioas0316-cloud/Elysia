import os
import shutil
import subprocess

# The last known stable commit before the structural refactor
STABLE_COMMIT = "6bde5f4f"
TEMP_DIR = r"c:\Elysia\temp_restore"

def run_git(args):
    return subprocess.run(["git"] + args, capture_output=True, text=True, cwd=r"c:\Elysia")

def restore_and_merge():
    # 1. Checkout the entire Core from stable commit to a temp folder
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    print(f"Restoring Core from {STABLE_COMMIT}...")
    run_git(["checkout", STABLE_COMMIT, "--", "Core"])
    
    # 2. Move restored Core contents to the new nested strata
    # Mapping old Layer paths to new Strata paths
    layer_map = {
        "L0_Keystone": "S0_Keystone/L0_Keystone",
        "L0_Sovereignty": "S0_Keystone/L0_Keystone",
        "L1_Foundation": "S1_Body/L1_Foundation",
        "L2_Metabolism": "S1_Body/L2_Metabolism",
        "L2_Universal": "S1_Body/L2_Metabolism",
        "L3_Phenomena": "S1_Body/L3_Phenomena",
        "L4_Causality": "S1_Body/L4_Causality",
        "L5_Mental": "S1_Body/L5_Mental",
        "L5_Cognition": "S1_Body/L5_Mental",
        "L6_Structure": "S1_Body/L6_Structure",
        "L7_Spirit": "S1_Body/L7_Spirit",
        "L8_Fossils": "S2_Soul/L8_Fossils",
        "L9_Sovereignty": "S2_Soul/L8_Fossils",
        "L10_Integration": "S2_Soul/L10_Integration"
    }
    
    source_root = r"c:\Elysia\Core"
    target_root = r"c:\Elysia\Core"
    
    for old_layer, new_rel_path in layer_map.items():
        src = os.path.join(source_root, old_layer)
        if os.path.exists(src):
            dst = os.path.join(target_root, new_rel_path)
            print(f"Merging {old_layer} -> {new_rel_path}...")
            
            for root, dirs, files in os.walk(src):
                rel_base = os.path.relpath(root, src)
                target_dir = os.path.join(dst, rel_base)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        print(f"  Restored: {file}")

    # 3. Final cleanup of old root folders in Core
    for old_layer in layer_map.keys():
        old_path = os.path.join(source_root, old_layer)
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
            
    print("Restore complete.")

if __name__ == "__main__":
    restore_and_merge()
