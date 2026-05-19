import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataPurification")

BASE_DIR = "c:/Elysia/data"

REORG_MAP = [
    # (Source relative to data/, Target relative to data/, Action)
    ("Logs", "L1_Foundation/M4_Logs", "merge"),
    ("L1_Foundation/Logs", "L1_Foundation/M4_Logs", "merge"),
    ("State", "L1_Foundation/M1_System", "merge"),
    ("Sandbox", "L2_Metabolism/M2_Incubation", "merge"),
    ("L1_Foundation/Sandbox", "L2_Metabolism/M2_Incubation", "merge"),
    ("Qualia", "L3_Phenomena/M1_Qualia", "merge"),
    ("L1_Foundation/Qualia", "L3_Phenomena/M1_Qualia", "merge"),
    ("L5_Mental/Qualia", "L3_Phenomena/M1_Qualia", "merge"),
    ("04_Causality", "L4_Causality/M4_Chronicles", "merge"),
    ("Memory", "L5_Mental/M1_Memory", "merge"),
    ("L5_Mental/Memory", "L5_Mental/M1_Memory", "merge"),
    ("L5_Mental/Memories", "L5_Mental/M1_Memory", "merge"),
    ("Identity", "L7_Spirit/M3_Sovereignty", "merge"),
    ("temp_test_rotor", None, "delete"),
    
    # Internal refinements
    ("L5_Mental/Learning", "L5_Mental/M4_Learning", "merge"),
    ("L5_Mental/Knowledge", "L5_Mental/M5_Knowledge", "merge"),
    ("L5_Mental/Intelligence", "L5_Mental/M5_Knowledge", "merge"),
    
    # Loose files
    ("psionic_state.json", "L6_Structure/M1_State/psionic_state.json", "move_file"),
    ("self_manifest.json", "L7_Spirit/M3_Sovereignty/self_manifest.json", "move_file"),
    ("L5_Mental/crystallized_wisdom.json", "L5_Mental/M5_Knowledge/crystallized_wisdom.json", "move_file"),
]

def purify():
    for src_rel, dst_rel, action in REORG_MAP:
        src = os.path.join(BASE_DIR, src_rel)
        if not os.path.exists(src):
            logger.debug(f"Source not found: {src}")
            continue

        if action == "delete":
            if os.path.isdir(src):
                shutil.rmtree(src)
            else:
                os.remove(src)
            logger.info(f"🗑️ Deleted: {src_rel}")
            continue

        dst = os.path.join(BASE_DIR, dst_rel)
        os.makedirs(os.path.dirname(dst) if action == "move_file" else dst, exist_ok=True)

        if action == "merge":
            logger.info(f"🧬 Merging: {src_rel} -> {dst_rel}")
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        # Simple recursive merge for flat structure
                        for sub_item in os.listdir(s):
                            shutil.move(os.path.join(s, sub_item), os.path.join(d, sub_item))
                        os.rmdir(s)
                    else:
                        shutil.move(s, d)
                else:
                    shutil.move(s, d)
            os.rmdir(src)
        
        elif action == "move_file":
            logger.info(f"🚚 Moving file: {src_rel} -> {dst_rel}")
            shutil.move(src, dst)

    logger.info("✨ Data Structural Purification Complete.")

if __name__ == "__main__":
    purify()
