import os
import shutil

DOCS = r"c:\Elysia\docs"

# Mapping: Source (Legacy _0*) -> Dest (Standard 0*)
MAPPING = {
    "_01_Origin": "01_Origin",
    "_02_Structure": "02_Structure",
    "_03_Operation": "03_Operation",
    "_04_Evolution": "04_Evolution",
    "_05_Echoes": "05_Echoes"
}

def clean_docs_structure():
    print("🧹 Cleaning docs structure...")
    
    for src, dst in MAPPING.items():
        src_path = os.path.join(DOCS, src)
        dst_path = os.path.join(DOCS, dst)
        
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                # Just rename
                print(f"   Moving {src} -> {dst}")
                shutil.move(src_path, dst_path)
            else:
                # Merge
                print(f"   Merging {src} -> {dst}")
                for item in os.listdir(src_path):
                    s = os.path.join(src_path, item)
                    d = os.path.join(dst_path, item)
                    if not os.path.exists(d):
                        if os.path.isdir(s):
                            shutil.copytree(s, d)
                        else:
                            shutil.copy2(s, d)
                # Remove source
                try:
                    shutil.rmtree(src_path)
                except Exception as e:
                    print(f"   Could not remove {src}: {e}")

if __name__ == "__main__":
    clean_docs_structure()
