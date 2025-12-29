import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DocsPolisher")

DOCS_ROOT = r"c:\Elysia\docs"

DEBRIS_PATTERNS = [
    "_01_Section", "_02_Section", "_03_Section", "_04_Section", "_05_Section"
]

def polish_docs():
    print("✨ Polishing Docs Structure...")
    count = 0
    
    for root, dirs, files in os.walk(DOCS_ROOT):
        for d in dirs:
            if d in DEBRIS_PATTERNS:
                path = os.path.join(root, d)
                try:
                    # Check if empty (ignoring .gitkeep etc? No, delete all debris)
                    if not os.listdir(path):
                        os.rmdir(path)
                        logger.info(f"   Deleted empty debris: {path}")
                        count += 1
                    else:
                        shutil.rmtree(path)
                        logger.info(f"   Deleted populated debris: {path}")
                        count += 1
                except Exception as e:
                    logger.error(f"   Failed to delete {path}: {e}")
                    
    print(f"✅ Removed {count} debris folders.")
    
    # Final Verification
    print("\n🔍 Final Structure Check (Depth 1):")
    for item in sorted(os.listdir(DOCS_ROOT)):
        path = os.path.join(DOCS_ROOT, item)
        if os.path.isdir(path):
             print(f"   📁 {item}")

if __name__ == "__main__":
    polish_docs()
