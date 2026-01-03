import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DocsArchiver")

ROOT = r"c:\Elysia"
DOCS_ECHOES = os.path.join(ROOT, "docs", "05_Echoes")
GLOBAL_ARCHIVE_DOCS = os.path.join(ROOT, "Archive", "Legacy_Docs")

TARGETS_TO_MOVE = ["archive", "legacy", "Legacy_Root_Archive"]

def move_docs_archive():
    print("📜 Consolidating Docs Archive...")
    
    if not os.path.exists(GLOBAL_ARCHIVE_DOCS):
        os.makedirs(GLOBAL_ARCHIVE_DOCS)
        
    for item in TARGETS_TO_MOVE:
        src = os.path.join(DOCS_ECHOES, item)
        dst = os.path.join(GLOBAL_ARCHIVE_DOCS, item)
        
        if os.path.exists(src):
            if os.path.exists(dst):
                logger.warning(f"⚠️ Conflict: {item} already exists in Global Archive. Merging/Renaming.")
                # Simple rename to avoid merge complexity for now, or assume unique
                dst = os.path.join(GLOBAL_ARCHIVE_DOCS, f"{item}_dup_{os.urandom(2).hex()}")
            
            try:
                shutil.move(src, dst)
                logger.info(f"✅ Moved {src} -> {dst}")
            except Exception as e:
                logger.error(f"❌ Failed to move {src}: {e}")
        else:
            logger.info(f"   Skipping {item} (Not found in Echoes)")

if __name__ == "__main__":
    move_docs_archive()
