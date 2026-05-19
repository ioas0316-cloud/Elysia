import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ArchiveUnifier")

ROOT = r"c:\Elysia"
GLOBAL_ARCHIVE = os.path.join(ROOT, "Archive")

SOURCES = [
    (os.path.join(ROOT, "Core", "Archive"), os.path.join(GLOBAL_ARCHIVE, "Legacy_Code")),
    (os.path.join(ROOT, "data", "Archive"), os.path.join(GLOBAL_ARCHIVE, "Legacy_Data"))
]

def unify_archives():
    print("🧹 Unifying Archives...")
    
    if not os.path.exists(GLOBAL_ARCHIVE):
        os.makedirs(GLOBAL_ARCHIVE)
        
    for src_path, dst_path in SOURCES:
        if os.path.exists(src_path):
            logger.info(f"   found local archive: {src_path}")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                
            # Move contents
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dst_path, item)
                
                try:
                    if os.path.exists(d):
                        logger.warning(f"   ⚠️ Conflict: {item} exists in global archive. Renaming.")
                        d = os.path.join(dst_path, f"{item}_dup")
                        
                    shutil.move(s, d)
                except Exception as e:
                    logger.error(f"   Error moving {item}: {e}")
            
            # Remove empty source
            try:
                os.rmdir(src_path)
                logger.info(f"✅ Removed empty local archive: {src_path}")
            except Exception as e:
                logger.warning(f"   Could not remove source (not empty?): {e}")
        else:
            logger.info(f"   Source not found (already moved?): {src_path}")

if __name__ == "__main__":
    unify_archives()
