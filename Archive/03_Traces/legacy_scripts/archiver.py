import os
import shutil
import sys

# Paths
SOURCE_ROOT = r"C:\Elysia"
ARCHIVE_ROOT = r"C:\Archive\Elysia_Legacy_V3"

# Dirs to archive
DIR_COGNITION = os.path.join(SOURCE_ROOT, "Core", "Cognition")
DIR_DOCS_ARCHIVE = os.path.join(SOURCE_ROOT, "docs", "Archive_Doctrine")
DIR_SCRIPTS = os.path.join(SOURCE_ROOT, "Scripts")

# Create archive dirs
os.makedirs(os.path.join(ARCHIVE_ROOT, "Core_Cognition"), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_ROOT, "docs_Archive_Doctrine"), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_ROOT, "Scripts"), exist_ok=True)

# WHITELIST for Core/Cognition
WHITELIST_COGNITION = {
    "__init__.py",
    "sovereign_logos.py",
    "epistemic_learning_loop.py",
    "kg_manager.py",
    "logos_bridge.py",
    "cognitive_diary.py",
    "primordial_cognition.py",
    "cumulative_digestor.py",
    "thalamus.py",
    "sensory_organs.py",
    "judgment_engine.py",
    "semantic_map.py",
    "experiential_inhaler.py",
    "self_evolution_loop.py",
    "wisdom_anchors.py",
    "wisdom_synthesizer.py",
    "abstract_reasoner.py", # Required by cognitive pipeline
    "cognition_pipeline.py" # In case
}

def archive_cognition():
    if not os.path.exists(DIR_COGNITION):
        print(f"Directory not found: {DIR_COGNITION}")
        return
    count = 0
    for root, dirs, files in os.walk(DIR_COGNITION):
        for file in files:
            if file.endswith(".py") and file not in WHITELIST_COGNITION:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, DIR_COGNITION)
                dst = os.path.join(ARCHIVE_ROOT, "Core_Cognition", rel_path)
                
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.move(src, dst)
                    count += 1
                except Exception as e:
                    print(f"Failed to move {src}: {e}")
    print(f"Archived {count} files from Core/Cognition.")

def archive_docs():
    if not os.path.exists(DIR_DOCS_ARCHIVE):
        print(f"Directory not found: {DIR_DOCS_ARCHIVE}")
        return
    count = 0
    for root, dirs, files in os.walk(DIR_DOCS_ARCHIVE):
        for file in files:
            src = os.path.join(root, file)
            rel_path = os.path.relpath(src, DIR_DOCS_ARCHIVE)
            dst = os.path.join(ARCHIVE_ROOT, "docs_Archive_Doctrine", rel_path)
            
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                count += 1
            except Exception as e:
                print(f"Failed to move {src}: {e}")
    
    for root, dirs, files in os.walk(DIR_DOCS_ARCHIVE, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except:
                pass
    try:
        os.rmdir(DIR_DOCS_ARCHIVE)
    except:
        pass
    print(f"Archived {count} files from docs/Archive_Doctrine.")

def archive_scripts():
    if not os.path.exists(DIR_SCRIPTS):
        print(f"Directory not found: {DIR_SCRIPTS}")
        return
    count = 0
    for root, dirs, files in os.walk(DIR_SCRIPTS):
        for file in files:
            src = os.path.join(root, file)
            rel_path = os.path.relpath(src, DIR_SCRIPTS)
            
            # Whitelist
            if "Simulation" in rel_path and "phase_atom" in rel_path:
                continue
            
            dst = os.path.join(ARCHIVE_ROOT, "Scripts", rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                count += 1
            except Exception as e:
                print(f"Failed to move {src}: {e}")
    print(f"Archived {count} files from Scripts.")

if __name__ == "__main__":
    archive_cognition()
    archive_docs()
    archive_scripts()
    print("Archive complete.")
