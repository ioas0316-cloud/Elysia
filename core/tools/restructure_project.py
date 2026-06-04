import os
import shutil
import re

ROOT_DIR = r"c:\Elysia"
CORE_DIR = os.path.join(ROOT_DIR, "core")
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
ARCHIVE_DIR = r"C:\Archive\Elysia_Archive"

# Create directories
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, "core"), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, "docs"), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, "tests"), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, "engines"), exist_ok=True)

CATEGORIES = {
    "brain": [
        "holographic_memory.py", "spacetime_rotor.py", "fractal_rotor.py", 
        "inverse_projector.py", "tensor_rotor.py", "wave_tensor.py", 
        "fractal_mirror.py", "causality_wave.py", "consciousness_stream.py"
    ],
    "cortex": [
        "gpu_sensory_cortex.py", "web_sensory_cortex.py", "vision_cortex.py", 
        "vision_utils.py", "omni_modal_sensor.py", "coding_cognition_sensor.py", 
        "autonomous_motor_cortex.py", "source_code_mirror.py", 
        "zero_distance_browser.py", "network_phase_snatcher.py"
    ],
    "nervous_system": [
        "autonomic_nervous_system.py", "elysia_omni_daemon.py", "autopoiesis_controller.py"
    ],
    "utils": [
        "math_utils.py", "actuator_utils.py"
    ]
}

# Create subdirs
for cat in CATEGORIES:
    os.makedirs(os.path.join(CORE_DIR, cat), exist_ok=True)

# 1. Move files to categorized folders
moved_files = {}  # original_filename -> new_category
for cat, files in CATEGORIES.items():
    for f in files:
        src = os.path.join(CORE_DIR, f)
        dst = os.path.join(CORE_DIR, cat, f)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_files[f] = cat

# 2. Archive remaining unused files in core
for item in os.listdir(CORE_DIR):
    item_path = os.path.join(CORE_DIR, item)
    if os.path.isdir(item_path):
        if item not in CATEGORIES and item != "__pycache__":
            # Archive dirs like legacy_daemons, engines
            dst = os.path.join(ARCHIVE_DIR, "core", item)
            shutil.move(item_path, dst)
        continue
        
    if item.endswith(".pkl"):
        shutil.move(item_path, os.path.join(DATA_DIR, item))
    elif item.endswith("_test.py") or "test" in item:
        shutil.move(item_path, os.path.join(ARCHIVE_DIR, "tests", item))
    elif item.endswith("_engine.py"):
        shutil.move(item_path, os.path.join(ARCHIVE_DIR, "engines", item))
    else:
        # Move any other python scripts, c/cpp files, etc to archive core
        shutil.move(item_path, os.path.join(ARCHIVE_DIR, "core", item))

# 3. Update imports in the newly moved files
def update_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    original_content = content
    # Replace 'from core.filename import' with 'from core.category.filename import'
    for filename, cat in moved_files.items():
        module_name = filename.replace(".py", "")
        # Regex to match exact import module to avoid partial matches
        pattern1 = r'from\s+core\.' + module_name + r'\s+import'
        repl1 = f'from core.{cat}.{module_name} import'
        content = re.sub(pattern1, repl1, content)
        
        pattern2 = r'import\s+core\.' + module_name + r'\b'
        repl2 = f'import core.{cat}.{module_name}'
        content = re.sub(pattern2, repl2, content)
        
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

for cat in CATEGORIES:
    cat_dir = os.path.join(CORE_DIR, cat)
    for f in os.listdir(cat_dir):
        if f.endswith(".py"):
            update_imports(os.path.join(cat_dir, f))

# 4. Archive docs
for item in os.listdir(DOCS_DIR):
    item_path = os.path.join(DOCS_DIR, item)
    if os.path.isfile(item_path) and item.startswith("PHASE"):
        shutil.move(item_path, os.path.join(ARCHIVE_DIR, "docs", item))

print("Restructuring Complete.")
