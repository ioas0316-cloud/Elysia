import os
import shutil
import re

ROOT_DIR = r"c:\Elysia"
CORE_DIR = os.path.join(ROOT_DIR, "core")
ARCHIVE_DIR = r"C:\Archive\Elysia_Archive\core"

MISSING_FILES = {
    "zero_distance_projector.py": "cortex",
    "static_oracle.py": "brain",
    "phase_mirror.py": "brain",
    "turbine_force_field.py": "nervous_system",
    "universal_language_cortex.py": "cortex",
    "evolution_sandbox.py": "nervous_system"
}

moved_files = {}

# Move back
for filename, cat in MISSING_FILES.items():
    src = os.path.join(ARCHIVE_DIR, filename)
    dst = os.path.join(CORE_DIR, cat, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        moved_files[filename] = cat
        print(f"Restored {filename} to {cat}")

# Add the previously moved files so we can update imports properly inside the newly restored files
# And also we need to update the daemon to point to these newly restored files.
CATEGORIES = {
    "brain": [
        "holographic_memory.py", "spacetime_rotor.py", "fractal_rotor.py", 
        "inverse_projector.py", "tensor_rotor.py", "wave_tensor.py", 
        "fractal_mirror.py", "causality_wave.py", "consciousness_stream.py",
        "static_oracle.py", "phase_mirror.py"
    ],
    "cortex": [
        "gpu_sensory_cortex.py", "web_sensory_cortex.py", "vision_cortex.py", 
        "vision_utils.py", "omni_modal_sensor.py", "coding_cognition_sensor.py", 
        "autonomous_motor_cortex.py", "source_code_mirror.py", 
        "zero_distance_browser.py", "network_phase_snatcher.py",
        "zero_distance_projector.py", "universal_language_cortex.py"
    ],
    "nervous_system": [
        "autonomic_nervous_system.py", "elysia_omni_daemon.py", "autopoiesis_controller.py",
        "turbine_force_field.py", "evolution_sandbox.py"
    ],
    "utils": [
        "math_utils.py", "actuator_utils.py"
    ]
}

all_moved = {}
for cat, files in CATEGORIES.items():
    for f in files:
        all_moved[f] = cat

def update_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    original_content = content
    for filename, cat in all_moved.items():
        module_name = filename.replace(".py", "")
        # replace from core.module import
        pattern1 = r'from\s+core\.' + module_name + r'\s+import'
        repl1 = f'from core.{cat}.{module_name} import'
        content = re.sub(pattern1, repl1, content)
        
        # replace import core.module
        pattern2 = r'import\s+core\.' + module_name + r'\b'
        repl2 = f'import core.{cat}.{module_name}'
        content = re.sub(pattern2, repl2, content)
        
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

# Update all python files in core
for cat in CATEGORIES:
    cat_dir = os.path.join(CORE_DIR, cat)
    for f in os.listdir(cat_dir):
        if f.endswith(".py"):
            update_imports(os.path.join(cat_dir, f))

print("Missing files restored and imports updated.")
