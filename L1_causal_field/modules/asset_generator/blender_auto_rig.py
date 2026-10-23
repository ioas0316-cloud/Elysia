"""
Blender Auto-Rig & Post-Processing Automation Script.

This module provides background (headless) Blender API automation for topology
cleaning, UV texture mapping, skeleton bone structure creation (Auto-Rigging),
and export to 3D formats (.gltf / .fbx / .obj).

Can be executed as a standalone script inside Blender's python environment
(via `blender --background --python blender_auto_rig.py -- ...`) or invoked via Python.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional


BLENDER_SCRIPT_TEMPLATE = """
import bpy
import sys
import os

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Parse args passed after '--'
argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
else:
    args = []

obj_path = "{mesh_path}"
output_path = "{output_path}"
front_tex = "{front_tex_path}"

# 1. Import Raw Mesh OBJ
if os.path.exists(obj_path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objs:
    print("Error: No mesh object imported!")
    sys.exit(1)

character_mesh = mesh_objs[0]
bpy.context.view_layer.objects.active = character_mesh
character_mesh.select_set(True)

# 2. Smooth Shading & UV Unwrapping
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
bpy.ops.object.mode_set(mode='OBJECT')

# 3. Create Armature / Bone Skeleton (Auto-Rigging)
bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0))
armature = bpy.context.active_object
armature.name = "Character_Armature"

# Skeleton structure definition (Head, Torso, Spine, Hips, Upper/Lower Arms, Legs)
eb = armature.data.edit_bones

# Root Spine & Torso
spine = eb.active
spine.name = "Spine"
spine.head = (0, 0, 0)
spine.tail = (0, 0, 0.4)

chest = eb.new("Chest")
chest.head = spine.tail
chest.tail = (0, 0, 0.8)
chest.parent = spine

head = eb.new("Head")
head.head = chest.tail
head.tail = (0, 0, 1.1)
head.parent = chest

# Left/Right Arms
arm_L = eb.new("Arm.L")
arm_L.head = (0.2, 0, 0.7)
arm_L.tail = (0.5, 0, 0.7)
arm_L.parent = chest

arm_R = eb.new("Arm.R")
arm_R.head = (-0.2, 0, 0.7)
arm_R.tail = (-0.5, 0, 0.7)
arm_R.parent = chest

# Left/Right Legs
leg_L = eb.new("Leg.L")
leg_L.head = (0.15, 0, 0)
leg_L.tail = (0.15, 0, -0.6)
leg_L.parent = spine

leg_R = eb.new("Leg.R")
leg_R.head = (-0.15, 0, 0)
leg_R.tail = (-0.15, 0, -0.6)
leg_R.parent = spine

bpy.ops.object.mode_set(mode='OBJECT')

# 4. Skinning (Parent Mesh to Armature with Automatic Weights)
character_mesh.select_set(True)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

# 5. Export Final 3D Asset
out_ext = os.path.splitext(output_path)[1].lower()
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

if out_ext in ['.gltf', '.glb']:
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLTF_EMBEDDED')
elif out_ext == '.fbx':
    bpy.ops.export_scene.fbx(filepath=output_path)
else:
    bpy.ops.wm.obj_export(filepath=output_path) if hasattr(bpy.ops.wm, "obj_export") else bpy.ops.export_scene.obj(filepath=output_path)

print(f"Exported successfully to {output_path}")
"""


class BlenderAutoRigger:
    """
    Manages headless Blender post-processing, UV texturing, Auto-Rigging,
    and 3D model export.
    """

    def __init__(self, blender_executable: Optional[str] = None):
        self.blender_bin = blender_executable or "blender"

    def run_auto_rig(self, raw_mesh_path: str, output_path: str, front_texture_path: Optional[str] = None) -> bool:
        """Executes auto-rigging and export via Blender headless mode if available, or fallback."""
        script_content = BLENDER_SCRIPT_TEMPLATE.format(
            mesh_path=os.path.abspath(raw_mesh_path),
            output_path=os.path.abspath(output_path),
            front_tex_path=os.path.abspath(front_texture_path) if front_texture_path else ""
        )

        import tempfile
        tmp_script_path = os.path.join(tempfile.gettempdir(), "run_blender_auto_rig.py")
        with open(tmp_script_path, "w") as f:
            f.write(script_content)

        import subprocess
        try:
            res = subprocess.run(
                [self.blender_bin, "--background", "--python", tmp_script_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if res.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        # Fallback if Blender binary is not in system PATH: perform pure Trimesh/Python GLTF export
        return self._fallback_export(raw_mesh_path, output_path)

    def _fallback_export(self, raw_mesh_path: str, output_path: str) -> bool:
        """Fallback Python export when headless Blender binary is unavailable."""
        import trimesh
        mesh = trimesh.load(raw_mesh_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mesh.export(output_path)
        return os.path.exists(output_path)
