# Asset Generator: 2D-to-3D Character Conversion Pipeline

## Overview
The `asset_generator` module provides an automated pipeline for transforming 2D multi-view character orthographic sheets (Front, Side, Back views) into textured, rigged, dynamic 3D models (`.gltf` / `.fbx`).

Rather than performing naive pixel rotations (which result in flat "cardboard doll" geometries), this system utilizes 3D spatial coordinate back-projection, mesh surface generation, UV texture mapping, and automatic skeleton bone rigging (Auto-Rigging).

---

## Directory Layout
```
modules/asset_generator/
├── README.md               # Architecture specification and documentation
├── __init__.py             # Module initialization
├── sheet_processor.py      # OpenCV/PIL 2D sheet auto-segmentation & alignment
├── mesh_reconstructor.py   # 3D spatial coordinate back-projection & mesh creation
├── blender_auto_rig.py     # Headless Blender script for UV mapping, auto-rigging & export
└── pipeline.py             # End-to-end pipeline orchestrator & CLI
```

---

## Pipeline Architecture & Data Flow

```
+-----------------------------+
| Input Sheet Image (.png/.jpg)|
+--------------+--------------+
               |
               v
+-----------------------------+
|     `sheet_processor.py`    |  1. Auto-detects bounding boxes for Front/Side/Back views
| (2D Image Segmentation &    |  2. Crops, resizes, aligns vertical feature axes (Y-axis)
|   Orthographic Alignment)   |  3. Produces normalized 2D multi-view masks & views
+--------------+--------------+
               |
               v  { front_img, side_img, back_img, height, width }
+-----------------------------+
|    `mesh_reconstructor.py`  |  1. Back-projects 2D orthographic points into 3D coordinates (X, Y, Z)
| (Spatial Point Cloud & Mesh |  2. Generates wireframe/triangulated mesh surface (Volume)
|        Reconstruction)      |  3. Exports intermediate raw mesh (`.obj` / `.ply`)
+--------------+--------------+
               |
               v  { raw_mesh_path, texture_views }
+-----------------------------+
|    `blender_auto_rig.py`    |  1. Loads raw mesh into Blender (Headless mode)
| (Headless Blender Post-Proc |  2. Cleans topology & calculates normal vectors
|   UV Mapping & Auto-Rigging)|  3. Performs UV Unwrapping & projects source texture views
|                             |  4. Constructs humanoid skeleton bone hierarchy & skinning weights
|                             |  5. Exports final model to `.gltf` / `.fbx`
+--------------+--------------+
               |
               v
+-----------------------------+
| Output 3D Asset (.gltf/.fbx)|
+-----------------------------+
```

---

## Interface Specifications

### 1. `sheet_processor.py`
- **Input:** `sheet_path: str` (Path to 2D character sheet)
- **Output:** `ProcessedViews` object containing:
  - `front`: `np.ndarray` (RGB / RGBA image)
  - `side`: `np.ndarray` (RGB / RGBA image)
  - `back`: `np.ndarray` (RGB / RGBA image)
  - `masks`: Dict of view masks for silhouette extraction

### 2. `mesh_reconstructor.py`
- **Input:** `ProcessedViews`
- **Output:** `raw_mesh_path: str` (Path to generated `.obj` mesh file)

### 3. `blender_auto_rig.py`
- **Input:** `mesh_path: str`, `output_path: str`, `texture_paths: Dict[str, str]`
- **Output:** `final_model_path: str` (`.gltf` or `.fbx`)

---

## Usage

```bash
python -m modules.asset_generator.pipeline --input path/to/character_sheet.png --output path/to/output_model.gltf
```
