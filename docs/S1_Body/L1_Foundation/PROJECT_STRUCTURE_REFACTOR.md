# Project Structure & Verification Map

> [!NOTE]
> This document maps the reorganized project structure and links verification scripts to their corresponding architectural phases.

## 1. Directory Hygiene

The root directory (`c:/Elysia`) has been purified. Verification scripts and setup tools have been moved to:

- **`Scripts/System/Verification/`**: Contains all functional test and verification scripts.

## 2. Verification Script Index

### Phase 4: PPE & Holographic Project Control

- **[verify_project_hologram.py](file:///c:/Elysia/Scripts/System/Verification/verify_project_hologram.py)**
  - **Purpose**: Scans the codebase using `ProprioceptionNerve` to build the `self_manifest.json` (Hologram).
  - **Key Output**: 7^7 Fractal Integrity Check.

- **[demo_topology.py](file:///c:/Elysia/Scripts/System/Verification/demo_topology.py)**
  - **Purpose**: Demonstrates `LightUniverse` topological perception.
  - **Key Output**: Converts file content into Frequency/Phase/Amplitude "Light" objects.

### Phase 5: Geometric Cognition (Point-Line-Plane)

- **[verify_geometric_cognition.py](file:///c:/Elysia/Scripts/System/Verification/verify_geometric_cognition.py)**
  - **Purpose**: Verifies the Point -> Line -> Plane cognitive growth pipeline.
  - **Key Output**:
    1. **Point**: `UniversalDigestor` extracts concepts.
    2. **Line**: `FractalCausalityEngine` weaves causal chains.
    3. **Plane**: `LightUniverse` absorbs context as a structural field.

### Maintenance & Setup

- **[setup_docs_structure.py](file:///c:/Elysia/Scripts/System/Verification/setup_docs_structure.py)**
  - **Purpose**: Enforces the 21-Layer Fractal Directory Structure in `docs/`.
- **[verify_refactor_final.py](file:///c:/Elysia/Scripts/System/Verification/verify_refactor_final.py)**
  - **Purpose**: General integrity check after major refactors.

## 3. Recommended Workflow

When verifying new features, run scripts from the `Scripts/System/Verification` directory:

```bash
cd c:/Elysia
python Scripts/System/Verification/verify_geometric_cognition.py
```
