"""
Unit and Integration Tests for Asset Generator Module.
"""

import os
import pytest
import numpy as np
from modules.asset_generator.sheet_processor import SheetProcessor, generate_synthetic_sheet
from modules.asset_generator.mesh_reconstructor import MeshReconstructor
from modules.asset_generator.blender_auto_rig import BlenderAutoRigger
from modules.asset_generator.pipeline import AssetGeneratorPipeline


@pytest.fixture
def synthetic_sheet(tmp_path):
    sheet_path = os.path.join(tmp_path, "synthetic_sheet.png")
    generate_synthetic_sheet(sheet_path)
    return sheet_path


def test_sheet_processor(synthetic_sheet):
    processor = SheetProcessor(target_size=(256, 256))
    views = processor.process_sheet(synthetic_sheet)

    assert views.front.shape == (256, 256, 4)
    assert views.side.shape == (256, 256, 4)
    assert views.back.shape == (256, 256, 4)
    assert views.front_mask.shape == (256, 256)
    assert np.count_nonzero(views.front_mask) > 0


def test_mesh_reconstructor(synthetic_sheet, tmp_path):
    processor = SheetProcessor(target_size=(256, 256))
    views = processor.process_sheet(synthetic_sheet)

    reconstructor = MeshReconstructor()
    out_mesh_path = os.path.join(tmp_path, "test_mesh.obj")
    result = reconstructor.reconstruct_3d_mesh(views, out_mesh_path)

    assert os.path.exists(out_mesh_path)
    assert result.num_vertices > 0
    assert result.num_faces > 0


def test_blender_auto_rigger(tmp_path):
    rigger = BlenderAutoRigger()
    # Create dummy mesh obj for testing
    obj_path = os.path.join(tmp_path, "dummy.obj")
    out_gltf = os.path.join(tmp_path, "dummy.gltf")
    with open(obj_path, "w") as f:
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    success = rigger.run_auto_rig(obj_path, out_gltf)
    assert success is True
    assert os.path.exists(out_gltf)


def test_asset_generator_pipeline(synthetic_sheet, tmp_path):
    output_model = os.path.join(tmp_path, "final_character.gltf")
    pipeline = AssetGeneratorPipeline(target_size=(256, 256))
    result_path = pipeline.run(synthetic_sheet, output_model)

    assert os.path.exists(result_path)
    assert result_path == output_model
