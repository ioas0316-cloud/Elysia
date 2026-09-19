"""
Asset Generator Pipeline Orchestrator

Connects 2D multi-view sheet segmentation, 3D orthographic projection mesh reconstruction,
and Blender auto-rigging into a unified end-to-end 2D-to-3D character pipeline.
"""

import os
import argparse
from typing import Tuple, Optional
from .sheet_processor import SheetProcessor, ProcessedViews, generate_synthetic_sheet
from .mesh_reconstructor import MeshReconstructor, ReconstructionResult
from .blender_auto_rig import BlenderAutoRigger


class AssetGeneratorPipeline:
    """
    End-to-end pipeline manager for converting 2D character sheets into rigged 3D models.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512), blender_bin: Optional[str] = None):
        self.sheet_processor = SheetProcessor(target_size=target_size)
        self.mesh_reconstructor = MeshReconstructor()
        self.auto_rigger = BlenderAutoRigger(blender_executable=blender_bin)

    def run(self, input_sheet_path: str, output_model_path: str) -> str:
        """
        Executes full 2D-to-3D transformation pipeline.
        1. Segment & align multi-view 2D orthographic views.
        2. Reconstruct 3D volumetric surface mesh (.obj).
        3. Post-process UV mapping, Auto-Rig skeleton, and export final model (.gltf/.fbx).
        """
        print(f"[Pipeline] 1/3 Processing 2D sheet: {input_sheet_path}")
        views = self.sheet_processor.process_sheet(input_sheet_path)

        # Save intermediate mesh OBJ
        intermediate_obj = os.path.join(
            os.path.dirname(os.path.abspath(output_model_path)),
            "_raw_reconstructed_mesh.obj"
        )
        print(f"[Pipeline] 2/3 Reconstructing 3D mesh...")
        recon_result = self.mesh_reconstructor.reconstruct_3d_mesh(views, intermediate_obj)
        print(f"            Reconstructed mesh with {recon_result.num_vertices} vertices, {recon_result.num_faces} faces.")

        # Save front texture view temporarily
        import cv2
        front_tex_path = os.path.join(
            os.path.dirname(os.path.abspath(output_model_path)),
            "_front_texture.png"
        )
        cv2.imwrite(front_tex_path, cv2.cvtColor(views.front, cv2.COLOR_RGBA2BGRA))

        print(f"[Pipeline] 3/3 Applying UV mapping & Auto-Rigging...")
        success = self.auto_rigger.run_auto_rig(intermediate_obj, output_model_path, front_texture_path=front_tex_path)

        if success and os.path.exists(output_model_path):
            print(f"[Pipeline] Successfully generated 3D asset: {output_model_path}")
            return output_model_path
        else:
            raise RuntimeError(f"Pipeline failed to generate output model at {output_model_path}")


def main():
    parser = argparse.ArgumentParser(description="2D Character Sheet to 3D Rigged Model Pipeline")
    parser.add_argument("--input", type=str, help="Path to input 2D character sheet image")
    parser.add_argument("--output", type=str, default="output_character.gltf", help="Path to output 3D asset (.gltf / .fbx)")
    parser.add_argument("--target-size", type=int, default=512, help="Resolution size for view alignment")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic test sheet and run demo")

    args = parser.parse_args()

    if args.demo or not args.input:
        print("[Demo] Generating synthetic character sheet...")
        demo_sheet = "/tmp/demo_character_sheet.png"
        generate_synthetic_sheet(demo_sheet)
        input_path = demo_sheet
    else:
        input_path = args.input

    pipeline = AssetGeneratorPipeline(target_size=(args.target_size, args.target_size))
    pipeline.run(input_path, args.output)


if __name__ == "__main__":
    main()
