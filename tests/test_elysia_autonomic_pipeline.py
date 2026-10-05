import os
import pytest
import subprocess

def test_elysia_autonomic_pipeline_headers():
    include_dir = os.path.join(os.path.dirname(__file__), "..", "include")
    headers = [
        "AutonomicStateController.hpp",
        "VramAdaptiveController.hpp",
        "CausalPhaseTensor.hpp",
        "ElysiaAutonomicCausalPipeline.hpp",
        "CausalGraphExtractor.hpp",
        "MitoticCausalTensorSpace.hpp",
        "FractalCausalTree.hpp",
        "CausalScaleVisualizer.hpp",
        "MetacognitiveFeedbackPipeline.hpp",
        "LockFreeCausalMemoryPool.cuh",
        "CausalDataNode.hpp",
        "CausalMeaningEvaluator.hpp",
        "AutonomicFrameSelector.hpp"
    ]
    for h in headers:
        path = os.path.join(include_dir, h)
        assert os.path.exists(path), f"Header {h} missing in include/"

def test_elysia_autonomic_pipeline_kernels():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    kernels = [
        "k_parasympathetic_consolidation.cu",
        "k_singularity_rotor_bypass.cu",
        "k_mitotic_tensor_branching.cu",
        "k_parasympathetic_node_fusion.cu",
        "k_mitotic_branching_fast.cu"
    ]
    for k in kernels:
        path = os.path.join(src_dir, k)
        assert os.path.exists(path), f"Kernel {k} missing in src/"
