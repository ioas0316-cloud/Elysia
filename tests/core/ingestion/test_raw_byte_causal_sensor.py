r"""
Tests for Raw Byte Causal Sensor, Zero-Convergence Equilibrium, and Transformation Provenance
================================================================================================
"""

import pytest
import numpy as np

from core.ingestion.raw_byte_causal_sensor import (
    RawByteCausalSensor,
    ZeroConvergenceTension,
    ByteStructuralGrounding,
    TransformationProvenanceLog,
    ByteProvenanceBranch,
    StructuralViolationError,
)
from core.physics.semantic_mass_engine import SemanticMassEngine
from core.memory.genealogical_memory_unit import DynamicDeconstructionEngine, GenealogicalMemoryUnit


def test_raw_byte_causal_sensor_scan():
    sensor = RawByteCausalSensor(adjacency_dim=16)

    # ASCII text bytes
    ascii_data = b"HELLO_WORLD_12345"
    res = sensor.scan(ascii_data)

    assert res["length"] == len(ascii_data)
    assert res["is_ascii"] is True
    assert res["is_utf8"] is True
    assert res["detected_signature"] == "ASCII"
    assert res["entropy"] > 0.0
    assert res["adjacency"].adj_matrix.shape == (16, 16)


def test_raw_byte_causal_sensor_binary_signatures():
    sensor = RawByteCausalSensor(adjacency_dim=16)

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    res = sensor.scan(png_bytes)
    assert res["detected_signature"] == "PNG_IMAGE"

    pe_bytes = b"MZ\x90\x00\x03\x00\x00\x00"
    res_pe = sensor.scan(pe_bytes)
    assert res_pe["detected_signature"] == "PE_EXEC"


def test_zero_convergence_tension_equilibrium():
    evaluator = ZeroConvergenceTension()
    sensor = RawByteCausalSensor(adjacency_dim=16)

    # Valid ASCII
    scan_ok = sensor.scan(b"Valid_ASCII_Stream_Testing_Zero_Convergence")
    tension, tension_vec, report = evaluator.evaluate_equilibrium(scan_ok, expected_format="ASCII")

    assert tension == 0.0
    assert report["is_zero_converged"] is True
    assert len(report["violations"]) == 0

    # Invalid ASCII (Non-ASCII byte b=200)
    scan_invalid = sensor.scan(b"Invalid_ASCII_\xc8_Byte")
    with pytest.raises(StructuralViolationError) as exc_info:
        evaluator.evaluate_equilibrium(scan_invalid, expected_format="ASCII", enforce_strict=True)

    assert "structural violation" in str(exc_info.value).lower()


def test_byte_structural_grounding_and_provenance():
    grounder = ByteStructuralGrounding(sensor_dim=16)
    valid_bytes = b"Valid UTF-8 stream: \xec\x95\x88\xeb\x83\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94"

    prov_log, payload = grounder.process_and_ground_bytes(
        raw_bytes=valid_bytes,
        provenance_id="Prov_Test_1",
        expected_format="UTF-8",
    )

    assert prov_log.provenance_id == "Prov_Test_1"
    assert prov_log.detected_format == "UTF-8"
    assert prov_log.is_valid_structure is True
    assert payload["residual_tension"] == 0.0
    assert payload["wave_spectrum"].shape == (16,)


def test_local_offset_delta_branching():
    grounder = ByteStructuralGrounding()

    base_data = b"ORIGINAL_DATA_BUFFER_HEADER_V1_PAYLOAD_ABC"
    modified_data = b"ORIGINAL_DATA_BUFFER_HEADER_V2_PAYLOAD_XYZ"

    branch = grounder.isolate_local_offset_delta(
        base_bytes=base_data,
        modified_bytes=modified_data,
        parent_provenance_id="Prov_Base",
        branch_id="Branch_V2",
    )

    assert branch.branch_id == "Branch_V2"
    assert len(branch.mismatched_offsets) > 0
    assert branch.local_delta_magnitude > 0.0
    assert branch.original_slice != b""


def test_integration_with_genealogical_memory_unit():
    deconstruction_engine = DynamicDeconstructionEngine()
    semantic_engine = SemanticMassEngine(dimensions=16)

    raw_bytes = b"Causal_Memory_Grounding_Bytes_101010110"

    unit = deconstruction_engine.create_and_register_unit(
        unit_id="Unit_Byte_001",
        label="ByteGroundedMemory",
        raw_friction=raw_bytes,
        semantic_engine=semantic_engine,
        expected_format="ASCII",
    )

    assert unit.unit_id == "Unit_Byte_001"
    assert unit.provenance_trace.transformation_provenance_log is not None
    assert unit.provenance_trace.transformation_provenance_log.detected_format == "ASCII"

    proof = unit.prove_genealogy()
    assert proof["provenance"]["transformation_provenance"]["detected_format"] == "ASCII"
