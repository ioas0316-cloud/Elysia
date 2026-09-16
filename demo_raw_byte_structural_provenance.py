r"""
Demo: Raw Byte Structural Provenance & Zero-Convergence Equilibrium Pipeline
================================================================================

Demonstrates the physical raw byte ingestion layer:
1. Grounding valid raw byte streams (ASCII, UTF-8, Binary magic headers) -> Zero-Convergence Equilibrium (\Delta -> 0).
2. Detection and rejection of corrupted/fake byte structures violating physical specifications (StructuralViolationError).
3. Local offset delta isolation (\Delta \neq 0) and lineage branching when raw bytes undergo partial variation.
4. Memory grounding into GenealogicalMemoryUnit with full Transformation Provenance Logs.
"""

import sys
import numpy as np

from core.ingestion.raw_byte_causal_sensor import (
    RawByteCausalSensor,
    ZeroConvergenceTension,
    ByteStructuralGrounding,
    StructuralViolationError,
)
from core.physics.semantic_mass_engine import SemanticMassEngine
from core.memory.genealogical_memory_unit import DynamicDeconstructionEngine


def main():
    print("=" * 80)
    print(" [ELYSIUS CAUSAL ENGINE] RAW BYTE STRUCTURAL PROVENANCE DEMONSTRATION")
    print("=" * 80)

    grounder = ByteStructuralGrounding()
    semantic_engine = SemanticMassEngine(dimensions=16)
    deconstruction_engine = DynamicDeconstructionEngine()

    # --------------------------------------------------------------------------
    # Scenario 1: Raw Bytes Ingestion & Zero-Convergence Equilibrium
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 1: Authentic UTF-8 Byte Stream Grounding] ---")
    utf8_raw = "원시 데이터 배열의 구조적 원본을 보존하고 인과를 스캔한다.".encode("utf-8")
    print(f"Input Raw Bytes (Hex): {utf8_raw[:30].hex()}... (Length: {len(utf8_raw)} bytes)")

    prov_1, payload_1 = grounder.process_and_ground_bytes(
        raw_bytes=utf8_raw,
        provenance_id="Prov_UTF8_Valid",
        expected_format="UTF-8",
    )

    print(f"Detected Format    : {prov_1.detected_format}")
    print(f"SHA256 Hash        : {prov_1.raw_sha256[:16]}...")
    print(f"Residual Tension   : {payload_1['residual_tension']:.4f} (Delta -> 0)")
    print(f"Zero-Converged?    : {payload_1['is_zero_converged']}")
    print(f"Transformation Rules: {prov_1.transformation_rules}")

    # Register into Genealogical Memory
    unit_1 = deconstruction_engine.create_and_register_unit(
        unit_id="Mem_UTF8_Authentic",
        label="StructuralTruth_Text",
        raw_friction=utf8_raw,
        semantic_engine=semantic_engine,
        expected_format="UTF-8",
    )
    print(f"Memory Unit Created: ID={unit_1.unit_id}, Semantic Mass={unit_1.semantic_mass:.3f}")

    # --------------------------------------------------------------------------
    # Scenario 2: Detection and Rejection of Corrupted / Fake Data Structure
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 2: Corrupted / Fake Specification Rejection] ---")
    fake_ascii_bytes = b"Standard_ASCII_Stream_\xFF\xFE_Corrupted_Non_ASCII_Bytes"
    print(f"Input Raw Bytes (Corrupted): {fake_ascii_bytes.hex()}")

    try:
        grounder.process_and_ground_bytes(
            raw_bytes=fake_ascii_bytes,
            provenance_id="Prov_Corrupted_ASCII",
            expected_format="ASCII",
            enforce_strict=True,
        )
        print("ERROR: Fake data was NOT rejected!")
    except StructuralViolationError as e:
        print(f" SUCCESS: Structural Violation Sensor Triggered!")
        print(f" Violation Error Detail: {e}")

    # --------------------------------------------------------------------------
    # Scenario 3: Local Offset Delta (\Delta \neq 0) & Provenance Branching
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 3: Local Byte Offset Variation & Branching] ---")
    base_data = b"SYSTEM_HEADER_V1.0_CONFIG_FLAGS_0001_PAYLOAD_DATA_BLOCK"
    modified_data = b"SYSTEM_HEADER_V1.0_CONFIG_FLAGS_9999_PAYLOAD_DATA_BLOCK"

    print(f"Base Data     : {base_data}")
    print(f"Modified Data : {modified_data}")

    branch = grounder.isolate_local_offset_delta(
        base_bytes=base_data,
        modified_bytes=modified_data,
        parent_provenance_id=prov_1.provenance_id,
        branch_id="Branch_Config_Delta",
    )

    print(f"Branch ID          : {branch.branch_id}")
    print(f"Mismatched Offsets : {branch.mismatched_offsets}")
    print(f"Original Slice     : {branch.original_slice}")
    print(f"Transformed Slice  : {branch.transformed_slice}")
    print(f"Delta Magnitude    : {branch.local_delta_magnitude:.4f}")

    # Attach branch to provenance log
    prov_1.add_branch(branch)
    print(f"Attached Branch to Parent Provenance Log (Total Branches: {len(prov_1.local_branches)})")

    # --------------------------------------------------------------------------
    # Scenario 4: Provenance Self-Proof via Genealogical Memory Unit
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 4: Genealogical Memory Unit Proof Verification] ---")
    genealogy_proof = unit_1.prove_genealogy()
    print(f"Memory Unit ID     : {genealogy_proof['unit_id']}")
    print(f"Is Justified?      : {genealogy_proof['justification_proof']['is_justified']}")
    print(f"Transformation Log : {genealogy_proof['provenance']['transformation_provenance']}")

    print("\n" + "=" * 80)
    print(" [SUMMARY] RAW BYTE CAUSAL SENSOR PIPELINE OPERATIONAL WITHOUT HALLUCINATIONS")
    print("=" * 80)


if __name__ == "__main__":
    main()
