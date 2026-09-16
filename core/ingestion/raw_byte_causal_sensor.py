r"""
Raw Byte Causal Sensor (저수준 바이트 인과 관측 센서 및 영점 수렴 모듈)
========================================================================

Implements the fundamental byte-level causal sensory pipeline:
1. StructuralViolationError: Custom exception for corrupted, fake, or spec-violating byte streams.
2. RawByteCausalSensor: Scans raw byte buffers for length, Shannon entropy, byte adjacency matrix,
   and physical byte pattern alignments.
3. ZeroConvergenceTension: Evaluates tension reduction (\Delta -> 0) against format templates.
   Achieves zero-convergence equilibrium when raw bit patterns match encoding/header laws.
4. ByteProvenanceBranch & TransformationProvenanceLog: Records transformation lineage,
   detects local offset deltas (\Delta \neq 0), and branches provenance history cleanly.
5. ByteStructuralGrounding: Bridges verified raw byte structures to SemanticMassEngine
   and GenealogicalMemoryUnit without information loss or hallucination.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class StructuralViolationError(Exception):
    """Raised when raw byte structure violates expected encoding/header rules or is corrupted/fake."""
    pass


@dataclass
class ByteAdjacencyMatrix:
    """
    Physical adjacency layout of bytes in a raw byte stream.
    Matrix represents transition probabilities or co-occurrence count between consecutive bytes.
    """
    adj_matrix: np.ndarray  # Shape (256, 256) or reduced dimensions
    transition_entropy: float
    total_transitions: int


@dataclass
class ByteProvenanceBranch:
    """
    Represents a local delta branch in byte provenance history when macro topology aligns
    but specific byte offsets contain local variations or transformations.
    """
    branch_id: str
    parent_provenance_id: str
    mismatched_offsets: List[int]
    original_slice: bytes
    transformed_slice: bytes
    local_delta_magnitude: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_provenance_id": self.parent_provenance_id,
            "mismatched_offsets": list(self.mismatched_offsets),
            "original_slice_hex": self.original_slice.hex(),
            "transformed_slice_hex": self.transformed_slice.hex(),
            "local_delta_magnitude": float(self.local_delta_magnitude),
            "timestamp": self.timestamp,
        }


@dataclass
class TransformationProvenanceLog:
    """
    Maintains the complete history of how raw bytes were created, transformed, or encoded:
    - raw_sha256: Hash of raw byte array
    - detected_format: Recognized format or encoding (e.g., 'ASCII', 'UTF-8', 'PNG', 'PE_EXEC', 'BINARY_RAW')
    - transformation_rules: Sequential list of applied rules (e.g., ['RawBytes', 'UTF-8 Decoding', 'JSON Parsing'])
    - local_branches: List of ByteProvenanceBranch instances for offset-level variations
    - is_valid_structure: Boolean flag indicating if byte layout is authentic and spec-compliant
    """
    provenance_id: str
    raw_sha256: str
    detected_format: str
    transformation_rules: List[str] = field(default_factory=list)
    local_branches: List[ByteProvenanceBranch] = field(default_factory=list)
    is_valid_structure: bool = True
    creation_timestamp: float = field(default_factory=time.time)

    def add_branch(self, branch: ByteProvenanceBranch):
        self.local_branches.append(branch)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "raw_sha256": self.raw_sha256,
            "detected_format": self.detected_format,
            "transformation_rules": list(self.transformation_rules),
            "is_valid_structure": self.is_valid_structure,
            "local_branches_count": len(self.local_branches),
            "local_branches": [b.to_dict() for b in self.local_branches],
            "creation_timestamp": self.creation_timestamp,
        }


class RawByteCausalSensor:
    """
    [Low-Level Byte Causal Sensor]
    Directly scans physical byte buffers (bytes/bytearray) to extract structural metrics:
    - Buffer length
    - Shannon entropy (Information density)
    - Byte Adjacency Matrix (Transition network of raw bytes)
    - Pattern matching against physical header signatures
    """
    # Known physical binary signatures
    HEADER_SIGNATURES: Dict[str, bytes] = {
        "PE_EXEC": b"MZ",
        "ELF_EXEC": b"\x7fELF",
        "PNG_IMAGE": b"\x89PNG\r\n\x1a\n",
        "ZIP_ARCHIVE": b"PK\x03\x04",
        "PDF_DOCUMENT": b"%PDF-",
    }

    def __init__(self, adjacency_dim: int = 16):
        self.adjacency_dim = adjacency_dim

    def scan(self, raw_data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """
        Scans raw byte buffer and returns low-level physical characteristics.
        """
        if not isinstance(raw_data, (bytes, bytearray, memoryview)):
            raise TypeError(f"RawByteCausalSensor expects bytes or bytearray, got {type(raw_data)}")

        buffer = bytes(raw_data)
        length = len(buffer)

        if length == 0:
            return {
                "length": 0,
                "entropy": 0.0,
                "adjacency": ByteAdjacencyMatrix(
                    adj_matrix=np.zeros((self.adjacency_dim, self.adjacency_dim), dtype=np.float32),
                    transition_entropy=0.0,
                    total_transitions=0,
                ),
                "detected_signature": "EMPTY",
                "is_ascii": True,
                "is_utf8": True,
                "raw_bytes": buffer,
            }

        # 1. Calculate Shannon Entropy
        entropy = self._calculate_entropy(buffer)

        # 2. Compute Byte Adjacency Matrix (reduced to adjacency_dim)
        adjacency = self._compute_adjacency_matrix(buffer)

        # 3. Detect Signature and Encodings
        detected_sig = self._detect_signature(buffer)
        is_ascii = self._check_ascii(buffer)
        is_utf8 = self._check_utf8(buffer)

        format_label = detected_sig
        if format_label == "RAW_UNKNOWN":
            if is_ascii:
                format_label = "ASCII"
            elif is_utf8:
                format_label = "UTF-8"
            else:
                format_label = "BINARY_RAW"

        return {
            "length": length,
            "entropy": float(entropy),
            "adjacency": adjacency,
            "detected_signature": format_label,
            "is_ascii": is_ascii,
            "is_utf8": is_utf8,
            "raw_bytes": buffer,
        }

    def _calculate_entropy(self, buffer: bytes) -> float:
        """Calculates Shannon Entropy in bits per byte (0.0 to 8.0)."""
        counts = [0] * 256
        for b in buffer:
            counts[b] += 1

        entropy = 0.0
        total = len(buffer)
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def _compute_adjacency_matrix(self, buffer: bytes) -> ByteAdjacencyMatrix:
        """Constructs a normalized byte transition matrix mapped to (dim, dim)."""
        dim = self.adjacency_dim
        matrix = np.zeros((dim, dim), dtype=np.float32)
        total_transitions = max(0, len(buffer) - 1)

        if total_transitions > 0:
            for i in range(total_transitions):
                b1 = buffer[i] % dim
                b2 = buffer[i + 1] % dim
                matrix[b1, b2] += 1.0

            # Normalize matrix
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            matrix = matrix / row_sums

        # Transition entropy
        non_zero = matrix[matrix > 0]
        trans_entropy = float(-np.sum(non_zero * np.log2(non_zero))) if len(non_zero) > 0 else 0.0

        return ByteAdjacencyMatrix(
            adj_matrix=matrix,
            transition_entropy=trans_entropy,
            total_transitions=total_transitions,
        )

    def _detect_signature(self, buffer: bytes) -> str:
        for fmt, sig in self.HEADER_SIGNATURES.items():
            if buffer.startswith(sig):
                return fmt
        return "RAW_UNKNOWN"

    def _check_ascii(self, buffer: bytes) -> bool:
        return all(b < 128 for b in buffer)

    def _check_utf8(self, buffer: bytes) -> bool:
        try:
            buffer.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False


class ZeroConvergenceTension:
    """
    [Zero-Convergence Equilibrium Model]
    Measures tension residual (\Delta) when raw bytes collide with format laws/templates:
    - Delta = 0: Perfect structural alignment, tension drops to 0 (Zero-Convergence Equilibrium).
    - Delta > 0: Local deviation or corruption.
    - Spec Violation: If severe encoding/header contradiction occurs, raises StructuralViolationError.
    """
    def __init__(self, max_allowed_residual: float = 0.8):
        self.max_allowed_residual = max_allowed_residual

    def evaluate_equilibrium(
        self,
        scan_report: Dict[str, Any],
        expected_format: Optional[str] = None,
        enforce_strict: bool = True,
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """
        Evaluates differential comparison (\Delta) against expected format laws.

        Returns:
            - residual_tension: Float scalar representing overall structural friction/delta (\Delta \to 0).
            - tension_vector: High-dimensional tension gradient vector.
            - equilibrium_report: Summary of structural alignment and zero-convergence status.
        """
        raw_bytes = scan_report["raw_bytes"]
        detected = scan_report["detected_signature"]
        is_utf8 = scan_report["is_utf8"]
        is_ascii = scan_report["is_ascii"]

        target_format = expected_format or detected
        residual_tension = 0.0
        violations = []

        # 1. Format Spec Verification
        if target_format in ("ASCII", "UTF-8"):
            if target_format == "ASCII" and not is_ascii:
                residual_tension += 0.9
                violations.append("Byte buffer contains non-ASCII values (b >= 128)")
            elif target_format == "UTF-8" and not is_utf8:
                residual_tension += 1.0
                violations.append("Byte buffer invalid UTF-8 sequence")

        elif target_format in RawByteCausalSensor.HEADER_SIGNATURES:
            sig = RawByteCausalSensor.HEADER_SIGNATURES[target_format]
            if not raw_bytes.startswith(sig):
                residual_tension += 1.0
                violations.append(f"Header magic mismatch for format {target_format}")

        # 2. Entropy / Noise Tension
        # Pure random noise has entropy near 8.0, structured formats usually have specific entropy ranges
        if scan_report["entropy"] > 7.95 and len(raw_bytes) > 64 and target_format in ("ASCII", "UTF-8"):
            residual_tension += 0.5
            violations.append("Unusual high entropy for text stream (Noise suspicion)")

        # Cap residual tension between 0.0 and 1.0
        residual_tension = float(np.clip(residual_tension, 0.0, 1.0))

        # Build tension vector (dim 16)
        adj_matrix = scan_report["adjacency"].adj_matrix
        tension_vector = np.mean(adj_matrix, axis=0) * residual_tension
        if np.linalg.norm(tension_vector) < 1e-9 and residual_tension > 0:
            tension_vector = np.ones(16, dtype=np.float32) * (residual_tension / 4.0)

        is_zero_converged = residual_tension < 1e-3

        report = {
            "target_format": target_format,
            "detected_format": detected,
            "residual_tension": residual_tension,
            "is_zero_converged": is_zero_converged,
            "violations": violations,
            "scan_report": {k: v for k, v in scan_report.items() if k != "raw_bytes"},
        }

        if enforce_strict and violations and residual_tension >= self.max_allowed_residual:
            raise StructuralViolationError(
                f"Raw byte structural violation detected [Tension={residual_tension:.2f}]: {'; '.join(violations)}"
            )

        return residual_tension, tension_vector, report


class ByteStructuralGrounding:
    """
    [Byte Structural Grounding]
    Connects low-level byte sensors, zero-convergence tension, and transformation provenance
    to higher-order cognitive components (SemanticMassEngine & GenealogicalMemoryUnit).
    """
    def __init__(self, sensor_dim: int = 16):
        self.sensor = RawByteCausalSensor(adjacency_dim=sensor_dim)
        self.tension_evaluator = ZeroConvergenceTension()

    def process_and_ground_bytes(
        self,
        raw_bytes: Union[bytes, bytearray],
        provenance_id: str,
        expected_format: Optional[str] = None,
        enforce_strict: bool = True,
        parent_provenance: Optional[TransformationProvenanceLog] = None,
    ) -> Tuple[TransformationProvenanceLog, Dict[str, Any]]:
        """
        Processes raw byte buffer through physical scan, tension equilibrium, and provenance creation.

        Returns:
            - provenance_log: TransformationProvenanceLog with full lineage and local branches.
            - grounding_payload: Dict containing wave vector, entropy, tension, and metadata
              ready for SemanticMassEngine consumption.
        """
        import hashlib

        raw_data = bytes(raw_bytes)
        raw_sha = hashlib.sha256(raw_data).hexdigest()

        # 1. Low-level scan
        scan_res = self.sensor.scan(raw_data)

        # 2. Tension evaluation & Zero-convergence check
        try:
            residual_tension, tension_vec, eq_report = self.tension_evaluator.evaluate_equilibrium(
                scan_res, expected_format=expected_format, enforce_strict=enforce_strict
            )
            is_valid = True
        except StructuralViolationError as e:
            if enforce_strict:
                raise
            is_valid = False
            residual_tension = 1.0
            tension_vec = np.ones(16, dtype=np.float32)
            eq_report = {
                "target_format": expected_format or "UNKNOWN",
                "detected_format": scan_res["detected_signature"],
                "residual_tension": 1.0,
                "is_zero_converged": False,
                "violations": [str(e)],
                "scan_report": {k: v for k, v in scan_res.items() if k != "raw_bytes"},
            }

        # 3. Form Transformation Provenance Log
        detected_fmt = scan_res["detected_signature"]
        provenance_log = TransformationProvenanceLog(
            provenance_id=provenance_id,
            raw_sha256=raw_sha,
            detected_format=detected_fmt,
            transformation_rules=["RawByteCapture", f"FormatScan({detected_fmt})", f"ZeroConvergenceTension(T={residual_tension:.3f})"],
            is_valid_structure=is_valid,
        )

        # 4. Build Grounding Payload for SemanticMassEngine
        adj_flat = scan_res["adjacency"].adj_matrix.flatten()
        if len(adj_flat) >= 16:
            wave_spectrum = adj_flat[:16]
        else:
            wave_spectrum = np.pad(adj_flat, (0, 16 - len(adj_flat)))

        wave_spectrum = wave_spectrum / (np.linalg.norm(wave_spectrum) + 1e-9)

        grounding_payload = {
            "provenance_id": provenance_id,
            "raw_sha256": raw_sha,
            "raw_bytes": raw_data,
            "detected_format": detected_fmt,
            "entropy": scan_res["entropy"],
            "residual_tension": residual_tension,
            "is_zero_converged": eq_report["is_zero_converged"],
            "wave_spectrum": wave_spectrum,
            "tension_vector": tension_vec,
            "equilibrium_report": eq_report,
            "adjacency_matrix": scan_res["adjacency"].adj_matrix,
        }

        return provenance_log, grounding_payload

    def isolate_local_offset_delta(
        self,
        base_bytes: bytes,
        modified_bytes: bytes,
        parent_provenance_id: str,
        branch_id: str,
    ) -> ByteProvenanceBranch:
        """
        Isolates specific byte offsets where modified_bytes differs from base_bytes,
        creating a ByteProvenanceBranch instead of marking the whole structure as fake.
        """
        min_len = min(len(base_bytes), len(modified_bytes))
        mismatches = []
        for i in range(min_len):
            if base_bytes[i] != modified_bytes[i]:
                mismatches.append(i)

        if len(base_bytes) != len(modified_bytes):
            mismatches.extend(range(min_len, max(len(base_bytes), len(modified_bytes))))

        delta_mag = len(mismatches) / float(max(1, max(len(base_bytes), len(modified_bytes))))

        first_offset = mismatches[0] if mismatches else 0
        last_offset = mismatches[-1] + 1 if mismatches else 0

        orig_slice = base_bytes[first_offset:last_offset] if mismatches else b""
        mod_slice = modified_bytes[first_offset:last_offset] if mismatches else b""

        return ByteProvenanceBranch(
            branch_id=branch_id,
            parent_provenance_id=parent_provenance_id,
            mismatched_offsets=mismatches,
            original_slice=orig_slice,
            transformed_slice=mod_slice,
            local_delta_magnitude=float(delta_mag),
        )
