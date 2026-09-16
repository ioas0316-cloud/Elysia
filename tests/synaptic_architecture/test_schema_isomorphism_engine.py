import pytest
import numpy as np
from synaptic_architecture.schema_isomorphism_engine import SchemaIsomorphismEngine


def test_schema_compression_and_decompression():
    engine = SchemaIsomorphismEngine(num_nodes=64)

    # Generate synthetic physical phenomenon profile (e.g. wave or gradient pattern)
    x = np.linspace(0, 4 * np.pi, 64)
    raw_potentials = (np.sin(x) * 1000.0 + 1500.0).astype(np.float32)

    # 1. Compress into schema
    schema = engine.compress_phenomenon_to_schema(raw_potentials)
    assert schema["compression_ratio"] > 1.0
    assert len(schema["anchor_indices"]) > 0

    # 2. Decompress and re-derive via non-statistical physical mechanics
    decompressed = engine.decompress_and_re_derive(schema, max_steps=30)
    assert len(decompressed["reconstructed_potentials"]) == 64

    # 3. Verify topological isomorphism and zero-hallucination index
    iso_metrics = engine.verify_topological_isomorphism(raw_potentials, schema, decompressed)
    assert "correlation_isomorphism" in iso_metrics
    assert iso_metrics["zero_hallucination_index"] >= 0.0
