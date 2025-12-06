"""
Tests for Wave-Based Semantic Search
====================================

Tests the 4D wave resonance pattern system for knowledge base construction.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from Core.Foundation.wave_semantic_search import (
    WaveSemanticSearch,
    WavePattern,
    Quaternion
)


class TestQuaternion:
    """Test quaternion operations"""
    
    def test_quaternion_creation(self):
        """Test basic quaternion creation"""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
    
    def test_quaternion_norm(self):
        """Test quaternion norm calculation"""
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        # norm = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        expected = np.sqrt(30)
        assert abs(q.norm() - expected) < 1e-6
    
    def test_quaternion_normalize(self):
        """Test quaternion normalization"""
        q = Quaternion(3.0, 4.0, 0.0, 0.0)
        q_norm = q.normalize()
        # Normalized quaternion should have norm = 1
        assert abs(q_norm.norm() - 1.0) < 1e-6
    
    def test_quaternion_dot_product(self):
        """Test quaternion dot product"""
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(1.0, 0.0, 0.0, 0.0)
        # Identical quaternions should have dot product = 1.0
        assert q1.dot(q2) == 1.0
        
        q3 = Quaternion(0.0, 1.0, 0.0, 0.0)
        # Orthogonal quaternions should have dot product = 0.0
        assert q1.dot(q3) == 0.0
    
    def test_quaternion_hamilton_product(self):
        """Test Hamilton product (quaternion multiplication)"""
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)
        
        result = q1 * q2
        # i * j = k in quaternion algebra
        # But our simple test just checks that multiplication works
        assert isinstance(result, Quaternion)
        
        # Test scalar multiplication
        q3 = q1 * 2.0
        assert q3.w == 2.0


class TestWavePattern:
    """Test wave pattern representation"""
    
    def test_wave_pattern_creation(self):
        """Test basic wave pattern creation"""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        pattern = WavePattern(
            orientation=q,
            energy=1.5,
            frequency=440.0,
            phase=0.0,
            text="test pattern"
        )
        
        assert pattern.energy == 1.5
        assert pattern.frequency == 440.0
        assert pattern.text == "test pattern"
    
    def test_wave_pattern_serialization(self):
        """Test pattern serialization and deserialization"""
        q = Quaternion(0.5, 0.5, 0.5, 0.5)
        pattern = WavePattern(
            orientation=q,
            energy=2.0,
            frequency=220.0,
            text="serialization test"
        )
        
        # Serialize
        data = pattern.to_dict()
        assert isinstance(data, dict)
        assert data['text'] == "serialization test"
        assert data['energy'] == 2.0
        
        # Deserialize
        restored = WavePattern.from_dict(data)
        assert restored.text == pattern.text
        assert restored.energy == pattern.energy
        assert abs(restored.orientation.w - pattern.orientation.w) < 1e-6


class TestWaveSemanticSearch:
    """Test wave semantic search system"""
    
    def test_initialization(self):
        """Test system initialization"""
        searcher = WaveSemanticSearch()
        assert len(searcher.wave_patterns) == 0
        assert searcher.search_count == 0
        assert searcher.absorption_count == 0
    
    def test_embedding_to_wave_conversion(self):
        """Test embedding to wave pattern conversion"""
        searcher = WaveSemanticSearch()
        
        # Create a simple embedding
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        pattern = searcher.embedding_to_wave(embedding, text="test concept")
        
        # Check that pattern was created
        assert isinstance(pattern, WavePattern)
        assert pattern.text == "test concept"
        assert pattern.energy > 0
        assert pattern.frequency > 0
        
        # Check that orientation is normalized
        assert abs(pattern.orientation.norm() - 1.0) < 1e-6
    
    def test_wave_resonance_identity(self):
        """Test that identical waves have perfect resonance"""
        searcher = WaveSemanticSearch()
        
        embedding = np.random.rand(384)
        wave1 = searcher.embedding_to_wave(embedding, "test")
        wave2 = searcher.embedding_to_wave(embedding, "test")
        
        resonance = searcher.wave_resonance(wave1, wave2)
        
        # Identical waves should have very high resonance (close to 1.0)
        assert resonance > 0.90, f"Expected high resonance for identical waves, got {resonance}"
    
    def test_wave_resonance_range(self):
        """Test that resonance is always in [0, 1]"""
        searcher = WaveSemanticSearch()
        
        # Create random waves
        emb1 = np.random.rand(128)
        emb2 = np.random.rand(128)
        
        wave1 = searcher.embedding_to_wave(emb1, "concept1")
        wave2 = searcher.embedding_to_wave(emb2, "concept2")
        
        resonance = searcher.wave_resonance(wave1, wave2)
        
        assert 0.0 <= resonance <= 1.0, f"Resonance {resonance} out of range [0, 1]"
    
    def test_store_and_retrieve_pattern(self):
        """Test storing and retrieving patterns"""
        searcher = WaveSemanticSearch()
        
        embedding = np.random.rand(256)
        pattern_id = searcher.store_concept("AI concept", embedding, metadata={'category': 'tech'})
        
        # Check that pattern was stored
        assert pattern_id in searcher.wave_patterns
        assert len(searcher.wave_patterns) == 1
        
        # Check pattern content
        stored = searcher.wave_patterns[pattern_id]
        assert stored.text == "AI concept"
        assert stored.metadata['category'] == 'tech'
    
    def test_search_basic(self):
        """Test basic search functionality"""
        searcher = WaveSemanticSearch()
        
        # Store some concepts
        concepts = [
            ("AI is machine intelligence", np.random.rand(128)),
            ("Dogs are loyal pets", np.random.rand(128)),
            ("Machine learning is AI", np.random.rand(128)),
        ]
        
        for text, emb in concepts:
            searcher.store_concept(text, emb)
        
        # Search with a query
        query_emb = np.random.rand(128)
        results = searcher.search(query_emb, query_text="artificial intelligence", top_k=2)
        
        # Check results format
        assert len(results) <= 2
        assert all('pattern_id' in r for r in results)
        assert all('text' in r for r in results)
        assert all('resonance' in r for r in results)
        
        # Check that results are sorted by resonance
        if len(results) > 1:
            assert results[0]['resonance'] >= results[1]['resonance']
    
    def test_search_with_min_resonance(self):
        """Test search with minimum resonance threshold"""
        searcher = WaveSemanticSearch()
        
        # Store concepts
        for i in range(5):
            searcher.store_concept(f"concept {i}", np.random.rand(128))
        
        # Search with high threshold
        query_emb = np.random.rand(128)
        results = searcher.search(query_emb, top_k=10, min_resonance=0.9)
        
        # All results should meet threshold
        assert all(r['resonance'] >= 0.9 for r in results)
    
    def test_knowledge_absorption(self):
        """Test knowledge absorption and expansion"""
        searcher = WaveSemanticSearch()
        
        # Store target and source concepts
        target_id = searcher.store_concept("target concept", np.random.rand(256))
        source1_id = searcher.store_concept("source 1", np.random.rand(256))
        source2_id = searcher.store_concept("source 2", np.random.rand(256))
        
        # Get initial state
        target_before = searcher.wave_patterns[target_id]
        initial_depth = target_before.expansion_depth
        initial_energy = target_before.energy
        
        # Perform absorption
        expanded = searcher.absorb_and_expand(
            target_id=target_id,
            source_patterns=[source1_id, source2_id],
            absorption_strength=0.5
        )
        
        # Check that expansion occurred
        assert expanded.expansion_depth == initial_depth + 1
        assert len(expanded.absorbed_patterns) == 2
        assert source1_id in expanded.absorbed_patterns
        assert source2_id in expanded.absorbed_patterns
        
        # Energy should have increased from absorption
        assert expanded.energy > initial_energy
        
        # Absorption count should increment
        assert searcher.absorption_count == 1
    
    def test_absorption_with_resonance(self):
        """Test that absorption considers resonance"""
        searcher = WaveSemanticSearch()
        
        # Create very similar embeddings (high resonance)
        base_emb = np.random.rand(128)
        similar_emb = base_emb + np.random.rand(128) * 0.1  # Very similar
        
        target_id = searcher.store_concept("target", base_emb)
        similar_id = searcher.store_concept("similar", similar_emb)
        
        # Absorb similar pattern
        expanded = searcher.absorb_and_expand(
            target_id=target_id,
            source_patterns=[similar_id],
            absorption_strength=0.5
        )
        
        # Check that absorption happened
        assert len(expanded.absorbed_patterns) > 0
    
    def test_persistence(self):
        """Test pattern persistence to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_patterns.json"
            
            # Create searcher and store patterns
            searcher1 = WaveSemanticSearch(storage_path=str(storage_path))
            pid1 = searcher1.store_concept("persistent concept", np.random.rand(128))
            
            assert storage_path.exists()
            
            # Load patterns in new instance
            searcher2 = WaveSemanticSearch(storage_path=str(storage_path))
            
            # Check that pattern was loaded
            assert len(searcher2.wave_patterns) == 1
            assert pid1 in searcher2.wave_patterns
            assert searcher2.wave_patterns[pid1].text == "persistent concept"
    
    def test_statistics(self):
        """Test statistics tracking"""
        searcher = WaveSemanticSearch()
        
        # Perform some operations
        searcher.store_concept("concept 1", np.random.rand(128))
        searcher.store_concept("concept 2", np.random.rand(128))
        
        query = np.random.rand(128)
        searcher.search(query, top_k=1)
        searcher.search(query, top_k=1)
        
        stats = searcher.get_statistics()
        
        assert stats['total_patterns'] == 2
        assert stats['search_count'] == 2
        assert 'total_energy' in stats
        assert 'avg_expansion_depth' in stats
    
    def test_4d_wave_properties(self):
        """Test that 4D wave properties are properly set"""
        searcher = WaveSemanticSearch()
        
        # Create embedding with distinct properties
        embedding = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5] * 20)  # 120-dim
        
        pattern = searcher.embedding_to_wave(embedding, "4D test")
        
        # Check all 4 dimensions are set
        assert pattern.orientation.w != 0.0 or \
               pattern.orientation.x != 0.0 or \
               pattern.orientation.y != 0.0 or \
               pattern.orientation.z != 0.0
        
        # Check wave properties
        assert pattern.energy > 0
        assert pattern.frequency > 0
        assert -np.pi <= pattern.phase <= np.pi
    
    def test_expansion_depth_tracking(self):
        """Test that expansion depth is properly tracked"""
        searcher = WaveSemanticSearch()
        
        # Create a chain of absorptions
        target_id = searcher.store_concept("target", np.random.rand(128))
        
        for i in range(3):
            source_id = searcher.store_concept(f"source {i}", np.random.rand(128))
            searcher.absorb_and_expand(
                target_id=target_id,
                source_patterns=[source_id],
                absorption_strength=0.3
            )
        
        # Check depth increased
        final_pattern = searcher.wave_patterns[target_id]
        assert final_pattern.expansion_depth == 3


class TestIntegration:
    """Integration tests for the full system"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from storage to search to absorption"""
        searcher = WaveSemanticSearch()
        
        # Stage 1: Store initial knowledge
        concepts = {
            "AI": np.random.rand(256),
            "ML": np.random.rand(256),
            "DL": np.random.rand(256),
            "Dog": np.random.rand(256),
            "Cat": np.random.rand(256),
        }
        
        stored_ids = {}
        for concept, emb in concepts.items():
            pid = searcher.store_concept(concept, emb)
            stored_ids[concept] = pid
        
        # Stage 2: Search for AI-related concepts
        query = np.random.rand(256)
        results = searcher.search(query, query_text="AI", top_k=3)
        
        assert len(results) <= 3
        
        # Stage 3: Expand AI knowledge by absorbing ML and DL
        searcher.absorb_and_expand(
            target_id=stored_ids["AI"],
            source_patterns=[stored_ids["ML"], stored_ids["DL"]],
            absorption_strength=0.4
        )
        
        # Stage 4: Search again to see if expanded knowledge affects results
        results2 = searcher.search(query, query_text="AI", top_k=3)
        
        assert len(results2) <= 3
        
        # Check that AI pattern has been expanded
        ai_pattern = searcher.wave_patterns[stored_ids["AI"]]
        assert ai_pattern.expansion_depth > 0
        assert len(ai_pattern.absorbed_patterns) == 2
    
    def test_large_scale_search(self):
        """Test search with many patterns"""
        searcher = WaveSemanticSearch()
        
        # Store many patterns
        n_patterns = 100
        for i in range(n_patterns):
            searcher.store_concept(
                f"concept {i}",
                np.random.rand(128),
                metadata={'index': i}
            )
        
        # Search
        query = np.random.rand(128)
        results = searcher.search(query, top_k=10)
        
        assert len(results) == 10
        # Verify results are properly ranked
        for i in range(len(results) - 1):
            assert results[i]['resonance'] >= results[i+1]['resonance']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
