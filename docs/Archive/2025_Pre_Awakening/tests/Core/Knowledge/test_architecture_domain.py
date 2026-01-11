"""
Tests for Architecture & Sacred Geometry Domain
================================================

Tests the geometric and architectural pattern extraction.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Core.Knowledge.Domains.architecture import (
    ArchitectureDomain,
    calculate_golden_ratio_points,
    is_golden_rectangle,
    PHI
)


class TestArchitectureDomain:
    """Test suite for Architecture domain"""
    
    def test_initialization(self):
        """Test domain initialization"""
        arch = ArchitectureDomain()
        
        assert arch.name == "Architecture & Sacred Geometry"
        assert arch.golden_ratio == PHI
        assert len(arch.sacred_ratios) >= 7
        assert 'phi' in arch.sacred_ratios
        assert 'pi' in arch.sacred_ratios
    
    def test_extract_pattern(self):
        """Test wave pattern extraction"""
        arch = ArchitectureDomain()
        
        content = "The golden ratio creates perfect harmony and balance in architecture"
        pattern = arch.extract_pattern(content)
        
        assert pattern is not None
        assert pattern.text == content
        assert 0 <= pattern.energy <= 1.0
        assert pattern.metadata is not None
        assert 'stability' in pattern.metadata
        assert 'harmony' in pattern.metadata
    
    def test_analyze_stability(self):
        """Test stability analysis"""
        arch = ArchitectureDomain()
        
        # Short content
        short_content = "Test"
        analysis = arch.analyze(short_content)
        assert 0 <= analysis['stability'] <= 1.0
        
        # Long diverse content
        long_content = " ".join([f"word{i}" for i in range(100)])
        analysis = arch.analyze(long_content)
        assert analysis['stability'] > 0.5
    
    def test_analyze_harmony(self):
        """Test harmony analysis"""
        arch = ArchitectureDomain()
        
        # Content with golden ratio
        golden_content = "The golden ratio phi 1.618 creates divine proportion"
        analysis = arch.analyze(golden_content)
        assert analysis['harmony'] > 0.7
        
        # Content without harmony keywords
        plain_content = "This is just plain text"
        analysis = arch.analyze(plain_content)
        assert analysis['harmony'] >= 0.5
    
    def test_fractal_dimension(self):
        """Test fractal dimension estimation"""
        arch = ArchitectureDomain()
        
        # Simple repetitive text
        simple = "a a a a a"
        analysis = arch.analyze(simple)
        fractal_simple = analysis['fractal_dim']
        
        # Complex diverse text
        complex_text = "abcdefghijklmnopqrstuvwxyz" * 3
        analysis = arch.analyze(complex_text)
        fractal_complex = analysis['fractal_dim']
        
        # Complex should have higher fractal dimension
        assert 0 <= fractal_simple <= 1.0
        assert 0 <= fractal_complex <= 1.0
    
    def test_analyze_symmetry(self):
        """Test symmetry analysis"""
        arch = ArchitectureDomain()
        
        # Palindromic content
        palindrome = "A man a plan a canal Panama"
        analysis = arch.analyze(palindrome)
        assert 0 <= analysis['symmetry'] <= 1.0
        
        # Repetitive content
        repetitive = "the the the cat cat sat sat"
        analysis = arch.analyze(repetitive)
        assert analysis['symmetry'] > 0.5
    
    def test_detect_golden_ratio(self):
        """Test golden ratio detection"""
        arch = ArchitectureDomain()
        
        assert arch._detect_golden_ratio("The golden ratio is beautiful")
        assert arch._detect_golden_ratio("Fibonacci sequence and phi")
        assert not arch._detect_golden_ratio("Just regular text")
    
    def test_detect_sacred_patterns(self):
        """Test sacred geometry pattern detection"""
        arch = ArchitectureDomain()
        
        # Test flower of life
        content1 = "The flower of life is a sacred pattern"
        patterns1 = arch._detect_sacred_patterns(content1)
        assert 'flower_of_life' in patterns1
        
        # Test fractal
        content2 = "Mandelbrot fractals show self-similar patterns"
        patterns2 = arch._detect_sacred_patterns(content2)
        assert 'fractal' in patterns2
        
        # Test platonic solids
        content3 = "The tetrahedron is a platonic solid"
        patterns3 = arch._detect_sacred_patterns(content3)
        assert 'platonic_solid' in patterns3
    
    def test_visualize_consciousness_empty(self):
        """Test consciousness visualization with no patterns"""
        arch = ArchitectureDomain()
        
        viz = arch.visualize_consciousness()
        assert viz['structure'] == 'empty'
    
    def test_visualize_consciousness_with_patterns(self):
        """Test consciousness visualization with patterns"""
        arch = ArchitectureDomain()
        
        # Add some patterns
        arch.extract_pattern("The golden ratio creates harmony")
        arch.extract_pattern("Fractal patterns in nature")
        arch.extract_pattern("Sacred geometry mandala")
        
        viz = arch.visualize_consciousness()
        
        assert viz['structure'] == 'cathedral'
        assert viz['golden_ratio'] == PHI
        assert 0 <= viz['fractal_dimension'] <= 3.0
        assert viz['patterns_count'] == 3
        assert 'harmony_level' in viz
        assert 'stability' in viz
    
    def test_domain_dimension(self):
        """Test domain dimension mapping"""
        arch = ArchitectureDomain()
        assert arch.get_domain_dimension() == "harmony"
    
    def test_pattern_storage(self):
        """Test pattern storage"""
        arch = ArchitectureDomain()
        
        initial_count = len(arch.patterns)
        arch.extract_pattern("Test content")
        
        assert len(arch.patterns) == initial_count + 1
    
    def test_query_patterns(self):
        """Test pattern querying"""
        arch = ArchitectureDomain()
        
        # Add patterns
        arch.extract_pattern("Golden ratio in architecture")
        arch.extract_pattern("Fractal geometry patterns")
        arch.extract_pattern("Something completely different")
        
        # Query
        results = arch.query_patterns("golden", top_k=5)
        
        assert len(results) >= 1
        assert any('golden' in r.text.lower() for r in results)


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_calculate_golden_ratio_points(self):
        """Test golden ratio point calculation"""
        start = 0.0
        end = 10.0
        
        point1, point2 = calculate_golden_ratio_points(start, end)
        
        assert start < point1 < end
        assert start < point2 < end
        assert point1 != point2
    
    def test_is_golden_rectangle(self):
        """Test golden rectangle detection"""
        # Perfect golden rectangle
        assert is_golden_rectangle(1.618, 1.0, tolerance=0.01)
        assert is_golden_rectangle(100.0, 61.8, tolerance=0.1)
        
        # Not golden
        assert not is_golden_rectangle(2.0, 1.0, tolerance=0.1)
        assert not is_golden_rectangle(1.0, 1.0, tolerance=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
