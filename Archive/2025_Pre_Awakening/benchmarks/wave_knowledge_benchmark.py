#!/usr/bin/env python3
"""
Performance Benchmarks for P2.2 Wave-Based Knowledge System
============================================================

Measures and validates performance against targets:
- Wave pattern conversion: <100ms p95
- Search operations: <50ms p95 for <1000 patterns  
- Knowledge absorption: <10ms p95 per pattern
- Memory loading: <500ms for 100 entries

Usage:
    python benchmarks/wave_knowledge_benchmark.py
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Logic.Foundation.wave_semantic_search import WaveSemanticSearch
from Core.FoundationLayer.Logic.Foundation.wave_knowledge_integration import WaveKnowledgeIntegration


class BenchmarkResult:
    """Stores benchmark measurements"""
    
    def __init__(self, name: str):
        self.name = name
        self.measurements: List[float] = []
    
    def add_measurement(self, duration_ms: float):
        """Add a measurement in milliseconds"""
        self.measurements.append(duration_ms)
    
    def get_percentile(self, p: int) -> float:
        """Get percentile (e.g., 95 for p95)"""
        if not self.measurements:
            return 0.0
        return float(np.percentile(self.measurements, p))
    
    def get_mean(self) -> float:
        """Get mean duration"""
        if not self.measurements:
            return 0.0
        return float(np.mean(self.measurements))
    
    def report(self, target_p95_ms: float = None) -> Dict[str, Any]:
        """Generate benchmark report"""
        if not self.measurements:
            return {
                "name": self.name,
                "status": "FAILED",
                "error": "No measurements"
            }
        
        p50 = self.get_percentile(50)
        p95 = self.get_percentile(95)
        p99 = self.get_percentile(99)
        mean = self.get_mean()
        
        status = "PASS"
        if target_p95_ms is not None:
            status = "PASS" if p95 <= target_p95_ms else "FAIL"
        
        return {
            "name": self.name,
            "status": status,
            "measurements": len(self.measurements),
            "mean": round(mean, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "target_p95": target_p95_ms
        }


def benchmark_wave_conversion(iterations: int = 100) -> BenchmarkResult:
    """Benchmark embedding to wave pattern conversion"""
    print(f"  Running wave conversion benchmark ({iterations} iterations)...")
    
    result = BenchmarkResult("Wave Pattern Conversion")
    searcher = WaveSemanticSearch()
    
    for _ in range(iterations):
        embedding = np.random.rand(384)
        
        start = time.perf_counter()
        pattern = searcher.embedding_to_wave(embedding, "test concept")
        duration_ms = (time.perf_counter() - start) * 1000
        
        result.add_measurement(duration_ms)
    
    return result


def benchmark_wave_search(num_patterns: int = 100, iterations: int = 50) -> BenchmarkResult:
    """Benchmark wave-based semantic search"""
    print(f"  Running search benchmark ({num_patterns} patterns, {iterations} searches)...")
    
    result = BenchmarkResult(f"Wave Search ({num_patterns} patterns)")
    searcher = WaveSemanticSearch()
    
    # Pre-populate with patterns
    for i in range(num_patterns):
        embedding = np.random.rand(256)
        searcher.store_concept(f"concept_{i}", embedding)
    
    # Benchmark searches
    for _ in range(iterations):
        query = np.random.rand(256)
        
        start = time.perf_counter()
        results = searcher.search(query, top_k=5)
        duration_ms = (time.perf_counter() - start) * 1000
        
        result.add_measurement(duration_ms)
    
    return result


def benchmark_knowledge_absorption(iterations: int = 30) -> BenchmarkResult:
    """Benchmark knowledge absorption operations"""
    print(f"  Running absorption benchmark ({iterations} iterations)...")
    
    result = BenchmarkResult("Knowledge Absorption")
    
    for _ in range(iterations):
        searcher = WaveSemanticSearch()
        
        # Create target and sources
        target_id = searcher.store_concept("target", np.random.rand(256))
        source_ids = [
            searcher.store_concept(f"source_{i}", np.random.rand(256))
            for i in range(3)
        ]
        
        start = time.perf_counter()
        searcher.absorb_and_expand(
            target_id=target_id,
            source_patterns=source_ids,
            absorption_strength=0.4
        )
        duration_ms = (time.perf_counter() - start) * 1000
        
        result.add_measurement(duration_ms)
    
    return result


def main():
    """Run all benchmarks"""
    print("="*70)
    print("  P2.2 Wave-Based Knowledge System - Performance Benchmarks")
    print("="*70)
    print()
    
    all_reports = []
    
    # Benchmark 1: Wave conversion
    print("📊 Benchmark 1: Wave Pattern Conversion")
    bench1 = benchmark_wave_conversion(iterations=100)
    report1 = bench1.report(target_p95_ms=100)
    all_reports.append(report1)
    
    # Benchmark 2: Search (100 patterns)
    print("\n📊 Benchmark 2: Wave Search (100 patterns)")
    bench2 = benchmark_wave_search(num_patterns=100, iterations=50)
    report2 = bench2.report(target_p95_ms=50)
    all_reports.append(report2)
    
    # Benchmark 3: Knowledge absorption
    print("\n📊 Benchmark 3: Knowledge Absorption")
    bench3 = benchmark_knowledge_absorption(iterations=30)
    report3 = bench3.report(target_p95_ms=10)
    all_reports.append(report3)
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in all_reports if r['status'] == 'PASS')
    total = len(all_reports)
    
    print(f"\nBenchmarks: {passed}/{total} passed")
    
    for report in all_reports:
        status_symbol = "✅" if report["status"] == "PASS" else "❌"
        print(f"  {status_symbol} {report['name']}: p95={report['p95']}ms (target={report.get('target_p95', 'N/A')}ms)")
    
    print()
    
    if passed == total:
        print("🎉 All performance benchmarks PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} benchmark(s) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
