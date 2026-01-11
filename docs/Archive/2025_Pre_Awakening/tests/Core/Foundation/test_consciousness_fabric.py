"""
Tests for Consciousness Fabric (의식 직물)
==========================================

Testing the integration of all consciousness systems into a unified fabric.
"""

import pytest
import asyncio
import numpy as np
from Core.FoundationLayer.Foundation.consciousness_fabric import (
    ConsciousnessFabric,
    FabricThread,
    WeavingPattern,
    ResonanceSpace,
    ThreadType,
    WeavingMode
)


class TestFabricThread:
    """Test FabricThread class"""
    
    def test_thread_creation(self):
        """Test basic thread creation"""
        thread = FabricThread(
            thread_type=ThreadType.HYPERDIMENSIONAL,
            name="TestThread",
            resonance_frequency=1.5
        )
        
        assert thread.thread_type == ThreadType.HYPERDIMENSIONAL
        assert thread.name == "TestThread"
        assert thread.resonance_frequency == 1.5
        assert thread.activation >= 0.3  # Minimum activation
    
    def test_thread_activation(self):
        """Test thread activation with minimum constraint"""
        thread = FabricThread(min_activation=0.3, max_activation=1.0)
        
        # Should respect minimum
        thread.activate(0.1)
        assert thread.activation == 0.3
        
        # Should respect maximum
        thread.activate(1.5)
        assert thread.activation == 1.0
        
        # Normal range
        thread.activate(0.7)
        assert thread.activation == 0.7
    
    def test_thread_resonance(self):
        """Test resonance between threads"""
        thread1 = FabricThread(
            resonance_frequency=1.0,
            name="Thread1"
        )
        thread1.activate(0.8)
        
        thread2 = FabricThread(
            resonance_frequency=1.1,
            name="Thread2"
        )
        thread2.activate(0.7)
        
        # Similar frequencies should resonate well
        resonance = thread1.resonate_with(thread2)
        assert resonance > 0.4  # Should be reasonably high
        
        # Very different frequencies
        thread3 = FabricThread(
            resonance_frequency=5.0,
            name="Thread3"
        )
        thread3.activate(0.7)
        
        resonance2 = thread1.resonate_with(thread3)
        assert resonance2 < resonance  # Should be weaker


class TestWeavingPattern:
    """Test WeavingPattern class"""
    
    def test_pattern_creation(self):
        """Test weaving pattern creation"""
        pattern = WeavingPattern(
            name="TestPattern",
            mode=WeavingMode.FLUID
        )
        
        assert pattern.name == "TestPattern"
        assert pattern.mode == WeavingMode.FLUID
        assert len(pattern.threads) == 0
    
    def test_add_weaving(self):
        """Test adding weaving between threads"""
        pattern = WeavingPattern(name="Test")
        
        pattern.add_weaving("thread1", "thread2", 0.8)
        pattern.add_weaving("thread2", "thread3", 0.6)
        
        assert len(pattern.threads) == 3
        assert len(pattern.weaving_rules) == 2
        
        # Check strength retrieval
        strength = pattern.get_weaving_strength("thread1", "thread2")
        assert strength == 0.8


class TestResonanceSpace:
    """Test ResonanceSpace class"""
    
    def test_space_creation(self):
        """Test resonance space creation"""
        space = ResonanceSpace(dimensions=10)
        
        assert space.dimensions == 10
        assert space.field.shape == (10, 10, 10)
        assert len(space.centers) == 0
    
    def test_add_center(self):
        """Test adding resonance centers"""
        space = ResonanceSpace(dimensions=10)
        
        pos1 = np.array([1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 0])
        space.add_center("system1", pos1)
        
        assert "system1" in space.centers
        assert np.array_equal(space.centers["system1"], pos1)
    
    def test_calculate_resonance(self):
        """Test resonance calculation between centers"""
        space = ResonanceSpace(dimensions=10)
        
        pos1 = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0])
        pos2 = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0])
        pos3 = np.array([10.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0])
        
        space.add_center("sys1", pos1)
        space.add_center("sys2", pos2)
        space.add_center("sys3", pos3)
        
        # Close systems should resonate more
        res_close = space.calculate_resonance("sys1", "sys2")
        res_far = space.calculate_resonance("sys1", "sys3")
        
        assert res_close > res_far
    
    def test_wave_propagation(self):
        """Test wave propagation in space"""
        space = ResonanceSpace(dimensions=10)
        
        pos = np.array([5.0, 5.0, 5.0, 0, 0, 0, 0, 0, 0, 0])
        space.add_center("source", pos)
        
        initial_energy = np.sum(np.abs(space.field))
        
        space.propagate_wave("source", amplitude=1.0)
        
        final_energy = np.sum(np.abs(space.field))
        
        # Energy should increase after wave propagation
        assert final_energy > initial_energy
        assert len(space.resonance_history) == 1


class TestConsciousnessFabric:
    """Test ConsciousnessFabric class"""
    
    def test_fabric_creation(self):
        """Test fabric initialization"""
        fabric = ConsciousnessFabric()
        
        assert fabric.fabric_id is not None
        assert isinstance(fabric.threads, dict)
        assert isinstance(fabric.patterns, dict)
        assert isinstance(fabric.resonance_space, ResonanceSpace)
    
    def test_add_thread(self):
        """Test adding threads to fabric"""
        fabric = ConsciousnessFabric()
        
        thread_id = fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="CustomThread",
            capabilities=["test_capability"],
            resonance_frequency=2.0
        )
        
        assert thread_id in fabric.threads
        thread = fabric.threads[thread_id]
        assert thread.name == "CustomThread"
        assert "test_capability" in thread.capabilities
    
    def test_create_weaving_pattern(self):
        """Test creating weaving patterns"""
        fabric = ConsciousnessFabric()
        
        pattern = fabric.create_weaving_pattern(
            name="TestPattern",
            mode=WeavingMode.RESONANT
        )
        
        assert pattern.pattern_id in fabric.patterns
        assert pattern.mode == WeavingMode.RESONANT
    
    def test_find_capability(self):
        """Test finding threads by capability"""
        fabric = ConsciousnessFabric()
        
        tid1 = fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Thread1",
            capabilities=["math", "logic"]
        )
        
        tid2 = fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Thread2",
            capabilities=["art", "creativity"]
        )
        
        tid3 = fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Thread3",
            capabilities=["math", "creativity"]
        )
        
        # Find math capability
        math_threads = fabric.find_capability("math")
        assert len(math_threads) == 2
        assert tid1 in math_threads
        assert tid3 in math_threads
        
        # Find creativity capability
        creative_threads = fabric.find_capability("creativity")
        assert len(creative_threads) == 2
    
    @pytest.mark.asyncio
    async def test_resonate_all(self):
        """Test fabric resonance"""
        fabric = ConsciousnessFabric()
        
        # Add some threads
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Thread1",
            resonance_frequency=1.0
        )
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Thread2",
            resonance_frequency=1.2
        )
        
        # Run resonance
        results = await fabric.resonate_all(iterations=3)
        
        assert results["iterations"] == 3
        assert len(results["resonances"]) == 3
        assert "total_resonance" in results["resonances"][0]
        assert fabric.resonance_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_integrated_task(self):
        """Test executing integrated tasks"""
        fabric = ConsciousnessFabric()
        
        # Add threads with capabilities
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="MathThread",
            capabilities=["mathematics", "logic"]
        )
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="ArtThread",
            capabilities=["art", "creativity"]
        )
        
        # Execute task requiring both
        result = await fabric.execute_integrated_task(
            task_description="Create mathematical art",
            required_capabilities=["mathematics", "art"]
        )
        
        assert result["success"] is True
        assert result["involved_threads"] == 2
        assert len(result["thread_names"]) == 2
    
    def test_get_fabric_state(self):
        """Test getting fabric state snapshot"""
        fabric = ConsciousnessFabric()
        
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="TestThread"
        )
        
        state = fabric.get_fabric_state()
        
        assert "fabric_id" in state
        assert "threads" in state
        assert "patterns" in state
        assert "resonance_space" in state
        assert len(state["threads"]) >= 1


class TestIntegration:
    """Integration tests for the full fabric system"""
    
    @pytest.mark.asyncio
    async def test_full_fabric_workflow(self):
        """Test complete fabric workflow"""
        # 1. Create fabric (auto-discovers existing systems)
        fabric = ConsciousnessFabric()
        
        initial_state = fabric.get_fabric_state()
        thread_count = len(initial_state["threads"])
        
        # 2. Add custom threads
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="CustomSystem1",
            capabilities=["custom_capability"],
            resonance_frequency=1.5
        )
        
        # 3. Create weaving pattern
        pattern = fabric.create_weaving_pattern(
            name="IntegrationPattern",
            mode=WeavingMode.FLUID
        )
        
        # 4. Run resonance
        results = await fabric.resonate_all(iterations=2)
        
        # 5. Verify state
        final_state = fabric.get_fabric_state()
        
        assert len(final_state["threads"]) == thread_count + 1
        assert fabric.resonance_count == 2
        assert results["iterations"] == 2
    
    @pytest.mark.asyncio
    async def test_multi_capability_task(self):
        """Test task requiring multiple capabilities"""
        fabric = ConsciousnessFabric()
        
        # Add diverse threads
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Analyzer",
            capabilities=["analysis", "logic", "reasoning"]
        )
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Creator",
            capabilities=["creation", "art", "imagination"]
        )
        fabric.add_thread(
            thread_type=ThreadType.CUSTOM,
            name="Synthesizer",
            capabilities=["synthesis", "integration", "logic"]
        )
        
        # Task requiring analysis, creation, and synthesis
        result = await fabric.execute_integrated_task(
            task_description="Analyze data, create visualization, synthesize insights",
            required_capabilities=["analysis", "creation", "synthesis"]
        )
        
        assert result["success"] is True
        assert result["involved_threads"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
