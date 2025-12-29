"""
Tests for Unified Knowledge System
==================================

Comprehensive test suite covering all knowledge operations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import tempfile
import shutil
from pathlib import Path

from Core.FoundationLayer.Foundation.unified_knowledge_system import (
    UnifiedKnowledgeSystem,
    KnowledgeEntry,
    KnowledgeType,
    KnowledgeSource,
    get_unified_knowledge_system
)


class TestUnifiedKnowledgeSystem(unittest.TestCase):
    """Test unified knowledge system."""
    
    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.system = UnifiedKnowledgeSystem(
            node_id="test_node",
            data_dir=self.test_dir,
            enable_web=False
        )
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test system initializes correctly."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.node_id, "test_node")
        self.assertIsInstance(self.system.knowledge_base, dict)
    
    def test_learn_concept(self):
        """Test learning a single concept."""
        entry = self.system.learn_concept(
            "Test Concept",
            "This is a test concept",
            tags=["test"]
        )
        
        self.assertIsInstance(entry, KnowledgeEntry)
        self.assertEqual(entry.concept, "Test Concept")
        self.assertIn(entry.knowledge_id, self.system.knowledge_base)
        self.assertEqual(self.system.stats["total_acquired"], 1)
    
    def test_learn_curriculum(self):
        """Test learning structured curriculum."""
        curriculum = [
            {"concept": "Concept 1", "description": "Description 1"},
            {"concept": "Concept 2", "description": "Description 2"},
            {"concept": "Concept 3", "description": "Description 3"},
        ]
        
        result = self.system.learn_curriculum(curriculum)
        
        self.assertEqual(result["total_concepts"], 3)
        self.assertEqual(result["successful"], 3)
        self.assertEqual(len(self.system.knowledge_base), 3)
    
    def test_query_knowledge(self):
        """Test querying knowledge base."""
        # Add some knowledge
        self.system.learn_concept("AI", "Artificial Intelligence")
        self.system.learn_concept("ML", "Machine Learning")
        self.system.learn_concept("DL", "Deep Learning")
        
        # Query all
        results = self.system.query_knowledge()
        self.assertEqual(len(results), 3)
        
        # Query by concept
        results = self.system.query_knowledge(concept="AI")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].concept, "AI")
    
    def test_query_with_filters(self):
        """Test querying with various filters."""
        # Add knowledge with different types
        self.system.learn_concept(
            "Pattern 1",
            "A pattern",
            knowledge_type=KnowledgeType.PATTERN
        )
        self.system.learn_concept(
            "Skill 1",
            "A skill",
            knowledge_type=KnowledgeType.SKILL
        )
        
        # Query by type
        patterns = self.system.query_knowledge(knowledge_type=KnowledgeType.PATTERN)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].knowledge_type, KnowledgeType.PATTERN)
        
        skills = self.system.query_knowledge(knowledge_type=KnowledgeType.SKILL)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].knowledge_type, KnowledgeType.SKILL)
    
    def test_query_by_tags(self):
        """Test querying by tags."""
        self.system.learn_concept("Tagged 1", "Description", tags=["tag1", "tag2"])
        self.system.learn_concept("Tagged 2", "Description", tags=["tag2", "tag3"])
        self.system.learn_concept("Untagged", "Description", tags=[])
        
        # Query by tag
        results = self.system.query_knowledge(tags=["tag1"])
        self.assertEqual(len(results), 1)
        
        results = self.system.query_knowledge(tags=["tag2"])
        self.assertEqual(len(results), 2)
    
    def test_query_by_quality(self):
        """Test querying with quality filter."""
        entry1 = self.system.learn_concept("High Quality", "Description")
        entry1.quality_score = 0.9
        
        entry2 = self.system.learn_concept("Low Quality", "Description")
        entry2.quality_score = 0.2
        
        # Query high quality only
        results = self.system.query_knowledge(min_quality=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].concept, "High Quality")
    
    def test_share_knowledge(self):
        """Test knowledge sharing."""
        entry = self.system.learn_concept("Shareable", "Description")
        
        success = self.system.share_knowledge(entry.knowledge_id)
        self.assertTrue(success)
        self.assertTrue(entry.content.get("shared"))
        self.assertEqual(self.system.stats["total_shared"], 1)
    
    def test_receive_knowledge(self):
        """Test receiving knowledge from another node."""
        # Create entry to receive
        entry_data = {
            "knowledge_id": "test_id",
            "concept": "Received Concept",
            "description": "From another node",
            "knowledge_type": "concept",
            "source": "network",
            "quality_score": 0.8,
            "confidence": 0.7,
            "usage_count": 0,
            "validation_count": 0,
            "source_node_id": "other_node",
            "timestamp": 1234567890.0,
            "tags": ["network"],
            "content": {},
            "validations": []
        }
        
        success = self.system.receive_knowledge(entry_data)
        self.assertTrue(success)
        self.assertIn("test_id", self.system.knowledge_base)
        self.assertEqual(self.system.stats["total_received"], 1)
    
    def test_reject_low_quality(self):
        """Test rejection of low quality knowledge."""
        entry_data = {
            "knowledge_id": "low_quality_id",
            "concept": "Bad Knowledge",
            "description": "Low quality",
            "knowledge_type": "concept",
            "source": "network",
            "quality_score": 0.1,  # Below threshold
            "confidence": 0.1,
            "usage_count": 0,
            "validation_count": 0,
            "source_node_id": "other_node",
            "timestamp": 1234567890.0,
            "tags": [],
            "content": {},
            "validations": []
        }
        
        success = self.system.receive_knowledge(entry_data)
        self.assertFalse(success)
        self.assertNotIn("low_quality_id", self.system.knowledge_base)
    
    def test_validate_knowledge(self):
        """Test knowledge validation."""
        entry = self.system.learn_concept("Validatable", "Description")
        initial_quality = entry.quality_score
        
        # Validate as useful
        self.system.validate_knowledge(entry.knowledge_id, is_useful=True)
        
        self.assertEqual(entry.validation_count, 1)
        self.assertGreater(entry.quality_score, initial_quality)
        self.assertEqual(len(entry.validations), 1)
    
    def test_use_knowledge(self):
        """Test knowledge usage tracking."""
        entry = self.system.learn_concept("Usable", "Description")
        
        self.assertEqual(entry.usage_count, 0)
        
        self.system.use_knowledge(entry.knowledge_id)
        self.assertEqual(entry.usage_count, 1)
        
        self.system.use_knowledge(entry.knowledge_id)
        self.assertEqual(entry.usage_count, 2)
    
    def test_statistics(self):
        """Test statistics generation."""
        # Add some knowledge
        self.system.learn_concept("Concept 1", "Description")
        self.system.learn_concept("Concept 2", "Description")
        
        stats = self.system.get_statistics()
        
        self.assertEqual(stats["total_knowledge"], 2)
        self.assertEqual(stats["total_acquired"], 2)
        self.assertIn("average_quality", stats)
        self.assertIn("by_type", stats)
    
    def test_cleanup_old_knowledge(self):
        """Test cleanup of old knowledge."""
        import time
        
        # Add old low-quality knowledge
        entry = self.system.learn_concept("Old", "Description")
        entry.timestamp = time.time() - (40 * 86400)  # 40 days old
        entry.quality_score = 0.3
        entry.usage_count = 0
        
        # Add recent knowledge
        self.system.learn_concept("Recent", "Description")
        
        # Cleanup
        removed = self.system.cleanup_old_knowledge(max_age_days=30)
        
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.system.knowledge_base), 1)
    
    def test_export_knowledge(self):
        """Test knowledge export."""
        self.system.learn_concept("Exportable", "Description")
        
        export_path = Path(self.test_dir) / "export.json"
        data = self.system.export_knowledge(str(export_path))
        
        self.assertTrue(export_path.exists())
        self.assertIn("knowledge_base", data)
        self.assertIn("statistics", data)
    
    def test_save_and_load_state(self):
        """Test persistence."""
        # Add knowledge
        entry = self.system.learn_concept("Persistent", "Description", tags=["test"])
        kid = entry.knowledge_id
        
        # Save
        self.system.save_state()
        
        # Create new system with same directory
        new_system = UnifiedKnowledgeSystem(
            node_id="test_node",
            data_dir=self.test_dir,
            enable_web=False
        )
        
        # Check knowledge loaded
        self.assertIn(kid, new_system.knowledge_base)
        loaded_entry = new_system.knowledge_base[kid]
        self.assertEqual(loaded_entry.concept, "Persistent")
        self.assertIn("test", loaded_entry.tags)
    
    def test_singleton_accessor(self):
        """Test singleton accessor function."""
        system1 = get_unified_knowledge_system()
        system2 = get_unified_knowledge_system()
        
        self.assertIs(system1, system2)


class TestKnowledgeEntry(unittest.TestCase):
    """Test KnowledgeEntry dataclass."""
    
    def test_creation(self):
        """Test creating knowledge entry."""
        entry = KnowledgeEntry(
            concept="Test",
            description="Test description",
            knowledge_type=KnowledgeType.CONCEPT,
            source=KnowledgeSource.CURRICULUM
        )
        
        self.assertEqual(entry.concept, "Test")
        self.assertEqual(entry.knowledge_type, KnowledgeType.CONCEPT)
        self.assertIsNotNone(entry.knowledge_id)
    
    def test_to_dict(self):
        """Test serialization to dict."""
        entry = KnowledgeEntry(
            concept="Test",
            description="Description",
            tags=["tag1", "tag2"]
        )
        
        data = entry.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["concept"], "Test")
        self.assertIn("knowledge_id", data)
        self.assertIn("timestamp", data)
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "knowledge_id": "test_id",
            "concept": "Test",
            "description": "Description",
            "knowledge_type": "concept",
            "source": "curriculum",
            "confidence": 0.8,
            "quality_score": 0.7,
            "usage_count": 5,
            "validation_count": 3,
            "source_node_id": "node1",
            "timestamp": 1234567890.0,
            "tags": ["tag1"],
            "validations": [],
            "content": {},
            "wave_coords": None
        }
        
        entry = KnowledgeEntry.from_dict(data)
        
        self.assertEqual(entry.concept, "Test")
        self.assertEqual(entry.knowledge_id, "test_id")
        self.assertEqual(entry.usage_count, 5)
        self.assertEqual(entry.knowledge_type, KnowledgeType.CONCEPT)


if __name__ == '__main__':
    unittest.main()
