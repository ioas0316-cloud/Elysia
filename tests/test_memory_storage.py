
import sys
import os
import shutil
sys.path.append(os.getcwd())

import unittest
import time
from Core.Memory.Mind.memory_storage import MemoryStorage
from Core.Memory.Mind.hippocampus import Hippocampus

class TestMemoryStorage(unittest.TestCase):
    def setUp(self):
        import uuid
        self.test_db = f"test_memory_{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.test_db)

    def tearDown(self):
        self.storage.close()
        # Give SQLite a moment to release locks
        time.sleep(0.1)
        for f in [self.test_db, self.test_db + "-wal", self.test_db + "-shm"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"Warning: Could not remove {f} (locked)")
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")

    def test_crud_concept(self):
        # Create
        data = {"key": "value", "num": 123}
        self.assertTrue(self.storage.add_concept("test_concept", data))
        
        # Read
        retrieved = self.storage.get_concept("test_concept")
        self.assertEqual(retrieved["key"], "value")
        
        # Update
        data["key"] = "new_value"
        self.storage.add_concept("test_concept", data)
        retrieved = self.storage.get_concept("test_concept")
        self.assertEqual(retrieved["key"], "new_value")

    # Relations test removed (Holographic Storage)
    # def test_relations(self): ...

class TestHippocampusIntegration(unittest.TestCase):
    def setUp(self):
        import uuid
        self.hippocampus = Hippocampus()
        self.db_path = f"test_hippo_{uuid.uuid4().hex}.db"
        self.hippocampus.storage = MemoryStorage(self.db_path)

    def tearDown(self):
        self.hippocampus.storage.close()
        time.sleep(0.1)
        for f in [self.db_path, self.db_path + "-wal", self.db_path + "-shm"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

    def test_add_concept_flow(self):
        concept = "TestConcept"
        self.hippocampus.add_concept(concept)
        
        # Verify in DB
        data = self.hippocampus.storage.get_concept(concept)
        self.assertIsNotNone(data)
        self.assertEqual(data['id'], concept)
        
        # Verify in Universe
        self.assertIn(concept, self.hippocampus.universe.spheres)

    # Causal link test removed (Holographic Storage)
    # def test_causal_link(self): ...

if __name__ == "__main__":
    unittest.main()
