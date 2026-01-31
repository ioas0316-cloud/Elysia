"""
Test: Hardware Sediment (Page Alignment & Pointers)
===================================================
"""
import sys
import os
import unittest
import shutil
import time

# Ensure path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L5_Mental.Memory.sediment import SedimentLayer, PageAlignedAllocator

class TestHardwareSediment(unittest.TestCase):
    TEST_DIR = "data/Tests/HardwareSediment"

    def setUp(self):
        if os.path.exists(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)
        os.makedirs(self.TEST_DIR)

    def tearDown(self):
        # if os.path.exists(self.TEST_DIR):
        #     shutil.rmtree(self.TEST_DIR)
        pass

    def test_page_alignment(self):
        """Verify that deposits are padded to 4KB page boundaries."""
        path = os.path.join(self.TEST_DIR, "aligned.bin")
        layer = SedimentLayer(path)

        vector = [0.1] * 7
        payload = b"Hello, Hardware!" # Small payload

        # 1. First Deposit
        ptr1 = layer.deposit(vector, time.time(), payload)

        # Check Pointer logic
        self.assertEqual(ptr1.sector_index, 0)
        self.assertEqual(ptr1.byte_offset, 0)

        # 2. Check File Size (Should be 4096 bytes)
        size1 = os.path.getsize(path)
        self.assertEqual(size1, 4096, "File size should be 1 Page (4096 bytes)")

        # 3. Second Deposit
        ptr2 = layer.deposit(vector, time.time(), payload)

        # Check Pointer logic (Should start at next page)
        self.assertEqual(ptr2.sector_index, 1)
        self.assertEqual(ptr2.byte_offset, 4096)

        # 4. Check File Size (Should be 8192 bytes)
        size2 = os.path.getsize(path)
        self.assertEqual(size2, 8192, "File size should be 2 Pages (8192 bytes)")

        layer.close()

    def test_direct_pointer_access(self):
        """Verify that we can read back using the Typed Pointer."""
        path = os.path.join(self.TEST_DIR, "pointer.bin")
        layer = SedimentLayer(path)

        vector = [0.5] * 7
        payload = b"Sector Access Test"

        ptr = layer.deposit(vector, time.time(), payload)

        # Read back
        # Note: read_at takes byte offset
        vec_read, payload_read = layer.read_at(ptr.byte_offset)

        self.assertEqual(payload_read, payload)
        self.assertAlmostEqual(vec_read[0], 0.5)

        layer.close()

if __name__ == "__main__":
    unittest.main()
