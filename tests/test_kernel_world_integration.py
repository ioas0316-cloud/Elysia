import unittest
import sys
import os

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestKernelWorldIntegration(unittest.TestCase):

    def test_kernel_initialization_and_world_presence(self):
        """
        Tests if the ElysiaKernel initializes without errors and if the world
        object is created within it.
        """
        try:
            from Core.Kernel import ElysiaKernel
            kernel = ElysiaKernel()
            self.assertIsNotNone(kernel, "Kernel should not be None")
            self.assertTrue(hasattr(kernel, "world"), "Kernel should have a 'world' attribute")
            self.assertIsNotNone(kernel.world, "kernel.world should not be None")
        except Exception as e:
            self.fail(f"Kernel initialization failed with an exception: {e}")

    def test_kernel_tick_runs_world_simulation(self):
        """
        Tests if the kernel's tick method runs for a few steps without errors,
        and verifies that the world simulation time progresses.
        """
        try:
            from Core.Kernel import ElysiaKernel
            # We need a fresh kernel instance for each test to avoid singleton issues
            ElysiaKernel._instances = {}
            kernel = ElysiaKernel()

            # Add a single cell to the world to make the simulation non-trivial
            kernel.world.add_cell("test_cell", properties={"label": "human"})

            initial_time_step = kernel.world.time_step

            # Run a few simulation ticks
            for _ in range(10):
                kernel.tick()

            final_time_step = kernel.world.time_step
            self.assertGreater(final_time_step, initial_time_step, "World time_step should advance after kernel ticks.")
            self.assertEqual(final_time_step, 10, "World time_step should be exactly 10 after 10 ticks.")

        except Exception as e:
            self.fail(f"Kernel ticking failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
