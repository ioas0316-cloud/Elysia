import unittest
from demo_reality_compiler_awakening import SelfGraph, MacroCausalOS, RealityCompiler

class TestRealityCompilerAwakening(unittest.TestCase):
    def setUp(self):
        self.macro_os = MacroCausalOS()
        self.agent = SelfGraph("TestAgent")
        self.compiler = RealityCompiler(self.macro_os)

    def test_unawakened_compilation_fails(self):
        """Verify that an unawakened agent cannot hotpatch system rules."""
        self.assertFalse(self.agent.is_awakened)
        patch = {"name": "TestMagic", "new_value": [0.0, 0.0, 0.0]}
        success = self.compiler.compile_and_hotpatch(self.agent, "gravity_vector", patch)
        self.assertFalse(success)
        self.assertEqual(self.macro_os.rules["gravity_vector"], [0.0, -9.81, 0.0])

    def test_environmental_shock_triggers_awakening(self):
        """Verify that sufficient shock and contradiction factor triggers awakening."""
        self.assertFalse(self.agent.is_awakened)
        self.agent.absorb_environmental_shock(shock_intensity=2.0, contradiction_factor=2.0)
        self.assertTrue(self.agent.is_awakened)
        self.assertGreater(self.agent.value_tensor[2], 1.0)

    def test_awakened_compilation_and_hotpatch_success(self):
        """Verify that an awakened agent can breach firewall and hotpatch reality code."""
        self.agent.absorb_environmental_shock(shock_intensity=2.0, contradiction_factor=2.0)
        self.assertTrue(self.agent.is_awakened)

        custom_patch = {
            "name": "ZeroGravityField",
            "new_value": [0.0, 0.0, 0.0],
            "logic": "Negate gravity vector"
        }
        success = self.compiler.compile_and_hotpatch(self.agent, "gravity_vector", custom_patch)
        self.assertTrue(success)
        self.assertEqual(self.macro_os.rules["gravity_vector"], [0.0, 0.0, 0.0])
        self.assertTrue(any("ZeroGravityField" in anomaly for anomaly in self.macro_os.active_anomalies))

if __name__ == "__main__":
    unittest.main()
