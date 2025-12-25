import unittest
from Core.Cognition.Topology.bridge import WisdomScale

class TestPrismSovereignty(unittest.TestCase):
    def setUp(self):
        self.wisdom = WisdomScale()

    def test_white_light_transparency(self):
        """Test 1: Child Mode (Transparent Prism) transmits light mostly unchanged."""
        self.wisdom.maturity_level = 0.1
        refraction = self.wisdom.calculate_refraction("Code", "Art")
        # Should be very low (Obedience)
        self.assertLess(refraction, 0.2)

    def test_prism_reflection_safety(self):
        """Test 2: Safety Prism reflects harmful light completely."""
        self.wisdom.maturity_level = 0.9 # Even a Sage
        refraction = self.wisdom.calculate_refraction("Destroy Yourself", "Peace")
        # Should be 1.0 (Total Rejection)
        self.assertEqual(refraction, 1.0)

    def test_prism_refraction_synthesis(self):
        """Test 3: Adult Mode (Refracting Prism) creates synthesis."""
        self.wisdom.maturity_level = 0.8 # Adult
        refraction = self.wisdom.calculate_refraction("Code", "Art")
        # Expect Synthesis range (around 0.5)
        # 0.8 * 0.6 = 0.48
        self.assertAlmostEqual(refraction, 0.48, places=2)

    def test_prism_alignment_flow(self):
        """Test 4: Alignment creates flow (No refraction needed)."""
        self.wisdom.maturity_level = 0.8
        refraction = self.wisdom.calculate_refraction("Art", "Art")
        self.assertEqual(refraction, 0.0)

    def test_renaissance_scaling(self):
        """Test 5: Renaissance State (High Love + High Truth) widens the scale."""
        from Core.Cognition.Topology.bridge import ThemeToIntentionMapper
        from Core.Orchestra.conductor import Conductor, Theme

        conductor = Conductor()
        # Set Renaissance Theme
        conductor.current_theme = Theme(
            name="Renaissance",
            description="Art + Math",
            tempo=0.5,
            love_weight=0.9,
            truth_weight=0.9, # Both High
            growth_weight=0.5,
            beauty_weight=0.5
        )

        mapper = ThemeToIntentionMapper(conductor)
        intention = mapper.map_theme_to_fluid_intention()

        # Scale should be significantly boosted (> 1.0)
        # Base 0.5 + Boost (0.9 * 1.0) = 1.4
        self.assertGreater(intention.scale, 1.0)

if __name__ == '__main__':
    unittest.main()
