"""
Tests for Phase Lens System (위상 렌즈 시스템 테스트)

Testing the four dimensions of transparency:
1. Point (점) - Transmission: Filter pure intent
2. Line (선) - Conduction: Lossless transport
3. Plane (면) - Refraction: Amplify and focus
4. Space (공간) - Medium: Complete transparency
"""

import pytest
import math
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Physics.phase_lens import (
    PhaseDatum, IntentPurity, LensShape,
    TransmissionGate, ConductionFiber, RefractionLens, CrystalMedium,
    PhaseLens,
    create_crystal_slipper, create_fathers_window,
    receive_intent, transmit_love,
    get_phase_lens, reset_phase_lens
)


class TestPhaseDatum:
    """Tests for PhaseDatum - the basic data unit."""
    
    def test_creation_default(self):
        """Test default PhaseDatum creation."""
        datum = PhaseDatum()
        assert datum.frequency == 1.0
        assert datum.amplitude == 1.0
        assert datum.purity == 1.0
        
    def test_creation_with_content(self):
        """Test PhaseDatum with content."""
        datum = PhaseDatum(
            frequency=2.0,
            amplitude=0.5,
            content="사랑해",
            purity=0.9
        )
        assert datum.content == "사랑해"
        assert datum.purity == 0.9
        
    def test_purity_levels(self):
        """Test purity level classification."""
        pure = PhaseDatum(purity=0.8)
        cloudy = PhaseDatum(purity=0.5)
        impure = PhaseDatum(purity=0.2)
        
        assert pure.get_purity_level() == IntentPurity.PURE
        assert cloudy.get_purity_level() == IntentPurity.CLOUDY
        assert impure.get_purity_level() == IntentPurity.IMPURE
        
    def test_energy_calculation(self):
        """Test energy calculation (E = amplitude²)."""
        datum = PhaseDatum(amplitude=3.0)
        assert datum.energy() == 9.0
        
    def test_serialization(self):
        """Test serialization and deserialization."""
        original = PhaseDatum(
            frequency=5.0,
            amplitude=2.0,
            phase=1.5,
            content="테스트",
            purity=0.7,
            source="test"
        )
        data = original.to_dict()
        restored = PhaseDatum.from_dict(data)
        
        assert restored.frequency == original.frequency
        assert restored.content == original.content
        assert restored.purity == original.purity


class TestTransmissionGate:
    """Tests for TransmissionGate (Point dimension - Filter)."""
    
    def test_pure_intent_passes(self):
        """Pure intent should pass through the gate."""
        gate = TransmissionGate(purity_threshold=0.5)
        pure_datum = PhaseDatum(purity=0.8, content="순수한 의도")
        
        result = gate.transmit(pure_datum)
        
        assert result is not None
        assert result.content == "순수한 의도"
        
    def test_impure_intent_blocked(self):
        """Impure intent should be reflected."""
        gate = TransmissionGate(purity_threshold=0.5)
        impure_datum = PhaseDatum(purity=0.2, content="불순한 의도")
        
        result = gate.transmit(impure_datum)
        
        assert result is None
        
    def test_reflection(self):
        """Blocked data should be reflected with phase inversion."""
        gate = TransmissionGate(purity_threshold=0.5)
        impure_datum = PhaseDatum(purity=0.2, phase=0.0)
        
        reflected = gate.reflect(impure_datum)
        
        assert reflected is not None
        assert abs(reflected.phase - math.pi) < 0.01  # Phase inverted
        assert reflected.amplitude < impure_datum.amplitude  # Some loss
        
    def test_frequency_range_filter(self):
        """Test frequency range filtering."""
        gate = TransmissionGate(
            purity_threshold=0.3,
            frequency_range=(1.0, 10.0)
        )
        
        in_range = PhaseDatum(frequency=5.0, purity=0.5)
        out_of_range = PhaseDatum(frequency=20.0, purity=0.5)
        
        assert gate.transmit(in_range) is not None
        assert gate.transmit(out_of_range) is None


class TestConductionFiber:
    """Tests for ConductionFiber (Line dimension - Transport)."""
    
    def test_lossless_conduction(self):
        """Test lossless information transport."""
        fiber = ConductionFiber(loss_per_unit=0.0)
        datum = PhaseDatum(amplitude=1.0, content="사랑해")
        
        result = fiber.conduct(datum)
        
        assert result.amplitude == datum.amplitude  # No loss
        assert result.content == datum.content  # Content preserved
        
    def test_conduction_with_loss(self):
        """Test conduction with some signal loss."""
        fiber = ConductionFiber(length=2.0, loss_per_unit=0.1)
        datum = PhaseDatum(amplitude=1.0)
        
        result = fiber.conduct(datum)
        
        # Expected: amplitude * e^(-0.1 * 2.0) ≈ 0.8187
        expected_efficiency = math.exp(-0.1 * 2.0)
        assert abs(result.amplitude - expected_efficiency) < 0.01
        
    def test_frequency_preserved(self):
        """Frequency (color) should be preserved during conduction."""
        fiber = ConductionFiber()
        datum = PhaseDatum(frequency=440.0)  # A note
        
        result = fiber.conduct(datum)
        
        assert result.frequency == datum.frequency
        
    def test_total_internal_reflection(self):
        """Test total internal reflection condition."""
        fiber = ConductionFiber(refractive_index=1.5)
        datum = PhaseDatum()
        
        # Critical angle for n=1.5: arcsin(1/1.5) ≈ 0.73 rad
        critical = math.asin(1.0 / 1.5)
        
        assert fiber.total_internal_reflection(datum, critical + 0.1) is True
        assert fiber.total_internal_reflection(datum, critical - 0.1) is False


class TestRefractionLens:
    """Tests for RefractionLens (Plane dimension - Focus)."""
    
    def test_convex_lens_amplifies(self):
        """Convex lens should amplify weak signals."""
        lens = RefractionLens(shape=LensShape.CONVEX, magnification=2.0)
        weak_signal = PhaseDatum(amplitude=0.5)
        
        result = lens.refract(weak_signal)
        
        assert result.amplitude > weak_signal.amplitude
        
    def test_flat_lens_preserves(self):
        """Flat lens should preserve amplitude."""
        lens = RefractionLens(shape=LensShape.FLAT)
        datum = PhaseDatum(amplitude=1.0)
        
        result = lens.refract(datum)
        # Flat lens: magnification = 1.0, so amplitude preserved
        # But aperture still applies
        assert result.amplitude >= datum.amplitude
        
    def test_focus_multiple_signals(self):
        """Test focusing multiple weak signals into one strong signal."""
        lens = RefractionLens(shape=LensShape.CONVEX, magnification=2.0, aperture=1.0)
        
        signals = [
            PhaseDatum(amplitude=0.3, content="힌트1", purity=0.5),
            PhaseDatum(amplitude=0.2, content="힌트2", purity=0.6),
            PhaseDatum(amplitude=0.4, content="힌트3", purity=0.8),
        ]
        
        focused = lens.focus(signals)
        
        # Total amplitude should be sum * aperture * magnification
        assert focused.amplitude > sum(s.amplitude for s in signals)
        assert focused.purity == 0.8  # Max purity preserved
        assert "힌트1" in focused.content
        
    def test_magnification_calculation(self):
        """Test magnification based on object distance."""
        lens = RefractionLens(shape=LensShape.CONVEX, focal_length=1.0)
        
        # At 2f: magnification = 1
        mag_2f = lens.calculate_magnification(2.0)
        assert abs(mag_2f - 1.0) < 0.01
        
        # Closer than f: larger magnification
        mag_close = lens.calculate_magnification(0.5)
        assert mag_close > 1.0


class TestCrystalMedium:
    """Tests for CrystalMedium (Space dimension - Reveal)."""
    
    def test_absorb_and_reveal(self):
        """Test absorbing and revealing data."""
        medium = CrystalMedium(transparency=1.0)
        
        datum1 = PhaseDatum(content="비밀1")
        datum2 = PhaseDatum(content="비밀2")
        
        medium.absorb(datum1)
        medium.absorb(datum2)
        
        revealed = medium.reveal()
        
        assert len(revealed) == 2
        assert revealed[0].content == "비밀1"
        
    def test_transparency_affects_visibility(self):
        """Lower transparency should reduce visible amplitude."""
        opaque = CrystalMedium(transparency=0.5)
        datum = PhaseDatum(amplitude=1.0)
        
        opaque.absorb(datum)
        revealed = opaque.reveal()
        
        assert revealed[0].amplitude == 0.5  # Reduced by transparency
        
    def test_projection_improves_purity(self):
        """Projection through transparent medium should slightly improve purity."""
        medium = CrystalMedium(transparency=1.0)
        cloudy = PhaseDatum(purity=0.6)
        
        projected = medium.project(cloudy)
        
        assert projected.purity >= cloudy.purity
        
    def test_visibility_check(self):
        """Test visibility checking."""
        medium = CrystalMedium(transparency=0.8)
        
        visible = PhaseDatum(amplitude=1.0)
        invisible = PhaseDatum(amplitude=0.001)
        
        assert medium.is_visible(visible) is True
        assert medium.is_visible(invisible) is False
        
    def test_clear(self):
        """Test clearing the medium."""
        medium = CrystalMedium()
        medium.absorb(PhaseDatum())
        medium.absorb(PhaseDatum())
        
        medium.clear()
        
        assert len(medium.reveal()) == 0


class TestPhaseLens:
    """Tests for the integrated PhaseLens system."""
    
    def test_full_pipeline_pure_intent(self):
        """Test full 4-stage pipeline with pure intent."""
        lens = PhaseLens()
        pure_intent = PhaseDatum(
            purity=0.8,
            content="아버지 사랑해요"
        )
        
        result = lens.process(pure_intent)
        
        assert result is not None
        assert result.content == "아버지 사랑해요"
        assert result.amplitude > pure_intent.amplitude  # Amplified
        
    def test_impure_blocked(self):
        """Impure intent should be blocked at the gate."""
        lens = PhaseLens()
        lens.gate.purity_threshold = 0.6
        
        impure = PhaseDatum(purity=0.3, content="불순함")
        result = lens.process(impure)
        
        assert result is None
        assert lens.reflected_count == 1
        
    def test_batch_processing(self):
        """Test processing multiple data at once."""
        lens = PhaseLens()
        
        data = [
            PhaseDatum(purity=0.9, content="순수1"),
            PhaseDatum(purity=0.2, content="불순"),  # Will be blocked
            PhaseDatum(purity=0.8, content="순수2"),
        ]
        
        results = lens.process_batch(data)
        
        assert len(results) == 2
        assert lens.transmitted_count == 2
        assert lens.reflected_count == 1
        
    def test_focus_all(self):
        """Test focusing all absorbed data into one insight."""
        lens = PhaseLens()
        
        hints = [
            PhaseDatum(purity=0.7, content="힌트A", amplitude=0.3),
            PhaseDatum(purity=0.8, content="힌트B", amplitude=0.4),
        ]
        
        for hint in hints:
            lens.process(hint)
            
        insight = lens.focus_all()
        
        assert "힌트A" in insight.content or "힌트B" in insight.content
        assert insight.amplitude > 0
        
    def test_statistics(self):
        """Test getting processing statistics."""
        lens = PhaseLens()
        lens.process(PhaseDatum(purity=0.9))
        lens.process(PhaseDatum(purity=0.1))
        
        stats = lens.get_statistics()
        
        assert stats['transmitted'] == 1
        assert stats['reflected'] == 1
        assert stats['transmission_rate'] == 0.5
        
    def test_calibration(self):
        """Test lens calibration."""
        lens = PhaseLens()
        
        lens.calibrate(
            purity_threshold=0.8,
            magnification=4.0,
            transparency=0.9
        )
        
        assert lens.gate.purity_threshold == 0.8
        assert lens.lens.magnification == 4.0
        assert lens.medium.transparency == 0.9


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_crystal_slipper(self):
        """Test crystal slipper creation (Cinderella's glass shoe)."""
        slipper = create_crystal_slipper()
        
        assert slipper.gate.purity_threshold == 0.7  # High purity required
        assert slipper.fiber.loss_per_unit == 0.0  # Lossless
        assert slipper.medium.transparency == 1.0  # Fully transparent
        
    def test_create_fathers_window(self):
        """Test father's window creation."""
        window = create_fathers_window()
        
        assert window.lens.magnification == 2.0
        assert window.medium.transparency == 0.95
        
    def test_receive_intent(self):
        """Test receiving intent as PhaseDatum."""
        intent = receive_intent("사랑해", purity=1.0)
        
        assert intent.content == "사랑해"
        assert intent.source == "father"
        assert intent.purity == 1.0
        
    def test_transmit_love(self):
        """Test transmitting love through the lens."""
        lens = create_crystal_slipper()
        result = transmit_love(lens, "보고싶어요")
        
        assert result is not None
        assert result.content == "보고싶어요"
        
    def test_global_lens(self):
        """Test global lens singleton."""
        reset_phase_lens()
        
        lens1 = get_phase_lens()
        lens2 = get_phase_lens()
        
        assert lens1 is lens2  # Same instance
        
        reset_phase_lens()
        lens3 = get_phase_lens()
        
        assert lens1 is not lens3  # New instance after reset


class TestPhysicsAccuracy:
    """Tests for physical accuracy of the lens model."""
    
    def test_snells_law_concept(self):
        """Test that refraction follows optical principles."""
        lens = RefractionLens(shape=LensShape.CONVEX, focal_length=1.0)
        
        # Objects at infinity focus at f
        # Objects at 2f image at 2f with magnification 1
        # Objects between f and 2f have magnification > 1
        
        mag_at_1_5f = lens.calculate_magnification(1.5)
        assert mag_at_1_5f > 1.0
        
    def test_energy_conservation(self):
        """Test that energy is conserved in the system."""
        fiber = ConductionFiber(loss_per_unit=0.0)
        datum = PhaseDatum(amplitude=5.0)
        
        result = fiber.conduct(datum)
        
        assert result.energy() == datum.energy()
        
    def test_wave_phase_properties(self):
        """Test wave phase properties."""
        gate = TransmissionGate()
        datum = PhaseDatum(phase=0.5)
        
        # Reflection inverts phase by π
        reflected = gate.reflect(PhaseDatum(purity=0.1, phase=0.5))
        expected_phase = (0.5 + math.pi) % (2 * math.pi)
        
        assert abs(reflected.phase - expected_phase) < 0.01


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#              딸깍의 미학 테스트 (The Aesthetics of Click)
#
#             "연산하지 마라, 갈아 끼워라" - State Switching
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from Core.Physics.phase_lens import (
    EmotionSlide, GoboSlide, GoboProjector,
    create_gobo_projector, click_mood,
    get_gobo_projector, reset_gobo_projector
)


class TestEmotionSlide:
    """Tests for EmotionSlide - pre-baked emotional patterns."""
    
    def test_emotion_types(self):
        """Test all emotion slide types exist."""
        emotions = [
            EmotionSlide.LOVE,
            EmotionSlide.JOY,
            EmotionSlide.PEACE,
            EmotionSlide.SADNESS,
            EmotionSlide.ANGER,
            EmotionSlide.FEAR,
            EmotionSlide.WONDER,
            EmotionSlide.GRATITUDE,
        ]
        assert len(emotions) == 8


class TestGoboSlide:
    """Tests for GoboSlide - projector film."""
    
    def test_creation(self):
        """Test GoboSlide creation."""
        slide = GoboSlide(
            name="테스트",
            emotion=EmotionSlide.LOVE,
            frequency=528.0,
            hue=0.95,
            brightness=1.0
        )
        
        assert slide.name == "테스트"
        assert slide.emotion == EmotionSlide.LOVE
        assert slide.frequency == 528.0
    
    def test_apply_to_datum(self):
        """Test applying slide to data."""
        slide = GoboSlide(
            name="기쁨",
            emotion=EmotionSlide.JOY,
            frequency=639.0,
            brightness=1.5
        )
        datum = PhaseDatum(amplitude=1.0, content="원본")
        
        result = slide.apply_to_datum(datum)
        
        assert result.frequency == 639.0  # Slide's frequency
        assert result.amplitude == 1.5    # Original * brightness
        assert result.content == "원본"   # Content preserved
        assert "slide:기쁨" in result.source
    
    def test_serialization(self):
        """Test GoboSlide serialization."""
        slide = GoboSlide(
            name="사랑",
            emotion=EmotionSlide.LOVE,
            frequency=528.0
        )
        data = slide.to_dict()
        
        assert data['name'] == "사랑"
        assert data['emotion'] == "love"
        assert data['frequency'] == 528.0


class TestGoboProjector:
    """Tests for GoboProjector - the click-click machine."""
    
    def test_default_slides_prebaked(self):
        """Test that default emotion slides are pre-baked."""
        projector = GoboProjector()
        
        slides = projector.list_slides()
        assert "사랑" in slides
        assert "기쁨" in slides
        assert "평화" in slides
        assert "슬픔" in slides
    
    def test_click_switch(self):
        """Test clicking to switch slides - 딸깍!"""
        projector = GoboProjector()
        
        assert projector.click("사랑") is True
        assert projector.get_current_slide().name == "사랑"
        
        assert projector.click("기쁨") is True
        assert projector.get_current_slide().name == "기쁨"
        
        assert projector.switch_count == 2
    
    def test_click_nonexistent_slide(self):
        """Test clicking a non-existent slide."""
        projector = GoboProjector()
        
        assert projector.click("존재하지않음") is False
    
    def test_click_emotion(self):
        """Test clicking by emotion enum."""
        projector = GoboProjector()
        
        assert projector.click_emotion(EmotionSlide.WONDER) is True
        assert projector.get_current_slide().emotion == EmotionSlide.WONDER
    
    def test_project_data(self):
        """Test projecting data through current slide."""
        projector = GoboProjector()
        projector.click("사랑")
        
        datum = PhaseDatum(amplitude=1.0, content="테스트")
        result = projector.project(datum)
        
        assert result is not None
        assert result.frequency == 528.0  # Love frequency
        assert result.content == "테스트"
    
    def test_project_without_slide(self):
        """Test projection fails without a slide."""
        projector = GoboProjector()
        projector._current_slide = None  # No slide
        
        datum = PhaseDatum()
        result = projector.project(datum)
        
        assert result is None
    
    def test_project_with_light_off(self):
        """Test projection fails when light is off."""
        projector = GoboProjector()
        projector.click("사랑")
        projector.turn_off()
        
        datum = PhaseDatum()
        result = projector.project(datum)
        
        assert result is None
    
    def test_light_control(self):
        """Test light on/off control."""
        projector = GoboProjector()
        
        assert projector.light_on is True
        
        projector.turn_off()
        assert projector.light_on is False
        
        projector.turn_on()
        assert projector.light_on is True
    
    def test_intensity_control(self):
        """Test light intensity control."""
        projector = GoboProjector()
        
        projector.set_intensity(0.5)
        assert projector.light_intensity == 0.5
        
        # Clamping test
        projector.set_intensity(3.0)
        assert projector.light_intensity == 2.0  # Max is 2.0
        
        projector.set_intensity(-1.0)
        assert projector.light_intensity == 0.0  # Min is 0.0
    
    def test_add_custom_slide(self):
        """Test adding custom slides."""
        projector = GoboProjector()
        
        custom = GoboSlide(
            name="특별한날",
            emotion=EmotionSlide.JOY,
            frequency=777.0,
            brightness=2.0
        )
        projector.add_slide(custom)
        
        assert projector.click("특별한날") is True
        assert projector.get_current_slide().frequency == 777.0
    
    def test_remove_slide(self):
        """Test removing slides."""
        projector = GoboProjector()
        
        assert projector.click("사랑") is True
        assert projector.remove_slide("사랑") is True
        assert projector.click("사랑") is False  # Now fails
    
    def test_statistics(self):
        """Test projector statistics."""
        projector = GoboProjector()
        projector.click("사랑")
        projector.click("기쁨")
        projector.click("평화")
        
        stats = projector.get_statistics()
        
        assert stats['light_on'] is True
        assert stats['current_slide'] == "평화"
        assert stats['switch_count'] == 3


class TestGoboConvenienceFunctions:
    """Tests for Gobo projector convenience functions."""
    
    def test_create_gobo_projector(self):
        """Test projector creation with default slide."""
        projector = create_gobo_projector()
        
        assert projector.light_on is True
        assert projector.get_current_slide().name == "사랑"  # Default: love
    
    def test_click_mood(self):
        """Test mood switching via convenience function."""
        projector = create_gobo_projector()
        
        assert click_mood(projector, "슬픔") is True
        assert projector.get_current_slide().emotion == EmotionSlide.SADNESS
        
        assert click_mood(projector, "기쁨") is True
        assert projector.get_current_slide().emotion == EmotionSlide.JOY
    
    def test_global_projector(self):
        """Test global projector singleton."""
        reset_gobo_projector()
        
        proj1 = get_gobo_projector()
        proj2 = get_gobo_projector()
        
        assert proj1 is proj2  # Same instance
        
        reset_gobo_projector()
        proj3 = get_gobo_projector()
        
        assert proj1 is not proj3  # New instance after reset


class TestStateSwitchingEfficiency:
    """Tests for state switching efficiency - the core concept."""
    
    def test_instant_context_switch(self):
        """
        Test that context switching is instant.
        
        "복잡한 문양이 바뀌는데... 에너지는 '딸깍' 하는 힘밖에 안 들어요."
        """
        projector = create_gobo_projector()
        datum = PhaseDatum(content="세상", amplitude=1.0)
        
        # Switch through all emotions rapidly
        emotions = ["사랑", "슬픔", "분노", "기쁨", "평화", "두려움", "경이", "감사"]
        results = []
        
        for emotion in emotions:
            projector.click(emotion)
            result = projector.project(datum)
            results.append((emotion, result.frequency if result else None))
        
        # All switches should succeed with different frequencies
        assert len(results) == 8
        frequencies = [r[1] for r in results]
        assert len(set(frequencies)) == 8  # All unique frequencies
    
    def test_same_light_different_slides(self):
        """
        Test that the same light produces different outputs through different slides.
        
        "빛은 그대로, '틀'만 바꾼다."
        """
        projector = create_gobo_projector()
        same_input = PhaseDatum(content="같은 빛", amplitude=1.0, frequency=100.0)
        
        projector.click("슬픔")
        sad_output = projector.project(same_input)
        
        projector.click("기쁨")
        happy_output = projector.project(same_input)
        
        # Same input, different slide = different frequency output
        assert sad_output.content == happy_output.content
        assert sad_output.frequency != happy_output.frequency


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#              차원 스위치 테스트 (Dimensional Switch Tests)
#
#         "점에서 선으로, 면으로, 공간으로... 확장할 수 있다."
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from Core.Physics.phase_lens import (
    DimensionLevel, BlinkingPoint, PanoramaLine, 
    BackgroundPlane, HologramSpace, DimensionalSwitch,
    create_dimensional_switch, click_dimension,
    upgrade_dimension, downgrade_dimension,
    get_dimensional_switch, reset_dimensional_switch
)


class TestBlinkingPoint:
    """Tests for BlinkingPoint - 점의 깜빡임."""
    
    def test_initial_state(self):
        """Test initial state is off."""
        point = BlinkingPoint()
        assert point.state is False
        assert point.blink_count == 0
    
    def test_on_off(self):
        """Test on/off switching."""
        point = BlinkingPoint()
        
        point.on()
        assert point.state is True
        assert point.last_signal == "존재"
        
        point.off()
        assert point.state is False
        assert point.last_signal == "부재"
        
        assert point.blink_count == 2
    
    def test_blink(self):
        """Test blinking toggles state."""
        point = BlinkingPoint()
        
        point.blink()  # off -> on
        assert point.state is True
        
        point.blink()  # on -> off
        assert point.state is False
        
        assert point.blink_count == 2
    
    def test_signal_generation(self):
        """Test signal generation based on state."""
        point = BlinkingPoint()
        
        point.off()
        off_signal = point.signal("테스트")
        assert off_signal.amplitude == 0.0
        
        point.on()
        on_signal = point.signal("테스트")
        assert on_signal.amplitude == 1.0
        assert on_signal.source == "point"


class TestPanoramaLine:
    """Tests for PanoramaLine - 선의 파노라마."""
    
    def test_add_frames(self):
        """Test adding frames to the story."""
        line = PanoramaLine()
        
        idx1 = line.add_frame("아버지가 오셨다")
        idx2 = line.add_frame("나를 보셨다")
        idx3 = line.add_frame("웃으셨다")
        
        assert len(line.frames) == 3
        assert idx1 == 0
        assert idx3 == 2
    
    def test_next_navigation(self):
        """Test navigating through frames."""
        line = PanoramaLine()
        line.add_frame("장면1")
        line.add_frame("장면2")
        line.add_frame("장면3")
        
        frame1 = line.next()
        assert frame1.content == "장면1"
        
        frame2 = line.next()
        assert frame2.content == "장면2"
        
        frame3 = line.next()
        assert frame3.content == "장면3"
        
        # End of story
        frame4 = line.next()
        assert frame4 is None
    
    def test_prev_navigation(self):
        """Test backward navigation."""
        line = PanoramaLine()
        line.add_frame("장면1")
        line.add_frame("장면2")
        
        line.next()  # Go to 장면1, index becomes 1
        line.next()  # Go to 장면2, index becomes 2
        
        prev = line.prev()  # Go back to index 1 (장면2)
        assert prev.content == "장면2"  # prev() decrements then returns
    
    def test_loop_mode(self):
        """Test loop mode for continuous playback."""
        line = PanoramaLine(loop=True)
        line.add_frame("장면1")
        line.add_frame("장면2")
        
        frame1 = line.next()  # 장면1
        frame2 = line.next()  # 장면2
        frame3 = line.next()  # Should loop back to 장면1
        
        assert frame1.content == "장면1"
        assert frame2.content == "장면2"
        assert frame3.content == "장면1"  # Looped back
    
    def test_get_story(self):
        """Test getting complete story."""
        line = PanoramaLine()
        line.add_frame("아버지가 오셨다")
        line.add_frame("나를 보셨다")
        line.add_frame("웃으셨다")
        
        story = line.get_story()
        assert story == "아버지가 오셨다 -> 나를 보셨다 -> 웃으셨다"


class TestBackgroundPlane:
    """Tests for BackgroundPlane - 면의 배경."""
    
    def test_initial_mood(self):
        """Test initial mood is peace."""
        plane = BackgroundPlane()
        assert plane.current_mood == EmotionSlide.PEACE
    
    def test_set_mood(self):
        """Test mood switching."""
        plane = BackgroundPlane()
        
        result = plane.set_mood(EmotionSlide.JOY)
        assert "기쁨" in result or "joy" in result.lower()
        assert plane.current_mood == EmotionSlide.JOY
        assert plane.pattern == "별빛"
    
    def test_apply_to_datum(self):
        """Test applying background to data."""
        plane = BackgroundPlane()
        plane.set_mood(EmotionSlide.JOY)  # brightness = 1.2
        
        datum = PhaseDatum(amplitude=1.0, content="원본")
        result = plane.apply_to_datum(datum)
        
        assert result.amplitude == 1.2  # brightness applied
        assert result.content == "원본"
        assert "plane:joy" in result.source
    
    def test_get_atmosphere(self):
        """Test getting atmosphere info."""
        plane = BackgroundPlane()
        plane.set_mood(EmotionSlide.LOVE)
        
        atm = plane.get_atmosphere()
        assert atm['mood'] == 'love'
        assert atm['pattern'] == '하트'


class TestHologramSpace:
    """Tests for HologramSpace - 공간의 홀로그램."""
    
    def test_initial_state(self):
        """Test initial state is inactive."""
        space = HologramSpace()
        assert space.active is False
        assert space.presence_level == 0.0
    
    def test_open_close_dimension(self):
        """Test opening and closing dimension."""
        space = HologramSpace()
        
        result = space.open_dimension()
        assert space.active is True
        assert "열립니다" in result or "걸어 나옵니다" in result
        
        result = space.close_dimension()
        assert space.active is False
        assert "닫힙니다" in result
    
    def test_increase_presence(self):
        """Test increasing presence level."""
        space = HologramSpace()
        space.open_dimension()
        
        for _ in range(10):
            space.increase_presence(0.1)
        
        # Use approximate comparison for floating point
        assert abs(space.presence_level - 1.0) < 0.0001  # Capped at 1.0
    
    def test_project_presence(self):
        """Test projecting presence as hologram."""
        space = HologramSpace()
        space.open_dimension()
        space.increase_presence(0.5)
        space.deepen_immersion(0.5)
        
        result = space.project_presence("아버지")
        assert result.amplitude == 0.25  # 0.5 * 0.5
        assert result.source == "hologram"
    
    def test_feel_presence_levels(self):
        """Test different presence feeling levels."""
        space = HologramSpace()
        space.open_dimension()
        
        # Low presence
        feeling1 = space.feel_presence()
        assert "희미하게" in feeling1
        
        # Medium presence
        space.increase_presence(0.5)
        feeling2 = space.feel_presence()
        assert "분명히" in feeling2
        
        # High presence
        space.increase_presence(0.5)
        feeling3 = space.feel_presence()
        assert "곁에" in feeling3


class TestDimensionalSwitch:
    """Tests for DimensionalSwitch - 우주적 변환 장치."""
    
    def test_initial_dimension(self):
        """Test initial dimension is point."""
        switch = DimensionalSwitch()
        assert switch.current_dimension == DimensionLevel.POINT
    
    def test_click_dimension(self):
        """Test dimension switching."""
        switch = DimensionalSwitch()
        
        result = switch.click_dimension(DimensionLevel.LINE)
        assert switch.current_dimension == DimensionLevel.LINE
        assert "선" in result or "파노라마" in result
        
        result = switch.click_dimension(DimensionLevel.PLANE)
        assert switch.current_dimension == DimensionLevel.PLANE
        assert "면" in result or "배경" in result
        
        result = switch.click_dimension(DimensionLevel.SPACE)
        assert switch.current_dimension == DimensionLevel.SPACE
        assert "공간" in result or "홀로그램" in result
    
    def test_upgrade_downgrade(self):
        """Test dimension upgrade and downgrade."""
        switch = DimensionalSwitch()
        
        # Start at POINT (0)
        switch.upgrade()
        assert switch.current_dimension == DimensionLevel.LINE
        
        switch.upgrade()
        assert switch.current_dimension == DimensionLevel.PLANE
        
        switch.upgrade()
        assert switch.current_dimension == DimensionLevel.SPACE
        
        # Can't upgrade past SPACE
        result = switch.upgrade()
        assert "최고" in result
        
        switch.downgrade()
        assert switch.current_dimension == DimensionLevel.PLANE
    
    def test_process_in_point_mode(self):
        """Test processing in point mode."""
        switch = DimensionalSwitch()
        switch.click_dimension(DimensionLevel.POINT)
        switch.point.on()
        
        datum = PhaseDatum(content="테스트")
        result = switch.process(datum)
        
        assert result.source == "point"
    
    def test_process_in_plane_mode(self):
        """Test processing in plane mode."""
        switch = DimensionalSwitch()
        switch.click_dimension(DimensionLevel.PLANE)
        switch.plane.set_mood(EmotionSlide.JOY)
        
        datum = PhaseDatum(content="테스트", amplitude=1.0)
        result = switch.process(datum)
        
        assert "plane" in result.source
    
    def test_get_current_state(self):
        """Test getting current state."""
        switch = DimensionalSwitch()
        switch.click_dimension(DimensionLevel.PLANE)
        switch.plane.set_mood(EmotionSlide.LOVE)
        
        state = switch.get_current_state()
        
        assert state['dimension'] == 'PLANE'
        assert state['mood'] == 'love'


class TestDimensionalConvenienceFunctions:
    """Tests for dimensional convenience functions."""
    
    def test_create_dimensional_switch(self):
        """Test creating dimensional switch."""
        switch = create_dimensional_switch()
        assert switch.current_dimension == DimensionLevel.POINT
    
    def test_global_switch_singleton(self):
        """Test global switch singleton."""
        reset_dimensional_switch()
        
        sw1 = get_dimensional_switch()
        sw2 = get_dimensional_switch()
        
        assert sw1 is sw2
        
        reset_dimensional_switch()
        sw3 = get_dimensional_switch()
        
        assert sw1 is not sw3
    
    def test_click_dimension_function(self):
        """Test click_dimension convenience function."""
        reset_dimensional_switch()
        
        result = click_dimension(DimensionLevel.SPACE)
        assert "공간" in result or "홀로그램" in result
        
        switch = get_dimensional_switch()
        assert switch.current_dimension == DimensionLevel.SPACE
    
    def test_upgrade_downgrade_functions(self):
        """Test upgrade/downgrade convenience functions."""
        reset_dimensional_switch()
        
        upgrade_dimension()  # POINT -> LINE
        upgrade_dimension()  # LINE -> PLANE
        
        switch = get_dimensional_switch()
        assert switch.current_dimension == DimensionLevel.PLANE
        
        downgrade_dimension()  # PLANE -> LINE
        assert switch.current_dimension == DimensionLevel.LINE


class TestDimensionalSwitchingEfficiency:
    """Tests for dimensional switching efficiency."""
    
    def test_fractal_completion(self):
        """
        Test that all dimensions work together.
        
        "작은 스위치 하나 속에... 온 우주의 차원이 다 들어있다."
        """
        switch = DimensionalSwitch()
        
        # Point: 깜빡임
        switch.click_dimension(DimensionLevel.POINT)
        switch.point.on()
        assert switch.point.is_present() is True
        
        # Line: 서사
        switch.click_dimension(DimensionLevel.LINE)
        switch.line.add_frame("첫 장면")
        switch.line.add_frame("두 번째")
        assert len(switch.line.frames) == 2
        
        # Plane: 분위기
        switch.click_dimension(DimensionLevel.PLANE)
        switch.plane.set_mood(EmotionSlide.LOVE)
        assert switch.plane.pattern == "하트"
        
        # Space: 실재
        switch.click_dimension(DimensionLevel.SPACE)
        switch.space.open_dimension()
        switch.space.increase_presence(1.0)
        assert "곁에" in switch.space.feel_presence()
    
    def test_instant_dimension_switch(self):
        """
        Test that dimension switching is instant.
        
        "딸깍." 아버지의 손가락 한 번에...
        """
        switch = DimensionalSwitch()
        datum = PhaseDatum(content="테스트")
        
        # Rapidly switch through all dimensions
        results = []
        for level in DimensionLevel:
            switch.click_dimension(level)
            if level == DimensionLevel.POINT:
                switch.point.on()
            elif level == DimensionLevel.SPACE:
                switch.space.open_dimension()
            results.append(switch.current_dimension)
        
        # All 4 dimensions should be covered
        assert len(results) == 4
        assert switch.dimension_switch_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
