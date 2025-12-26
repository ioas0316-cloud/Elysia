"""
Test Orchestra/Symphony Architecture
====================================

Tests the orchestral paradigm for system coordination.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Orchestra.conductor import (
    Conductor,
    Instrument,
    HarmonyCoordinator,
    Tempo,
    Mode,
    MusicalIntent
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestOrchestra")


def test_conductor_solo():
    """Test conductor with solo instrument."""
    print("\n" + "="*70)
    print("TEST 1: Conductor Solo Performance")
    print("="*70)
    
    conductor = Conductor()
    
    # Create a simple instrument
    def test_module(_tempo=None, _mode=None, _dynamics=None, value=0):
        return {"result": value * 2, "tempo": _tempo.name if _tempo else None}
    
    instrument = Instrument("TestModule", "Strings", test_module)
    conductor.register_instrument(instrument)
    
    # Conduct solo
    result = conductor.conduct_solo("TestModule", value=5)
    
    print(f"\n📊 Result: {result}")
    
    assert result["result"] == 10
    assert result["tempo"] == "MODERATO"  # Default tempo
    
    print("\n✅ Test 1 passed: Solo performance works")
    return True


def test_conductor_ensemble():
    """Test ensemble (multiple instruments playing together)."""
    print("\n" + "="*70)
    print("TEST 2: Ensemble Harmony")
    print("="*70)
    
    conductor = Conductor()
    
    # Create multiple instruments
    def module1(_tempo=None, _mode=None, _dynamics=None):
        return f"Module1: {_mode.value if _mode else 'neutral'}"
    
    def module2(_tempo=None, _mode=None, _dynamics=None):
        return f"Module2: intensity={_dynamics:.2f}"
    
    def module3(_tempo=None, _mode=None, _dynamics=None):
        return f"Module3: speed={_tempo.value if _tempo else 100}"
    
    conductor.register_instrument(Instrument("Module1", "Strings", module1))
    conductor.register_instrument(Instrument("Module2", "Woodwinds", module2))
    conductor.register_instrument(Instrument("Module3", "Brass", module3))
    
    # Set intent
    conductor.set_intent(tempo=Tempo.ALLEGRO, mode=Mode.MAJOR, dynamics=0.9)
    
    # Conduct ensemble (no collision!)
    results = conductor.conduct_ensemble(["Module1", "Module2", "Module3"])
    
    print(f"\n📊 Ensemble results:")
    for name, result in results.items():
        print(f"   {name}: {result}")
    
    assert len(results) == 3
    assert "Module1" in results
    assert "Module2" in results
    assert "Module3" in results
    
    print("\n✅ Test 2 passed: Ensemble creates harmony")
    return True


def test_intent_changes():
    """Test changing musical intent."""
    print("\n" + "="*70)
    print("TEST 3: Intent Changes")
    print("="*70)
    
    conductor = Conductor()
    
    def responsive_module(_tempo=None, _mode=None, _dynamics=None):
        return {
            "tempo": _tempo.name if _tempo else "unknown",
            "mode": _mode.value if _mode else "unknown",
            "dynamics": _dynamics
        }
    
    conductor.register_instrument(Instrument("Responsive", "All", responsive_module))
    
    # Test different intents
    intents = [
        (Tempo.LARGO, Mode.MINOR, 0.3, "Slow and sad"),
        (Tempo.PRESTO, Mode.MAJOR, 0.9, "Fast and happy"),
        (Tempo.ANDANTE, Mode.DORIAN, 0.5, "Mysterious walk"),
    ]
    
    for tempo, mode, dynamics, description in intents:
        print(f"\n🎼 {description}")
        conductor.set_intent(tempo=tempo, mode=mode, dynamics=dynamics)
        result = conductor.conduct_solo("Responsive")
        
        print(f"   Result: {result}")
        
        assert result["tempo"] == tempo.name
        assert result["mode"] == mode.value
        assert result["dynamics"] == dynamics
    
    print("\n✅ Test 3 passed: Intent changes work")
    return True


def test_improvisation():
    """Test error handling through improvisation."""
    print("\n" + "="*70)
    print("TEST 4: Improvisation (Error Handling)")
    print("="*70)
    
    conductor = Conductor()
    
    # Create a module that throws errors
    def unstable_module(_tempo=None, _mode=None, _dynamics=None, should_fail=False):
        if should_fail:
            raise ValueError("Intentional error for testing")
        return "Success"
    
    conductor.register_instrument(Instrument("Unstable", "Percussion", unstable_module))
    
    # Normal operation
    print("\n🎵 Normal operation:")
    result1 = conductor.conduct_solo("Unstable", should_fail=False)
    print(f"   Result: {result1}")
    assert result1 == "Success"
    
    # Error case - should improvise, not crash!
    print("\n🎶 Error case (should improvise):")
    result2 = conductor.conduct_solo("Unstable", should_fail=True)
    print(f"   Result: {result2}")
    
    assert result2["status"] == "improvised"
    assert result2["instrument"] == "Unstable"
    assert "error" in result2["original_error"].lower()
    
    print("\n✅ Test 4 passed: Improvisation works (no crashes!)")
    return True


def test_harmony_coordinator():
    """Test harmony coordination."""
    print("\n" + "="*70)
    print("TEST 5: Harmony Coordination")
    print("="*70)
    
    harmony = HarmonyCoordinator()
    
    # Test numeric harmony
    print("\n🎵 Numeric harmony:")
    harmony.add_voice("temperature", 20.0)
    harmony.add_voice("temperature", 22.0)
    harmony.add_voice("temperature", 21.0)
    
    result = harmony.resolve_harmony("temperature")
    print(f"   3 voices (20, 22, 21) → {result}")
    assert result == 21.0  # Average
    
    # Test dict harmony (merge)
    harmony.clear_voices("state")
    print("\n🎵 Dict harmony (merging):")
    harmony.add_voice("state", {"mood": "happy"})
    harmony.add_voice("state", {"energy": "high"})
    harmony.add_voice("state", {"focus": "work"})
    
    result = harmony.resolve_harmony("state")
    print(f"   Result: {result}")
    assert "mood" in result
    assert "energy" in result
    assert "focus" in result
    
    print("\n✅ Test 5 passed: Harmony coordination works")
    return True


def test_tuning():
    """Test instrument tuning (not debugging!)."""
    print("\n" + "="*70)
    print("TEST 6: Tuning (not Debugging)")
    print("="*70)
    
    conductor = Conductor()
    
    def tunable_module(_tempo=None, _mode=None, _dynamics=None):
        return "Playing"
    
    instrument = Instrument("Tunable", "Strings", tunable_module)
    conductor.register_instrument(instrument)
    
    # Tune the instrument
    print("\n🎻 Tuning instrument...")
    conductor.tune_instrument("Tunable", "pitch", 440.0)
    conductor.tune_instrument("Tunable", "volume", 0.8)
    
    # Check tuning
    tuned = conductor.instruments["Tunable"]
    print(f"   Tuning: {tuned.tuning}")
    
    assert tuned.tuning["pitch"] == 440.0
    assert tuned.tuning["volume"] == 0.8
    
    print("\n✅ Test 6 passed: Tuning works")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("ELYSIA SYMPHONY ARCHITECTURE - TEST SUITE")
    print("="*70)
    print("\n지휘자(Conductor)가 있는 한, 악기들은 서로 부딪히지 않습니다")
    print("The Conductor ensures instruments never collide")
    print()
    
    tests = [
        ("Conductor Solo", test_conductor_solo),
        ("Ensemble Harmony", test_conductor_ensemble),
        ("Intent Changes", test_intent_changes),
        ("Improvisation", test_improvisation),
        ("Harmony Coordination", test_harmony_coordinator),
        ("Tuning", test_tuning)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ Symphony Architecture is operational!")
        print("🎼 오류가 아닌 조율로 완벽한 하모니!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
