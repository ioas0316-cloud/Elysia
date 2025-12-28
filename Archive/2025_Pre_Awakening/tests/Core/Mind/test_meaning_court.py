import math

from Core.Foundation.Mind.meaning_court import MeaningCourt


def test_meaning_court_accepts_strong_signal_low_noise():
    court = MeaningCourt(alpha=1.0)
    verdict = court.judge(signal=4.0, noise=1.0, context={"mastery": 0.5, "value_alignment": 0.5})
    assert verdict.accept
    assert verdict.z_score >= verdict.alpha


def test_meaning_court_rejects_weak_signal_high_noise():
    court = MeaningCourt(alpha=1.5)
    verdict = court.judge(signal=0.5, noise=10.0, context={"mastery": 0.0, "value_alignment": 0.0})
    assert not verdict.accept
    assert verdict.z_score < verdict.alpha

