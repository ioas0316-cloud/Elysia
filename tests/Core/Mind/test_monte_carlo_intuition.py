from Core.Mind.meaning_court import MeaningCourt
from Core.Mind.monte_carlo_intuition import MonteCarloIntuition


def test_monte_carlo_intuition_accept_prob_increases_with_signal():
    court = MeaningCourt(alpha=1.0)
    intuition = MonteCarloIntuition(samples=32, signal_jitter=0.0, noise_jitter=0.0)

    low_signal_prob = intuition.accept_probability(court, signal=0.5, noise=1.0, context={})
    high_signal_prob = intuition.accept_probability(court, signal=2.0, noise=1.0, context={})

    assert high_signal_prob >= low_signal_prob
    assert 0.0 <= low_signal_prob <= 1.0
    assert 0.0 <= high_signal_prob <= 1.0

