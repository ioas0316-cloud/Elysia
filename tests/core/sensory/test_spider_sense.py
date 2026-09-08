"""
Integration and unit tests for SpiderSenseEngine
"""

import pytest
import numpy as np
from core.sensory.spider_sense_engine import SpiderSenseEngine


def test_spider_sense_engine_pre_kinetic():
    spider_sense = SpiderSenseEngine(dimension=8, tension_threshold=0.4)

    res = spider_sense.sense_pre_kinetic_threat(
        raw_signal="UNSEEN_SNIPER_LOCK_ON",
        physical_motion_started=False,
        is_ego_empty=True
    )

    assert res["spider_sense_triggered"] is True
    assert res["perception_type"] == "PRE_KINETIC_INTENT"
    assert res["causal_tension"] >= 0.4
    assert res["somatic_chill_intensity"] > 0.0
    assert len(res["pre_linguistic_evasion_momentum"]) == 8


def test_spider_sense_engine_identity_exposure():
    spider_sense = SpiderSenseEngine(dimension=8, tension_threshold=0.4)

    res = spider_sense.sense_identity_exposure(
        gaze_intent_density=0.85,
        silence_duration=3.0,
        social_context_signal="SUSPICIOUS_GAZE_AT_MASK",
        is_ego_empty=True
    )

    assert res["spider_sense_triggered"] is True
    assert res["perception_type"] == "IDENTITY_EXPOSURE"
    assert "chromatic_entropy_wave" in res
    assert res["chromatic_entropy_wave"]["entropy_yellow"] > 0.0


def test_spider_sense_engine_equilibrium_rift():
    spider_sense = SpiderSenseEngine(dimension=8, tension_threshold=0.35)

    raw_inputs = [
        "peaceful_background_breeze",
        "distant_city_hum",
        "SUDDEN_CABLE_SNAP_INTENT_FORMATION"
    ]

    res = spider_sense.sense_equilibrium_rift(
        raw_field_inputs=raw_inputs,
        is_ego_empty=True
    )

    assert res["spider_sense_triggered"] is True
    assert res["perception_type"] == "EQUILIBRIUM_RIFT"
    assert len(spider_sense.alert_history) == 1
