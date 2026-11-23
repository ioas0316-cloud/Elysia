import math
from typing import Any, Dict, Optional

import numpy as np

from core.orchestration.sensory_orchestra_os import (
    ElysiaOrchestraOS,
    HarmonicAcousticInstrument,
    InstrumentPerformance,
    OpticalPrismLightInstrument,
    SensoryInstrument,
    SpacetimeGlobeState,
    TactileResistanceInstrument,
    cauchy_refractive_index,
    planck_spectrum,
)
from core.physics.causal_field import CausalField
from core.physics.causal_mechanism_palette import CausalMechanismPalette
from core.physics.spectral_scale_rotor_manifold import SpectralBalanceScale


def test_spacetime_globe_non_sequential_scrub_and_exact_rewind() -> None:
    os_engine = ElysiaOrchestraOS()

    # Jump arbitrarily forward, backward, and back to origin without 1->t stepping
    s0 = os_engine.conduct_slice(0.0, manifest=False)
    s_fwd = os_engine.conduct_slice(100.7, manifest=False)
    s_back = os_engine.conduct_slice(-42.3, manifest=False)
    assert s_fwd.globe_state.time == 100.7
    assert s_back.globe_state.time == -42.3
    assert not np.allclose(s0.globe_state.macro_pos, s_fwd.globe_state.macro_pos)

    err_check = os_engine.scrub_reversibility_check(0.0, 100.7)
    assert err_check["globe_position_error"] < 1e-12
    assert err_check["manifold_field_error"] < 1e-12

    err_neg = os_engine.scrub_reversibility_check(-42.3, 250.0)
    assert err_neg["globe_position_error"] < 1e-12
    assert err_neg["manifold_field_error"] < 1e-12


def test_optical_prism_physical_laws_planck_and_cauchy() -> None:
    # 1. Planck blackbody shifts blue/red energy ratio monotonically from 2800K to 6500K
    spec_2800 = planck_spectrum(2800.0)
    spec_6500 = planck_spectrum(6500.0)
    assert abs(float(np.sum(spec_2800)) - 1.0) < 1e-12
    assert abs(float(np.sum(spec_6500)) - 1.0) < 1e-12
    assert (spec_6500[0] / spec_6500[-1]) > (spec_2800[0] / spec_2800[-1]) * 5.0

    # 2. Cauchy dispersion law: n(430nm) > n(550nm) > n(670nm)
    n_vals = cauchy_refractive_index(dispersion_strength=0.9)
    assert n_vals[0] > n_vals[3] > n_vals[-1]

    # 3. Rendered Dawn (t=0.0), Noon (t=2.5), and Twilight (t=7.5) images are distinct & chromatic
    os_engine = ElysiaOrchestraOS()
    dawn = os_engine.conduct_slice(0.0, manifest=True, resolution=64)
    noon = os_engine.conduct_slice(2.5, manifest=True, resolution=64)
    twilight = os_engine.conduct_slice(7.5, manifest=True, resolution=64)

    img_dawn = dawn.performances["vision.optical_prism"].manifestation
    img_noon = noon.performances["vision.optical_prism"].manifestation
    img_twilight = twilight.performances["vision.optical_prism"].manifestation

    for img in (img_dawn, img_noon, img_twilight):
        assert img is not None
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert float(np.std(img)) > 5.0
        # Non-grayscale chromatic prism dispersion check
        chroma_spread = float(np.mean(np.max(img, axis=-1) - np.min(img, axis=-1)))
        assert chroma_spread > 8.0

    # Dawn, Noon, and Twilight must be physically distinct
    assert float(np.mean(np.abs(img_noon.astype(np.float64) - img_dawn.astype(np.float64)))) > 10.0
    assert float(np.mean(np.abs(img_noon.astype(np.float64) - img_twilight.astype(np.float64)))) > 15.0


def test_spectral_balance_scale_controls_all_modalities_coherently() -> None:
    os_engine = ElysiaOrchestraOS()

    bal_low = SpectralBalanceScale(scale_tilt=-0.8, contrast_tilt=-0.6)
    bal_high = SpectralBalanceScale(scale_tilt=+0.8, contrast_tilt=+0.6)

    s_low = os_engine.conduct_slice(2.5, balance=bal_low, manifest=False)
    s_high = os_engine.conduct_slice(2.5, balance=bal_high, manifest=False)

    # Vision micro-to-macro detail ratio rises with positive scale_tilt
    v_low = s_low.performances["vision.optical_prism"].rotor_manifold.spectral_axis_profile()["micro_to_macro_ratio"]  # type: ignore[union-attr]
    v_high = s_high.performances["vision.optical_prism"].rotor_manifold.spectral_axis_profile()["micro_to_macro_ratio"]  # type: ignore[union-attr]
    assert v_high > v_low

    # Hearing brightness rises with positive scale_tilt
    h_low = s_low.performances["hearing.harmonic_synth"].causal_state["spectral_brightness"]
    h_high = s_high.performances["hearing.harmonic_synth"].causal_state["spectral_brightness"]
    assert h_high > h_low

    # Touch structural tension rises coherently with positive contrast/scale tilt
    t_low = s_low.performances["touch.tactile_resistance"].causal_state["structural_tension"]
    t_high = s_high.performances["touch.tactile_resistance"].causal_state["structural_tension"]
    assert t_high > t_low


def test_harmonic_acoustic_and_tactile_instruments() -> None:
    os_engine = ElysiaOrchestraOS()
    noon_m = os_engine.conduct_slice(2.5, manifest=True, resolution=48)
    twilight_nom = os_engine.conduct_slice(7.5, manifest=False)

    # Acoustic PCM waveform is finite, non-silent, and consonant
    pcm = noon_m.performances["hearing.harmonic_synth"].manifestation
    assert pcm is not None
    assert pcm.ndim == 1 and pcm.size > 1000
    assert np.all(np.isfinite(pcm))
    assert float(np.std(pcm)) > 0.05
    assert noon_m.performances["hearing.harmonic_synth"].causal_state["overtone_consonance"] > 0.4

    # Tactile instrument requires zero pixel rasterization (manifest=False) and tracks warmth
    assert twilight_nom.performances["vision.optical_prism"].manifestation is None
    assert twilight_nom.performances["touch.tactile_resistance"].manifestation is None
    noon_temp = noon_m.performances["touch.tactile_resistance"].causal_state["thermal_warmth_celsius"]
    twi_temp = twilight_nom.performances["touch.tactile_resistance"].causal_state["thermal_warmth_celsius"]
    assert noon_temp > twi_temp + 8.0


class _DummyCustomInstrument(SensoryInstrument):
    def __init__(self) -> None:
        self.instrument_id = "custom.barometer"
        self.modality = "atmosphere"

    def perform(
        self,
        globe_state: SpacetimeGlobeState,
        palette: CausalMechanismPalette,
        balance: SpectralBalanceScale,
        manifest: bool = False,
        resolution: int = 128,
        context: Optional[Dict[str, InstrumentPerformance]] = None,
    ) -> InstrumentPerformance:
        pressure = 1013.25 - 6.0 * math.sin(globe_state.macro_angle)
        return InstrumentPerformance(
            instrument_id=self.instrument_id,
            modality=self.modality,
            causal_state={"pressure_hpa": round(pressure, 2)},
            rotor_manifold=None,
            manifestation=None,
            chromatic_triad=(0.3, 0.5, 0.2),
            explanation=f"Atmospheric pressure {pressure:.1f} hPa.",
        )


def test_orchestra_os_conductor_explanation_and_causal_field_sync() -> None:
    os_engine = ElysiaOrchestraOS()
    os_engine.register_instrument(_DummyCustomInstrument())

    res = os_engine.conduct_slice(2.5, manifest=False)
    assert res.cross_modal_coherence > 0.65
    assert "Sunlight" in res.human_explanation
    assert "Harmonic chord" in res.human_explanation
    assert "Tactile sensation" in res.human_explanation
    assert "Atmospheric pressure" in res.human_explanation

    cf = CausalField(dimensions=3)
    sync = os_engine.sync_to_causal_field(res, cf, prefix="orch_test")
    assert len(sync["tier_voxels"]) == 3
    assert len(sync["instrument_voxels"]) == 4
    assert len(cf.voxels) == 7
    assert len(cf.engrams) == 7
    assert sync["beams_linked"] >= 8
    cf.step(dt=0.05)
