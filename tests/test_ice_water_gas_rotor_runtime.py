import math
from pathlib import Path

import numpy as np

from core.orchestration.ice_water_gas_rotor_runtime import (
    HardwarePhaseController,
    IceCrystalStorage,
    IceWaterGasRotorRuntime,
    extract_macro_signature,
)
from core.orchestration.sensory_orchestra_os import ElysiaOrchestraOS
from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
)


def _make_sample_1d(idx: int, freq_mult: float = 1.0) -> SpectralScaleRotorManifold:
    w0_val = 2.0 * math.pi * (1.0 + 0.7 * idx) * freq_mult
    phase = 0.23 * idx
    b0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=np.array([[w0_val], [w0_val * 2.1]], dtype=np.float64),
        temporal_omegas=np.array([w0_val, w0_val * 2.1], dtype=np.float64),
        amplitudes=np.array(
            [
                (1.0 + 0.05 * idx) * np.exp(1j * phase),
                0.5 * np.exp(-1j * (phase + 0.4)),
            ],
            dtype=np.complex128,
        ),
        centers=np.zeros((2, 1), dtype=np.float64),
        sigmas=np.full(2, np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=np.array([-1, -1], dtype=np.int32),
        parent_couplings=np.ones(2, dtype=np.float64),
    )
    b1 = ScaleRotorBand(
        level=1,
        dim=1,
        wavevectors=np.array([[w0_val * 3.0]], dtype=np.float64),
        temporal_omegas=np.array([w0_val * 3.0], dtype=np.float64),
        amplitudes=np.array([0.4 * np.exp(1j * 0.5 * phase)], dtype=np.complex128),
        centers=np.array([[0.2 + 0.02 * (idx % 25)]], dtype=np.float64),
        sigmas=np.array([0.05], dtype=np.float64),
        is_global=False,
        parent_indices=np.array([0], dtype=np.int32),
        parent_couplings=np.array([0.85], dtype=np.float64),
    )
    return SpectralScaleRotorManifold(
        dim=1,
        domain_bounds=([0.0], [1.0]),
        bands=[b0, b1],
        balance=SpectralBalanceScale(scale_tilt=0.1, contrast_tilt=-0.1),
        current_time=0.5,
    )


def _make_sample_2d(idx: int) -> SpectralScaleRotorManifold:
    wy = 2.0 * math.pi * (1.0 + 0.5 * idx)
    wx = 2.0 * math.pi * (2.0 + 0.3 * idx)
    b0 = ScaleRotorBand(
        level=0,
        dim=2,
        wavevectors=np.array([[wy, wx], [-wx, wy]], dtype=np.float64),
        temporal_omegas=np.array([math.hypot(wy, wx), math.hypot(wy, wx)], dtype=np.float64),
        amplitudes=np.array([1.2 * np.exp(1j * 0.3 * idx), 0.7 - 0.2j], dtype=np.complex128),
        centers=np.zeros((2, 2), dtype=np.float64),
        sigmas=np.full(2, np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=np.array([-1, -1], dtype=np.int32),
        parent_couplings=np.ones(2, dtype=np.float64),
    )
    return SpectralScaleRotorManifold(dim=2, domain_bounds=([0.0, 0.0], [1.0, 1.0]), bands=[b0])


def test_ice_crystal_mmap_lossless_freeze_melt_and_zero_scan_lookup(tmp_path: Path) -> None:
    slab = tmp_path / "ice_crystals.slab"
    # Small initial capacity forces real Windows mmap expansion cleanly
    with IceCrystalStorage(slab, initial_capacity_bytes=64 * 1024) as ice:
        manifolds = {}
        for i in range(16):
            m1 = _make_sample_1d(i)
            m2 = _make_sample_2d(i)
            manifolds[f"c1_{i}"] = m1
            manifolds[f"c2_{i}"] = m2
            ice.freeze(f"c1_{i}", m1)
            ice.freeze(f"c2_{i}", m2)

        assert ice.num_crystals == 32
        assert ice.bytes_read_from_ssd == 0

        # Query target crystal purely against RAM Macro Phase Index -> 0 SSD bytes read!
        target_id = "c1_11"
        q_sig = extract_macro_signature(manifolds[target_id])
        found_id, score = ice.query_resonant_slot(q_sig, dim=1)
        assert found_id == target_id
        assert score > 0.9999
        assert ice.bytes_read_from_ssd == 0

        # Melt ONLY that 1 crystal -> reads exact slice << total_used_bytes
        melted = ice.melt(target_id)
        meta = ice.entries[target_id]
        assert ice.bytes_read_from_ssd == meta.byte_len
        assert ice.bytes_read_from_ssd < ice.used_bytes * 0.1

        probe = np.linspace(0.0, 1.0, 256)[:, None]
        err = float(
            np.max(
                np.abs(
                    melted.slice_manifestation(probe)
                    - manifolds[target_id].slice_manifestation(probe)
                )
            )
        )
        assert err < 1e-13
        assert melted.balance == manifolds[target_id].balance


def test_water_pool_budget_and_automatic_freeze_eviction(tmp_path: Path) -> None:
    slab = tmp_path / "water_evict.slab"
    # Each sample manifold is ~210 bytes; 450 bytes allows ~2 entries max in Water
    with IceWaterGasRotorRuntime(
        slab_path=slab,
        ice_capacity_bytes=128 * 1024,
        max_water_bytes=450,
        max_gas_bytes=32 * 1024,
    ) as rt:
        samples = {f"s_{i}": _make_sample_1d(i) for i in range(6)}
        for sid, m in samples.items():
            rt.ingest_to_ice(sid, m)

        # Activate all 6 sequentially; Water pool must stay bounded by auto-freezing cold entries
        for i in range(6):
            sid = f"s_{i}"
            best_id, _, melted_from_ice = rt.activate_by_resonance(
                extract_macro_signature(samples[sid]), dim=1
            )
            assert best_id == sid
            assert melted_from_ice
            assert rt.water.active_bytes <= rt.water.max_water_bytes

        assert rt.water.eviction_freeze_count >= 4
        assert "s_0" not in rt.water.entries

        # Re-activating evicted crystal s_0 restores it losslessly from SSD Ice
        best_id, _, melted_again = rt.activate_by_resonance(
            extract_macro_signature(samples["s_0"]), dim=1
        )
        assert best_id == "s_0"
        assert melted_again
        probe = np.linspace(0.0, 1.0, 128)[:, None]
        err = float(
            np.max(
                np.abs(
                    rt.water.entries["s_0"].manifold.slice_manifestation(probe)
                    - samples["s_0"].slice_manifestation(probe)
                )
            )
        )
        assert err < 1e-13


def test_gas_tier_strict_l1_cache_budget_and_zero_ssd_burst(tmp_path: Path) -> None:
    slab = tmp_path / "gas_burst.slab"
    with IceWaterGasRotorRuntime(
        slab_path=slab,
        ice_capacity_bytes=128 * 1024,
        max_water_bytes=256 * 1024,
        max_gas_bytes=32 * 1024,
    ) as rt:
        m_ref = _make_sample_1d(3)
        for i in range(12):
            rt.ingest_to_ice(f"cold_{i}", _make_sample_1d(i + 20))
        rt.ingest_to_ice("hot_target", m_ref)

        # Activate ONLY hot_target into Water RAM
        rt.activate_by_resonance(extract_macro_signature(m_ref), dim=1)
        vel = np.array([0.002], dtype=np.float64)

        # Run 100 frames strictly inside L1-resident Gas SoA -> 0 SSD bytes read!
        burst = rt.run_gas_burst(num_frames=100, dt=0.25, dim=1, velocity=vel)
        assert burst["ssd_read_bytes_delta"] == 0
        assert burst["gas_bytes"] <= rt.gas.max_gas_bytes
        assert burst["hot_modes"] == m_ref.total_modes

        # Condensed Water manifold must match exact closed-form dt=25.0 evolution!
        expected = m_ref.evolve_spacetime(dt=25.0, velocity=vel)
        actual = rt.water.entries["hot_target"].manifold
        probe = np.linspace(0.0, 1.0, 256)[:, None]
        err = float(
            np.max(
                np.abs(actual.slice_manifestation(probe) - expected.slice_manifestation(probe))
            )
        )
        assert err < 1e-10
        assert abs(actual.current_time - expected.current_time) < 1e-10

        # Also verify tight 4 KB Gas ceiling strictly respected under large mode counts
        for i in range(12):
            rt.activate_by_resonance(extract_macro_signature(_make_sample_1d(i + 20)), dim=1)
        packed = rt.gas.evaporate_from_water(
            list(rt.water.entries.values()), effective_gas_budget=4096, dim=1
        )
        assert packed > 0
        assert rt.gas.active_bytes <= 4096


def test_hardware_phase_controller_real_telemetry_and_stress_throttling() -> None:
    ctrl = HardwarePhaseController(history_window=16)
    base_gas = 32 * 1024
    base_water = 512 * 1024

    # 8 steady fast frames (~20 us)
    t_base = None
    for _ in range(8):
        t_base = ctrl.observe_and_regulate(20_000, base_gas, base_water)
    assert t_base is not None
    assert t_base["rss_mb"] > 0.0
    assert 0.0 < t_base["sys_ram_percent"] < 100.0

    # Latency spike (~4000 us) triggers positive latency slope & protective throttling
    t_surge = None
    for _ in range(4):
        t_surge = ctrl.observe_and_regulate(4_000_000, base_gas, base_water)
    assert t_surge is not None
    assert t_surge["latency_slope_us"] > 0.0
    assert t_surge["stress_index"] > t_base["stress_index"]
    assert t_surge["effective_gas_bytes"] < base_gas
    assert t_surge["effective_water_bytes"] < base_water
    assert t_surge["recommended_granularity"] > 1.0


def test_orchestra_and_causal_field_end_to_end_integration(tmp_path: Path) -> None:
    slab = tmp_path / "orchestra_iwg.slab"
    os_engine = ElysiaOrchestraOS()
    slices = {
        "dawn": os_engine.conduct_slice(0.0, manifest=False),
        "noon": os_engine.conduct_slice(2.5, manifest=False),
        "dusk": os_engine.conduct_slice(5.0, manifest=False),
        "twilight": os_engine.conduct_slice(7.5, manifest=False),
    }

    with IceWaterGasRotorRuntime(slab_path=slab, max_gas_bytes=32 * 1024) as rt:
        for name, res in slices.items():
            vis_manifold = res.performances["vision.optical_prism"].rotor_manifold
            assert vis_manifold is not None
            rt.ingest_to_ice(f"sky_{name}", vis_manifold)

        noon_manifold = slices["noon"].performances["vision.optical_prism"].rotor_manifold
        assert noon_manifold is not None
        matched_id, score, melted = rt.activate_by_resonance(
            extract_macro_signature(noon_manifold), dim=2
        )
        assert matched_id == "sky_noon"
        assert score > 0.99
        assert melted

        frame_metrics = rt.step_spacetime_frame(dt=0.5, dim=2, diffusivity=0.0005)
        assert frame_metrics["ssd_read_bytes_delta"] == 0
        assert frame_metrics["gas_bytes"] <= 32 * 1024

        cf = CausalField(dimensions=3)
        sync = rt.sync_to_causal_field(cf, last_metrics=frame_metrics, prefix="iwg")
        assert len(cf.voxels) == 3
        assert len(cf.engrams) == 3
        assert sync["beams_linked"] == 2
        cf.step(dt=0.05)
