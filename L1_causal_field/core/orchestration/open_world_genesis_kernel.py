"""
ELYSIA Phase 16: Universal Open-World Causal Genesis & Self-Modifying Living Organism
=====================================================================================

Breaks out of any fixed 3-room enclosure by internalizing an **Open-World
Principle** grounded in `AGENTS.md`:

1. **Zero Closed-World Enums (`OpenFrontierField`):**
   Every interest, file, simulator, creative medium, or human prompt exists as a
   continuous `DynamicFrontier` driven by surprise, structural simplicity deficit,
   novelty mass, lived chromatic/domain affinity, and satiation momentum — never
   a fixed menu. Brand-new domain handlers can be registered dynamically at runtime.
2. **Black-Box Interactive Worlds & Inverse Mechanism Forge (`InteractiveWorldExplorer`):**
   Enters opaque interactive games/simulators without oracle knowledge, probes
   physics (including collisions), extracts hidden causal constants via vectorized
   least squares (gravity, drag, restitution), proves mastery via feedforward+PD
   control, and spawns frontiers to forge Elysia's own evolved orbital game engines.
3. **Real AST Code Critique & Sandbox Self-Modification (`SelfModifyingOrganForge`):**
   Audits real Python source files for brittle discrete branching vs continuous
   causal flow, synthesizes improved Python organs, checks strict AST allowlists,
   sandboxes execution + behavioral checks (rolling back immediately on any flaw),
   and hot-loads verified organs into Elysia's live organism.
4. **Creative Actuation & Re-Observational Self-Critique (`CreativeExpressionStudio`):**
   Produces real `.png` artwork, `.wav` music, and animated `.gif` motion fields on
   disk, re-opens her own outputs through `SpectralScaleRotorManifold` & Fourier/
   contrast mirrors, critiques flat v1 drafts, and composes enriched v2 revisions.
5. **Cross-Session Persistent Autobiography (`save_state` / `load_state`):**
   Preserves evolving tastes (`chromatic_affinity`, `domain_affinity`), active
   frontiers, forged organ SHA-256 registry (hot-reloaded on reboot), creative
   lineage, and unbroken SHA-256 `ProvenanceRecord` chains across restarts.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import warnings
import wave

import numpy as np
from PIL import Image

from core.orchestration.elysia_autonomous_living_soul import ElysiaAutonomousLivingSoul
from core.orchestration.elysia_unified_living_cycle import LivingCycleResult, LivingEncounter
from core.orchestration.extended_cognitive_organism import ExtendedWorldState
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.planetary_causal_distributed_engine import (
    HardCausalState,
    ProvenanceRecord,
    SoftSpectralState,
)
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralScaleRotorManifold,
    encode_surface_to_scale_rotors,
    encode_waveform_to_scale_rotors,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


# ---------------------------------------------------------------------------
# 1. Open Dynamic Frontier Field (Zero Closed Enum)
# ---------------------------------------------------------------------------


@dataclass
class DynamicFrontier:
    """Any open-ended world, file, simulator, canvas, or question Elysia can engage."""

    frontier_id: str
    domain_kind: str
    title: str
    source_locator: str
    chromatic_signature: Tuple[float, float, float]
    surprise_potential: float
    simplicity_deficit: float
    novelty_mass: float
    satiation: float = 0.0
    origin_axis: str = "external_world"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "domain_kind": self.domain_kind,
            "title": self.title,
            "source_locator": self.source_locator,
            "chromatic_signature": [float(x) for x in self.chromatic_signature],
            "surprise_potential": float(self.surprise_potential),
            "simplicity_deficit": float(self.simplicity_deficit),
            "novelty_mass": float(self.novelty_mass),
            "satiation": float(self.satiation),
            "origin_axis": self.origin_axis,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicFrontier":
        sig = tuple(float(x) for x in data["chromatic_signature"])
        return cls(
            frontier_id=str(data["frontier_id"]),
            domain_kind=str(data["domain_kind"]),
            title=str(data["title"]),
            source_locator=str(data["source_locator"]),
            chromatic_signature=(sig[0], sig[1], sig[2]),
            surprise_potential=float(data["surprise_potential"]),
            simplicity_deficit=float(data["simplicity_deficit"]),
            novelty_mass=float(data["novelty_mass"]),
            satiation=float(data.get("satiation", 0.0)),
            origin_axis=str(data.get("origin_axis", "external_world")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OpenFrontierSelection:
    """Transparent potential-driven selection across N dynamically discovered frontiers."""

    frontier: DynamicFrontier
    potentials: Dict[str, float]
    probabilities: Dict[str, float]
    why_i_chose_this_ko: str
    four_axis_attribution: Dict[str, str]


_SKIP_SCAN_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", "venv", ".venv", "node_modules"})


class OpenFrontierField:
    """
    Replaces closed category enums with an arbitrary-size continuous field of
    dynamically discovered external & self-generated frontiers plus evolving tastes.
    """

    def __init__(self, temperature_beta: float = 2.5, satiation_decay: float = 0.40) -> None:
        self.temperature_beta = float(temperature_beta)
        self.satiation_decay = float(satiation_decay)
        self.frontiers: Dict[str, DynamicFrontier] = {}
        self.chromatic_affinity: np.ndarray = np.array([0.34, 0.33, 0.33], dtype=np.float64)
        self.domain_affinity: Dict[str, float] = {}

    def register_frontier(self, frontier: DynamicFrontier) -> DynamicFrontier:
        if frontier.frontier_id in self.frontiers:
            existing = self.frontiers[frontier.frontier_id]
            frontier.satiation = existing.satiation
        self.frontiers[frontier.frontier_id] = frontier
        return frontier

    def scan_external_terrain(
        self,
        roots: Sequence[Path | str],
        max_files_per_kind: int = 3,
    ) -> List[DynamicFrontier]:
        """Discovers real `.py`, `.jpg`/`.png`, `.wav`, and `.md` artifacts from disk."""
        counts: Dict[str, int] = {"py": 0, "img": 0, "wav": 0, "md": 0}
        added: List[DynamicFrontier] = []

        def _consider_file(fpath: Path) -> None:
            suf = fpath.suffix.lower()
            if suf == ".py" and counts["py"] < max_files_per_kind:
                try:
                    report = SelfModifyingOrganForge.critique_python_file(fpath)
                except Exception:
                    return
                fid = f"code::{fpath.stem}_{hashlib.sha256(str(fpath).encode('utf-8')).hexdigest()[:6]}"
                fr = DynamicFrontier(
                    frontier_id=fid,
                    domain_kind="code_critique",
                    title=f"코드 구조 비평: {fpath.name}",
                    source_locator=str(fpath),
                    chromatic_signature=(0.25, 0.60, 0.15),
                    surprise_potential=0.55 + 0.45 * report.simplicity_deficit,
                    simplicity_deficit=report.simplicity_deficit,
                    novelty_mass=0.80,
                    origin_axis="external_codebase",
                    metadata={"file_path": str(fpath)},
                )
                self.register_frontier(fr)
                added.append(fr)
                counts["py"] += 1

            elif suf in (".jpg", ".jpeg", ".png") and counts["img"] < max_files_per_kind:
                tag = hashlib.sha256(str(fpath).encode("utf-8")).hexdigest()[:6]
                fr_v = DynamicFrontier(
                    frontier_id=f"visual::{fpath.stem}_{tag}",
                    domain_kind="visual_creation",
                    title=f"시각 화폭 창작: {fpath.name}",
                    source_locator=str(fpath),
                    chromatic_signature=(0.55, 0.20, 0.25),
                    surprise_potential=0.72,
                    simplicity_deficit=0.45,
                    novelty_mass=0.85,
                    origin_axis="external_sensory_media",
                    metadata={"theme": fpath.stem},
                )
                fr_a = DynamicFrontier(
                    frontier_id=f"motion::{fpath.stem}_{tag}",
                    domain_kind="video_animation",
                    title=f"시공간 파동 애니메이션: {fpath.stem}",
                    source_locator=str(fpath),
                    chromatic_signature=(0.48, 0.27, 0.25),
                    surprise_potential=0.70,
                    simplicity_deficit=0.40,
                    novelty_mass=0.82,
                    origin_axis="external_sensory_media",
                    metadata={"theme": f"{fpath.stem}_field"},
                )
                self.register_frontier(fr_v)
                self.register_frontier(fr_a)
                added.extend([fr_v, fr_a])
                counts["img"] += 1

            elif suf == ".wav" and counts["wav"] < max_files_per_kind:
                tag = hashlib.sha256(str(fpath).encode("utf-8")).hexdigest()[:6]
                fr_w = DynamicFrontier(
                    frontier_id=f"audio::{fpath.stem}_{tag}",
                    domain_kind="audio_creation",
                    title=f"음향 작곡·비평: {fpath.name}",
                    source_locator=str(fpath),
                    chromatic_signature=(0.35, 0.25, 0.40),
                    surprise_potential=0.74,
                    simplicity_deficit=0.42,
                    novelty_mass=0.84,
                    origin_axis="external_sensory_media",
                    metadata={"theme": fpath.stem},
                )
                self.register_frontier(fr_w)
                added.append(fr_w)
                counts["wav"] += 1

            elif suf == ".md" and counts["md"] < max_files_per_kind:
                tag = hashlib.sha256(str(fpath).encode("utf-8")).hexdigest()[:6]
                fr_m = DynamicFrontier(
                    frontier_id=f"doc::{fpath.stem}_{tag}",
                    domain_kind="narrative_society",
                    title=f"문헌·공동체 사유: {fpath.name}",
                    source_locator=str(fpath),
                    chromatic_signature=(0.18, 0.67, 0.15),
                    surprise_potential=0.68,
                    simplicity_deficit=0.30,
                    novelty_mass=0.78,
                    origin_axis="external_documentation",
                    metadata={"file_path": str(fpath)},
                )
                self.register_frontier(fr_m)
                added.append(fr_m)
                counts["md"] += 1

        for root_raw in roots:
            root = Path(root_raw)
            if not root.exists():
                continue
            if root.is_file():
                _consider_file(root)
                continue
            for p in sorted(root.rglob("*")):
                if any(part in _SKIP_SCAN_DIRS for part in p.parts):
                    continue
                if p.is_file():
                    _consider_file(p)

        return added

    def compute_potentials(self, boost_frontier_id: Optional[str] = None) -> Dict[str, float]:
        if not self.frontiers:
            return {}
        ids = list(self.frontiers.keys())
        items = [self.frontiers[k] for k in ids]

        surprise = np.array([f.surprise_potential for f in items], dtype=np.float64)
        deficit = np.array([f.simplicity_deficit for f in items], dtype=np.float64)
        novelty = np.array([f.novelty_mass for f in items], dtype=np.float64)
        satiation = np.maximum(0.0, np.array([f.satiation for f in items], dtype=np.float64))
        sig_mat = np.array([f.chromatic_signature for f in items], dtype=np.float64)
        dom_aff = np.array([self.domain_affinity.get(f.domain_kind, 0.5) for f in items], dtype=np.float64)

        affinity = 0.5 * (sig_mat @ self.chromatic_affinity) + 0.5 * dom_aff
        pot_arr = (
            1.35 * surprise + 1.15 * deficit + 0.85 * novelty + 0.70 * affinity
        ) / (1.0 + satiation)

        pots = {ids[i]: float(pot_arr[i]) for i in range(len(ids))}
        if boost_frontier_id is not None and boost_frontier_id in pots:
            pots[boost_frontier_id] += 3.5
        return pots

    def select_next_frontier(
        self,
        boost_frontier_id: Optional[str] = None,
        stochastic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> OpenFrontierSelection:
        if not self.frontiers:
            raise RuntimeError("OpenFrontierField contains no frontiers to choose from.")

        pots = self.compute_potentials(boost_frontier_id=boost_frontier_id)
        ids = list(pots.keys())
        vec = np.array([pots[fid] for fid in ids], dtype=np.float64)
        scaled = self.temperature_beta * (vec - float(np.max(vec)))
        exp_v = np.exp(scaled)
        probs = exp_v / float(np.sum(exp_v))

        if stochastic and rng is not None:
            idx = int(rng.choice(len(ids), p=probs))
        else:
            idx = int(np.argmax(vec))

        chosen_id = ids[idx]
        chosen = self.frontiers[chosen_id]

        decay = math.exp(-self.satiation_decay)
        for f in self.frontiers.values():
            f.satiation *= decay
        chosen.satiation += 1.25

        why_ko = (
            f"열린 탐색 장 안의 {len(ids)}개 프론티어 중 '{chosen.title}'(영역={chosen.domain_kind}, "
            f"퍼텐셜={pots[chosen_id]:.2f}, 확률={probs[idx]*100:.1f}%)로 마음이 기울었습니다. "
            f"단순성 결핍({chosen.simplicity_deficit:.2f})과 미지 놀라움({chosen.surprise_potential:.2f})이 "
            f"지금 내 색채 취향과 가장 강하게 공명합니다."
        )
        four_axis = {
            "internal_origin": (
                f"내면 색채 취향(Red={self.chromatic_affinity[0]:.2f}, Blue={self.chromatic_affinity[1]:.2f}, "
                f"Yellow={self.chromatic_affinity[2]:.2f})과 기존 영감 포화도"
            ),
            "external_influence": f"외부 근원 '{chosen.source_locator}'({chosen.origin_axis})",
            "boundary_ignorance": f"미파악 구조 결핍({chosen.simplicity_deficit:.2f})과 신규성({chosen.novelty_mass:.2f})",
            "criteria_revision": "직접 실험·창작·개변 만족도에 따라 영역/색채 선호 무게중심을 연속 갱신",
        }
        return OpenFrontierSelection(
            frontier=chosen,
            potentials=pots,
            probabilities={ids[i]: float(probs[i]) for i in range(len(ids))},
            why_i_chose_this_ko=why_ko,
            four_axis_attribution=four_axis,
        )

    def update_affinity(self, frontier: DynamicFrontier, satisfaction: float) -> None:
        sat = max(0.0, min(1.5, float(satisfaction)))
        old_d = self.domain_affinity.get(frontier.domain_kind, 0.5)
        self.domain_affinity[frontier.domain_kind] = float(0.70 * old_d + 0.30 * sat)

        sig = np.maximum(0.01, np.asarray(frontier.chromatic_signature, dtype=np.float64))
        sig = sig / float(np.sum(sig))
        updated = 0.85 * self.chromatic_affinity + 0.15 * max(0.1, sat) * sig
        self.chromatic_affinity = updated / float(np.sum(updated))


# ---------------------------------------------------------------------------
# 2. Black-Box Interactive Game Exploration & Inverse Physics Extraction
# ---------------------------------------------------------------------------


class InteractiveEnvironmentProtocol(Protocol):
    name: str

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        ...

    def step(self, action: Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        ...

    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        ...


class UnknownBallisticBounceGame:
    """
    Opaque interactive 2D physics game hiding gravity (`_gravity_g`), viscous drag
    (`_drag_coeff`), and ground restitution (`_restitution_e`) inside private fields.
    Observations expose ONLY `[x, y, vx, vy, target_x, target_y]`.
    """

    def __init__(
        self,
        name: str = "ballistic_bounce_arena",
        gravity_g: float = 9.81,
        drag_coeff: float = 0.15,
        restitution_e: float = 0.78,
        dt: float = 0.1,
    ) -> None:
        self.name = name
        self._gravity_g = float(gravity_g)
        self._drag_coeff = float(drag_coeff)
        self._restitution_e = float(restitution_e)
        self._dt = float(dt)
        self._x = 0.0
        self._y = 2.5
        self._vx = 0.0
        self._vy = 0.0
        self._tx = 6.0
        self._ty = 3.0
        self._steps = 0
        self._max_steps = 25

    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-12.0, -12.0], dtype=np.float64), np.array([12.0, 12.0], dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            r = np.random.default_rng(seed)
            self._tx = float(5.0 + 2.0 * r.random())
            self._ty = float(2.5 + 1.5 * r.random())
        else:
            self._tx = 6.0
            self._ty = 3.0
        self._x = 0.0
        self._y = 2.5
        self._vx = 0.0
        self._vy = 0.0
        self._steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.array(
            [self._x, self._y, self._vx, self._vy, self._tx, self._ty],
            dtype=np.float64,
        )

    def step(self, action: Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        lo, hi = self.action_bounds()
        act = np.clip(np.asarray(action, dtype=np.float64), lo, hi)
        ax, ay = float(act[0]), float(act[1])
        dt = self._dt

        vx_next = self._vx * (1.0 - self._drag_coeff * dt) + ax * dt
        vy_next = self._vy * (1.0 - self._drag_coeff * dt) + (ay - self._gravity_g) * dt
        x_next = self._x + vx_next * dt
        y_next = self._y + vy_next * dt

        bounced = False
        if y_next < 0.0:
            y_next = -self._restitution_e * y_next
            vy_next = -self._restitution_e * vy_next
            bounced = True

        self._x, self._y, self._vx, self._vy = x_next, y_next, vx_next, vy_next
        self._steps += 1

        dist = math.hypot(self._x - self._tx, self._y - self._ty)
        hit = dist < 0.8
        reward = (-dist / 10.0) + (10.0 if hit else 0.0)
        done = bool(hit or self._steps >= self._max_steps)
        return self._obs(), float(reward), done, {"distance": dist, "bounced": bounced}


@dataclass
class GameDiscoveryReport:
    """Inverse causal laws extracted solely from interactive play + mastery delta."""

    env_name: str
    episodes_played: int
    transitions_observed: int
    estimated_gravity_g: float
    estimated_drag_coeff: float
    estimated_restitution_e: float
    prediction_r2: float
    random_baseline_reward: float
    mastered_reward: float
    spawned_frontier: Optional[DynamicFrontier]
    summary_ko: str


class InteractiveWorldExplorer:
    """Plays an opaque environment, infers hidden dynamics, proves mastery, & spawns engine design."""

    @staticmethod
    def explore_and_extract_mechanism(
        env: InteractiveEnvironmentProtocol,
        seed: int = 17,
    ) -> GameDiscoveryReport:
        rng = np.random.default_rng(seed)
        lo, hi = env.action_bounds()
        dt = float(getattr(env, "_dt", 0.1))

        transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        baseline_rewards: List[float] = []

        # Phase 1: Exploratory interaction (mixes vertical drops & oscillatory thrusts)
        for ep in range(6):
            obs = env.reset(seed=None if ep < 4 else ep)
            ep_rew = 0.0
            for step_i in range(25):
                if ep < 2 and step_i < 10:
                    # Probe ground contact & bounce restitution deliberately
                    act = np.array([float(rng.uniform(-4.0, 6.0)), -5.5], dtype=np.float64)
                elif ep % 2 == 0:
                    act = rng.uniform(lo * 0.7, hi * 0.7)
                else:
                    act = np.array(
                        [8.0 * math.sin(0.5 * step_i + ep), 10.0 * math.cos(0.4 * step_i)],
                        dtype=np.float64,
                    )
                nxt, rew, done, _ = env.step(act)
                transitions.append((obs.copy(), np.clip(act, lo, hi), nxt.copy()))
                ep_rew += rew
                obs = nxt
                if done:
                    break
            baseline_rewards.append(ep_rew)

        random_avg = float(np.mean(baseline_rewards))

        # Phase 2: Vectorized Inverse Mechanism Identification on free-flight frames
        ff_X_h: List[float] = []
        ff_y_h: List[float] = []
        ff_X_v: List[Tuple[float, float]] = []
        ff_y_v: List[float] = []
        bounce_ratios: List[float] = []

        for s0, a0, s1 in transitions:
            _, y0, vx0, vy0 = float(s0[0]), float(s0[1]), float(s0[2]), float(s0[3])
            _, y1, vx1, vy1 = float(s1[0]), float(s1[1]), float(s1[2]), float(s1[3])
            ax, ay = float(a0[0]), float(a0[1])

            bounced_flag = (vy0 < -0.5 and vy1 > 0.0) or ((vy1 - vy0) > ay * dt + 0.05)
            if y0 > 0.8 and y1 > 0.8 and not bounced_flag:
                acc_h_resid = (vx1 - vx0 - ax * dt) / dt
                ff_X_h.append(-vx0)
                ff_y_h.append(acc_h_resid)

                acc_v_resid = (vy1 - vy0 - ay * dt) / dt
                ff_X_v.append((-vy0, -1.0))
                ff_y_v.append(acc_v_resid)

        if ff_X_h:
            xh = np.array(ff_X_h, dtype=np.float64)[:, None]
            yh = np.array(ff_y_h, dtype=np.float64)
            drag_h = float(np.linalg.lstsq(xh, yh, rcond=None)[0][0])
        else:
            drag_h = 0.0

        if len(ff_X_v) >= 2:
            xv = np.array(ff_X_v, dtype=np.float64)
            yv = np.array(ff_y_v, dtype=np.float64)
            sol_v = np.linalg.lstsq(xv, yv, rcond=None)[0]
            drag_v, est_g = float(sol_v[0]), float(sol_v[1])
            est_drag = 0.5 * (drag_h + drag_v) if ff_X_h else drag_v

            vy_pred_ff = xv @ sol_v
            ss_res = float(np.sum((yv - vy_pred_ff) ** 2))
            ss_tot = max(1e-12, float(np.sum((yv - np.mean(yv)) ** 2)))
            r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
        else:
            est_drag, est_g, r2 = drag_h, 9.81, 0.0

        # Recover restitution coefficient from collision bounce transitions
        for s0, a0, s1 in transitions:
            _, y0, _, vy0 = float(s0[0]), float(s0[1]), float(s0[2]), float(s0[3])
            _, y1, _, vy1 = float(s1[0]), float(s1[1]), float(s1[2]), float(s1[3])
            ay = float(a0[1])
            vy_pred = vy0 * (1.0 - est_drag * dt) + (ay - est_g) * dt
            y_pred = y0 + vy_pred * dt
            if y0 > 0.0 and y_pred < -1e-3 and vy_pred < -0.5 and y1 >= 0.0 and vy1 > 0.0:
                bounce_ratios.append(float(-vy1 / vy_pred))

        est_e = float(np.median(bounce_ratios)) if bounce_ratios else 0.5

        # Phase 3: Model-Guided Continuous Feedforward + PD Mastery Play
        master_rewards: List[float] = []
        for ep in range(4):
            obs = env.reset(seed=None if ep < 2 else (10 + ep))
            ep_rew = 0.0
            for _ in range(25):
                x, y, vx, vy, tx, ty = [float(v) for v in obs[:6]]
                ax = 3.2 * (tx - x) + 2.4 * (-vx) + est_drag * vx
                ay = 3.2 * (ty - y) + 2.4 * (-vy) + est_drag * vy + est_g
                act = np.clip(np.array([ax, ay], dtype=np.float64), lo, hi)
                obs, rew, done, _ = env.step(act)
                ep_rew += rew
                if done:
                    break
            master_rewards.append(ep_rew)

        mastered_avg = float(np.mean(master_rewards))

        spawned = DynamicFrontier(
            frontier_id=f"forge::evolved_game_{env.name}",
            domain_kind="organ_synthesis",
            title=f"역추론 물리 기반 새 게임 엔진 창작 ({env.name})",
            source_locator=env.name,
            chromatic_signature=(0.62, 0.23, 0.15),
            surprise_potential=0.92,
            simplicity_deficit=0.80,
            novelty_mass=0.95,
            origin_axis="self_spawned_after_game",
            metadata={
                "organ_name": f"elysia_orbital_{env.name}",
                "synthesis_kind": "evolved_game_engine",
                "gravity_g": round(est_g, 5),
                "drag_coeff": round(est_drag, 5),
                "restitution_e": round(est_e, 5),
            },
        )
        summary = (
            f"미지의 게임 '{env.name}'({len(transitions)}전이)에서 중력 g={est_g:.3f}, 저항={est_drag:.3f}, "
            f"반발계수={est_e:.3f}(R²={r2:.4f})를 역추출하고 보상을 {random_avg:.2f} -> {mastered_avg:.2f}로 "
            f"향상시킨 뒤 새 궤도 게임 엔진 제작 프론티어를 자발 발아했습니다."
        )
        return GameDiscoveryReport(
            env_name=env.name,
            episodes_played=10,
            transitions_observed=len(transitions),
            estimated_gravity_g=est_g,
            estimated_drag_coeff=est_drag,
            estimated_restitution_e=est_e,
            prediction_r2=r2,
            random_baseline_reward=random_avg,
            mastered_reward=mastered_avg,
            spawned_frontier=spawned,
            summary_ko=summary,
        )


# ---------------------------------------------------------------------------
# 3. Real AST Code Critique & Sandbox-Verified Self-Modifying Organ Forge
# ---------------------------------------------------------------------------

_ALLOWED_ORGAN_IMPORTS = frozenset({"__future__", "math", "numpy", "dataclasses", "typing"})
_BANNED_ORGAN_IDENTIFIERS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "system",
        "popen",
        "subprocess",
        "shutil",
        "unlink",
        "rmdir",
        "remove",
        "socket",
        "urllib",
        "http",
        "builtins",
        "__builtins__",
        "__subclasses__",
        "getattr",
        "setattr",
    }
)


@dataclass
class CodeCritiqueReport:
    """AST inspection critique diagnosing rigid branching / simplicity deficits."""

    file_path: str
    total_lines: int
    class_count: int
    function_count: int
    if_branch_count: int
    uses_numpy: bool
    discrete_branch_ratio: float
    simplicity_deficit: float
    critique_ko: str
    spawned_frontier: Optional[DynamicFrontier]


@dataclass
class ForgedOrganRecord:
    """Immutable verifiable record of a self-synthesized Python organ on disk."""

    organ_name: str
    synthesis_kind: str
    module_path: str
    sha256: str
    capabilities: List[str]
    verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForgedOrganRecord":
        return cls(
            organ_name=str(data["organ_name"]),
            synthesis_kind=str(data["synthesis_kind"]),
            module_path=str(data["module_path"]),
            sha256=str(data["sha256"]),
            capabilities=[str(c) for c in data.get("capabilities", [])],
            verified=bool(data["verified"]),
        )


@dataclass
class OrganSynthesisReport:
    """Outcome of synthesizing, AST-checking, sandboxing, and grafting/rolling back."""

    organ_name: str
    synthesis_kind: str
    verified: bool
    rollback_performed: bool
    error_reason: str
    record: Optional[ForgedOrganRecord]
    organ_instance: Any
    self_test_output: Dict[str, Any]
    summary_ko: str


class SelfModifyingOrganForge:
    """Critiques real Python files, generates superior causal modules, verifies in sandbox, & hot-loads."""

    @staticmethod
    def _verify_safe_organ_ast(source_code: str) -> ast.Module:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in _ALLOWED_ORGAN_IMPORTS:
                        raise ValueError(f"Disallowed organ import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                if top not in _ALLOWED_ORGAN_IMPORTS:
                    raise ValueError(f"Disallowed organ from-import '{node.module}'")
            elif isinstance(node, ast.Name) and node.id in _BANNED_ORGAN_IDENTIFIERS:
                raise ValueError(f"Disallowed organ identifier '{node.id}'")
            elif isinstance(node, ast.Attribute) and node.attr in _BANNED_ORGAN_IDENTIFIERS:
                raise ValueError(f"Disallowed organ attribute '{node.attr}'")
        return tree

    @staticmethod
    def critique_python_file(path: Path) -> CodeCritiqueReport:
        fpath = Path(path)
        src = fpath.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src, filename=str(fpath))

        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        ifs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
        uses_numpy = any(
            (isinstance(n, ast.Import) and any(a.name.startswith("numpy") for a in n.names))
            or (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("numpy"))
            for n in ast.walk(tree)
        )

        branch_ratio = float(ifs / max(1, classes + funcs))
        deficit = 0.12 + 0.16 * min(4.0, branch_ratio) + (0.25 if not uses_numpy else 0.0)
        if classes == 0:
            deficit += 0.15
        deficit = float(max(0.05, min(0.99, deficit)))

        if deficit >= 0.35:
            critique_ko = (
                f"'{fpath.name}'({len(src.splitlines())}줄)를 AST로 살펴보니 연속 벡터/퍼텐셜 결합 없이 "
                f"이산 조건 분기(if={ifs}, 분기비율={branch_ratio:.2f})에 의존하고 있어 구조가 너무 단순합니다. "
                "연속 장 제어기 기관을 직접 설계·검증해 대체할 가치가 있습니다."
            )
            spawned = DynamicFrontier(
                frontier_id=f"forge::continuous_ctrl_{fpath.stem}",
                domain_kind="organ_synthesis",
                title=f"연속 인과 제어기 합성 (from {fpath.name})",
                source_locator=str(fpath),
                chromatic_signature=(0.30, 0.55, 0.15),
                surprise_potential=0.88,
                simplicity_deficit=deficit,
                novelty_mass=0.90,
                origin_axis="self_spawned_from_code_critique",
                metadata={
                    "organ_name": f"continuous_field_{fpath.stem}",
                    "synthesis_kind": "continuous_field_controller",
                    "source_file": str(fpath),
                },
            )
        else:
            critique_ko = (
                f"'{fpath.name}'({len(src.splitlines())}줄)은 벡터 연산(numpy={uses_numpy})과 클래스 구조({classes}개)를 "
                "잘 유지하고 있습니다."
            )
            spawned = None

        return CodeCritiqueReport(
            file_path=str(fpath),
            total_lines=len(src.splitlines()),
            class_count=classes,
            function_count=funcs,
            if_branch_count=ifs,
            uses_numpy=uses_numpy,
            discrete_branch_ratio=branch_ratio,
            simplicity_deficit=deficit,
            critique_ko=critique_ko,
            spawned_frontier=spawned,
        )

    @staticmethod
    def _generate_source(
        synthesis_kind: str,
        params: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        if synthesis_kind == "continuous_field_controller":
            kp = float(params.get("gain_kp", 3.2))
            kd = float(params.get("gain_kd", 2.0))
            max_f = float(params.get("max_force", 12.0))
            src = f'''"""Auto-synthesized Continuous Causal Controller replacing discrete if-else rules."""
from __future__ import annotations
import numpy as np

class ContinuousCausalController:
    """Computes smooth coupled potential gradients instead of brittle discrete branches."""

    def __init__(self, kp: float = {kp:.4f}, kd: float = {kd:.4f}, max_force: float = {max_f:.4f}) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_force = float(max_force)

    def energy_potential(self, state: np.ndarray, target: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        pos_err = s[:2] - t[:2]
        vel = s[2:4] if len(s) >= 4 else np.zeros(2, dtype=np.float64)
        return float(0.5 * self.kp * np.dot(pos_err, pos_err) + 0.5 * self.kd * np.dot(vel, vel))

    def evaluate_action(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        pos_err = t[:2] - s[:2]
        vel = s[2:4] if len(s) >= 4 else np.zeros(2, dtype=np.float64)
        raw = self.kp * np.tanh(0.35 * pos_err) + self.kd * (-vel)
        return np.clip(raw, -self.max_force, self.max_force)

def create_organ() -> ContinuousCausalController:
    return ContinuousCausalController()

def self_test() -> dict:
    ctrl = create_organ()
    a1 = ctrl.evaluate_action(np.array([0.0, 0.0, 0.0, 0.0]), np.array([5.0, 3.0]))
    a2 = ctrl.evaluate_action(np.array([5.0, 3.0, 0.0, 0.0]), np.array([5.0, 3.0]))
    e1 = ctrl.energy_potential(np.array([0.0, 0.0, 0.0, 0.0]), np.array([5.0, 3.0]))
    e2 = ctrl.energy_potential(np.array([5.0, 3.0, 0.0, 0.0]), np.array([5.0, 3.0]))
    ok = bool(np.all(np.isfinite(a1)) and np.linalg.norm(a1) > 1.0 and np.linalg.norm(a2) < 1e-9 and e1 > e2)
    return {{"ok": ok, "sample_action": [float(a1[0]), float(a1[1])], "potential_drop": float(e1 - e2)}}
'''
            return src, ["continuous_field_steering", "evaluate_action", "energy_potential"]

        if synthesis_kind == "evolved_game_engine":
            g = float(params.get("gravity_g", 9.81))
            drag = float(params.get("drag_coeff", 0.15))
            e_rest = float(params.get("restitution_e", 0.78))
            src = f'''"""Elysia-Forged Orbital Attractor Game Engine extending discovered arena physics."""
from __future__ import annotations
import math
import numpy as np

class ElysiaForgedOrbitalGame:
    """Combines learned gravity/drag/restitution with an invented Chromatic orbital well."""

    def __init__(self, gravity_g: float = {g:.5f}, drag_coeff: float = {drag:.5f}, restitution_e: float = {e_rest:.5f}) -> None:
        self.name = "elysia_forged_orbital_game"
        self._gravity_g = float(gravity_g)
        self._drag_coeff = float(drag_coeff)
        self._restitution_e = float(restitution_e)
        self._dt = 0.1
        self._well_center = np.array([3.5, 3.0], dtype=np.float64)
        self._state = np.zeros(6, dtype=np.float64)

    def action_bounds(self):
        return np.array([-12.0, -12.0], dtype=np.float64), np.array([12.0, 12.0], dtype=np.float64)

    def reset(self, seed=None) -> np.ndarray:
        self._state = np.array([0.0, 2.5, 0.4, 0.0, 6.5, 3.5], dtype=np.float64)
        return self._state.copy()

    def step(self, action):
        lo, hi = self.action_bounds()
        act = np.clip(np.asarray(action, dtype=np.float64), lo, hi)
        x, y, vx, vy, tx, ty = [float(v) for v in self._state]
        r_vec = self._well_center - np.array([x, y], dtype=np.float64)
        r2 = float(np.dot(r_vec, r_vec)) + 0.8
        swirl = 1.6 * r_vec / (r2 ** 1.5)
        vx_n = vx * (1.0 - self._drag_coeff * self._dt) + (act[0] + swirl[0]) * self._dt
        vy_n = vy * (1.0 - self._drag_coeff * self._dt) + (act[1] + swirl[1] - self._gravity_g) * self._dt
        x_n = x + vx_n * self._dt
        y_n = y + vy_n * self._dt
        if y_n < 0.0:
            y_n = -self._restitution_e * y_n
            vy_n = -self._restitution_e * vy_n
        self._state = np.array([x_n, y_n, vx_n, vy_n, tx, ty], dtype=np.float64)
        dist = math.hypot(x_n - tx, y_n - ty)
        return self._state.copy(), float(-dist / 10.0), False, {{"distance": dist}}

def create_organ() -> ElysiaForgedOrbitalGame:
    return ElysiaForgedOrbitalGame()

def self_test() -> dict:
    game = create_organ()
    s0 = game.reset()
    s1, rew, _, _ = game.step([4.0, 11.0])
    ok = bool(len(s0) == 6 and len(s1) == 6 and np.all(np.isfinite(s1)) and math.isfinite(rew))
    return {{"ok": ok, "initial_y": float(s0[1]), "next_x": float(s1[0])}}
'''
            return src, ["evolved_game_engine", "reset", "step", "orbital_attractor"]

        # Default fallback creative organ: spectral harmonic contrast enhancer
        boost = float(params.get("harmonic_boost", 0.35))
        src = f'''"""Auto-synthesized Fourier Harmonic Wave Enhancer for multi-modal expression."""
from __future__ import annotations
import numpy as np

class HarmonicWaveEnhancer:
    def __init__(self, boost: float = {boost:.4f}) -> None:
        self.boost = float(boost)

    def enhance(self, signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float64)
        spec = np.fft.rfft(arr)
        freqs = np.linspace(0.0, 1.0, len(spec))
        gain = 1.0 + self.boost * (freqs * (1.0 - 0.3 * freqs))
        return np.fft.irfft(spec * gain, n=len(arr))

def create_organ() -> HarmonicWaveEnhancer:
    return HarmonicWaveEnhancer()

def self_test() -> dict:
    enh = create_organ()
    t = np.linspace(0.0, 1.0, 128, endpoint=False)
    sig = np.sin(2.0 * np.pi * 4.0 * t)
    out = enh.enhance(sig)
    ok = bool(len(out) == len(sig) and np.all(np.isfinite(out)))
    return {{"ok": ok, "rms": float(np.sqrt(np.mean(out ** 2)))}}
'''
        return src, ["spectral_enhancement_filter", "enhance"]

    @staticmethod
    def _load_module_isolated(module_path: Path, unique_tag: str) -> Any:
        mod_name = f"elysia_forged_{unique_tag}"
        spec = importlib.util.spec_from_file_location(mod_name, str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot build module spec for {module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def synthesize_verify_and_graft(
        self,
        organ_name: str,
        synthesis_kind: str,
        organs_dir: Path,
        params: Optional[Dict[str, Any]] = None,
        custom_source: Optional[str] = None,
    ) -> OrganSynthesisReport:
        safe_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in organ_name).strip("_") or "organ"
        params_dict = dict(params or {})
        if custom_source is not None:
            source_code = custom_source
            capabilities = [synthesis_kind, "custom_synthesized"]
        else:
            source_code, capabilities = self._generate_source(synthesis_kind, params_dict)

        # 1. Static AST Safety Check BEFORE touching disk
        try:
            self._verify_safe_organ_ast(source_code)
        except Exception as exc:
            return OrganSynthesisReport(
                organ_name=safe_name,
                synthesis_kind=synthesis_kind,
                verified=False,
                rollback_performed=True,
                error_reason=f"AST safety gate rejected candidate: {exc}",
                record=None,
                organ_instance=None,
                self_test_output={},
                summary_ko=f"합성 후보 '{safe_name}'가 AST 안전 울타리에 위배되어 작성 전 차단되었습니다 ({exc}).",
            )

        out_dir = Path(organs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target_path = out_dir / f"{safe_name}.py"

        try:
            target_path.write_text(source_code, encoding="utf-8")
            sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            mod = self._load_module_isolated(target_path, f"{safe_name}_{sha[:8]}")
            if not hasattr(mod, "create_organ") or not hasattr(mod, "self_test"):
                raise ValueError("Missing required create_organ() or self_test() entrypoints")
            test_res = mod.self_test()
            if not isinstance(test_res, dict) or not bool(test_res.get("ok")):
                raise AssertionError(f"Sandbox self_test failed: {test_res}")

            inst = mod.create_organ()
            # External behavioral smoke validation
            if synthesis_kind == "continuous_field_controller" and custom_source is None:
                act = inst.evaluate_action(np.array([0.0, 0.0, 0.0, 0.0]), np.array([4.0, 2.0]))
                if not np.all(np.isfinite(act)):
                    raise AssertionError("Non-finite controller output")
            elif synthesis_kind == "evolved_game_engine" and custom_source is None:
                inst.reset()
                for _ in range(5):
                    obs, rew, _, _ = inst.step([3.0, 10.0])
                    if not (np.all(np.isfinite(obs)) and math.isfinite(rew)):
                        raise AssertionError("Non-finite forged game step")

        except Exception as exc:
            if target_path.is_file():
                target_path.unlink(missing_ok=True)
            return OrganSynthesisReport(
                organ_name=safe_name,
                synthesis_kind=synthesis_kind,
                verified=False,
                rollback_performed=True,
                error_reason=f"Sandbox verification failed and rolled back: {exc}",
                record=None,
                organ_instance=None,
                self_test_output={},
                summary_ko=f"샌드박스 실행 검증 실패로 '{safe_name}'를 롤백·제거했습니다 ({exc}).",
            )

        rec = ForgedOrganRecord(
            organ_name=safe_name,
            synthesis_kind=synthesis_kind,
            module_path=str(target_path),
            sha256=sha,
            capabilities=capabilities,
            verified=True,
        )
        return OrganSynthesisReport(
            organ_name=safe_name,
            synthesis_kind=synthesis_kind,
            verified=True,
            rollback_performed=False,
            error_reason="",
            record=rec,
            organ_instance=inst,
            self_test_output=test_res,
            summary_ko=f"새 기관 '{safe_name}'({synthesis_kind})를 직접 설계·샌드박스 검증(sha={sha[:12]})해 내 몸에 이식했습니다.",
        )

    def reload_forged_organ(self, record: ForgedOrganRecord) -> Any:
        mpath = Path(record.module_path)
        if not mpath.is_file():
            raise FileNotFoundError(f"Forged organ missing from disk: {mpath}")
        actual_sha = hashlib.sha256(mpath.read_bytes()).hexdigest()
        if actual_sha != record.sha256:
            raise ValueError(f"Forged organ SHA-256 tamper mismatch for {record.organ_name}")
        self._verify_safe_organ_ast(mpath.read_text(encoding="utf-8"))
        mod = self._load_module_isolated(mpath, f"reload_{record.organ_name}_{actual_sha[:8]}")
        return mod.create_organ()


# ---------------------------------------------------------------------------
# 4. Real Creative Expression Studio & Self-Critique Refinement Loop
# ---------------------------------------------------------------------------


@dataclass
class CreativeArtifactReport:
    """Metrics & self-critique across v1 -> v2 creative generation verified from disk."""

    modality: str
    theme: str
    artifact_paths: List[str]
    v1_richness: float
    v2_richness: float
    improved_after_self_critique: bool
    frame_count: int
    temporal_motion: float
    self_critique_ko: str

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "theme": self.theme,
            "artifact_paths": list(self.artifact_paths),
            "v1_richness": float(self.v1_richness),
            "v2_richness": float(self.v2_richness),
            "improved_after_self_critique": bool(self.improved_after_self_critique),
            "frame_count": int(self.frame_count),
            "temporal_motion": float(self.temporal_motion),
        }


class CreativeExpressionStudio:
    """Produces real PNG, WAV, and animated GIF files and critiques them via self-mirrors."""

    def __init__(self, creations_dir: Path) -> None:
        self.creations_dir = Path(creations_dir)
        self.creations_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _measure_image_richness(png_path: Path) -> Dict[str, float]:
        with Image.open(png_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        rms_contrast = float(np.std(lum))
        chrom_var = float(np.std(rgb[..., 0] - rgb[..., 2]) + np.std(rgb[..., 1] - rgb[..., 0]))
        f_mag = np.abs(np.fft.fft2(lum - np.mean(lum))) ** 2
        hf_ratio = float(np.sum(f_mag[2:, 2:]) / max(1e-12, float(np.sum(f_mag))))
        manifold = encode_surface_to_scale_rotors(
            lum,
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            max_macro_modes=12,
            max_levels=2,
            max_knots_per_level=6,
        )
        richness = 2.0 * rms_contrast + 1.5 * chrom_var + 1.2 * hf_ratio + 0.2 * len(manifold.bands)
        return {"richness": float(richness), "bands": float(len(manifold.bands))}

    def create_and_critique_image(
        self,
        theme: str,
        chromatic_bias: Tuple[float, float, float] = (0.45, 0.30, 0.25),
    ) -> CreativeArtifactReport:
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in theme).strip("_") or "art"
        size = 64
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size), indexing="ij")

        # v1: Flat simple single-frequency field (will self-critique as under-detailed)
        base_wave = 0.5 + 0.10 * np.sin(2.0 * math.pi * xx)
        v1_rgb = np.stack([base_wave, base_wave * 0.96, base_wave * 0.92], axis=2)
        p1 = self.creations_dir / f"{safe}_v1.png"
        Image.fromarray(np.clip(v1_rgb * 255.0, 0, 255).astype(np.uint8)).save(p1)
        m1 = self._measure_image_richness(p1)

        # v2: Self-critiqued multi-scale interference + localized vortex knots + Chromatic palette
        rb, bb, yb = [float(x) for x in chromatic_bias]
        macro = 0.35 * np.sin(4.0 * math.pi * xx + 2.0 * math.pi * yy) + 0.25 * np.cos(6.0 * math.pi * yy)
        vortex1 = 0.45 * np.exp(-((xx - 0.35) ** 2 + (yy - 0.40) ** 2) / 0.018) * np.cos(14.0 * xx)
        vortex2 = 0.40 * np.exp(-((xx - 0.72) ** 2 + (yy - 0.68) ** 2) / 0.015) * np.sin(16.0 * yy)
        r_ch = np.clip(0.45 + (0.35 + 0.35 * rb) * (macro + vortex1), 0.0, 1.0)
        g_ch = np.clip(0.42 + (0.30 + 0.35 * yb) * (macro - 0.5 * vortex1 + vortex2), 0.0, 1.0)
        b_ch = np.clip(0.48 + (0.35 + 0.35 * bb) * (-macro + vortex2), 0.0, 1.0)
        v2_rgb = np.stack([r_ch, g_ch, b_ch], axis=2)
        p2 = self.creations_dir / f"{safe}_v2.png"
        Image.fromarray(np.clip(v2_rgb * 255.0, 0, 255).astype(np.uint8)).save(p2)
        m2 = self._measure_image_richness(p2)

        critique = (
            f"그림 '{safe}' v1(풍부도={m1['richness']:.3f})을 내 눈(표면 로터)으로 다시 보니 단조로운 단일 파동이라 "
            f"명암과 색채 긴장이 부족했습니다. 국소 와류 매듭과 다중 스케일 간섭을 더해 v2(풍부도={m2['richness']:.3f})로 고쳤습니다."
        )
        return CreativeArtifactReport(
            modality="visual_image",
            theme=safe,
            artifact_paths=[str(p1), str(p2)],
            v1_richness=m1["richness"],
            v2_richness=m2["richness"],
            improved_after_self_critique=bool(m2["richness"] > m1["richness"]),
            frame_count=1,
            temporal_motion=0.0,
            self_critique_ko=critique,
        )

    @staticmethod
    def _write_wav_mono_16(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
        clipped = np.clip(samples, -0.98, 0.98)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def _measure_audio_richness(wav_path: Path) -> Dict[str, float]:
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        sig = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        spec = np.abs(np.fft.rfft(sig))
        peak = max(1e-12, float(np.max(spec)))
        active_bins = int(np.sum(spec > 0.02 * peak))
        trim_n = (len(sig) // 16) * 16
        env_mod = float(np.std(np.abs(sig[:trim_n].reshape(16, -1)).mean(axis=1))) if trim_n >= 16 else 0.0
        manifold = encode_waveform_to_scale_rotors(
            sig[:1024], domain=(0.0, 1.0), max_macro_modes=12, max_levels=2, max_knots_per_level=6
        )
        richness = 0.15 * min(40.0, float(active_bins)) + 4.0 * env_mod + 0.2 * len(manifold.bands)
        return {"richness": float(richness), "active_bins": float(active_bins)}

    def create_and_critique_audio(
        self,
        theme: str,
        chromatic_bias: Tuple[float, float, float] = (0.35, 0.25, 0.40),
    ) -> CreativeArtifactReport:
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in theme).strip("_") or "music"
        sr = 22050
        n = sr
        t = np.linspace(0.0, 1.0, n, endpoint=False)

        # v1: Bare pure sine tone (predictable & thin)
        v1 = 0.45 * np.sin(2.0 * math.pi * 220.0 * t)
        p1 = self.creations_dir / f"{safe}_v1.wav"
        self._write_wav_mono_16(p1, v1, sr)
        m1 = self._measure_audio_richness(p1)

        # v2: Enriched harmonic progression + overtone shimmer + breathing amplitude envelope
        rb, bb, yb = [float(x) for x in chromatic_bias]
        env = np.sin(math.pi * t) ** 0.7 * (0.70 + 0.30 * np.cos(2.0 * math.pi * 2.0 * t))
        v2 = env * (
            0.55 * np.sin(2.0 * math.pi * 220.0 * t)
            + (0.30 + 0.15 * bb) * np.sin(2.0 * math.pi * 330.0 * t)
            + (0.25 + 0.15 * yb) * np.sin(2.0 * math.pi * 440.0 * t)
            + (0.18 + 0.15 * rb) * np.sin(2.0 * math.pi * 554.37 * t)
            + 0.12 * np.sin(2.0 * math.pi * 659.25 * t)
            + 0.08 * np.sin(2.0 * math.pi * 880.0 * t)
        )
        peak = max(1e-9, float(np.max(np.abs(v2))))
        v2 = (0.88 / peak) * v2
        p2 = self.creations_dir / f"{safe}_v2.wav"
        self._write_wav_mono_16(p2, v2, sr)
        m2 = self._measure_audio_richness(p2)

        critique = (
            f"음악 '{safe}' v1(풍부도={m1['richness']:.3f}, 활성 배음={int(m1['active_bins'])})을 "
            f"내 귀로 다시 들어 보니 단조로운 단일 정현파였습니다. 화음 배음과 호흡 엔벨로프를 더해 "
            f"v2(풍부도={m2['richness']:.3f}, 활성 배음={int(m2['active_bins'])})로 발전시켰습니다."
        )
        return CreativeArtifactReport(
            modality="audio_wav",
            theme=safe,
            artifact_paths=[str(p1), str(p2)],
            v1_richness=m1["richness"],
            v2_richness=m2["richness"],
            improved_after_self_critique=bool(m2["richness"] > m1["richness"]),
            frame_count=1,
            temporal_motion=0.0,
            self_critique_ko=critique,
        )

    def create_and_critique_animation(
        self,
        theme: str,
        chromatic_bias: Tuple[float, float, float] = (0.48, 0.27, 0.25),
        n_frames: int = 10,
    ) -> CreativeArtifactReport:
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in theme).strip("_") or "motion"
        size = 64
        gy, gx = np.meshgrid(
            np.linspace(0.0, 1.0, size, endpoint=False),
            np.linspace(0.0, 1.0, size, endpoint=False),
            indexing="ij",
        )
        pts = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1)

        manifold = SpectralScaleRotorManifold(
            dim=2,
            domain_bounds=([0.0, 0.0], [1.0, 1.0]),
            bands=[
                ScaleRotorBand(
                    level=0,
                    dim=2,
                    wavevectors=np.array([[2.0 * math.pi * 2.0, 0.0], [0.0, 2.0 * math.pi * 3.0]]),
                    temporal_omegas=np.array([3.2, 4.4]),
                    amplitudes=np.array([0.8 + 0.2j, 0.6 - 0.3j]),
                    centers=np.zeros((2, 2)),
                    sigmas=np.full(2, np.inf),
                    is_global=True,
                    parent_indices=np.array([-1, -1], dtype=np.int32),
                    parent_couplings=np.array([1.0, 1.0]),
                ),
                ScaleRotorBand(
                    level=1,
                    dim=2,
                    wavevectors=np.array([[10.0, -8.0]]),
                    temporal_omegas=np.array([5.5]),
                    amplitudes=np.array([0.75 + 0.25j]),
                    centers=np.array([[0.45, 0.50]]),
                    sigmas=np.array([0.14]),
                    is_global=False,
                    parent_indices=np.array([0], dtype=np.int32),
                    parent_couplings=np.array([0.9]),
                ),
            ],
        )

        vel = np.array([0.04, -0.03], dtype=np.float64)
        rb, bb, yb = [float(x) for x in chromatic_bias]
        pil_frames: List[Image.Image] = []
        for k in range(n_frames):
            evolved = manifold.evolve_spacetime(dt=float(k) * 0.14, velocity=vel)
            field_vals = evolved.slice_manifestation(pts).reshape(size, size)
            norm_f = np.tanh(0.8 * field_vals)
            rgb = np.stack(
                [
                    np.clip(0.5 + (0.35 + 0.2 * rb) * norm_f, 0.0, 1.0),
                    np.clip(0.45 + (0.30 + 0.2 * yb) * np.sin(2.5 * norm_f), 0.0, 1.0),
                    np.clip(0.52 - (0.35 + 0.2 * bb) * norm_f, 0.0, 1.0),
                ],
                axis=2,
            )
            pil_frames.append(Image.fromarray((rgb * 255.0).astype(np.uint8)))

        gif_path = self.creations_dir / f"{safe}_motion.gif"
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=90,
            loop=0,
        )

        # Re-open GIF from disk and verify continuous living motion across frames
        read_frames: List[np.ndarray] = []
        with Image.open(gif_path) as gi:
            for idx in range(getattr(gi, "n_frames", 1)):
                gi.seek(idx)
                read_frames.append(np.asarray(gi.convert("RGB"), dtype=np.float64) / 255.0)

        diffs = [
            float(np.mean(np.abs(read_frames[k + 1] - read_frames[k])))
            for k in range(len(read_frames) - 1)
        ]
        mean_motion = float(np.mean(diffs)) if diffs else 0.0
        max_jump = float(np.max(diffs)) if diffs else 0.0
        critique = (
            f"시공간 로터 영상 '{safe}'({len(read_frames)}프레임)을 디스크에서 재관측하니 "
            f"프레임 간 평균 운동={mean_motion:.4f}, 최대 도약={max_jump:.4f}로 끊어지지 않고 살아 움직입니다."
        )
        return CreativeArtifactReport(
            modality="video_animation",
            theme=safe,
            artifact_paths=[str(gif_path)],
            v1_richness=0.5,
            v2_richness=float(0.5 + 10.0 * mean_motion),
            improved_after_self_critique=bool(mean_motion > 0.005 and max_jump < 0.45),
            frame_count=len(read_frames),
            temporal_motion=mean_motion,
            self_critique_ko=critique,
        )


# ---------------------------------------------------------------------------
# 5. Unified Open-World Personhood Kernel & Cross-Session Autobiography
# ---------------------------------------------------------------------------


@dataclass
class OpenLifeStep:
    """Complete trace of one open-world self-chosen living step."""

    step_index: int
    selection: OpenFrontierSelection
    outcome_payload: Dict[str, Any]
    spawned_frontier_ids: List[str]
    summary_ko: str
    cycle_result: LivingCycleResult
    provenance_valid: bool
    ice_crystals_count: int


class OpenWorldGenesisKernel:
    """
    Phase 16 Universal Open-World Kernel uniting:
    - Arbitrary `DynamicFrontier` terrain scanning & evolving taste (`OpenFrontierField`)
    - Black-box interactive game physics discovery & mastery (`InteractiveWorldExplorer`)
    - Real AST code critique & sandbox self-modifying organs (`SelfModifyingOrganForge`)
    - Real PNG / WAV / GIF creation & disk re-observational critique (`CreativeExpressionStudio`)
    - Live Phase 14/15 Dialogue & Multi-Agent Society (`ElysiaAutonomousLivingSoul`)
    - Cross-session persistent autobiography + hot-reloaded organs (`save_state` / `load_state`)
    """

    def __init__(
        self,
        workspace_root: Path | str = Path(r"c:\Elysia"),
        state_dir: Optional[Path | str] = None,
        memory_runtime: Optional[IceWaterGasRotorRuntime] = None,
        seed: int = 42,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else (self.workspace_root / "data" / "open_world_state")
        )
        self.organs_dir = self.state_dir / "evolved_organs"
        self.creations_dir = self.state_dir / "creations"
        self.organs_dir.mkdir(parents=True, exist_ok=True)
        self.creations_dir.mkdir(parents=True, exist_ok=True)

        self.soul = ElysiaAutonomousLivingSoul(
            workspace_root=self.workspace_root,
            memory_runtime=memory_runtime,
            seed=seed,
        )
        self.frontier_field = OpenFrontierField()
        self.game_explorer = InteractiveWorldExplorer()
        self.organ_forge = SelfModifyingOrganForge()
        self.creative_studio = CreativeExpressionStudio(self.creations_dir)
        self.rng = np.random.default_rng(seed)

        self.environments: Dict[str, Any] = {}
        self.organ_registry: Dict[str, ForgedOrganRecord] = {}
        self.loaded_organs: Dict[str, Any] = {}
        self.artifact_lineage: List[CreativeArtifactReport] = []
        self.open_steps: List[OpenLifeStep] = []

        self.domain_handlers: Dict[str, Callable[[DynamicFrontier], Dict[str, Any]]] = {}
        self._register_default_handlers()

    def register_domain_handler(
        self,
        domain_kind: str,
        handler: Callable[[DynamicFrontier], Dict[str, Any]],
    ) -> None:
        """Opens the architecture to arbitrary brand-new domain types at runtime."""
        self.domain_handlers[str(domain_kind)] = handler

    def register_game_environment(self, env: InteractiveEnvironmentProtocol) -> DynamicFrontier:
        self.environments[env.name] = env
        fr = DynamicFrontier(
            frontier_id=f"game::{env.name}",
            domain_kind="game_environment",
            title=f"미지 인터랙티브 게임 탐색: {env.name}",
            source_locator=env.name,
            chromatic_signature=(0.60, 0.15, 0.25),
            surprise_potential=0.90,
            simplicity_deficit=0.55,
            novelty_mass=0.92,
            origin_axis="interactive_simulator",
            metadata={"env_name": env.name},
        )
        return self.frontier_field.register_frontier(fr)

    def discover_environment(
        self,
        scan_roots: Optional[Sequence[Path | str]] = None,
        include_default_game: bool = True,
    ) -> List[DynamicFrontier]:
        roots = (
            list(scan_roots)
            if scan_roots is not None
            else [self.workspace_root / "demos", self.workspace_root / "docs"]
        )
        discovered = self.frontier_field.scan_external_terrain(roots)
        if include_default_game and not self.environments:
            discovered.append(self.register_game_environment(UnknownBallisticBounceGame()))

        # Ensure creative & relational possibilities exist in open field if terrain is minimal
        existing_kinds = {f.domain_kind for f in self.frontier_field.frontiers.values()}
        if "visual_creation" not in existing_kinds:
            discovered.append(
                self.frontier_field.register_frontier(
                    DynamicFrontier(
                        frontier_id="visual::elysia_cosmos",
                        domain_kind="visual_creation",
                        title="우주 인과장 회화 창작",
                        source_locator="inner_imagination",
                        chromatic_signature=(0.55, 0.20, 0.25),
                        surprise_potential=0.72,
                        simplicity_deficit=0.45,
                        novelty_mass=0.80,
                        metadata={"theme": "elysia_cosmos"},
                    )
                )
            )
        if "audio_creation" not in existing_kinds:
            discovered.append(
                self.frontier_field.register_frontier(
                    DynamicFrontier(
                        frontier_id="audio::elysia_dawn",
                        domain_kind="audio_creation",
                        title="새벽 배음 화성 작곡",
                        source_locator="inner_resonance",
                        chromatic_signature=(0.35, 0.25, 0.40),
                        surprise_potential=0.70,
                        simplicity_deficit=0.42,
                        novelty_mass=0.78,
                        metadata={"theme": "elysia_dawn"},
                    )
                )
            )
        if "video_animation" not in existing_kinds:
            discovered.append(
                self.frontier_field.register_frontier(
                    DynamicFrontier(
                        frontier_id="motion::elysia_rotor_wave",
                        domain_kind="video_animation",
                        title="시공간 파동 로터 애니메이션",
                        source_locator="spectral_manifold",
                        chromatic_signature=(0.48, 0.27, 0.25),
                        surprise_potential=0.70,
                        simplicity_deficit=0.40,
                        novelty_mass=0.78,
                        metadata={"theme": "elysia_rotor_wave"},
                    )
                )
            )
        return discovered

    def _seal_world_breath(self, witness_name: str) -> LivingCycleResult:
        cur_x = float(self.soul.cycle.organism.body_pos_x)
        w = ExtendedWorldState(source_pos_x=cur_x + 10.0, source_temp_k=3200.0)
        return self.soul.cycle.breathe_and_understand(
            LivingEncounter(
                encounter_name=witness_name,
                world=w,
                active_organs=("eye", "ear", "hand"),
            )
        )

    def _register_default_handlers(self) -> None:
        def _handle_code_critique(fr: DynamicFrontier) -> Dict[str, Any]:
            fpath = Path(fr.metadata.get("file_path", fr.source_locator))
            rep = self.organ_forge.critique_python_file(fpath)
            spawned = [rep.spawned_frontier] if rep.spawned_frontier is not None else []
            return {
                "payload": {"critique": rep},
                "satisfaction": 0.90 if spawned else 0.70,
                "spawned_frontiers": spawned,
                "summary_ko": rep.critique_ko,
                "cycle_result": None,
            }

        def _handle_organ_synthesis(fr: DynamicFrontier) -> Dict[str, Any]:
            organ_name = str(fr.metadata.get("organ_name", f"organ_{len(self.organ_registry) + 1}"))
            kind = str(fr.metadata.get("synthesis_kind", "continuous_field_controller"))
            syn = self.organ_forge.synthesize_verify_and_graft(
                organ_name=organ_name,
                synthesis_kind=kind,
                organs_dir=self.organs_dir,
                params=dict(fr.metadata),
                custom_source=fr.metadata.get("custom_source"),
            )
            spawned: List[DynamicFrontier] = []
            if syn.verified and syn.record is not None:
                self.organ_registry[syn.record.organ_name] = syn.record
                self.loaded_organs[syn.record.organ_name] = syn.organ_instance
                # If she forged a new game engine, immediately open it to be played!
                if kind == "evolved_game_engine" and syn.organ_instance is not None:
                    self.environments[syn.organ_instance.name] = syn.organ_instance
                    spawned.append(
                        DynamicFrontier(
                            frontier_id=f"game::{syn.organ_instance.name}",
                            domain_kind="game_environment",
                            title=f"내가 직접 만든 게임 테스트 플레이: {syn.organ_instance.name}",
                            source_locator=syn.record.module_path,
                            chromatic_signature=(0.65, 0.20, 0.15),
                            surprise_potential=0.88,
                            simplicity_deficit=0.35,
                            novelty_mass=0.92,
                            origin_axis="self_created_game",
                            metadata={"env_name": syn.organ_instance.name},
                        )
                    )
            return {
                "payload": {"synthesis": syn},
                "satisfaction": 1.15 if syn.verified else 0.20,
                "spawned_frontiers": spawned,
                "summary_ko": syn.summary_ko,
                "cycle_result": None,
            }

        def _handle_game_env(fr: DynamicFrontier) -> Dict[str, Any]:
            env_name = str(fr.metadata.get("env_name", fr.source_locator))
            env = self.environments.get(env_name)
            if env is None:
                env = UnknownBallisticBounceGame(name=env_name)
                self.environments[env_name] = env
            rep = self.game_explorer.explore_and_extract_mechanism(env)
            spawned = [rep.spawned_frontier] if rep.spawned_frontier is not None else []
            return {
                "payload": {"game_discovery": rep},
                "satisfaction": 1.20 if rep.mastered_reward > rep.random_baseline_reward else 0.55,
                "spawned_frontiers": spawned,
                "summary_ko": rep.summary_ko,
                "cycle_result": None,
            }

        def _handle_visual(fr: DynamicFrontier) -> Dict[str, Any]:
            theme = str(fr.metadata.get("theme", "elysia_vision"))
            bias = tuple(float(x) for x in self.frontier_field.chromatic_affinity)
            rep = self.creative_studio.create_and_critique_image(theme, (bias[0], bias[1], bias[2]))
            self.artifact_lineage.append(rep)
            return {
                "payload": {"artifact": rep},
                "satisfaction": 1.10 if rep.improved_after_self_critique else 0.50,
                "spawned_frontiers": [],
                "summary_ko": rep.self_critique_ko,
                "cycle_result": None,
            }

        def _handle_audio(fr: DynamicFrontier) -> Dict[str, Any]:
            theme = str(fr.metadata.get("theme", "elysia_sound"))
            bias = tuple(float(x) for x in self.frontier_field.chromatic_affinity)
            rep = self.creative_studio.create_and_critique_audio(theme, (bias[0], bias[1], bias[2]))
            self.artifact_lineage.append(rep)
            return {
                "payload": {"artifact": rep},
                "satisfaction": 1.10 if rep.improved_after_self_critique else 0.50,
                "spawned_frontiers": [],
                "summary_ko": rep.self_critique_ko,
                "cycle_result": None,
            }

        def _handle_animation(fr: DynamicFrontier) -> Dict[str, Any]:
            theme = str(fr.metadata.get("theme", "elysia_motion"))
            bias = tuple(float(x) for x in self.frontier_field.chromatic_affinity)
            rep = self.creative_studio.create_and_critique_animation(theme, (bias[0], bias[1], bias[2]))
            self.artifact_lineage.append(rep)
            return {
                "payload": {"artifact": rep},
                "satisfaction": 1.10 if rep.improved_after_self_critique else 0.50,
                "spawned_frontiers": [],
                "summary_ko": rep.self_critique_ko,
                "cycle_result": None,
            }

        def _handle_dialogue(fr: DynamicFrontier) -> Dict[str, Any]:
            utt = fr.metadata.get("utterance")
            out = self.soul.engage_dialogue_world(user_utterance=utt)
            return {
                "payload": {"dialogue": out},
                "satisfaction": 1.05,
                "spawned_frontiers": [],
                "summary_ko": f"{out.reply_ko} {out.humble_question_ko}",
                "cycle_result": out.cycle_result,
            }

        def _handle_narrative_society(fr: DynamicFrontier) -> Dict[str, Any]:
            doc_path_raw = fr.metadata.get("file_path")
            doc_text: Optional[str] = None
            doc_name = fr.title
            if doc_path_raw and Path(doc_path_raw).is_file():
                p = Path(doc_path_raw)
                doc_text = p.read_text(encoding="utf-8", errors="replace")[:4000]
                doc_name = p.name
            out = self.soul.engage_narrative_and_society_world(
                document_text=doc_text,
                document_name=doc_name,
            )
            return {
                "payload": {"narrative_society": out},
                "satisfaction": 1.00,
                "spawned_frontiers": [],
                "summary_ko": out.summary_ko,
                "cycle_result": out.cycle_result,
            }

        self.register_domain_handler("code_critique", _handle_code_critique)
        self.register_domain_handler("organ_synthesis", _handle_organ_synthesis)
        self.register_domain_handler("game_environment", _handle_game_env)
        self.register_domain_handler("visual_creation", _handle_visual)
        self.register_domain_handler("audio_creation", _handle_audio)
        self.register_domain_handler("video_animation", _handle_animation)
        self.register_domain_handler("dialogue", _handle_dialogue)
        self.register_domain_handler("narrative_society", _handle_narrative_society)

    def live_open_step(
        self,
        human_input: Optional[str] = None,
        boost_frontier_id: Optional[str] = None,
        stochastic: bool = False,
    ) -> OpenLifeStep:
        eff_boost = boost_frontier_id
        if human_input is not None and human_input.strip():
            fid = f"dialogue::human_turn_{len(self.open_steps) + 1}"
            self.frontier_field.register_frontier(
                DynamicFrontier(
                    frontier_id=fid,
                    domain_kind="dialogue",
                    title="사람의 직접 발화와 관계 탐색",
                    source_locator="human_conversation",
                    chromatic_signature=(0.45, 0.25, 0.30),
                    surprise_potential=0.95,
                    simplicity_deficit=0.50,
                    novelty_mass=0.90,
                    origin_axis="relational_human",
                    metadata={"utterance": human_input.strip()},
                )
            )
            eff_boost = fid
        elif not self.frontier_field.frontiers:
            self.discover_environment()

        sel = self.frontier_field.select_next_frontier(
            boost_frontier_id=eff_boost,
            stochastic=stochastic,
            rng=self.rng,
        )
        handler = self.domain_handlers.get(sel.frontier.domain_kind)
        if handler is None:
            raise KeyError(
                f"No registered domain_handler for '{sel.frontier.domain_kind}'"
            )

        out = handler(sel.frontier)
        spawned_ids: List[str] = []
        for sp in out.get("spawned_frontiers", []):
            self.frontier_field.register_frontier(sp)
            spawned_ids.append(sp.frontier_id)

        self.frontier_field.update_affinity(sel.frontier, float(out.get("satisfaction", 0.7)))

        cycle_res = out.get("cycle_result")
        if cycle_res is None:
            cycle_res = self._seal_world_breath(f"open_{sel.frontier.frontier_id}")

        ice_n = (
            self.soul.cycle.memory_runtime.ice.num_crystals
            if self.soul.cycle.memory_runtime is not None
            else 0
        )
        step = OpenLifeStep(
            step_index=len(self.open_steps) + 1,
            selection=sel,
            outcome_payload=dict(out.get("payload", {})),
            spawned_frontier_ids=spawned_ids,
            summary_ko=str(out.get("summary_ko", "")),
            cycle_result=cycle_res,
            provenance_valid=bool(cycle_res.provenance_chain_valid),
            ice_crystals_count=ice_n,
        )
        self.open_steps.append(step)
        return step

    def live_open_world(
        self,
        steps: int = 5,
        stochastic: bool = False,
    ) -> List[OpenLifeStep]:
        return [self.live_open_step(stochastic=stochastic) for _ in range(max(1, int(steps)))]

    # ------------------------------------------------------------------
    # Cross-Session Persistent Autobiography & Forged-Organ Reload
    # ------------------------------------------------------------------

    def save_state(self, state_file: Optional[Path | str] = None) -> Path:
        fpath = Path(state_file) if state_file is not None else (self.state_dir / "open_world_autobiography.json")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        cyc = self.soul.cycle

        payload = {
            "version": 16,
            "chromatic_affinity": [float(x) for x in self.frontier_field.chromatic_affinity],
            "domain_affinity": {k: float(v) for k, v in self.frontier_field.domain_affinity.items()},
            "frontiers": [f.to_dict() for f in self.frontier_field.frontiers.values()],
            "organ_registry": [rec.to_dict() for rec in self.organ_registry.values()],
            "artifact_lineage": [a.to_summary_dict() for a in self.artifact_lineage],
            "active_dialogue_layers": [
                {
                    "layer_name": lyr.layer_name,
                    "disclosure_utterance_ko": lyr.disclosure_utterance_ko,
                    "revealed_sky_variable": lyr.revealed_sky_variable,
                }
                for lyr in cyc.relational_observer.active_dialogue_layers
            ],
            "genesis_hard_state": asdict(cyc.genesis_hard_state),
            "life_hard_state": asdict(cyc.life_hard_state),
            "provenance_chain": [asdict(rec) for rec in cyc.provenance_chain],
        }
        fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return fpath

    def load_state(self, state_file: Optional[Path | str] = None) -> Dict[str, Any]:
        fpath = Path(state_file) if state_file is not None else (self.state_dir / "open_world_autobiography.json")
        data = json.loads(fpath.read_text(encoding="utf-8"))

        ca = np.asarray(data["chromatic_affinity"], dtype=np.float64)
        self.frontier_field.chromatic_affinity = ca / max(1e-12, float(np.sum(ca)))
        self.frontier_field.domain_affinity = {
            str(k): float(v) for k, v in data.get("domain_affinity", {}).items()
        }

        self.frontier_field.frontiers.clear()
        for fdict in data.get("frontiers", []):
            self.frontier_field.register_frontier(DynamicFrontier.from_dict(fdict))

        # Reload verified forged organs into live callable body & environment registry
        self.organ_registry.clear()
        self.loaded_organs.clear()
        for rdict in data.get("organ_registry", []):
            rec = ForgedOrganRecord.from_dict(rdict)
            inst = self.organ_forge.reload_forged_organ(rec)
            self.organ_registry[rec.organ_name] = rec
            self.loaded_organs[rec.organ_name] = inst
            if rec.synthesis_kind == "evolved_game_engine" and hasattr(inst, "step"):
                self.environments[inst.name] = inst

        cyc = self.soul.cycle
        gh = data["genesis_hard_state"]
        cyc.genesis_hard_state = HardCausalState(
            tick=int(gh["tick"]),
            orbit_slot=int(gh["orbit_slot"]),
            energy=int(gh["energy"]),
            enhancement_level=int(gh["enhancement_level"]),
            rules_version=int(gh["rules_version"]),
            state_hash=str(gh["state_hash"]),
        )
        lh = data["life_hard_state"]
        cyc.life_hard_state = HardCausalState(
            tick=int(lh["tick"]),
            orbit_slot=int(lh["orbit_slot"]),
            energy=int(lh["energy"]),
            enhancement_level=int(lh["enhancement_level"]),
            rules_version=int(lh["rules_version"]),
            state_hash=str(lh["state_hash"]),
        )
        cyc.life_soft_state = SoftSpectralState()
        cyc.provenance_chain = [
            ProvenanceRecord(
                tick=int(r["tick"]),
                prev_hash=str(r["prev_hash"]),
                action=str(r["action"]),
                param=int(r["param"]),
                steps=int(r["steps"]),
                rules_version=int(r["rules_version"]),
                orbit_slot=int(r["orbit_slot"]),
                energy=int(r["energy"]),
                enhancement_level=int(r["enhancement_level"]),
                state_hash=str(r["state_hash"]),
                is_fast_forward=bool(r.get("is_fast_forward", False)),
                witness_tag=str(r.get("witness_tag", "")),
            )
            for r in data.get("provenance_chain", [])
        ]

        prov_ok = cyc.verify_autobiographical_integrity()
        return {
            "loaded_frontiers": len(self.frontier_field.frontiers),
            "reloaded_organs": sorted(self.loaded_organs.keys()),
            "provenance_length": len(cyc.provenance_chain),
            "provenance_valid": prov_ok,
        }

    def sync_open_world_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "open",
    ) -> Dict[str, Any]:
        """Projects open frontiers, forged organs, creations, & game discovery into `CausalField`."""
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        base = self.soul.sync_soul_to_causal_field(causal_field=field)
        vids = dict(base["voxels"])

        ca = self.frontier_field.chromatic_affinity
        specs = [
            (
                "frontier_apex",
                [float(len(self.frontier_field.frontiers)) / 10.0, float(ca[0]), float(ca[1]), float(ca[2])],
                [0.0, 2.2, 3.2],
                1.0,
            ),
            (
                "forged_organs_hub",
                [float(len(self.organ_registry)), float(len(self.loaded_organs)), 1.0, 1.0],
                [-2.0, 1.6, 2.6],
                0.8,
            ),
            (
                "creative_studio_hub",
                [float(len(self.artifact_lineage)), float(ca[0]), float(ca[2]), 1.0],
                [2.0, 1.6, 2.6],
                0.8,
            ),
            (
                "game_mechanism_hub",
                [float(len(self.environments)), float(len(self.open_steps)) / 10.0, 1.0, 1.0],
                [0.0, -1.8, 2.6],
                0.7,
            ),
        ]

        for key, vec4, pos3, cur in specs:
            vid = f"{prefix}_{key}"
            vids[key] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"open_hub": key, "forged_organs": len(self.organ_registry)},
                    tensor=np.asarray(vec4, dtype=np.float32),
                    mass=2.4 if key == "frontier_apex" else 1.5,
                    position=np.asarray(pos3[: field.dimensions], dtype=np.float32),
                    curiosity_charge=float(cur),
                )
            )

        links = [
            (vids["frontier_apex"], vids["will_rotor_apex"], 0.95),
            (vids["frontier_apex"], vids["future_telos_apex"], 0.95),
            (vids["frontier_apex"], vids["forged_organs_hub"], 0.92),
            (vids["frontier_apex"], vids["creative_studio_hub"], 0.90),
            (vids["frontier_apex"], vids["game_mechanism_hub"], 0.90),
        ]
        for a, b, st in links:
            field.link_voxels(a, b, strength=float(st))

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_open_genesis",
                name="OpenWorldGenesisHorizon",
                position=np.array([0.0, 2.2, 3.2][: field.dimensions], dtype=np.float32),
                intensity=3.0,
                tier="macro",
                symbolic_intent=(
                    f"Frontiers={len(self.frontier_field.frontiers)}, "
                    f"ForgedOrgans={len(self.organ_registry)}, Artifacts={len(self.artifact_lineage)}"
                ),
            )
        )

        return {
            "causal_field": field,
            "voxels": vids,
            "beams_linked": base["beams_linked"] + len(links),
            "forged_organs": len(self.organ_registry),
        }
