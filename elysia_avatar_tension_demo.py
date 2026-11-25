r"""
ELYSIA Boundary-Tension Avatar Physics & Scale-Axis Spectral O(1) Reproducer Demo
================================================================================

Demonstrates and benchmarks:
  1. Multi-joint skeleton-muscle-skin physics (boundary tension gradient ->
     volume-preserving bulge -> proximal load sharing -> obstacle contact
     compliance -> fascia-sliding skin & compression-driven wrinkles).
  2. Scale-Axis Spectral Reclassification (`Cl(3,0)` rotor manifold): solves
     novel boundary regimes once and reproduces repeated/nearby poses in O(1).
  3. Generates a standalone interactive HTML5 Canvas visualizer (`--export-html`).

Usage:
  python c:\Elysia\elysia_avatar_tension_demo.py --benchmark
  python c:\Elysia\elysia_avatar_tension_demo.py --benchmark --export-html c:\Elysia\data\avatar_tension_simulator.html
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment.boundary_tension_avatar_body import (
    ArticulatedTensionChain,
    ScaleSpectralMotorManifold,
)


def run_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Runs 24 realistic multi-joint boundary tension queries (initial novel constraint
    regimes followed by subtle nearby perturbations & repeats), comparing full
    iterative physics vs Cl(3,0) Scale-Axis Spectral O(1) reproduction.
    """
    manifold = ScaleSpectralMotorManifold()
    ref_chain = ArticulatedTensionChain()

    base_regimes: List[Tuple[List[float], List[float], Tuple[float, float], Optional[Tuple[float, float, float]], str]] = [
        ([0.15, 0.20, 0.10], [0.25, 0.30, 0.20], (0.0, 0.0), None, "relaxed_reach"),
        ([0.85, 0.90, 0.45], [0.10, 0.15, 0.25], (0.0, 0.0), None, "strong_arm_curl"),
        ([0.20, 0.10, 0.75], [0.65, 0.75, 0.10], (0.0, 0.0), None, "extend_with_wrist_flex"),
        ([0.75, 0.80, 0.35], [0.15, 0.20, 0.20], (0.0, -7.5), None, "heavy_downward_load"),
        ([0.80, 0.85, 0.30], [0.15, 0.20, 0.20], (0.0, 0.0), (0.36, 0.22, 0.07), "obstacle_contact"),
        ([0.55, 0.65, 0.40], [0.35, 0.35, 0.30], (3.0, 4.0), None, "co_contract_push"),
    ]

    queries: List[Tuple[List[float], List[float], Tuple[float, float], Optional[Tuple[float, float, float]], str]] = []
    # Phase A: initial encounters of all 6 constraint regimes -> expected physics_solve
    queries.extend(base_regimes)
    # Phase B: 18 nearby variations & recalls -> expected Cl(3,0) spectral_reproduction
    rng = np.random.default_rng(42)
    for k in range(18):
        bf, be, bforce, bobs, label = base_regimes[k % len(base_regimes)]
        nf = [float(np.clip(x + rng.uniform(-0.025, 0.025), 0.0, 1.0)) for x in bf]
        ne = [float(np.clip(x + rng.uniform(-0.025, 0.025), 0.0, 1.0)) for x in be]
        nforce = (bforce[0] + float(rng.uniform(-0.2, 0.2)), bforce[1] + float(rng.uniform(-0.2, 0.2)))
        queries.append((nf, ne, nforce, bobs, f"{label}_var_{k}"))

    angle_errors_deg: List[float] = []
    skin_errors_mm: List[float] = []
    physics_times_ms: List[float] = []
    spectral_times_ms: List[float] = []

    for idx, (f_drv, e_drv, ext_f, obs, label) in enumerate(queries):
        t0 = time.perf_counter()
        pose = manifold.evaluate_or_reproduce(f_drv, e_drv, ext_force_xy=ext_f, obstacle=obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        t_ref0 = time.perf_counter()
        gt = ref_chain.solve_to_equilibrium(f_drv, e_drv, ext_force_xy=ext_f, obstacle=obs)
        gt_ms = (time.perf_counter() - t_ref0) * 1000.0
        physics_times_ms.append(gt_ms)

        if pose.source_mode == "spectral_reproduction":
            spectral_times_ms.append(dt_ms)
            ang_err = float(np.mean(np.abs(np.degrees(pose.joint_angles - gt.joint_angles))))
            skin_err_u = float(np.mean(np.linalg.norm(pose.skin.upper_pts - gt.skin.upper_pts, axis=1))) * 1000.0
            skin_err_l = float(np.mean(np.linalg.norm(pose.skin.lower_pts - gt.skin.lower_pts, axis=1))) * 1000.0
            angle_errors_deg.append(ang_err)
            skin_errors_mm.append(0.5 * (skin_err_u + skin_err_l))

        if verbose and idx in (0, 1, 4, 6, 7, 10, 23):
            degs = [round(math.degrees(a), 1) for a in pose.joint_angles]
            b_elbow = pose.muscles[1].flex_bulge
            w_max = float(np.max(pose.skin.upper_wrinkle))
            print(
                f"  [{idx+1:2d}] {label:24s} | mode={pose.source_mode:21s} | "
                f"aff={pose.docking_affinity:.3f} tq={pose.residual_torque:.3f} | "
                f"angles={degs} | elbow_bulge={b_elbow:.3f} | max_wrinkle={w_max:.2f}"
            )

    st = manifold.stats()
    mean_phys_ms = float(np.mean(physics_times_ms)) if physics_times_ms else 1.0
    mean_spec_ms = float(np.mean(spectral_times_ms)) if spectral_times_ms else mean_phys_ms
    speedup = mean_phys_ms / max(1e-6, mean_spec_ms)
    mean_ang_err = float(np.mean(angle_errors_deg)) if angle_errors_deg else 0.0
    mean_skin_err = float(np.mean(skin_errors_mm)) if skin_errors_mm else 0.0

    summary = {
        **st,
        "mean_angle_error_deg": round(mean_ang_err, 3),
        "mean_skin_error_mm": round(mean_skin_err, 3),
        "mean_physics_ms": round(mean_phys_ms, 3),
        "mean_spectral_hit_ms": round(mean_spec_ms, 3),
        "spectral_speedup_x": round(speedup, 2),
    }
    if verbose:
        print("-" * 88)
        print(
            f"  [결과 요약] 총={st['total_queries']}회 | 물리적분={st['physics_solves']}회 | "
            f"O(1) 스펙트럼재현={st['spectral_hits']}회 (적중률 {st['hit_rate']*100:.1f}%) | "
            f"매니폴드={st['manifold_size']}개"
        )
        print(
            f"              평균각도오차={summary['mean_angle_error_deg']:.3f}도 | "
            f"피부윤곽오차={summary['mean_skin_error_mm']:.3f}mm | "
            f"속도={summary['mean_physics_ms']:.3f}ms -> {summary['mean_spectral_hit_ms']:.3f}ms ({summary['spectral_speedup_x']:.1f}x)"
        )
        print("=" * 88)
    return summary


def run_trajectory_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Demonstrates Trajectory-as-a-Frame (`R(tau)`) + Visibility-Culled Micro Hydration:
    solves a curl rollout ONCE, samples 101 continuous phases (`d_tau = 0.01`) in O(1)
    vs ground-truth physics steps, and compares full-body vs elbow-zoomed micro hydration.
    """
    manifold = ScaleSpectralMotorManifold()
    f_drv = [0.65, 0.92, 0.48]
    e_drv = [0.12, 0.08, 0.18]

    # Reference step-by-step rollout (ground truth)
    gt_chain = ArticulatedTensionChain()
    n_steps = 40
    gt_angles = np.zeros((n_steps + 1, gt_chain.n_joints), dtype=np.float64)
    for k in range(1, n_steps + 1):
        gt_chain.step(f_drv, e_drv, dt=0.02)
        gt_angles[k] = gt_chain.angles

    # Capture TrajectoryScaleRotor once, then verify O(1) hit on slight variant
    traj_rotor, cap_mode = manifold.capture_or_get_trajectory_rotor(f_drv, e_drv, steps=n_steps, dt=0.02)
    traj_hit, hit_mode = manifold.capture_or_get_trajectory_rotor([0.64, 0.91, 0.49], [0.12, 0.09, 0.18], steps=n_steps, dt=0.02)

    # Sample all 41 rollout steps via continuous phase tau without stepping
    o1_errs: List[float] = []
    t_samp0 = time.perf_counter()
    for k in range(n_steps + 1):
        tau = k / float(n_steps)
        fr = traj_rotor.sample_at_phase(manifold.chain, tau, resolution_scale=1.0)
        o1_errs.append(float(np.mean(np.abs(np.degrees(fr.joint_angles - gt_angles[k])))))
    sample_41_ms = (time.perf_counter() - t_samp0) * 1000.0

    # Fine resolution & viewport culling comparison at tau = 1.0 (full curl)
    full_coarse = traj_hit.sample_at_phase(manifold.chain, 1.0, viewport=None, resolution_scale=1.0, tear_threshold=1.14)
    full_fine = traj_hit.sample_at_phase(manifold.chain, 1.0, viewport=None, resolution_scale=0.1, tear_threshold=1.14)

    elbow_xy = full_coarse.bone_points[1]
    elbow_vp = (
        float(elbow_xy[0] - 0.16),
        float(elbow_xy[0] + 0.16),
        float(elbow_xy[1] - 0.16),
        float(elbow_xy[1] + 0.16),
    )
    culled_fine = traj_hit.sample_at_phase(manifold.chain, 1.0, viewport=elbow_vp, resolution_scale=0.1, tear_threshold=1.14)

    res = {
        "capture_mode": cap_mode,
        "recall_mode": hit_mode,
        "mean_trajectory_angle_error_deg": round(float(np.mean(o1_errs)), 4),
        "sample_41_frames_ms": round(sample_41_ms, 3),
        "full_coarse_cells": full_coarse.micro_surface.hydrated_micro_cells,
        "full_fine_cells": full_fine.micro_surface.hydrated_micro_cells,
        "culled_fine_cells": culled_fine.micro_surface.hydrated_micro_cells,
        "visible_segments": culled_fine.micro_surface.visible_segments,
        "culled_segments": culled_fine.micro_surface.culled_segments,
        "max_wrinkle": round(float(np.max(full_fine.micro_surface.upper_wrinkle)), 3),
        "max_sheerness": round(float(np.max(full_fine.micro_surface.lower_sheerness)), 3),
        "max_tear": round(float(np.max(full_fine.micro_surface.lower_tear)), 3),
    }
    if verbose:
        print(
            f"  [궤적 로터 R(tau)] 최초={res['capture_mode']} -> 재조회={res['recall_mode']} | "
            f"41프레임 O(1) 복원={res['sample_41_frames_ms']:.2f}ms (전구간 각도오차 {res['mean_trajectory_angle_error_deg']:.3f}°)"
        )
        print(
            f"  [스케일·시야절삭] 전체(scale=1.0)={res['full_coarse_cells']}셀 | "
            f"전체(scale=0.1)={res['full_fine_cells']}셀 -> 팔꿈치 시야확대(scale=0.1)={res['culled_fine_cells']}셀 "
            f"(가시세그먼트={res['visible_segments']}, 절삭={res['culled_segments']})"
        )
        print(
            f"  [연속 미세스펙트럼] 최대주름={res['max_wrinkle']:.3f} | "
            f"최대비침(신장)={res['max_sheerness']:.3f} | 최대찢어짐(임계초과)={res['max_tear']:.3f}"
        )
        print("=" * 88)
    return res


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>ELYSIA · Boundary-Tension Skeleton-Muscle-Skin &amp; Scale-Axis Spectral Reproducer</title>
<style>
  :root {
    --bg: #080d18;
    --panel: #101829;
    --border: #1e2d4d;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --warm: #f97316;
    --wrinkle: #fbbf24;
    --good: #34d399;
  }
  * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); display: flex; min-height: 100vh; }
  .layout { display: grid; grid-template-columns: 360px 1fr; width: 100%; gap: 16px; padding: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  h1 { margin: 0; font-size: 15px; color: var(--accent); letter-spacing: 0.3px; }
  h2 { margin: 8px 0 4px; font-size: 12px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .control { display: grid; grid-template-columns: 112px 1fr 44px; align-items: center; gap: 8px; font-size: 12px; }
  input[type="range"] { accent-color: var(--accent); cursor: pointer; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button { background: #1d4ed8; color: #fff; border: 0; border-radius: 7px; padding: 7px 11px; font-size: 12px; font-weight: 600; cursor: pointer; }
  button.sec { background: #1e293b; color: var(--text); border: 1px solid var(--border); }
  button.active { background: #047857; }
  .stat-box { background: #0b1220; border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-size: 12px; line-height: 1.55; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
  .badge-phys { background: rgba(249,115,22,0.18); color: #fdba74; border: 1px solid rgba(249,115,22,0.4); }
  .badge-spec { background: rgba(52,211,153,0.18); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.4); }
  .canvas-wrap { position: relative; background: radial-gradient(circle at 50% 45%, #111c33 0%, #070b14 100%); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
  canvas { width: 100%; height: 100%; flex: 1; cursor: grab; display: block; }
  canvas:active { cursor: grabbing; }
  .legend { position: absolute; bottom: 12px; left: 14px; font-size: 11px; color: var(--muted); background: rgba(8,13,24,0.78); padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); pointer-events: none; }
</style>
</head>
<body>
<div class="layout">
  <div class="panel">
    <h1>ELYSIA · 경계 장력 근육-골격-피부 아바타</h1>
    <div style="font-size:12px;color:var(--muted);line-height:1.45;">
      관절 각도를 직접 입력하지 않고 <b>경계 장력 차이(ΔT) → 부피 보존 팽창 → 근위부 하중 분산 → 접촉 반력 → 근막 슬라이딩 피부 &amp; 주름</b>으로 구동되며, 수렴 결과를 <b>Cl(3,0) 스케일 스펙트럼</b>에 재분류해 재현합니다.
    </div>

    <h2>어깨 · 팔꿈치 · 손목 근육 구동</h2>
    <div id="sliders"></div>

    <h2>궤적 프레임 (τ) &amp; 스케일 해상도 · 시야 절삭</h2>
    <div class="control">
      <span>궤적 위상 τ</span>
      <input id="tauSlider" type="range" min="0" max="1" step="0.01" value="1.00" />
      <span id="tauVal">1.00</span>
    </div>
    <div class="control">
      <span>찢어짐 임계 λ</span>
      <input id="tearSlider" type="range" min="1.04" max="1.40" step="0.02" value="1.14" />
      <span id="tearVal">1.14</span>
    </div>
    <div class="row">
      <button id="scale1" class="sec">스케일 1.0 (거시)</button>
      <button id="scale025" class="active">스케일 0.25 (중시)</button>
      <button id="scale01" class="sec">스케일 0.1 (미시)</button>
    </div>
    <div class="row">
      <button id="toggleCull" class="sec">시야 절삭 (팔꿈치 카메라): OFF</button>
    </div>

    <h2>외부 접촉 &amp; 스펙트럼 모드</h2>
    <div class="row">
      <button id="toggleSpec" class="active">Cl(3,0) 스펙트럼 O(1) 재현: ON</button>
      <button id="toggleObs" class="sec">접촉 장애물: ON</button>
      <button id="presetCurl" class="sec">강한 굽힘</button>
      <button id="presetRelax" class="sec">이완</button>
    </div>

    <div class="stat-box" id="readout"></div>
    <div class="stat-box" id="cacheStats" style="color:var(--muted)"></div>
  </div>

  <div class="canvas-wrap">
    <canvas id="view" width="900" height="620"></canvas>
    <div class="legend">
      ● <span style="color:#93c5fd">청색 선</span>: 골격 &nbsp;|&nbsp;
      ● <span style="color:#fb7185">적색 결</span>: 부피보존 근육 &nbsp;|&nbsp;
      ● <span style="color:#fbbf24">금색</span>: 압축 주름 &nbsp;|&nbsp;
      ● <span style="color:#38bdf8">투명 청록</span>: 신장 비침 &nbsp;|&nbsp;
      ● <span style="color:#ef4444">적색 섬광</span>: 임계 찢어짐 &nbsp;|&nbsp;
      <b>점선 박스</b>: 가시 시야(미시 하이드레이션)
    </div>
  </div>
</div>

<script>
const SPECS = [
  { name: "어깨 (Shoulder)", len: 0.30, minA: -0.45, maxA: 1.85, rad: 0.048, I: 0.14, damp: 0.85, stiff: 7.5, maxF: 26.0, arm: 0.085 },
  { name: "팔꿈치 (Elbow)",  len: 0.26, minA: -0.20, maxA: 2.35, rad: 0.038, I: 0.09, damp: 0.65, stiff: 7.0, maxF: 24.0, arm: 0.075 },
  { name: "손목 (Wrist)",   len: 0.13, minA: -0.95, maxA: 1.25, rad: 0.024, I: 0.04, damp: 0.40, stiff: 5.5, maxF: 16.0, arm: 0.045 },
];

const state = {
  flexDrives: [0.55, 0.88, 0.45],
  extDrives:  [0.15, 0.10, 0.20],
  tau: 1.0,
  resScale: 0.25,
  tearThresh: 1.14,
  cullOn: false,
  useSpectralCache: true,
  obstacleOn: true,
  obs: { x: 0.36, y: 0.22, r: 0.072 },
  draggingObs: false,
  manifold: [],
  hits: 0,
  solves: 0,
  lastMode: "physics_solve",
  lastAff: 1.0,
};

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function fk(angles) {
  const pts = [[0, 0]];
  const cum = [];
  let phi = 0, x = 0, y = 0;
  for (let j = 0; j < SPECS.length; j++) {
    phi += angles[j];
    cum.push(phi);
    x += SPECS[j].len * Math.cos(phi);
    y += SPECS[j].len * Math.sin(phi);
    pts.push([x, y]);
  }
  return { pts, cum };
}

function encodeRotor(fDrv, eDrv, obsOn, obs) {
  // Exact 8-component Cl(3,0) rotor composition matching Python ScaleSpectralMotorManifold
  const mv = (s, e12, e23, e31) => [s, 0, 0, 0, e12, e23, e31, 0];
  function mul(a, b) {
    const [a0,a1,a2,a3,a12,a23,a31,a123] = a;
    const [b0,b1,b2,b3,b12,b23,b31,b123] = b;
    return [
      a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a23*b23 - a31*b31 - a123*b123,
      a0*b1 + a1*b0 - a2*b12 + a12*b2 + a3*b31 - a31*b3 - a23*b123 - a123*b23,
      a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3 - a31*b123 - a123*b31,
      a0*b3 + a3*b0 - a1*b31 + a31*b1 + a2*b23 - a23*b2 - a12*b123 - a123*b12,
      a0*b12 + a12*b0 + a1*b2 - a2*b1 + a23*b31 - a31*b23 + a3*b123 + a123*b3,
      a0*b23 + a23*b0 + a2*b3 - a3*b2 + a31*b12 - a12*b31 + a1*b123 + a123*b1,
      a0*b31 + a31*b0 + a3*b1 - a1*b3 + a12*b23 - a23*b12 + a2*b123 + a123*b2,
      a0*b123 + a123*b0 + a1*b23 + a23*b1 + a2*b31 + a31*b2 + a3*b12 + a12*b3,
    ];
  }
  function norm(v) {
    const n = Math.hypot(...v) || 1.0;
    return v.map(x => x / n);
  }
  let R = [1,0,0,0,0,0,0,0];
  let coSum = 0;
  for (let j = 0; j < 3; j++) {
    const diff = clamp(fDrv[j] - eDrv[j], -1, 1);
    coSum += 0.5 * (fDrv[j] + eDrv[j]);
    const ha = diff * (Math.PI / 2.6);
    const rj = [Math.cos(ha), 0, 0, 0, j===0?Math.sin(ha):0, j===1?Math.sin(ha):0, j===2?Math.sin(ha):0, 0];
    R = norm(mul(R, rj));
  }
  const fx = obsOn ? obs.x * 0.45 : 0;
  const fy = obsOn ? obs.y * 0.45 : 0;
  R[1] += Math.tanh(fx) * 0.42;
  R[2] += Math.tanh(fy) * 0.42;
  R[3] += (coSum / 3 - 0.35) * 0.28 + (obsOn ? 0.42 : 0);
  R[7] += 0.18 * Math.tanh(fx - fy);
  return norm(R);
}

function dockAffinity(cand, query) {
  // M_cand * M_query~
  const qRev = [query[0], query[1], query[2], query[3], -query[4], -query[5], -query[6], -query[7]];
  const [a0,a1,a2,a3,a12,a23,a31,a123] = cand;
  const [b0,b1,b2,b3,b12,b23,b31,b123] = qRev;
  const d0 = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a23*b23 - a31*b31 - a123*b123;
  const d12 = a0*b12 + a12*b0 + a1*b2 - a2*b1 + a23*b31 - a31*b23 + a3*b123 + a123*b3;
  const d23 = a0*b23 + a23*b0 + a2*b3 - a3*b2 + a31*b12 - a12*b31 + a1*b123 + a123*b1;
  const d31 = a0*b31 + a31*b0 + a3*b1 - a1*b3 + a12*b23 - a23*b12 + a2*b123 + a123*b2;
  const tq = Math.hypot(d12, d23, d31);
  return { aff: clamp(d0 - 0.35 * tq, -1, 1), tq };
}

function solveTrajectoryRollout(fDrv, eDrv, obsOn, obs) {
  const ang = [0, 0, 0], vel = [0, 0, 0], ft = [0, 0, 0], et = [0, 0, 0];
  let contact = [0, 0];
  const dt = 0.02, steps = 40;
  const seq = [{ angles: [0,0,0], flexT: [0,0,0], extT: [0,0,0], contact: [0,0] }];
  for (let step = 0; step < steps; step++) {
    const rawT = [0, 0, 0];
    for (let j = 0; j < 3; j++) {
      const s = SPECS[j];
      const bias = Math.sin(ang[j]) * s.stiff;
      const tf = Math.max(0, fDrv[j] * s.maxF - bias);
      const te = Math.max(0, eDrv[j] * s.maxF + bias);
      ft[j] += (tf - ft[j]) * Math.min(1, (tf > ft[j] ? 14 : 7.5) * dt);
      et[j] += (te - et[j]) * Math.min(1, (te > et[j] ? 14 : 7.5) * dt);
      rawT[j] = (ft[j] - et[j]) * s.arm * (0.7 + 0.3 * Math.cos(ang[j]));
    }
    const { pts } = fk(ang);
    const extT = [0, 0, 0];
    if (obsOn) {
      let bestPen = 0, bestN = [0, 0], bestPt = pts[3], bestSeg = 2;
      for (let j = 0; j < 3; j++) {
        const a = pts[j], b = pts[j+1];
        const abx = b[0]-a[0], aby = b[1]-a[1];
        const t = clamp(((obs.x-a[0])*abx + (obs.y-a[1])*aby) / Math.max(1e-9, abx*abx + aby*aby), 0, 1);
        const cx = a[0] + t*abx, cy = a[1] + t*aby;
        const dx = cx - obs.x, dy = cy - obs.y;
        const dist = Math.hypot(dx, dy);
        const pen = (obs.r + SPECS[j].rad * 0.85) - dist;
        if (pen > bestPen) {
          bestPen = pen;
          bestN = dist > 1e-6 ? [dx/dist, dy/dist] : [0, 1];
          bestPt = [cx, cy];
          bestSeg = j;
        }
      }
      if (bestPen > 0) {
        let vx = 0, vy = 0;
        for (let j = 0; j <= bestSeg; j++) {
          const rx = bestPt[0] - pts[j][0], ry = bestPt[1] - pts[j][1];
          vx += -vel[j] * ry; vy += vel[j] * rx;
        }
        const vn = vx*bestN[0] + vy*bestN[1];
        const fMag = Math.max(0, 180 * bestPen - 14 * vn);
        contact = [bestN[0]*fMag, bestN[1]*fMag];
        for (let j = 0; j < 3; j++) {
          const rx = bestPt[0] - pts[j][0], ry = bestPt[1] - pts[j][1];
          extT[j] = rx * contact[1] - ry * contact[0];
        }
      } else contact = [0, 0];
    }
    const ovf = [0, 0, 0];
    for (let j = 2; j >= 1; j--) {
      const ex = rawT[j] + 0.55 * extT[j];
      if (Math.abs(ex) > 0.15) ovf[j-1] += 0.24 * ex;
    }
    for (let j = 0; j < 3; j++) {
      const s = SPECS[j];
      const tau = rawT[j] + ovf[j] + extT[j] - s.damp * vel[j];
      vel[j] += (tau / s.I) * dt;
      ang[j] += vel[j] * dt;
      if (ang[j] < s.minA) { ang[j] = s.minA; vel[j] = Math.max(0, vel[j]); }
      if (ang[j] > s.maxA) { ang[j] = s.maxA; vel[j] = Math.min(0, vel[j]); }
    }
    seq.push({ angles: [...ang], flexT: [...ft], extT: [...et], contact: [...contact] });
  }
  return seq;
}

function sampleTrajectorySeq(seq, tau) {
  const fIdx = clamp(tau, 0, 1) * (seq.length - 1);
  const i0 = Math.floor(fIdx), i1 = Math.min(seq.length - 1, i0 + 1);
  const w = fIdx - i0;
  const lerpArr = (a, b) => a.map((v, k) => (1 - w)*v + w*b[k]);
  return {
    angles: lerpArr(seq[i0].angles, seq[i1].angles),
    flexT: lerpArr(seq[i0].flexT, seq[i1].flexT),
    extT: lerpArr(seq[i0].extT, seq[i1].extT),
    contact: lerpArr(seq[i0].contact, seq[i1].contact),
  };
}

function buildCulledSkin(angles, ft, et, resScale, tearThresh, vp) {
  const { pts, cum } = fk(angles);
  const NSEG = Math.max(2, Math.min(60, Math.round(6 / resScale)));
  const visSegs = [], culledSegs = [];
  const segPatches = [];
  let cells = 0;

  for (let j = 0; j < 3; j++) {
    const p0 = pts[j], p1 = pts[j+1];
    const s = SPECS[j];
    const pad = s.rad * 2.2;
    const inVp = !vp || !(
      Math.max(p0[0], p1[0]) + pad < vp[0] ||
      Math.min(p0[0], p1[0]) - pad > vp[1] ||
      Math.max(p0[1], p1[1]) + pad < vp[2] ||
      Math.min(p0[1], p1[1]) - pad > vp[3]
    );
    if (!inVp) { culledSegs.push(j); continue; }
    visSegs.push(j);

    const cf = clamp(ft[j] / s.maxF, 0, 1.25);
    const ce = clamp(et[j] / s.maxF, 0, 1.25);
    const bf = 1 / Math.sqrt(clamp(1 - 0.22*cf - 0.14*Math.sin(angles[j]), 0.64, 1.32));
    const be = 1 / Math.sqrt(clamp(1 - 0.22*ce + 0.14*Math.sin(angles[j]), 0.64, 1.32));
    const nxtRad = j < 2 ? SPECS[j+1].rad : s.rad * 0.72;
    const kb = Math.tanh(angles[j]), kn = j < 2 ? Math.tanh(angles[j+1]) : 0;
    const tx0 = Math.cos(cum[j]), ty0 = Math.sin(cum[j]);

    const up = [], lo = [];
    for (let i = 0; i <= NSEG; i++) {
      const t = i / NSEG;
      const cx = (1-t)*p0[0] + t*p1[0], cy = (1-t)*p0[1] + t*p1[1];
      const rInt = (1-t)*s.rad + t*nxtRad;
      const belly = Math.sin(Math.PI * t);
      const rUp = rInt * (1 + 0.55*(bf - 1)*(0.25 + 0.75*belly));
      const rLo = rInt * (1 + 0.55*(be - 1)*(0.25 + 0.75*belly));
      let tx = tx0, ty = ty0;
      if (j > 0 && t < 0.25) {
        const w = t / 0.25;
        tx = (1 - 0.5*(1-w))*tx0 + 0.5*(1-w)*Math.cos(cum[j-1]);
        ty = (1 - 0.5*(1-w))*ty0 + 0.5*(1-w)*Math.sin(cum[j-1]);
      } else if (j < 2 && t > 0.75) {
        const w = (t - 0.75) / 0.25;
        tx = (1 - 0.5*w)*tx0 + 0.5*w*Math.cos(cum[j+1]);
        ty = (1 - 0.5*w)*ty0 + 0.5*w*Math.sin(cum[j+1]);
      }
      const tn = Math.hypot(tx, ty) || 1; tx /= tn; ty /= tn;
      const nx = -ty, ny = tx;
      const sl = 0.012*kb*Math.exp(-3.5*t) - 0.012*kn*Math.exp(-3.5*(1-t)) + 0.004*Math.tanh((ft[j]-et[j])/14)*belly;
      up.push([cx + rUp*nx + sl*tx, cy + rUp*ny + sl*ty]);
      lo.push([cx - rLo*nx - 0.5*sl*tx, cy - rLo*ny - 0.5*sl*ty]);
    }
    const restStep = Math.max(1e-6, Math.hypot(s.len / NSEG, (nxtRad - s.rad) / NSEG));
    const uW = [], lW = [], uS = [], lS = [], uT = [], lT = [];
    for (let i = 0; i < NSEG; i++) {
      const su = Math.hypot(up[i+1][0]-up[i][0], up[i+1][1]-up[i][1]) / restStep;
      const slv = Math.hypot(lo[i+1][0]-lo[i][0], lo[i+1][1]-lo[i][1]) / restStep;
      uW.push(clamp((0.96 - su) / 0.28, 0, 1));
      lW.push(clamp((0.96 - slv) / 0.28, 0, 1));
      uS.push(clamp((su - 1.02) / 0.22, 0, 1));
      lS.push(clamp((slv - 1.02) / 0.22, 0, 1));
      uT.push(clamp((su - tearThresh) / 0.16, 0, 1));
      lT.push(clamp((slv - tearThresh) / 0.16, 0, 1));
    }
    cells += 2 * NSEG;
    segPatches.push({ up, lo, uW, lW, uS, lS, uT, lT });
  }
  return { pts, visSegs, culledSegs, segPatches, cells };
}

function evaluate() {
  const rotor = encodeRotor(state.flexDrives, state.extDrives, state.obstacleOn, state.obs);
  let seq = null;
  if (state.useSpectralCache) {
    const cands = state.manifold.filter(e => e.obsOn === state.obstacleOn);
    if (cands.length > 0) {
      let best = null, bestAff = -2, bestTq = 9;
      for (const e of cands) {
        const { aff, tq } = dockAffinity(e.rotor, rotor);
        if (aff > bestAff) { bestAff = aff; bestTq = tq; best = e; }
      }
      if (best && bestAff >= 0.94 && bestTq <= 0.19) {
        seq = best.seq;
        state.hits++;
        state.lastMode = "spectral_reproduction";
        state.lastAff = bestAff;
      }
    }
  }
  if (!seq) {
    seq = solveTrajectoryRollout(state.flexDrives, state.extDrives, state.obstacleOn, state.obs);
    state.manifold.push({ rotor, obsOn: state.obstacleOn, seq });
    if (state.manifold.length > 120) state.manifold.shift();
    state.solves++;
    state.lastMode = "physics_solve";
    state.lastAff = 1.0;
  }
  return sampleTrajectorySeq(seq, state.tau);
}

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const ORIGIN_X = 200, ORIGIN_Y = 390, SCALE = 690;
const toPx = (x, y) => [ORIGIN_X + x * SCALE, ORIGIN_Y - y * SCALE];
const toWorld = (px, py) => [(px - ORIGIN_X) / SCALE, (ORIGIN_Y - py) / SCALE];

function render() {
  const pose = evaluate();
  const { pts } = fk(pose.angles);
  const vp = state.cullOn
    ? [pts[1][0] - 0.15, pts[1][0] + 0.15, pts[1][1] - 0.15, pts[1][1] + 0.15]
    : null;
  const sk = buildCulledSkin(pose.angles, pose.flexT, pose.extT, state.resScale, state.tearThresh, vp);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(148,163,184,0.08)";
  ctx.lineWidth = 1;
  for (let gx = 0; gx < canvas.width; gx += 45) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, canvas.height); ctx.stroke(); }
  for (let gy = 0; gy < canvas.height; gy += 45) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(canvas.width, gy); ctx.stroke(); }

  if (state.obstacleOn) {
    const [ox, oy] = toPx(state.obs.x, state.obs.y);
    ctx.beginPath();
    ctx.arc(ox, oy, state.obs.r * SCALE, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(100,116,139,0.28)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#94a3b8";
    ctx.stroke();
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px sans-serif";
    ctx.fillText("접촉 장애물 (드래그)", ox - 44, oy + 4);
  }

  let maxW = 0, maxS = 0, maxT = 0;
  for (const p of sk.segPatches) {
    ctx.beginPath();
    for (let i = 0; i < p.up.length; i++) {
      const [x, y] = toPx(p.up[i][0], p.up[i][1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = p.lo.length - 1; i >= 0; i--) {
      const [x, y] = toPx(p.lo[i][0], p.lo[i][1]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(244, 63, 94, 0.22)";
    ctx.fill();

    for (let i = 0; i < p.up.length - 1; i++) {
      const drawEdge = (ptA, ptB, w, sh, tr) => {
        maxW = Math.max(maxW, w); maxS = Math.max(maxS, sh); maxT = Math.max(maxT, tr);
        const [x1, y1] = toPx(ptA[0], ptA[1]);
        const [x2, y2] = toPx(ptB[0], ptB[1]);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        if (tr > 0.05) {
          ctx.lineWidth = 4.5 + 3.0 * tr;
          ctx.strokeStyle = `rgba(239, 68, 68, ${0.65 + 0.35*tr})`;
        } else if (w > 0.12) {
          ctx.lineWidth = 2.5 + 3.5 * w;
          ctx.strokeStyle = `rgba(251, 191, 36, ${0.45 + 0.55*w})`;
        } else if (sh > 0.10) {
          ctx.lineWidth = Math.max(1.2, 2.5 - 1.2 * sh);
          ctx.strokeStyle = `rgba(56, 189, 248, ${0.85 - 0.55*sh})`;
        } else {
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = "#e2e8f0";
        }
        ctx.stroke();
      };
      drawEdge(p.up[i], p.up[i+1], p.uW[i], p.uS[i], p.uT[i]);
      drawEdge(p.lo[i], p.lo[i+1], p.lW[i], p.lS[i], p.lT[i]);
    }
  }

  // Bones & Joints (shown even for culled segments cleanly)
  ctx.beginPath();
  for (let j = 0; j < sk.pts.length; j++) {
    const [bx, by] = toPx(sk.pts[j][0], sk.pts[j][1]);
    if (j === 0) ctx.moveTo(bx, by); else ctx.lineTo(bx, by);
  }
  ctx.strokeStyle = "#60a5fa";
  ctx.lineWidth = 5;
  ctx.stroke();
  for (let j = 0; j < sk.pts.length; j++) {
    const [bx, by] = toPx(sk.pts[j][0], sk.pts[j][1]);
    ctx.beginPath(); ctx.arc(bx, by, j === 3 ? 6 : 8, 0, Math.PI * 2);
    ctx.fillStyle = "#93c5fd"; ctx.fill();
  }

  if (vp) {
    const [vx0, vy1] = toPx(vp[0], vp[3]);
    const [vx1, vy0] = toPx(vp[1], vp[2]);
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.strokeRect(vx0, vy1, vx1 - vx0, vy0 - vy1);
    ctx.fillStyle = "#38bdf8";
    ctx.font = "11px sans-serif";
    ctx.fillText("가시 시야 (나머지 절삭)", vx0 + 6, vy1 + 14);
    ctx.restore();
  }

  const badge = state.lastMode === "spectral_reproduction"
    ? `<span class="badge badge-spec">O(1) Cl(3,0) 궤적 스펙트럼 재현 (aff=${state.lastAff.toFixed(3)})</span>`
    : `<span class="badge badge-phys">전체 궤적 물리 적분 (매니폴드 신규 등록)</span>`;
  const degs = pose.angles.map(a => (a * 180 / Math.PI).toFixed(1) + "°");
  document.getElementById("readout").innerHTML = `
    <div>${badge}</div>
    <div style="margin-top:6px;">위상 τ=<b>${state.tau.toFixed(2)}</b> · 각도: <b>${degs.join(" · ")}</b></div>
    <div>하이드레이션: <b>${sk.cells}셀</b> (스케일=${state.resScale}, 가시=[${sk.visSegs}], 절삭=[${sk.culledSegs}])</div>
    <div>주름=<b>${maxW.toFixed(2)}</b> · 비침=<b>${maxS.toFixed(2)}</b> · 찢어짐=<b>${maxT.toFixed(2)}</b> · 반력=(${pose.contact[0].toFixed(1)}, ${pose.contact[1].toFixed(1)})N</div>
  `;
  const total = state.hits + state.solves;
  document.getElementById("cacheStats").innerHTML =
    `궤적 로터 매니폴드: <b>${state.manifold.length}개</b> | 물리풀이: <b>${state.solves}회</b> · 스펙트럼 재현: <b>${state.hits}회</b> (${total ? ((state.hits/total)*100).toFixed(1) : 0}%)`;
}

// Build Sliders
const slidersEl = document.getElementById("sliders");
SPECS.forEach((s, j) => {
  ["flex", "ext"].forEach(kind => {
    const row = document.createElement("div");
    row.className = "control";
    const lbl = document.createElement("span");
    lbl.textContent = `${s.name.split(" ")[0]} ${kind === "flex" ? "굴근" : "신근"}`;
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = 0; inp.max = 1; inp.step = 0.02;
    inp.value = kind === "flex" ? state.flexDrives[j] : state.extDrives[j];
    const val = document.createElement("span");
    val.textContent = Number(inp.value).toFixed(2);
    inp.addEventListener("input", () => {
      if (kind === "flex") state.flexDrives[j] = Number(inp.value);
      else state.extDrives[j] = Number(inp.value);
      val.textContent = Number(inp.value).toFixed(2);
      render();
    });
    row.append(lbl, inp, val);
    slidersEl.appendChild(row);
  });
});

document.getElementById("tauSlider").addEventListener("input", (e) => {
  state.tau = Number(e.target.value);
  document.getElementById("tauVal").textContent = state.tau.toFixed(2);
  render();
});
document.getElementById("tearSlider").addEventListener("input", (e) => {
  state.tearThresh = Number(e.target.value);
  document.getElementById("tearVal").textContent = state.tearThresh.toFixed(2);
  render();
});
[["scale1", 1.0], ["scale025", 0.25], ["scale01", 0.1]].forEach(([id, sc]) => {
  document.getElementById(id).addEventListener("click", () => {
    state.resScale = sc;
    ["scale1", "scale025", "scale01"].forEach(k => document.getElementById(k).className = (k === id ? "active" : "sec"));
    render();
  });
});
document.getElementById("toggleCull").addEventListener("click", (e) => {
  state.cullOn = !state.cullOn;
  e.target.textContent = `시야 절삭 (팔꿈치 카메라): ${state.cullOn ? "ON" : "OFF"}`;
  e.target.className = state.cullOn ? "active" : "sec";
  render();
});

document.getElementById("toggleSpec").addEventListener("click", (e) => {
  state.useSpectralCache = !state.useSpectralCache;
  e.target.textContent = `Cl(3,0) 스펙트럼 O(1) 재현: ${state.useSpectralCache ? "ON" : "OFF"}`;
  e.target.classList.toggle("active", state.useSpectralCache);
  render();
});
document.getElementById("toggleObs").addEventListener("click", (e) => {
  state.obstacleOn = !state.obstacleOn;
  e.target.textContent = `접촉 장애물: ${state.obstacleOn ? "ON" : "OFF"}`;
  render();
});
document.getElementById("presetCurl").addEventListener("click", () => {
  state.flexDrives = [0.75, 0.92, 0.55]; state.extDrives = [0.10, 0.08, 0.20];
  document.querySelectorAll("#sliders input").forEach((el, idx) => {
    const j = Math.floor(idx / 2), k = idx % 2;
    el.value = k === 0 ? state.flexDrives[j] : state.extDrives[j];
    el.nextElementSibling.textContent = Number(el.value).toFixed(2);
  });
  render();
});
document.getElementById("presetRelax").addEventListener("click", () => {
  state.flexDrives = [0.15, 0.15, 0.10]; state.extDrives = [0.25, 0.30, 0.20];
  document.querySelectorAll("#sliders input").forEach((el, idx) => {
    const j = Math.floor(idx / 2), k = idx % 2;
    el.value = k === 0 ? state.flexDrives[j] : state.extDrives[j];
    el.nextElementSibling.textContent = Number(el.value).toFixed(2);
  });
  render();
});

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvas.width / rect.width);
  const py = (e.clientY - rect.top) * (canvas.height / rect.height);
  const [wx, wy] = toWorld(px, py);
  if (Math.hypot(wx - state.obs.x, wy - state.obs.y) <= state.obs.r + 0.04) state.draggingObs = true;
});
window.addEventListener("mousemove", (e) => {
  if (!state.draggingObs) return;
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvas.width / rect.width);
  const py = (e.clientY - rect.top) * (canvas.height / rect.height);
  const [wx, wy] = toWorld(px, py);
  state.obs.x = clamp(wx, 0.05, 0.72);
  state.obs.y = clamp(wy, -0.35, 0.55);
  render();
});
window.addEventListener("mouseup", () => { state.draggingObs = false; });

render();
</script>
</body>
</html>
"""


def export_html(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_HTML_TEMPLATE, encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ELYSIA Boundary-Tension Avatar Physics & Spectral O(1) Demo")
    parser.add_argument("--benchmark", action="store_true", help="Run 24-query Physics vs Cl(3,0) Spectral benchmark")
    parser.add_argument(
        "--export-html",
        default="",
        help="Path to export interactive HTML5 Canvas visualizer",
    )
    args = parser.parse_args(argv)

    print("=" * 88)
    print("  ELYSIA — BOUNDARY-TENSION AVATAR BODY & Cl(3,0) SPECTRAL REPRODUCER")
    print("=" * 88)
    if args.benchmark or not args.export_html:
        run_benchmark(verbose=True)
        run_trajectory_benchmark(verbose=True)
    if args.export_html:
        out = export_html(args.export_html)
        print(f"  [Interactive HTML Exported] {out.resolve()}")
        print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
