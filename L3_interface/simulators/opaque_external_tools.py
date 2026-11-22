"""
opaque_external_tools.py

Strictly opaque external tools, physical worlds, and digital protocols with
ZERO semantic or algebraic formula leakage to ELYSIA:
1. `OpaqueParametricCADKernel`: Black-box 3D SDF CAD software exposing only opaque
   operator IDs (`op_k0`..`op_k4`), rendered RGB views (`rgb_iso`), and 3D physical scans.
2. `ContinuousSpeechAcousticWorld`: Time-varying multi-formant vocalic/prosodic PCM
   generator containing exact time-reversed glides and dual-formant contrasts where
   static peak FFT organs fail mathematically.
3. `OpaquePacketProtocol`: Raw byte-frame communication channel supporting distinct
   wire protocols (`xor`, `add_mod_256`) verified strictly via black-box `accepts(bytes)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, map_coordinates


def render_occ_to_rgb(occ: np.ndarray, size: int = 96) -> np.ndarray:
    """
    Pure physics 3D orthographic/oblique Lambertian raycast renderer turning any
    `(G, G, G)` boolean occupancy grid into a sharp `size x size x 3` uint8 RGB image
    with shading & bottom-plane contact shadow, observable by OcularScaleOrgan.
    """
    occ_b = np.asarray(occ, dtype=bool)
    g = occ_b.shape[0]
    if not np.any(occ_b):
        return np.full((size, size, 3), 24, dtype=np.uint8)

    # Smooth pseudo-SDF gradients for natural surface normals
    occ_f = occ_b.astype(np.float32)
    gx, gy, gz = np.gradient(occ_f)

    # Orthographic oblique view camera looking down-forward from (+0.48, -0.58, +0.65)
    cam_dir = np.array([-0.48, 0.58, -0.66], dtype=np.float32)
    cam_dir /= float(np.linalg.norm(cam_dir))
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(cam_dir, up_hint)
    right /= float(np.linalg.norm(right) + 1e-8)
    cam_up = np.cross(right, cam_dir)
    cam_up /= float(np.linalg.norm(cam_up) + 1e-8)

    light_dir = np.array([0.45, -0.45, 0.77], dtype=np.float32)
    light_dir /= float(np.linalg.norm(light_dir))

    # Sample ray grid
    u = np.linspace(-1.35, 1.35, size, dtype=np.float32)
    v = np.linspace(1.35, -1.35, size, dtype=np.float32)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    n_steps = 44
    ts = np.linspace(-1.6, 1.6, n_steps, dtype=np.float32)[:, None, None]

    # World coordinates [-1, 1]^3 -> index coordinates [0, g-1]
    wx = ts * cam_dir[0] + uu[None, :, :] * right[0] + vv[None, :, :] * cam_up[0]
    wy = ts * cam_dir[1] + uu[None, :, :] * right[1] + vv[None, :, :] * cam_up[1]
    wz = ts * cam_dir[2] + uu[None, :, :] * right[2] + vv[None, :, :] * cam_up[2]

    in_box = (
        (wx >= -1.0) & (wx <= 1.0)
        & (wy >= -1.0) & (wy <= 1.0)
        & (wz >= -1.0) & (wz <= 1.0)
    )
    ix = np.clip(np.round((wx + 1.0) * 0.5 * (g - 1)).astype(np.int32), 0, g - 1)
    iy = np.clip(np.round((wy + 1.0) * 0.5 * (g - 1)).astype(np.int32), 0, g - 1)
    iz = np.clip(np.round((wz + 1.0) * 0.5 * (g - 1)).astype(np.int32), 0, g - 1)

    hits = in_box & occ_b[ix, iy, iz]
    any_hit = np.any(hits, axis=0)
    first_step = np.argmax(hits, axis=0)

    rr, cc = np.indices((size, size))
    hx = ix[first_step, rr, cc]
    hy = iy[first_step, rr, cc]
    hz = iz[first_step, rr, cc]

    # Surface normal points toward outward decreasing occupancy (-grad)
    nx = -gx[hx, hy, hz]
    ny = -gy[hx, hy, hz]
    nz = -gz[hx, hy, hz]
    n_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= n_len
    ny /= n_len
    nz /= n_len

    lambert = np.clip(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0.0, 1.0)
    height_tint = 0.5 * (wz[first_step, rr, cc] + 1.0)
    shade = 0.30 + 0.65 * lambert + 0.12 * height_tint

    img = np.zeros((size, size, 3), dtype=np.float32)
    # Neutral cool sky/slate studio background
    bg_grad = np.linspace(46.0, 26.0, size, dtype=np.float32)[:, None]
    img[:, :, 0] = bg_grad
    img[:, :, 1] = bg_grad + 4.0
    img[:, :, 2] = bg_grad + 10.0

    # Shaded terracotta/titanium solid body
    body_r = np.clip(215.0 * shade, 0.0, 255.0)
    body_g = np.clip(180.0 * shade + 18.0 * height_tint, 0.0, 255.0)
    body_b = np.clip(150.0 * shade + 28.0 * height_tint, 0.0, 255.0)
    img[any_hit, 0] = body_r[any_hit]
    img[any_hit, 1] = body_g[any_hit]
    img[any_hit, 2] = body_b[any_hit]

    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def evaluate_occ_physics(occ: np.ndarray) -> Dict[str, Any]:
    """Compute geometric, topological cavity, and static gravity metrics from an occupancy grid."""
    occ_b = np.asarray(occ, dtype=bool)
    g = occ_b.shape[0]
    voxel_vol = (2.0 / g) ** 3
    n_occ = int(np.sum(occ_b))
    if n_occ == 0:
        return {
            "volume": 0.0,
            "bbox_xyz": (0.0, 0.0, 0.0),
            "com_xyz": (0.0, 0.0, 0.0),
            "hollow_cavity_voxels": 0,
            "stable_under_gravity": False,
            "stability_margin": -1.0,
            "min_wall_voxels": 0.0,
        }

    coords = np.linspace(-1.0, 1.0, g, dtype=np.float64)
    idx = np.argwhere(occ_b)
    pts_x = coords[idx[:, 0]]
    pts_y = coords[idx[:, 1]]
    pts_z = coords[idx[:, 2]]

    dx = float((idx[:, 0].max() - idx[:, 0].min() + 1) * (2.0 / g))
    dy = float((idx[:, 1].max() - idx[:, 1].min() + 1) * (2.0 / g))
    dz = float((idx[:, 2].max() - idx[:, 2].min() + 1) * (2.0 / g))
    com = (float(np.mean(pts_x)), float(np.mean(pts_y)), float(np.mean(pts_z)))

    # Detect closed 3D voids AND open-ended slice cavities (through-holes / vessels)
    filled_3d = binary_fill_holes(occ_b)
    cav_3d = int(np.sum(filled_3d & (~occ_b)))
    cav_2d = 0
    for k in range(g):
        sl = occ_b[:, :, k]
        if np.any(sl):
            sl_fill = binary_fill_holes(sl)
            cav_2d += int(np.sum(sl_fill & (~sl)))
    hollow_voxels = max(cav_3d, cav_2d)

    # Bottom support layer stability check (COM projection inside base polygon)
    k_min = int(idx[:, 2].min())
    base_pts = idx[idx[:, 2] == k_min]
    base_xmin = float(coords[base_pts[:, 0]].min())
    base_xmax = float(coords[base_pts[:, 0]].max())
    base_ymin = float(coords[base_pts[:, 1]].min())
    base_ymax = float(coords[base_pts[:, 1]].max())
    margin_x = min(com[0] - base_xmin, base_xmax - com[0])
    margin_y = min(com[1] - base_ymin, base_ymax - com[1])
    margin = float(min(margin_x, margin_y))
    stable = bool(margin >= -1e-6)

    dist_in = distance_transform_edt(occ_b)
    max_half_wall = float(np.max(dist_in))
    return {
        "volume": float(n_occ * voxel_vol),
        "bbox_xyz": (dx, dy, dz),
        "com_xyz": com,
        "hollow_cavity_voxels": hollow_voxels,
        "stable_under_gravity": stable,
        "stability_margin": margin,
        "min_wall_voxels": float(max_half_wall),
    }


@dataclass
class CADShapeObservation:
    """Observable physical/visual output returned by OpaqueParametricCADKernel."""

    shape_id: str
    rgb_iso: np.ndarray
    occ_grid: np.ndarray
    volume: float
    bbox_xyz: Tuple[float, float, float]
    com_xyz: Tuple[float, float, float]
    hollow_cavity_voxels: int
    stable_under_gravity: bool
    stability_margin: float
    min_wall_voxels: float


class OpaqueParametricCADKernel:
    """
    Black-box 3D CAD geometry engine.
    Under `blind_mode=True`, all operators are exposed strictly as opaque IDs
    (`op_k0`..`op_k4`) with zero internal SDF equations leaked.
    """

    def __init__(self, blind_mode: bool = True, grid_size: int = 24) -> None:
        self.blind_mode = bool(blind_mode)
        self.grid_size = int(grid_size)
        c = np.linspace(-1.0, 1.0, self.grid_size, dtype=np.float32)
        self._X, self._Y, self._Z = np.meshgrid(c, c, c, indexing="ij")
        self._sdfs: Dict[str, np.ndarray] = {}
        self._seq = 0

        if self.blind_mode:
            self._op_map = {
                "op_k0": "box_solid",
                "op_k1": "axial_extrude",
                "op_k2": "boolean_union",
                "op_k3": "boolean_diff",
                "op_k4": "uniform_scale",
            }
        else:
            self._op_map = {
                "box_solid": "box_solid",
                "axial_extrude": "axial_extrude",
                "boolean_union": "boolean_union",
                "boolean_diff": "boolean_diff",
                "uniform_scale": "uniform_scale",
            }

    def available_ops(self) -> List[str]:
        return list(self._op_map.keys())

    def _register_sdf(self, sdf: np.ndarray) -> CADShapeObservation:
        sid = f"cad_obj_{self._seq:04d}"
        self._seq += 1
        self._sdfs[sid] = sdf.astype(np.float32)
        occ = sdf <= 0.0
        phys = evaluate_occ_physics(occ)
        rgb = render_occ_to_rgb(occ, size=96)
        return CADShapeObservation(
            shape_id=sid,
            rgb_iso=rgb,
            occ_grid=occ,
            volume=phys["volume"],
            bbox_xyz=phys["bbox_xyz"],
            com_xyz=phys["com_xyz"],
            hollow_cavity_voxels=phys["hollow_cavity_voxels"],
            stable_under_gravity=phys["stable_under_gravity"],
            stability_margin=phys["stability_margin"],
            min_wall_voxels=phys["min_wall_voxels"],
        )

    def _resample_scaled(self, sdf: np.ndarray, sx: float, sy: float, sz: float) -> np.ndarray:
        g = self.grid_size
        mid = 0.5 * (g - 1)
        idx = np.arange(g, dtype=np.float32)
        gx, gy, gz = np.meshgrid(
            mid + (idx - mid) / max(sx, 1e-4),
            mid + (idx - mid) / max(sy, 1e-4),
            mid + (idx - mid) / max(sz, 1e-4),
            indexing="ij",
        )
        return map_coordinates(sdf, [gx, gy, gz], order=1, mode="constant", cval=1.0).astype(np.float32)

    def execute(
        self,
        op_id: str,
        shape_a: Optional[str] = None,
        shape_b: Optional[str] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> CADShapeObservation:
        if op_id not in self._op_map:
            raise ValueError(f"Unknown opaque CAD operator: {op_id}")
        kind = self._op_map[op_id]
        p = params or {}

        if kind == "box_solid":
            hx = float(p.get("hx", 0.35))
            hy = float(p.get("hy", 0.35))
            hz = float(p.get("hz", 0.35))
            ox = float(p.get("ox", 0.0))
            oy = float(p.get("oy", 0.0))
            oz = float(p.get("oz", 0.0))
            sdf = np.maximum(
                np.maximum(np.abs(self._X - ox) - hx, np.abs(self._Y - oy) - hy),
                np.abs(self._Z - oz) - hz,
            )
            return self._register_sdf(sdf)

        if kind == "axial_extrude":
            if shape_a is None or shape_a not in self._sdfs:
                raise ValueError("Missing shape_a for unary operator")
            fz = float(p.get("factor", p.get("fz", 1.5)))
            sdf_new = self._resample_scaled(self._sdfs[shape_a], 1.0, 1.0, fz)
            return self._register_sdf(sdf_new)

        if kind == "uniform_scale":
            if shape_a is None or shape_a not in self._sdfs:
                raise ValueError("Missing shape_a for unary operator")
            s = float(p.get("factor", p.get("scale", 1.25)))
            sdf_new = self._resample_scaled(self._sdfs[shape_a], s, s, s) * s
            return self._register_sdf(sdf_new)

        if kind == "boolean_union":
            if shape_a not in self._sdfs or shape_b not in self._sdfs:
                raise ValueError("Missing shape_a or shape_b for binary operator")
            return self._register_sdf(np.minimum(self._sdfs[shape_a], self._sdfs[shape_b]))

        if kind == "boolean_diff":
            if shape_a not in self._sdfs or shape_b not in self._sdfs:
                raise ValueError("Missing shape_a or shape_b for binary operator")
            return self._register_sdf(np.maximum(self._sdfs[shape_a], -self._sdfs[shape_b]))

        raise RuntimeError(f"Unhandled operator kind: {kind}")


class ContinuousSpeechAcousticWorld:
    """
    Generates realistic 16 kHz time-varying multi-formant vocalic/prosodic PCM waveforms
    with pitch variation and noise:
    - Class 0: Rising formant glide (F1: 500->680 Hz, F2: 1100->1450 Hz)
    - Class 1: Falling formant glide (EXACT time-reversal of Class 0 -> identical global |rfft|)
    - Class 2: Steady mid-F1 (590 Hz) + strong high F2 front vowel (2400 Hz)
    - Class 3: Steady mid-F1 (590 Hz) + low F2 (900 Hz) + double-syllable envelope modulation
    """

    CLASS_NAMES: Tuple[str, ...] = (
        "rising_glide",
        "falling_glide",
        "high_f2_front",
        "double_pulse_low_f2",
    )

    def __init__(self, sample_rate: int = 16000, duration_s: float = 0.25) -> None:
        self.sample_rate = int(sample_rate)
        self.duration_s = float(duration_s)
        self.n_samples = int(round(self.sample_rate * self.duration_s))

    def _synthesize_base_glide(self, f0_hz: float) -> np.ndarray:
        t = np.linspace(0.0, self.duration_s, self.n_samples, endpoint=False, dtype=np.float64)
        u = t / self.duration_s
        f1_inst = 500.0 + 180.0 * u
        f2_inst = 1100.0 + 350.0 * u
        phi1 = 2.0 * np.pi * np.cumsum(f1_inst) / self.sample_rate
        phi2 = 2.0 * np.pi * np.cumsum(f2_inst) / self.sample_rate
        glottal = 0.55 + 0.45 * np.sin(2.0 * np.pi * f0_hz * t)
        sym_win = np.sin(np.pi * u) ** 0.6
        wav = sym_win * glottal * (1.0 * np.sin(phi1) + 0.55 * np.sin(phi2))
        wav /= max(float(np.max(np.abs(wav))), 1e-6)
        return wav.astype(np.float32)

    def _synthesize_clip(
        self, cls_idx: int, rng: np.random.Generator, pitch_shift_hz: float, noise_std: float
    ) -> np.ndarray:
        f0 = 135.0 + pitch_shift_hz + float(rng.uniform(-12.0, 12.0))
        t = np.linspace(0.0, self.duration_s, self.n_samples, endpoint=False, dtype=np.float64)
        u = t / self.duration_s

        if cls_idx == 0:
            wav = self._synthesize_base_glide(f0)
        elif cls_idx == 1:
            # Exact time reversal guarantees identical global magnitude spectrum |rfft|!
            wav = self._synthesize_base_glide(f0)[::-1].copy()
        elif cls_idx == 2:
            glottal = 0.55 + 0.45 * np.sin(2.0 * np.pi * f0 * t)
            sym_win = np.sin(np.pi * u) ** 0.6
            wav = sym_win * glottal * (
                1.0 * np.sin(2.0 * np.pi * 590.0 * t) + 0.85 * np.sin(2.0 * np.pi * 2400.0 * t)
            )
            wav = (wav / max(float(np.max(np.abs(wav))), 1e-6)).astype(np.float32)
        elif cls_idx == 3:
            glottal = 0.55 + 0.45 * np.sin(2.0 * np.pi * f0 * t)
            double_env = np.abs(np.sin(2.0 * np.pi * u)) ** 1.6
            wav = double_env * glottal * (
                1.0 * np.sin(2.0 * np.pi * 590.0 * t) + 0.45 * np.sin(2.0 * np.pi * 900.0 * t)
            )
            wav = (wav / max(float(np.max(np.abs(wav))), 1e-6)).astype(np.float32)
        else:
            raise ValueError(f"Invalid speech class {cls_idx}")

        if noise_std > 0.0:
            wav = wav + rng.normal(0.0, noise_std, size=wav.shape).astype(np.float32)
        return wav.astype(np.float32)

    def sample_dataset(
        self,
        seed: int = 42,
        n_per_class: int = 6,
        pitch_shift_hz: float = 0.0,
        noise_std: float = 0.015,
    ) -> List[Tuple[np.ndarray, int]]:
        rng = np.random.default_rng(seed)
        data: List[Tuple[np.ndarray, int]] = []
        for cls_idx in range(len(self.CLASS_NAMES)):
            for _ in range(n_per_class):
                data.append((self._synthesize_clip(cls_idx, rng, pitch_shift_hz, noise_std), cls_idx))
        return data


class OpaquePacketProtocol:
    """
    Black-box digital communication channel supporting distinct hidden protocols
    (`xor` or `add_mod_256`). Only raw sample frames and boolean `accepts(bytes)` are exposed.
    """

    def __init__(self, variant: str = "xor") -> None:
        if variant not in ("xor", "add_mod_256"):
            raise ValueError(f"Unknown protocol variant: {variant}")
        self._variant = variant
        if variant == "xor":
            self._magic = bytes([0xA5, 0x5A, 0x01])
        else:
            self._magic = bytes([0x7E, 0x3C, 0x02])

    def _encode_internal(self, payload: bytes) -> bytes:
        n = len(payload)
        if not (1 <= n <= 200):
            raise ValueError("Invalid payload length")
        if self._variant == "xor":
            chk = self._magic[2] ^ n
            for b in payload:
                chk ^= b
            return self._magic + bytes([n]) + payload + bytes([chk & 0xFF])
        chk = (self._magic[2] + n + sum(payload)) % 256
        return self._magic + bytes([n]) + payload + bytes([chk])

    def sample_valid_packets(self, n: int = 12, seed: int = 7) -> List[bytes]:
        rng = np.random.default_rng(seed)
        pkts: List[bytes] = []
        for i in range(n):
            plen = int(3 + (i * 3) % 9)
            payload = bytes(int(x) for x in rng.integers(0, 256, size=plen))
            pkts.append(self._encode_internal(payload))
        return pkts

    def accepts(self, packet: bytes) -> bool:
        if len(packet) < 6:
            return False
        if packet[:3] != self._magic:
            return False
        plen = packet[3]
        if len(packet) != 5 + plen:
            return False
        payload = packet[4 : 4 + plen]
        if self._variant == "xor":
            expected = packet[2] ^ plen
            for b in payload:
                expected ^= b
            return packet[-1] == (expected & 0xFF)
        expected = (packet[2] + plen + sum(payload)) % 256
        return packet[-1] == expected

    def extract_payload_oracle(self, packet: bytes) -> Optional[bytes]:
        """Oracle ONLY for external verification tests (never called by inverter)."""
        if not self.accepts(packet):
            return None
        return bytes(packet[4 : 4 + packet[3]])
