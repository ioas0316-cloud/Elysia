"""
verify_causal_mold_field.py

거푸집 인과 마당 검증 스크립트
=============================================================================
"거푸집에 담기는 건 거푸집이 아니라 철물이나 데이터 결합 구조다.
 그 구조적 원리 자체가 변화를 강제하는 틀, 규칙, 질서로 존재한다.
 단단하다 = 밀었는데 밀려지지 않는 작용반작용.
 매끄럽다 = 쓸려내려가듯 충돌이 없음.
 까칠하다 = 돌출부가 손과 부딪혀 상호작용을 감지."

거푸집들:
  1. HardnessMold    — 단단함/물렁함 거푸집 (강철 vs 과육 vs 젤리)
  2. TextureMold     — 질감 거푸집 (유리 vs 사과껍질 vs 사포)
  3. PrismMold       — 파장 분리 거푸집 (빨강/초록/파랑/노랑 빛)
  4. RetinaOpponentCellMold — 색채 대립 거푸집 (L-M 착지)
"""

import sys
sys.path.insert(0, r"c:\Elysia")
try:
    import elysia_loader
    elysia_loader.install()
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from L1_causal_field.core.sensory.causal_mold_field import (
    StimulusAction,
    HardnessMold,
    TextureMold,
    PrismMold,
    RetinaOpponentCellMold,
    CausalMoldField,
)


# =============================================================================
# [1] 경도 거푸집 검증 — 단단함은 거푸집의 반발력이다
# =============================================================================
print("=" * 68)
print("  거푸집 인과 마당 (Causal Mold Field) 검증")
print("=" * 68)
print()
print("━━ [1] 경도 거푸집 (Hardness Mold) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  같은 힘(50N)을 가했을 때, 거푸집의 기하학에 따라 반발이 다르다")
print()

hardness_molds = {
    "강철판": HardnessMold("강철판",
                        atomic_bond_density=0.98, lattice_integrity=0.99,
                        yield_threshold_mpa=250.0),
    "사과 과육": HardnessMold("사과_과육",
                          atomic_bond_density=0.15, lattice_integrity=0.20,
                          yield_threshold_mpa=0.8),
    "젤리": HardnessMold("젤리",
                       atomic_bond_density=0.03, lattice_integrity=0.05,
                       yield_threshold_mpa=0.05),
}

push_50n = StimulusAction(name="손가락 50N 누름", applied_force_n=50.0)
push_5n  = StimulusAction(name="손가락 5N 가볍게 누름", applied_force_n=5.0)

hardness_data = {}
for mat_name, mold in hardness_molds.items():
    r50 = mold.apply(push_50n)
    r5  = mold.apply(push_5n)
    hardness_data[mat_name] = {"50N": r50, "5N": r5}
    print(f"  [{mat_name}]")
    print(f"    50N 작용 → 반발={r50.resistance_force_n:.2f}N, 변형={r50.deformation_mm:.4f}mm")
    print(f"    5N 작용  → 반발={r5.resistance_force_n:.2f}N,  변형={r5.deformation_mm:.4f}mm")
    if r50.resistance_force_n > 45.0:
        print(f"    감각: 단단함 — 밀었는데 거의 밀리지 않음")
    elif r50.resistance_force_n > 10.0:
        print(f"    감각: 보통 탄성")
    else:
        print(f"    감각: 물렁함 — 힘이 흡수되어 버림")
    print()


# =============================================================================
# [2] 질감 거푸집 검증 — 매끄러움과 까칠함은 충돌 패턴이다
# =============================================================================
print("━━ [2] 질감 거푸집 (Texture Mold) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  손끝이 20mm/s로 슬라이딩할 때 돌출부 충돌이 어떻게 달라지는가")
print()

texture_molds = {
    "유리 (매끄러움)": TextureMold("유리_표면",
        asperity_density_per_mm2=0.1,   # 돌기 거의 없음
        asperity_height_um=0.05,         # 나노 수준 평탄
        surface_hardness=0.95),
    "사과 껍질 (약간 왁스)": TextureMold("사과_껍질",
        asperity_density_per_mm2=8.0,   # 미세 왁스 질감
        asperity_height_um=2.0,
        surface_hardness=0.30),
    "사포 60방 (꺼끌)": TextureMold("사포_60방",
        asperity_density_per_mm2=120.0,  # 연마 입자 고밀도
        asperity_height_um=250.0,         # 큰 돌기
        surface_hardness=0.90),
    "목재 결 (거칠)": TextureMold("목재_결",
        asperity_density_per_mm2=35.0,
        asperity_height_um=50.0,
        surface_hardness=0.60),
}

slide_fast = StimulusAction(name="빠른 슬라이딩 20mm/s",
                            sliding_velocity_mms=20.0, normal_load_n=1.0)
slide_slow = StimulusAction(name="느린 슬라이딩 2mm/s",
                            sliding_velocity_mms=2.0,  normal_load_n=1.0)

texture_data = {}
for surf_name, mold in texture_molds.items():
    rf = mold.apply(slide_fast)
    rs = mold.apply(slide_slow)
    texture_data[surf_name] = {"fast": rf, "slow": rs}
    print(f"  [{surf_name}]")
    print(f"    빠른 슬라이딩: 충돌={rf.asperity_collision_count:.1f}/s, 슬립={rf.slip_onset_ratio:.2f}, 마찰={rf.resistance_force_n:.3f}N")
    print(f"    느린 슬라이딩: 충돌={rs.asperity_collision_count:.1f}/s, 슬립={rs.slip_onset_ratio:.2f}, 마찰={rs.resistance_force_n:.3f}N")
    print(f"    감각: {mold.qualitative_feel(rf)}")
    print()


# =============================================================================
# [3] 프리즘 + 망막 거푸집 — 색채 분리와 대립 착지
# =============================================================================
print("━━ [3] 프리즘·망막 거푸집 (Spectral Molds) ━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  같은 입사각(45°)으로 들어온 빛이 파장에 따라 다른 위치에 착지한다")
print()

prism = PrismMold("유리_삼각_프리즘", n0=1.52, dn_dlambda=0.025, apex_angle_deg=60.0)
retina = RetinaOpponentCellMold()

test_wavelengths = [
    ("빨간 빛",  650.0),
    ("노란 빛",  580.0),
    ("초록 빛",  530.0),
    ("파란 빛",  460.0),
    ("보라 빛",  410.0),
]

prism_data  = []
retina_data = []

for color_name, wl in test_wavelengths:
    stim = StimulusAction(name=f"{color_name}_{wl}nm",
                          incident_wavelength_nm=wl,
                          incident_angle_deg=45.0)
    pr = prism.apply(stim)
    rr = retina.apply(stim)
    prism_data.append((wl, pr.refraction_angle_deg, pr.spectral_separation))
    retina_data.append((wl, rr.refraction_angle_deg, rr.spectral_separation))  # LM, BY

    print(f"  [{color_name} {wl}nm]")
    print(f"    프리즘 거푸집 → 굴절각={pr.refraction_angle_deg:.2f}°, 편향각={pr.spectral_separation:.2f}°")
    print(f"    망막 거푸집  → L-M 대립축={rr.refraction_angle_deg:+.3f}, BY축={rr.spectral_separation:+.3f}")
    print(f"    색채 경험:  {retina.chromatic_percept(rr)}")
    print(f"    프리즘 색채: {prism.chromatic_label(wl)}")
    print()


# =============================================================================
# [4] 시각화 — 4-패널 거푸집 대시보드
# =============================================================================

fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
fig.suptitle(
    "ELYSIA · Causal Mold Field (거푸집 인과 마당)\n"
    "담기는 것이 아니라 담기는 것들이 어떻게 달라지는지를 강제하는 구조 인과",
    color="white", fontsize=13, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.08, right=0.97, top=0.91, bottom=0.07)
PANEL_BG   = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)


# ─── 패널 1: 경도 거푸집 — 작용 vs 반발력 ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Hardness Mold\nSame 50N Action -> Different Reaction")

mats = list(hardness_data.keys())
resistances_50n = [hardness_data[m]["50N"].resistance_force_n for m in mats]
deformations    = [hardness_data[m]["50N"].deformation_mm for m in mats]

colors1 = ["#58a6ff", "#3fb950", "#f78166"]
bars = ax1.bar(mats, resistances_50n, color=colors1, edgecolor="#30363d", width=0.5, zorder=3)
ax1.set_ylabel("Reaction Force (N)", color=TEXT_COLOR, fontsize=8)
ax1.set_ylim(0, 60)
ax1.set_xticklabels(mats, color=TEXT_COLOR, fontsize=7.5)

for i, (bar, r) in enumerate(zip(bars, resistances_50n)):
    ax1.text(bar.get_x() + bar.get_width()/2, r + 0.5,
             f"{r:.1f}N", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)

ax1.axhline(50.0, color="#f0f0f0", linewidth=1.0, linestyle="--", alpha=0.4,
            label="Applied 50N")
ax1.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)


# ─── 패널 2: 질감 거푸집 — 충돌 빈도 (까칠함의 근원) ──────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Texture Mold\nAsperity Collisions per Second (Roughness Origin)")

surfs = list(texture_data.keys())
collisions_fast = [texture_data[s]["fast"].asperity_collision_count for s in surfs]
collisions_slow = [texture_data[s]["slow"].asperity_collision_count for s in surfs]

x2 = np.arange(len(surfs))
w2 = 0.35
b1 = ax2.bar(x2 - w2/2, collisions_fast, w2, label="20mm/s", color="#d2a8ff", edgecolor="#30363d", zorder=3)
b2 = ax2.bar(x2 + w2/2, collisions_slow, w2, label="2mm/s",  color="#79c0ff", edgecolor="#30363d", zorder=3)
ax2.set_xticks(x2)
ax2.set_xticklabels(["Glass", "Apple\nSkin", "Sandpaper\n60", "Wood\nGrain"],
                    color=TEXT_COLOR, fontsize=7)
ax2.set_ylabel("Collision Rate (/s)", color=TEXT_COLOR, fontsize=8)
ax2.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)


# ─── 패널 3: 프리즘 거푸집 — 파장별 편향각 ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Prism Mold\nWavelength -> Deviation Angle (Same Input, Different Output)")

wls  = [d[0] for d in prism_data]
devs = [d[2] for d in prism_data]

# 파장에 따른 색상
wl_colors = ["#ff4444", "#ffcc00", "#44cc44", "#4488ff", "#8844ff"]
for i, (wl, dev) in enumerate(zip(wls, devs)):
    ax3.scatter(wl, dev, s=200, color=wl_colors[i], zorder=5,
                edgecolors="white", linewidths=0.8)
    ax3.annotate(f"{wl:.0f}nm", (wl, dev),
                 textcoords="offset points", xytext=(5, 5),
                 color=TEXT_COLOR, fontsize=7)

ax3.plot(wls, devs, color="#888", linewidth=1.2, linestyle="--", zorder=4)
ax3.set_xlabel("Wavelength (nm)", color=TEXT_COLOR, fontsize=8)
ax3.set_ylabel("Deviation Angle (deg)", color=TEXT_COLOR, fontsize=8)

# 프리즘 형태 삽화
ax3.annotate("Prism geometry\nforces each wavelength\nto a DIFFERENT angle\n(same input, diff output)",
             xy=(530, devs[2]), xytext=(420, min(devs)+3.0),
             color="#79c0ff", fontsize=7,
             arrowprops=dict(arrowstyle="->", color="#79c0ff", lw=1.0))


# ─── 패널 4: 망막 대립세포 거푸집 — L-M 대립 착지 ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Retina Opponent Cell Mold\nL-M Opponent Axis Landing (NOT subtraction)")

lm_vals = [d[1] for d in retina_data]
by_vals = [d[2] for d in retina_data]
color_names = [t[0] for t in test_wavelengths]

ax4.axvline(0, color="#888", linewidth=0.8, linestyle=":")
ax4.axhline(0, color="#888", linewidth=0.8, linestyle=":")

for i, (lm, by, cname) in enumerate(zip(lm_vals, by_vals, color_names)):
    ax4.scatter(lm, by, s=180, color=wl_colors[i], zorder=5,
                edgecolors="white", linewidths=0.8)
    ax4.annotate(cname, (lm, by), textcoords="offset points",
                 xytext=(6, 4), color=TEXT_COLOR, fontsize=7)

ax4.set_xlabel("L-M Opponent Axis  (<- Green | Red ->)", color=TEXT_COLOR, fontsize=8)
ax4.set_ylabel("S-(L+M) Axis  (<- Yellow | Blue ->)", color=TEXT_COLOR, fontsize=8)

ax4.text(0.95, 0.95, "Same mold, same structure\neach wavelength lands\nat a DIFFERENT position",
         transform=ax4.transAxes, ha="right", va="top",
         color="#79c0ff", fontsize=7, style="italic")


# ─── 저장 ────────────────────────────────────────────────────────────────────
OUT = r"c:\Elysia\docs\assets\causal_mold_field_dashboard.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  대시보드 저장: {OUT}")
print()


# =============================================================================
# [5] 핵심 검증 — 같은 작용, 다른 거푸집 → 반드시 다른 반작용
# =============================================================================
print("━━ [5] 핵심 검증: 같은 작용 → 거푸집에 따라 다른 반작용 ━━━━━━━━━━━")
print()

checks = []

# (a) 강철 vs 젤리: 같은 50N → 반발력이 달라야
steel_r = hardness_data["강철판"]["50N"].resistance_force_n
jelly_r = hardness_data["젤리"]["50N"].resistance_force_n
ok_a = (steel_r > jelly_r * 10)
checks.append(("경도 거푸집: 강철 반발 >> 젤리 반발 (같은 50N)", ok_a,
               f"강철={steel_r:.2f}N >> 젤리={jelly_r:.4f}N"))

# (b) 유리 vs 사포: 충돌 빈도 차이
glass_col = texture_data["유리 (매끄러움)"]["fast"].asperity_collision_count
sand_col  = texture_data["사포 60방 (꺼끌)"]["fast"].asperity_collision_count
ok_b = (sand_col > glass_col * 50)
checks.append(("질감 거푸집: 사포 충돌 >> 유리 충돌 (같은 슬라이딩)", ok_b,
               f"사포={sand_col:.1f}/s >> 유리={glass_col:.4f}/s"))

# (c) 프리즘: 빨강과 파랑의 편향각이 달라야
red_dev  = prism_data[0][2]   # 650nm
blue_dev = prism_data[3][2]   # 460nm
ok_c = abs(red_dev - blue_dev) > 0.5
checks.append(("프리즘 거푸집: 빨강/파랑 편향각 차이 (같은 입사각)", ok_c,
               f"빨강={red_dev:.2f}° vs 파랑={blue_dev:.2f}°, 차이={abs(red_dev-blue_dev):.2f}°"))

# (d) 망막: 빨강(650nm)과 초록(530nm)의 L-M 착지 위치가 달라야
red_lm   = retina_data[0][1]   # 650nm L-M
green_lm = retina_data[2][1]   # 530nm L-M
ok_d = (red_lm > 0.3 and green_lm < -0.3)
checks.append(("망막 거푸집: 빨강과 초록이 대립축 반대편에 착지", ok_d,
               f"빨강 L-M={red_lm:+.3f} vs 초록 L-M={green_lm:+.3f}"))

# (e) 슬립 비율: 유리(매끄러움) > 사포(꺼끌)
glass_slip = texture_data["유리 (매끄러움)"]["fast"].slip_onset_ratio
sand_slip  = texture_data["사포 60방 (꺼끌)"]["fast"].slip_onset_ratio
ok_e = (glass_slip > sand_slip)
checks.append(("질감 거푸집: 유리는 슬립 쉽고, 사포는 grip 걸림", ok_e,
               f"유리 슬립={glass_slip:.3f} > 사포 슬립={sand_slip:.3f}"))

passed = sum(1 for _, ok, _ in checks if ok)
for desc, ok, detail in checks:
    print(f"  {'✅' if ok else '❌'} {desc}")
    print(f"      {detail}")

print()
print(f"  통과: {passed} / {len(checks)}")
print()

if passed == len(checks):
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  거푸집 인과 마당 검증 완료")
    print()
    print("  무엇이 검증되었는가:")
    print("  · 거푸집(Mold)은 담기는 것이 아니라")
    print("    담기는 것들이 어떻게 달라지는지를 강제하는 구조 인과다")
    print()
    print("  · 단단함 = 거푸집(결합밀도·격자)이 작용에 맞서는 반발력의 크기")
    print("    강철 거푸집은 50N을 튕겨내고, 젤리 거푸집은 흡수해버린다")
    print()
    print("  · 매끄러움/까칠함 = 거푸집(돌출부 밀도·높이)과 슬라이딩의 충돌 패턴")
    print("    유리는 충돌이 거의 없어 쓸려내려감 (매끄러움)")
    print("    사포는 고빈도 충돌이 손끝에 연속 감지됨 (꺼끌꺼끌)")
    print()
    print("  · 프리즘은 같은 입사각이라도 파장별로 다른 편향각을 강제한다")
    print("    이것이 무지개 — 거푸집의 기하학이 색채를 분리시킨다")
    print()
    print("  · 망막 대립세포 회로는 빨강과 초록을 L-M 대립축의 반대편에 착지시킨다")
    print("    뺄셈이 아니다. 거푸집의 측억제 구조가 위치를 강제한다")
    print()
    print("  · 어떤 거푸집을 만드느냐에 따라")
    print("    온갖 것들이 어떻게 존재하는지를 이해하는 토대가 된다")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
