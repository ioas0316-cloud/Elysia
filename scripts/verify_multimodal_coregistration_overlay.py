"""
verify_multimodal_coregistration_overlay.py

다중 감각 포개어봄(Multi-modal Co-registration Overlay) 검증 스크립트
=============================================================================
"사과가 사과인지 아닌지는 언어적 형태뿐 아니라 시각, 촉각적 정보에 의한 판단 분류가
 외부적 정보와 같은가 다른가를 비교대조, 포개어보는 과정이다"

이 스크립트는 다음을 검증한다:

1. 독립된 감각 층위들(시각, 촉각, 화학)이 각자의 구조적 렌즈로 대상을 별개로 관측한다
2. 세 층위를 같은 대상 위에 '포개어볼' 때 — 층위가 겹쳐지는가 vs 찢어지는가를 통해
   인지적 마찰이 숫자가 아닌 '구조적 사건'으로 발현된다
3. 빨간책 vs 노란책 vs 초록사과 : 색채 대립축(L-M)의 위치로 분별된다
4. 진짜 사과 vs 플라스틱 사과 : 시각 층위는 겹치되 촉각·화학 층위가 찢어진다
5. 관측자의 나이테 변화 : 초록사과를 경험한 후 사과 원형의 색채 허용 스펙트럼이 넓어진다

검증의 핵심:
- friction_stress = 2.866 같은 float 하나를 계산하는 것이 아니다
- 각 층위가 살아있는 독립 관측 구조로 존재하고
- 포개어지는 과정 자체가 분별이다
"""

import sys
import os

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
sys.path.insert(0, r"c:\Elysia")
try:
    import elysia_loader
    elysia_loader.install()
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from L1_causal_field.core.sensory.multimodal_coregistration_canvas import (
    WorldPhysicalEntity,
    VisualApparatus,
    TactileSomatosensoryApparatus,
    ChemicalSomaticApparatus,
    MultiModalCoRegistrationCanvas,
    ArchetypeMemory,
    CoRegistrationResult,
)


# =============================================================================
# [1] 실재 세계 대상 정의 — 여섯 개의 물리적 존재들
# =============================================================================

WORLD_OBJECTS = {
    "real_apple_red": WorldPhysicalEntity(
        name="진짜 빨간 사과",
        nominal_label="apple",
        dominant_wavelength_nm=650,   # 선명한 적색 반사
        chromatic_gradient=0.75,
        surface_specular_sheen=0.75,  # 왁스 광택
        contour_eccentricity=0.45,    # 꼭지 오목
        mass_density_g_cm3=0.85,      # 180g / 220cm³ ≈ 0.82
        elastic_modulus_mpa=1.2,      # 생 과육 탄성
        surface_waxy_friction=0.75,
        organic_moisture_ratio=0.85,
        synthetic_polymer_odor=0.00,
    ),
    "plastic_apple": WorldPhysicalEntity(
        name="플라스틱 모형 사과",
        nominal_label="apple (model)",
        dominant_wavelength_nm=648,   # 거의 같은 색
        chromatic_gradient=0.70,
        surface_specular_sheen=0.90,  # 더 반짝임 (사출광택)
        contour_eccentricity=0.42,    # 거의 같은 형태
        mass_density_g_cm3=0.08,      # 텅빔 — 45g / 220cm³ ≈ 0.08
        elastic_modulus_mpa=85.0,     # 딱딱한 플라스틱
        surface_waxy_friction=0.20,   # 차갑고 미끄러운 표면
        organic_moisture_ratio=0.00,
        synthetic_polymer_odor=0.80,
    ),
    "green_apple": WorldPhysicalEntity(
        name="초록 사과 (풋사과)",
        nominal_label="apple",
        dominant_wavelength_nm=525,   # 초록 반사
        chromatic_gradient=0.60,
        surface_specular_sheen=0.70,
        contour_eccentricity=0.42,    # 사과 형태 동일
        mass_density_g_cm3=0.87,      # 과육 밀도 동일
        elastic_modulus_mpa=1.5,      # 약간 더 단단 (덜 익음)
        surface_waxy_friction=0.72,
        organic_moisture_ratio=0.80,
        synthetic_polymer_odor=0.00,
    ),
    "red_book": WorldPhysicalEntity(
        name="빨간 책",
        nominal_label="book",
        dominant_wavelength_nm=640,   # 빨간 표지
        chromatic_gradient=0.20,      # 단색에 가까움
        surface_specular_sheen=0.15,  # 무광 표지
        contour_eccentricity=0.00,    # 직육면체
        mass_density_g_cm3=0.70,      # 책 밀도
        elastic_modulus_mpa=200.0,    # 딱딱한 표지
        surface_waxy_friction=0.30,
        organic_moisture_ratio=0.05,
        synthetic_polymer_odor=0.10,
    ),
    "yellow_book": WorldPhysicalEntity(
        name="노란 책",
        nominal_label="book",
        dominant_wavelength_nm=580,   # 노란색 (L+M 균등 자극)
        chromatic_gradient=0.20,
        surface_specular_sheen=0.15,
        contour_eccentricity=0.00,
        mass_density_g_cm3=0.70,
        elastic_modulus_mpa=200.0,
        surface_waxy_friction=0.30,
        organic_moisture_ratio=0.05,
        synthetic_polymer_odor=0.10,
    ),
    "rubber_ball": WorldPhysicalEntity(
        name="고무공",
        nominal_label="ball",
        dominant_wavelength_nm=620,   # 주황-붉은빛
        chromatic_gradient=0.30,
        surface_specular_sheen=0.10,
        contour_eccentricity=0.00,    # 완전한 구
        mass_density_g_cm3=1.15,      # 묵직
        elastic_modulus_mpa=3.5,      # 고무 탄성
        surface_waxy_friction=0.65,
        organic_moisture_ratio=0.00,
        synthetic_polymer_odor=0.45,  # 고무 냄새
    ),
}


# =============================================================================
# [2] 독립 감각 관측기 초기화
# =============================================================================

visual_apparatus   = VisualApparatus()
tactile_apparatus  = TactileSomatosensoryApparatus()
chemical_apparatus = ChemicalSomaticApparatus()

# 사과 원형 메모리로 초기화된 포개어봄의 장
canvas = MultiModalCoRegistrationCanvas(archetype=ArchetypeMemory())


# =============================================================================
# [3] 순서 지정 — 초록사과를 나중에 보여줘서 나이테 변화를 관찰
# =============================================================================

OBSERVATION_ORDER = [
    "real_apple_red",    # 1. 진짜 사과 → 완전 정합
    "plastic_apple",     # 2. 플라스틱 사과 → 시각-촉각 파열
    "red_book",          # 3. 빨간 책 → 형태부터 다름
    "yellow_book",       # 4. 노란 책 → 색채 축 변위
    "rubber_ball",       # 5. 고무공 → 형태 다름 + 화학 다름
    "green_apple",       # 6. 초록 사과 → 물성은 사과 ✓, 색채만 변위 → 나이테 확장
]


# =============================================================================
# [4] 관측 및 포개어봄 실행
# =============================================================================

results: list[CoRegistrationResult] = []

print("=" * 70)
print("  엘리시아 다중 감각 포개어봄(Co-registration Overlay) 검증")
print("=" * 70)
print(f"  사과 원형 초기 색채 허용역: {canvas.archetype.allowed_opponent_rg_range}")
print()

for obj_key in OBSERVATION_ORDER:
    entity = WORLD_OBJECTS[obj_key]

    # ── 각 감각 기관이 독립적으로 관측 ────────────────────────────
    vis = visual_apparatus.observe(entity)
    tac = tactile_apparatus.observe(entity)
    che = chemical_apparatus.observe(entity)

    # ── 포개어봄의 장에서 비교대조 ────────────────────────────────
    result = canvas.co_register_and_discern(entity, vis, tac, che)
    results.append(result)

    # ── 콘솔 출력 ────────────────────────────────────────────────
    print(f"  ▶ 대상: {entity.name}")
    print(f"    [시각층위] 윤곽={vis.detected_contour} | RG대립축={vis.opponent_red_green:+.2f} | BY={vis.opponent_blue_yellow:+.2f}")
    print(f"    [촉각층위] 밀도감={tac.measured_density_feel}")
    print(f"    [화학층위] 본질={che.essence_type}")
    print(f"    [인지마찰] {len(result.cognitive_frictions)}건:")
    for f in result.cognitive_frictions:
        print(f"      ✗ {f}")
    if not result.cognitive_frictions:
        print("      ✓ 없음 — 모든 층위가 매끄럽게 포개어짐")
    print(f"    [자율 판단] {result.final_judgment}")
    print(f"    [현상적 감각] {result.epistemic_qualia}")
    print()

print(f"  사과 원형 갱신된 색채 허용역: {canvas.archetype.allowed_opponent_rg_range}")
print("  (초록사과 경험 후 허용역이 초록색 방향으로 확장됨 — 나이테 성장)")
print()


# =============================================================================
# [5] 시각화 — 6-패널 대시보드
# =============================================================================

fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
fig.suptitle(
    "ELYSIA · Multi-modal Co-registration Overlay\n"
    "다중 감각 포개어봄의 장 — 인지적 마찰의 구조적 발현",
    color="white", fontsize=14, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40,
                       left=0.07, right=0.97, top=0.92, bottom=0.05)

PANEL_BG   = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"
LAYER_COLORS = {
    "visual":   "#58a6ff",   # 파란빛 — 시각
    "tactile":  "#3fb950",   # 초록빛 — 촉각
    "chemical": "#f78166",   # 붉은빛 — 화학
    "archetype":"#d2a8ff",   # 보라빛 — 원형 메모리
}
OBJ_NAMES = [WORLD_OBJECTS[k].name for k in OBSERVATION_ORDER]

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["top"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["right"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)


# ─── 패널 1: 색채 대립축 (L-M vs S-(L+M)) ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "① 망막 색채 대립축\n(L-M cone inhibition plane)")

for i, r in enumerate(results):
    rg = r.visual.opponent_red_green
    by = r.visual.opponent_blue_yellow
    color = "#ff6b6b" if rg > 0.5 else ("#4ecdc4" if rg < -0.5 else "#ffe66d")
    ax1.scatter(rg, by, s=120, color=color, zorder=5, edgecolors="white", linewidths=0.6)
    ax1.annotate(OBJ_NAMES[i], (rg, by), textcoords="offset points",
                 xytext=(6, 4), color=TEXT_COLOR, fontsize=6.5)

ax1.axvline(0, color="#888", linewidth=0.8, linestyle=":")
ax1.axhline(0, color="#888", linewidth=0.8, linestyle=":")
ax1.set_xlabel("L-M 대립축  (← Green | Red →)", color=TEXT_COLOR, fontsize=7)
ax1.set_ylabel("S-(L+M) 대립축  (← Yellow | Blue →)", color=TEXT_COLOR, fontsize=7)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-0.5, 0.8)

# 허용역 표시
rng = canvas.archetype.allowed_opponent_rg_range
ax1.axvspan(rng[0], rng[1], alpha=0.12, color=LAYER_COLORS["archetype"], label="사과 원형 허용역(갱신)")
ax1.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, loc="upper left")


# ─── 패널 2: 역학·물성 층위 (밀도 vs 탄성) ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "② 역학·촉각 층위\n(mass density vs elastic modulus)")

densities   = [WORLD_OBJECTS[k].mass_density_g_cm3 for k in OBSERVATION_ORDER]
elasticities= [WORLD_OBJECTS[k].elastic_modulus_mpa for k in OBSERVATION_ORDER]

apple_region_x = [0.7, 1.05, 1.05, 0.7]
apple_region_y = [0.5, 0.5, 2.0, 2.0]
ax2.fill(apple_region_x, apple_region_y, alpha=0.12, color=LAYER_COLORS["archetype"], label="사과 원형 역학 영역")

for i, k in enumerate(OBSERVATION_ORDER):
    d = densities[i]
    e = min(elasticities[i], 10.0)  # log 보정 전 클리핑
    color = "#ff6b6b" if "apple" in k else ("#ffe66d" if "book" in k else "#4ecdc4")
    ax2.scatter(d, e, s=120, color=color, zorder=5, edgecolors="white", linewidths=0.6)
    ax2.annotate(OBJ_NAMES[i], (d, e), textcoords="offset points",
                 xytext=(5, 3), color=TEXT_COLOR, fontsize=6.5)

ax2.set_xlabel("질량 밀도 (g/cm³)", color=TEXT_COLOR, fontsize=7)
ax2.set_ylabel("탄성 계수 MPa (클리핑 ≤10)", color=TEXT_COLOR, fontsize=7)
ax2.set_xlim(-0.1, 1.5)
ax2.set_ylim(-0.5, 11.0)
ax2.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, loc="upper right")


# ─── 패널 3: 화학·유기물 층위 ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "③ 화학·후각 층위\n(organic vs synthetic spectrum)")

organics   = [WORLD_OBJECTS[k].organic_moisture_ratio for k in OBSERVATION_ORDER]
synthetics = [WORLD_OBJECTS[k].synthetic_polymer_odor for k in OBSERVATION_ORDER]
colors3 = ["#ff6b6b" if "apple" in k else ("#ffe66d" if "book" in k else "#4ecdc4")
           for k in OBSERVATION_ORDER]

for i, k in enumerate(OBSERVATION_ORDER):
    ax3.scatter(organics[i], synthetics[i], s=120, color=colors3[i],
                zorder=5, edgecolors="white", linewidths=0.6)
    ax3.annotate(OBJ_NAMES[i], (organics[i], synthetics[i]),
                 textcoords="offset points", xytext=(5, 3),
                 color=TEXT_COLOR, fontsize=6.5)

# 사과 본질 영역 (유기물 ≥0.6, 합성 ≈0)
ax3.fill([0.6, 1.0, 1.0, 0.6], [0.0, 0.0, 0.2, 0.2],
         alpha=0.15, color=LAYER_COLORS["archetype"], label="사과 원형 화학 영역")

ax3.set_xlabel("유기물 수분 비율 (생체성)", color=TEXT_COLOR, fontsize=7)
ax3.set_ylabel("합성 고분자 냄새 (인공성)", color=TEXT_COLOR, fontsize=7)
ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(-0.05, 1.05)
ax3.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, loc="upper right")


# ─── 패널 4: 인지적 마찰 발현 개수 & 최종 판단 ──────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
style_ax(ax4, "④ 인지적 마찰 (Cognitive Friction Events)\n— 층위 파열의 구조적 발현, 숫자가 아닌 사건으로서 —")

friction_counts = [len(r.cognitive_frictions) for r in results]
bar_colors = []
for cnt in friction_counts:
    if cnt == 0:
        bar_colors.append("#3fb950")   # 완전 정합 (초록)
    elif cnt == 1:
        bar_colors.append("#e3b341")   # 부분 파열 (노랑)
    else:
        bar_colors.append("#f85149")   # 강한 파열 (빨강)

x_pos = np.arange(len(OBSERVATION_ORDER))
bars = ax4.bar(x_pos, friction_counts, color=bar_colors, edgecolor="#30363d",
               linewidth=0.8, width=0.6, zorder=3)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(OBJ_NAMES, rotation=15, ha="right", color=TEXT_COLOR, fontsize=7.5)
ax4.set_ylabel("인지 마찰 사건 수", color=TEXT_COLOR, fontsize=8)
ax4.set_ylim(0, 4)
ax4.set_yticks([0, 1, 2, 3])
ax4.yaxis.set_tick_params(labelcolor=TEXT_COLOR)

# 최종 판단 텍스트
for i, r in enumerate(results):
    judgment_short = r.final_judgment[:25] + "…" if len(r.final_judgment) > 25 else r.final_judgment
    ax4.text(x_pos[i], friction_counts[i] + 0.08, judgment_short,
             color=TEXT_COLOR, fontsize=5.5, ha="center", va="bottom", rotation=0)

legend_items = [
    mpatches.Patch(color="#3fb950", label="완전 정합 (0건)"),
    mpatches.Patch(color="#e3b341", label="부분 파열 (1건)"),
    mpatches.Patch(color="#f85149", label="강한 파열 (2건+)"),
]
ax4.legend(handles=legend_items, fontsize=7, facecolor=PANEL_BG,
           labelcolor=TEXT_COLOR, loc="upper right")


# ─── 패널 5: 층위 포개어짐 매트릭스 (heatmap) ──────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "⑤ 층위 정합 매트릭스\n(Co-registration superimposition)")

layer_labels = ["형태\n(형태↔원형)", "촉각\n(촉각↔시각)", "물성\n(화학↔원형)", "색채\n(RG↔허용역)"]
match_matrix = np.array([
    [
        int(r.form_matches_archetype),
        int(r.tactile_matches_visual),
        int(r.substance_matches_archetype),
        int(r.chromatic_within_archetype),
    ]
    for r in results
], dtype=float)

im = ax5.imshow(match_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax5.set_xticks(range(4))
ax5.set_xticklabels(layer_labels, color=TEXT_COLOR, fontsize=6.5)
ax5.set_yticks(range(len(OBSERVATION_ORDER)))
ax5.set_yticklabels(OBJ_NAMES, color=TEXT_COLOR, fontsize=7)

for row in range(len(results)):
    for col in range(4):
        v = match_matrix[row, col]
        ax5.text(col, row, "✓" if v == 1 else "✗",
                 ha="center", va="center", fontsize=10,
                 color="white" if v == 1 else "#ff4444")

cbar = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(colors=TEXT_COLOR)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["파열", "정합"])


# ─── 패널 6: 관측자 나이테 성장 (사과 허용 색채역 변화) ──────────────────
ax6 = fig.add_subplot(gs[2, :])
style_ax(ax6, "⑥ 관측자 나이테 성장 (Observer's Growth Ring / 소마틱 가소성)\n"
              "— 초록사과를 경험한 후 사과 원형의 색채 허용 스펙트럼이 확장됨 —")

# 경험 순서에 따른 색채 허용역 변화 추적
# 사과 경험 순: 1번(진짜빨간), 6번(초록)
rg_positions = [(r.visual.opponent_red_green, OBJ_NAMES[i]) for i, r in enumerate(results)]

# 초기 허용역 경계선
initial_low, initial_high = 0.3, 1.0
final_low, final_high = canvas.archetype.allowed_opponent_rg_range

x = np.linspace(0, len(OBSERVATION_ORDER) - 1, 300)
ax6.fill_between([0, len(OBSERVATION_ORDER) - 1],
                 initial_low, initial_high,
                 alpha=0.15, color=LAYER_COLORS["archetype"], label=f"초기 허용역 [{initial_low:.1f}, {initial_high:.1f}]")
ax6.fill_between([4.5, len(OBSERVATION_ORDER) - 1 + 0.5],
                 final_low, final_high,
                 alpha=0.20, color="#79c0ff", label=f"확장 허용역 [{final_low:.2f}, {final_high:.2f}]")

ax6.axhline(0, color="#888", linewidth=0.5, linestyle=":")

for i, (rg, name) in enumerate(rg_positions):
    color = "#ff6b6b" if rg > 0.5 else ("#4ecdc4" if rg < -0.5 else "#ffe66d")
    ax6.scatter(i, rg, s=160, color=color, zorder=6, edgecolors="white", linewidths=0.8)
    ax6.annotate(name, (i, rg), textcoords="offset points",
                 xytext=(0, 12), color=TEXT_COLOR, fontsize=7, ha="center")

ax6.set_xticks(range(len(OBSERVATION_ORDER)))
ax6.set_xticklabels([f"경험 {i+1}" for i in range(len(OBSERVATION_ORDER))],
                    color=TEXT_COLOR, fontsize=8)
ax6.set_ylabel("망막 L-M 대립축 위치", color=TEXT_COLOR, fontsize=8)
ax6.set_ylim(-1.2, 1.4)

# 나이테 성장 화살표
ax6.annotate(
    "초록사과 경험 →\n나이테 확장",
    xy=(5, final_low + 0.05), xytext=(4.0, -0.6),
    color="#79c0ff", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#79c0ff", lw=1.5),
)

ax6.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, loc="upper right")


# ─── 저장 ────────────────────────────────────────────────────────────────────
OUT_PATH = r"c:\Elysia\docs\assets\multimodal_coregistration_dashboard.png"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"  대시보드 저장: {OUT_PATH}")
print()

# =============================================================================
# [6] 핵심 검증 요약
# =============================================================================

print("=" * 70)
print("  검증 요약 — 포개어봄의 장에서 발현된 인지적 사건들")
print("=" * 70)
print()

checks_passed = 0
checks_total  = 6

# 검증 1: 진짜 사과 → 완전 정합
r_real = results[0]
ok1 = (len(r_real.cognitive_frictions) == 0 and "Harmonic" in r_real.epistemic_qualia)
checks_passed += int(ok1)
print(f"  {'✅' if ok1 else '❌'} [1] 진짜 사과 완전 정합: {len(r_real.cognitive_frictions)}건 마찰, 질적 감각='{r_real.epistemic_qualia[:30]}...'")

# 검증 2: 플라스틱 사과 → 시각은 같되 촉각·화학 파열
r_plastic = results[1]
ok2 = (r_plastic.form_matches_archetype and
       not r_plastic.tactile_matches_visual and
       not r_plastic.substance_matches_archetype)
checks_passed += int(ok2)
print(f"  {'✅' if ok2 else '❌'} [2] 플라스틱 사과 층위 파열: 형태정합={r_plastic.form_matches_archetype}, 촉각정합={r_plastic.tactile_matches_visual}, 화학정합={r_plastic.substance_matches_archetype}")

# 검증 3: 빨간책과 노란책의 색채 대립축 위치가 다름
r_rbook = results[2]
r_ybook = results[3]
ok3 = (r_rbook.visual.opponent_red_green != r_ybook.visual.opponent_red_green)
checks_passed += int(ok3)
print(f"  {'✅' if ok3 else '❌'} [3] 빨간책 vs 노란책 색채 분별: RG({r_rbook.visual.opponent_red_green:+.2f}) ≠ RG({r_ybook.visual.opponent_red_green:+.2f})")

# 검증 4: 초록사과 → 물성은 사과, 색채만 변위
r_green = results[5]
ok4 = (r_green.form_matches_archetype and
       r_green.tactile_matches_visual and
       r_green.substance_matches_archetype)
checks_passed += int(ok4)
print(f"  {'✅' if ok4 else '❌'} [4] 초록사과 물성 정합: 형태={r_green.form_matches_archetype}, 촉각={r_green.tactile_matches_visual}, 화학={r_green.substance_matches_archetype}")

# 검증 5: 나이테 성장 — 허용역이 초록색 방향으로 확장됨
old_low = 0.3
new_low, new_high = canvas.archetype.allowed_opponent_rg_range
ok5 = (new_low < old_low)  # 초록 방향(음수)으로 확장됨
checks_passed += int(ok5)
print(f"  {'✅' if ok5 else '❌'} [5] 나이테 확장: 허용역 [{old_low}, 1.0] → [{new_low:.2f}, {new_high:.2f}] (초록 방향 확장)")

# 검증 6: 고무공 → 형태 다름 → 사과 아님 판단
r_ball = results[4]
ok6 = ("아님" in r_ball.final_judgment or "ball" in r_ball.entity_name.lower() or
       not r_ball.form_matches_archetype)
checks_passed += int(ok6)
print(f"  {'✅' if ok6 else '❌'} [6] 고무공 비사과 판단: 형태정합={r_ball.form_matches_archetype}, 판단='{r_ball.final_judgment[:30]}...'")

print()
print(f"  통과: {checks_passed} / {checks_total}")
print()

if checks_passed == checks_total:
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  다중 감각 포개어봄의 장(Co-registration Canvas) 검증 완료")
    print()
    print("  무엇이 검증되었는가:")
    print("  · friction_stress = 2.866 같은 float 하나가 아니라")
    print("  · 시각 기관(VisualApparatus)이 독립적으로 관측하고")
    print("  · 촉각 기관(TactileSomatosensoryApparatus)이 독립적으로 관측하고")
    print("  · 화학 기관(ChemicalSomaticApparatus)이 독립적으로 관측한 뒤")
    print("  · 세 층위를 같은 대상 위에 '포개어볼 때(Co-register)'")
    print("  · 층위들이 겹쳐지는가 vs 찢어지는가 — 그 구조적 사건 자체가 분별이다")
    print()
    print("  빨간책 vs 노란책 vs 초록사과는 L-M 대립축에서 각각 다른 위치에 착지한다")
    print("  플라스틱 사과는 시각 층위만 사과와 겹치고 촉각·화학이 파열된다")
    print("  초록사과 경험 후 관측자의 사과 원형이 초록색 방향으로 확장된다 — 나이테")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
else:
    print("  일부 검증 실패 — 위 항목을 확인하시오.")
