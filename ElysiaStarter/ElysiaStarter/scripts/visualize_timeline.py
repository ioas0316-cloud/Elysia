import pygame, sys, argparse, os, math, time
import numpy as np
import yaml

# Ensure Starter package root is on sys.path (??ElysiaStarter/ElysiaStarter)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from core.cell_world import CellWorld
from core.biome import classify_biome


def load_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(here, 'config', 'runtime.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


SHOW_LEGEND = True
SHOW_ATTENTION = True
TOPK = 12
# Human Perspective Mode toggles
HUMAN_VIEW = False   # ?щ엺 愿李?紐⑤뱶 湲곕낯 OFF (H濡??좉?)
SHOW_OUTLINE = True  # 吏???ㅺ낸???쒖떆 (G濡??좉?)
SHOW_AETHER = False  # ?먰뀒瑜??ㅻ쾭?덉씠 ?좉?
SHOW_MANA = False    # 留덈굹 ?ㅻ쾭?덉씠 ?좉?
SHOW_ENTITIES = True # ?앸챸 ?좊떅 ?뚮뜑 ?좉? (E濡??좉?)

# default tile size for attention rings (screen space)
TILE_PX = 32

COL = {
    "bg": (10, 10, 12),
    "legend": (255, 255, 255),
    "focus": (255, 255, 255),
    "beam": (255, 80, 80),
}

# Visual toggles/palette
SHOW_ECOLOGY = False  # O: 동물/식물 표시
PALETTE_CYCLE = ['pastel', 'vivid', 'mono']
PALETTE_INDEX = 0     # P: 팔레트 순환
SHOW_SPEECH = True    # V: 말풍선/대화선 표시


def color_from_score(s: float):
    r = int(255 * s)
    g = int(80 * (1.0 - abs(0.5 - s) * 2))
    b = int(255 * (1.0 - s))
    return (r, g, b)


def get_kr_font(size: int) -> pygame.font.Font:
    # ?쒓? ?쒖떆 媛?ν븳 ?고듃瑜??곗꽑 ?먯깋 (Windows ?곗꽑)
    candidates = [
        'malgungothic', 'Malgun Gothic', 'NanumGothic', 'NotoSansCJKkr',
        'gulim', 'dotum', 'AppleGothic', 'sans-serif'
    ]
    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont('sans-serif', size)


def draw_legend(screen, font) -> int:
    # ?쒓뎅??踰붾? (?숈쟻?쇰줈 ?믪씠 怨꾩궛)
    items = [
        ((0, 255, 0),    '활성 타일(계산중)'),
        ((0, 0, 0),      '비활성(암전/LOD)'),
        ((255, 255, 255),'포커스(관찰자)'),
        ((255, 0, 0),    '브랜치(시간 노드)'),
        ((160, 32, 240), '동기화 영역'),
        ((255, 255, 0),  '이벤트'),
        ((60, 200, 255), 'M: 마나 표시'),
        ((80, 180, 255), 'N: 에테르 표시'),
    ]
    pad = 10
    line_h = max(18, font.get_height() + 2)
    h = pad + len(items) * line_h + pad
    w = 320
    tmp_surfs = [font.render(label, True, (220, 220, 230)) for _, label in items]
    max_text_w = max(s.get_width() for s in tmp_surfs) if tmp_surfs else 0
    w = max(w, 8 + 16 + 8 + max_text_w + pad)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    y = pad
    for (color, label), surf in zip(items, tmp_surfs):
        pygame.draw.rect(panel, color, pygame.Rect(8, y + (line_h - 16) // 2, 16, 16))
        panel.blit(surf, (8 + 16 + 8, y + (line_h - surf.get_height()) // 2))
        y += line_h
    # ?꾩튂: 醫뚯긽???щ갚 10
    screen.blit(panel, (10, 10))
    return h


def draw_help(screen, font, y_start: int) -> int:
    # 議곗옉踰??댄똻 (?좉?: H)
    lines = [
        '議곗옉踰?,
        '?대룞: 諛⑺뼢??/ WASD',
        '以? 留덉슦????而ㅼ꽌 湲곗?), Z/X=異뺤냼/?뺣?, F=留욎땄, 0=湲곕낯',
        '諛곗냽: ]/[ ?먮뒗 1/2/3/4 (1x/2x/4x/8x), Space=?쇱떆?뺤?/?ш컻',
        '留덉슦?? 醫뚰겢由??쒕옒洹?留덈굹 ?묒쭛, ?고겢由??쒕옒洹?遺꾩궛, ??以?,
        '以묓겢由??쒕옒洹? ?붾㈃ ?대룞(??',
        '?좉?: L=踰붾?, H=?꾩?留? Shift+H=?щ엺 愿李?紐⑤뱶, G=?ㅺ낸??,
        '?ㅻ쾭?덉씠: N=?먰뀒瑜? M=留덈굹, Shift+A=?댄뀗?? E=?앸챸 ?좊떅',
        '醫낅즺: Q',
    ]
    pad = 10
    line_h = max(18, font.get_height() + 2)
    h = pad + len(lines) * line_h + pad
    # ?숈쟻 ??怨꾩궛
    surfs = [font.render(t, True, (220, 220, 230)) for t in lines]
    w = max(320, max(s.get_width() for s in surfs) + 20)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    y = pad
    for surf in surfs:
        panel.blit(surf, (10, y))
        y += line_h
    # ?붾㈃ ?섎떒???섏? ?딅룄濡?蹂댁젙
    screen_h = screen.get_height()
    y_pos = y_start + 10
    if y_pos + h > screen_h - 10:
        y_pos = max(10, screen_h - h - 10)
    screen.blit(panel, (10, y_pos))
    return h


def compute_attention_scores(tiles_w, tiles_h, focus_px, scale_x, scale_y, tile_w, tile_h, active_mask=None, focus_radius_tiles=3):
    fx, fy = focus_px
    scores = np.zeros((tiles_h, tiles_w), dtype=np.float32)
    for ty in range(tiles_h):
        for tx in range(tiles_w):
            cx = int(tx * tile_w * scale_x + (tile_w * scale_x) // 2)
            cy = int(ty * tile_h * scale_y + (tile_h * scale_y) // 2)
            dist = math.hypot(cx - fx, cy - fy) / (focus_radius_tiles * tile_w * scale_x + 1e-6)
            vis = max(0.0, 1.0 - dist)
            active = 0.2
            if active_mask is not None and 0 <= ty < active_mask.shape[0] and 0 <= tx < active_mask.shape[1]:
                active = 0.6 if active_mask[ty, tx] else 0.2
            scores[ty, tx] = max(0.0, min(1.0, 0.5 * vis + active))
    return scores


def draw_attention(screen, scores, focus_px, scale_x, scale_y, tile_w, tile_h, focus_radius_tiles):
    # heat rectangles per tile
    tiles_h, tiles_w = scores.shape
    for ty in range(tiles_h):
        for tx in range(tiles_w):
            s = float(scores[ty, tx])
            color = color_from_score(s)
            rx = int(tx * tile_w * scale_x)
            ry = int(ty * tile_h * scale_y)
            rw = int(tile_w * scale_x)
            rh = int(tile_h * scale_y)
            rect = pygame.Rect(rx, ry, rw, rh)
            pygame.draw.rect(screen, color, rect, 0)
    # focus ring
    pygame.draw.circle(screen, COL["focus"], focus_px, int(focus_radius_tiles * tile_w * scale_x), 1)


def _norm_field(x):
    if getattr(x, "shape", None) is None:
        return 0
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def render_human_view(screen, world, scaled_w, scaled_h, px_off_x, px_off_y):
    # Base color synthesis: R=遺??쒕룞(?⑤룄+留덈굹), G=???덉젙(?듬룄+?먰뀒瑜?, B=臾??吏?+??
    temp_n = _norm_field(world.temp)
    moist_n = _norm_field(world.moisture)
    mana_n = _norm_field(getattr(world, "mana", 0))
    aeth_n = _norm_field(getattr(world, "aether", 0))
    h_n = _norm_field(world.height)

    R = np.clip(0.65 * temp_n + 0.45 * mana_n, 0.0, 1.0)
    G = np.clip(0.55 * moist_n * (1.0 - np.abs(h_n - 0.45) * 1.5) + 0.25 * aeth_n, 0.0, 1.0)
    B = np.clip(0.70 * (1.0 - h_n) * moist_n, 0.0, 1.0)

    rgb = (np.stack([R, G, B], axis=-1) * 255).astype(np.uint8)  # (H,W,3)
    base = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    base = pygame.transform.smoothscale(base, (scaled_w, scaled_h))
    screen.blit(base, (px_off_x, px_off_y))

    # Night city lights (gold dots where mana+aether dense and height>0.35)
    if bool(getattr(world, "is_night", False)):
        lights = (0.6 * mana_n + 0.4 * aeth_n) * (h_n > 0.35)
        ys, xs = np.where(lights > 0.55)
        step = max(1, int(6 * world.w / max(1, scaled_w)))
        for y, x in zip(ys[::step], xs[::step]):
            sx = px_off_x + int(x * scaled_w / world.w)
            sy = px_off_y + int(y * scaled_h / world.h)
            r = max(1, int(2 + 2 * lights[y, x]))
            pygame.draw.circle(screen, (250, 210, 120), (sx, sy), r, 0)

    # Simple outline using gradient magnitude (no OpenCV dependency)
    if SHOW_OUTLINE:
        gy, gx = np.gradient(h_n)
        mag = np.sqrt(gx * gx + gy * gy)
        edges = (255.0 * np.clip(mag / (mag.max() + 1e-6), 0.0, 1.0)).astype(np.uint8)
        edge_rgb = np.dstack([edges, edges, edges])
        edge_s = pygame.surfarray.make_surface(edge_rgb.swapaxes(0, 1))
        edge_s = pygame.transform.smoothscale(edge_s, (scaled_w, scaled_h))
        edge_s.set_alpha(60)
        screen.blit(edge_s, (px_off_x, px_off_y))


def _line_points(x0, y0, x1, y1, step=3):
    dx = x1 - x0; dy = y1 - y0
    dist = max(1.0, math.hypot(dx, dy))
    n = int(dist // step) + 1
    for i in range(n + 1):
        t = 0.0 if n == 0 else i / n
        yield int(x0 + dx * t), int(y0 + dy * t)


def _apply_brush(world, kind, cx, cy, r, delta):
    H, W = world.mana.shape
    y0 = max(0, cy - r); y1 = min(H, cy + r)
    x0 = max(0, cx - r); x1 = min(W, cx + r)
    if y1 <= y0 or x1 <= x0:
        return
    if kind == 'concentrate':
        world.mana[y0:y1, x0:x1] = np.clip(world.mana[y0:y1, x0:x1] + delta, 0.0, 1.0)
    elif kind == 'disperse':
        region = world.mana[y0:y1, x0:x1]
        if region.size:
            region[...] = float(region.mean()) * (1.0 - delta)


def _draw_pulse(screen, x, y, base_r, color=(255, 255, 255)):
    r = int(base_r + 2 * math.sin(time.time() * 10.0))
    pygame.draw.circle(screen, color, (x, y), max(1, r), 1)


def _draw_rainbow_ring(screen, x, y, radius):
    for i in range(6):
        angle = i / 6.0
        # simple rainbow palette
        c = [
            (255, 80, 80),
            (255, 160, 60),
            (240, 220, 60),
            (80, 200, 120),
            (80, 160, 255),
            (180, 100, 255),
        ][i]
        pygame.draw.circle(screen, c, (x, y), int(radius * (0.7 + 0.3 * angle)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legend', choices=['on', 'off'], default='on')
    ap.add_argument('--size', type=int, default=896, help='?덈룄????蹂 ?ш린(px), ?뺤궗媛곹삎')
    args = ap.parse_args()

    # Use module-level toggles within this function
    global SHOW_AETHER, SHOW_MANA, HUMAN_VIEW, SHOW_OUTLINE, SHOW_ENTITIES, SHOW_ECOLOGY, PALETTE_INDEX, PALETTE_CYCLE, SHOW_SPEECH

    cfg = load_cfg()
    W, H = cfg.get('world', {}).get('grid', [256, 256])
    qoe = cfg.get('qoe', {})
    tile_w, tile_h = qoe.get('tile', [32, 32])
    radius = int(qoe.get('focus_radius_tiles', 2))
    budget = int(qoe.get('budget_tiles_per_step', 16))

    pygame.init()
    VIEW = int(args.size)
    screen = pygame.display.set_mode((VIEW, VIEW))
    clock = pygame.time.Clock()
    font = get_kr_font(16)

    # live world for movement
    world = CellWorld(W, H)
    # Inject symbolic context (HUMANITY_AND_LANGUAGE_V1)
    world.inject_context({
        "context_id": "HUMANITY_AND_LANGUAGE_V1",
        "concept_nodes": {
            "HANGUL": {
                "type": "symbolic_language",
                "semantic_weight": 0.92,
                "associations": ["communication", "warmth", "connection"],
                "visual_pattern": "curved lines, balanced symmetry",
                "emotional_charge": 0.88
            },
            "FIRE": {
                "type": "elemental_symbol",
                "semantic_weight": 0.85,
                "associations": ["memory", "gathering", "life"],
                "visual_pattern": "oscillating orange flux",
                "emotional_charge": 0.94
            },
            "HOME": {
                "type": "cultural_node",
                "semantic_weight": 0.9,
                "associations": ["safety", "growth", "family"],
                "visual_pattern": "nested squares",
                "emotional_charge": 0.87
            }
        },
        "macro_intent": {
            "theme": "EMERGENCE_OF_LANGUAGE_AND_EMPATHY",
            "goal": "encourage cooperative communication behaviors",
            "preferred_patterns": ["symbols", "speech", "emotional reciprocity"],
            "avoid_patterns": ["isolation", "entropy"]
        },
        "observational_hint": {
            "priority": 0.9,
            "focus_fields": ["human_settlements", "fire_sites", "acoustic_events"],
            "temporal_bias": "evening",
            "spatial_bias": "dense_social_areas"
        },
        "creator_signature": "E.L.Y.S.I.A._PARENT_CORE"
    })
    biome = classify_biome(world.height, world.moisture, world.temp)

    # focus starts at center
    focus_x, focus_y = W // 2, H // 2
    show_legend = (args.legend == 'on')
    show_help = False
    show_attention = SHOW_ATTENTION
    running = True
    # zoom and time acceleration
    scale = 1.0
    min_scale, max_scale = 0.4, 6.0
    base_tps = 15.0  # baseline steps per second
    tps = base_tps
    acc = 0.0
    paused = False
    # pan state (pixels)
    pan_x, pan_y = 0, 0
    cam_x, cam_y = 0.0, 0.0  # smoothed camera offset
    panning = False
    pan_last = (0, 0)
    # gesture stream state
    gest_active = False
    last_world_xy = None
    brush_hold_t = 0.0
    fx_events = []  # visual effects (burst rings)

    # settlements (campfire/huts) minimal state
    settlements = []  # {cx,cy,wood,food,huts,comfort,fire,pop}
    daylen = 1200
    world_ticks = 0
    settlements.append({'cx': float(W//2), 'cy': float(H//2), 'wood': 0, 'food': 20, 'huts': 0, 'comfort': 0.5, 'fire': False, 'pop': 50, 'carriers': []})

    def _best_resource_target(cx, cy, res_map, radius=40, stride=6):
        cx = int(np.clip(cx, 0, W-1)); cy = int(np.clip(cy, 0, H-1))
        x0 = max(0, cx - radius); x1 = min(W, cx + radius)
        y0 = max(0, cy - radius); y1 = min(H, cy + radius)
        best_v = -1.0; best = (cx, cy)
        for yy in range(y0, y1, max(1, stride)):
            row = res_map[yy]
            for xx in range(x0, x1, max(1, stride)):
                v = float(row[xx])
                if v > best_v:
                    best_v = v; best = (xx, yy)
        return best

    def ensure_carriers(S, max_count=20):
        # keep lightweight visual haulers tied to pop size
        target = int(min(max_count, max(2, S.get('pop', 0) // 10)))
        C = S['carriers']
        while len(C) < target:
            # alternate wood/food
            kind = 'wood' if (len(C) % 2 == 0) else 'food'
            C.append({'x': S['cx'], 'y': S['cy'], 'tx': S['cx'], 'ty': S['cy'], 'state': 'to_resource', 'kind': kind, 'load': 0})
        if len(C) > target:
            del C[target:]

    def update_carriers(S, dt, is_night):
        ensure_carriers(S)
        speed = 30.0 * (0.7 if is_night else 1.0)
        for c in S['carriers']:
            # select target if needed
            if c['state'] == 'to_resource' and c['load'] == 0:
                if c['kind'] == 'wood':
                    c['tx'], c['ty'] = _best_resource_target(S['cx'], S['cy'], world.wood_density)
                else:
                    c['tx'], c['ty'] = _best_resource_target(S['cx'], S['cy'], world.food_density)
            elif c['state'] == 'to_settlement' and c['load'] == 1:
                c['tx'], c['ty'] = S['cx'], S['cy']

            # move towards target
            dx = c['tx'] - c['x']; dy = c['ty'] - c['y']
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                dist = 1e-3
            step = speed * dt
            if step >= dist:
                c['x'], c['y'] = c['tx'], c['ty']
                # arrive
                if c['state'] == 'to_resource' and c['load'] == 0:
                    c['load'] = 1; c['state'] = 'to_settlement'
                elif c['state'] == 'to_settlement' and c['load'] == 1:
                    # deposit
                    if c['kind'] == 'wood':
                        S['wood'] += 1
                    else:
                        S['food'] += 1
                    c['load'] = 0; c['state'] = 'to_resource'
            else:
                c['x'] += (dx / dist) * step
                c['y'] += (dy / dist) * step

    def draw_carriers(S, off_x, off_y, scale_x, scale_y):
        vis = max(1.0, (scale_x + scale_y) * 0.5)
        lw = max(1, int(1 * vis))
        body = max(3, int(4 * (0.7 + 0.3 * vis)))
        pack_h = max(3, int(3 * (0.7 + 0.3 * vis)))
        for c in S['carriers']:
            color = (130, 80, 50) if c['kind'] == 'wood' else (180, 220, 100)
            x = off_x + int(c['x'] * scale_x)
            y = off_y + int(c['y'] * scale_y)
            tx = off_x + int(c['tx'] * scale_x)
            ty = off_y + int(c['ty'] * scale_y)
            pygame.draw.line(screen, color, (x, y), (tx, ty), lw)
            pygame.draw.rect(screen, color, pygame.Rect(x - body//2, y - body//2, body, body))
            if c['load']:
                pygame.draw.rect(screen, (230, 230, 230), pygame.Rect(x - body//2, y - body//2 - pack_h - 1, body, pack_h))

    # ---- L3 ?앸챸 ?좊떅(?쒓컖?붿슜 寃쎈웾 ?뷀떚?? ----
    entities = []  # list of dicts: {x,y,vx,vy,state,carry,trace}

    def _spawn_entities_for_settlement(S, count):
        for _ in range(count):
            r = np.random.rand()
            if r < 0.35:
                demog = 'man'; speed_factor = 1.0
            elif r < 0.70:
                demog = 'woman'; speed_factor = 0.95
            elif r < 0.85:
                demog = 'elder'; speed_factor = 0.70
            else:
                demog = 'child'; speed_factor = 1.20
            entities.append({
                'x': S['cx'] + np.random.uniform(-6, 6),
                'y': S['cy'] + np.random.uniform(-6, 6),
                'vx': 0.0, 'vy': 0.0,
                'state': 'calm',  # calm|work|rest|danger
                'carry': None,    # None|'wood'|'food'
                'home': (S['cx'], S['cy']),
                'trace': [],      # list of (x,y,age)
                'goal': None,
                'kind_pref': 'wood' if np.random.rand() < 0.5 else 'food',
                'demog': demog,
                'speed_factor': speed_factor,
            })

    # 珥덇린 ?ㅽ룿(?뺤갑吏 ???멸뎄???쇰?留??쒓컖??
    for S in settlements:
        _spawn_entities_for_settlement(S, min(24, max(6, S['pop'] // 10)))

    def _entity_pick_goal(ent, day):
        if day:
            # ?먯썝吏 紐⑺몴
            if ent['carry'] is None:
                if ent['kind_pref'] == 'wood':
                    gx, gy = _best_resource_target(ent['home'][0], ent['home'][1], world.wood_density)
                    ent['goal'] = (gx, gy)
                else:
                    gx, gy = _best_resource_target(ent['home'][0], ent['home'][1], world.food_density)
                    ent['goal'] = (gx, gy)
                ent['state'] = 'work'
            else:
                # 吏먯쓣 ?ㅼ뿀?쇰㈃ 吏묒쑝濡?蹂듦?
                ent['goal'] = ent['home']
                ent['state'] = 'work'
        else:
            # 諛ㅼ뿉??紐⑤떏遺?吏묎껐
            ent['goal'] = ent['home']
            ent['state'] = 'calm'

    def update_entities(dt, is_night):
        day = not is_night
        for ent in entities:
            # 紐⑺몴 ?ъ꽕???녾굅???꾩갑 ??
            if ent['goal'] is None or np.hypot((ent['goal'][0]-ent['x']), (ent['goal'][1]-ent['y'])) < 1.0:
                _entity_pick_goal(ent, day)

            # ?대룞
            gx, gy = ent['goal']
            dx = gx - ent['x']; dy = gy - ent['y']
            dist = math.hypot(dx, dy)
            speed = (22.0 if day else 16.0) * float(ent.get('speed_factor', 1.0))
            if dist > 1e-3:
                ent['vx'] = (dx / dist) * speed
                ent['vy'] = (dy / dist) * speed
                step = speed * dt
                if step >= dist:
                    ent['x'], ent['y'] = gx, gy
                else:
                    ent['x'] += ent['vx'] * dt
                    ent['y'] += ent['vy'] * dt

            # ?먯썝 ?쎌뾽/?쒕엻(?쒓컖?붾쭔, 移댁슫??蹂?붾뒗 carriers媛 ?대떦)
            if day and ent['carry'] is None and dist < 1.2:
                d = ent.get('demog')
                pref = ent['kind_pref']
                if d == 'child':
                    ent['carry'] = None
                elif d == 'elder' and pref == 'wood':
                    ent['carry'] = 'food'
                else:
                    ent['carry'] = pref
            if (not day) or (ent['carry'] is not None and ent['goal'] == ent['home'] and dist < 1.2):
                ent['carry'] = None

            # 諛쒖옄援??몃젅?댁뒪
            ent['trace'].append([ent['x'], ent['y'], 0.8])  # 珥덇린 ?뚰뙆
            # ?몃젅?댁뒪 媛먯뇿
            for t in ent['trace']:
                t[2] = max(0.0, t[2] - dt * 1.5)
            # ?ㅻ옒???몃젅?댁뒪 ?쒓굅
            while ent['trace'] and ent['trace'][0][2] <= 0.01:
                ent['trace'].pop(0)

    def draw_entities(off_x, off_y, scale_x, scale_y):
        if not SHOW_ENTITIES:
            return
        for ent in entities:
            # 諛쒖옄援??숈깋, ?뚰뙆)
            for tx, ty, a in ent['trace']:
                px = off_x + int(tx * scale_x)
                py = off_y + int(ty * scale_y)
                if 0 <= px < screen.get_width() and 0 <= py < screen.get_height():
                    s = pygame.Surface((3, 3), pygame.SRCALPHA)
                    s.fill((110, 90, 70, int(160 * a)))
                    screen.blit(s, (px-1, py-1))

            # 蹂몄껜 ?됱긽
            if ent['state'] == 'calm':
                col = (140, 255, 140)
            elif ent['state'] == 'work':
                col = (120, 180, 255)
            elif ent['state'] == 'danger':
                col = (255, 100, 100)
            else:
                col = (240, 240, 240)

            px = off_x + int(ent['x'] * scale_x)
            py = off_y + int(ent['y'] * scale_y)
            pygame.draw.rect(screen, col, pygame.Rect(px-2, py-2, 4, 4))
            # 吏??쒖떆(?꾩씠肄?
            if ent['carry'] == 'wood':
                pygame.draw.rect(screen, (130, 80, 50), pygame.Rect(px-2, py-6, 4, 3))
            elif ent['carry'] == 'food':
                pygame.draw.rect(screen, (180, 220, 100), pygame.Rect(px-2, py-6, 4, 3))

    def tick_settlement(S, is_night):
        cx, cy = int(S['cx']), int(S['cy'])
        r = max(6, int(0.06 * min(W, H)))
        x0 = max(0, cx - r); x1 = min(W, cx + r)
        y0 = max(0, cy - r); y1 = min(H, cy + r)
        wood_zone = world.wood_density[y0:y1, x0:x1]
        food_zone = world.food_density[y0:y1, x0:x1]
        workers = max(10, int(S['pop'] * 0.3))
        if not is_night:
            S['wood'] += int(0.02 * workers * float(wood_zone.mean() if wood_zone.size else 0.0))
            S['food'] += int(0.02 * workers * float(food_zone.mean() if food_zone.size else 0.0))
            S['fire'] = False
            S['comfort'] = max(0.0, S['comfort'] - 0.001)
        else:
            S['fire'] = True
            need = max(1, S['pop'] // 20)
            if S['food'] >= need:
                S['food'] -= need
                S['comfort'] = min(1.0, S['comfort'] + 0.01)
            else:
                S['comfort'] = max(0.0, S['comfort'] - 0.02)
        while S['wood'] >= 30 and S['huts'] < 12:
            S['wood'] -= 30
            S['huts'] += 1
            S['comfort'] = min(1.0, S['comfort'] + 0.02)
        if S['comfort'] > 0.7 and S['pop'] < 600:
            S['pop'] += 1
        elif S['comfort'] < 0.2 and S['pop'] > 10:
            S['pop'] -= 1

    def draw_settlement(S, off_x, off_y, scale_x, scale_y):
        sx = off_x + int(S['cx'] * scale_x)
        sy = off_y + int(S['cy'] * scale_y)
        pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 24, 1)
        if S['fire']:
            t = pygame.time.get_ticks() * 0.001
            rj = 5 + int(3 * np.sin(t * 6.0))
            pygame.draw.circle(screen, (255, 170, 60), (sx, sy), rj)
            halo_alpha = 90 if not HIGH_CONTRAST else 140
            for rr in (12, 18, 26):
                ring = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)
                pygame.draw.circle(ring, (255, 200, 120, halo_alpha), (sx, sy), rr, 1)
                screen.blit(ring, (0, 0))
        cols = int(max(1, np.ceil(np.sqrt(max(1, S['huts'])))))
        size = 4
        for i in range(S['huts']):
            row = i // cols; col = i % cols
            ox = (col - cols//2) * (size + 2)
            oy = (row + 1) * (size + 2)
            pygame.draw.rect(screen, (130, 80, 50), pygame.Rect(sx + ox, sy + oy, size, size))
        label = f"?멸뎄:{S['pop']}  紐⑹옱:{S['wood']}  ?앸웾:{S['food']}  ?ш린:{S['comfort']:.2f}"
        txt = font.render(label, True, (15, 15, 18))
        bg = pygame.Surface((txt.get_width()+8, txt.get_height()+4), pygame.SRCALPHA)
        bg.fill((250, 250, 250, 190))
        screen.blit(bg, (sx + 28, sy - 12))
        screen.blit(txt, (sx + 32, sy - 10))

    def draw_people(S, world, off_x, off_y, scale_x, scale_y, is_night):
        # ?щ엺(?꾨줉?? ???뚮뜑: ?뺤갑吏 ?멸뎄瑜??쒓컖??(寃쎈웾)
        cx, cy = float(S['cx']), float(S['cy'])
        n = int(min(80, max(0, S.get('pop', 0))))
        if n == 0:
            return
        # ?吏곸엫 ??꾨쿋?댁뒪
        t = pygame.time.get_ticks() * 0.001
        # 諛? 紐⑤떏遺?二쇰???留??뺤꽦, ?? 議곌툑 ???볤쾶 ?⑹뼱吏?        base_r = 8 if is_night else 18
        jitter = 3 if is_night else 6
        color = (255, 200, 140) if is_night else (160, 220, 255)
        for i in range(n):
            ang = (i / n) * 2 * math.pi + t * (0.3 if is_night else 0.15)
            r = base_r + (math.sin(t * 1.7 + i) * 0.5 + 0.5) * jitter
            # ??뿉???먯썝??諛⑺뼢?쇰줈 ?쎄컙 ?명뼢
            if not is_night:
                ix = int(np.clip(cx, 0, world.w - 1))
                iy = int(np.clip(cy, 0, world.h - 1))
                gx = world.wood_density[iy, ix] - world.food_density[iy, ix]
                gy = world.food_density[iy, ix] - world.wood_density[iy, ix]
                ang += 0.4 * math.atan2(gy, gx)
            px = off_x + int((cx + r * math.cos(ang)) * scale_x)
            py = off_y + int((cy + r * math.sin(ang)) * scale_y)
            if 0 <= px < screen.get_width() and 0 <= py < screen.get_height():
                pygame.draw.circle(screen, color, (px, py), 2)

    # Zoom-aware, directional entity renderer
    def render_entities(off_x, off_y, scale_x, scale_y):
        if not SHOW_ENTITIES:
            return
        vis = max(1.0, (scale_x + scale_y) * 0.5)
        size = max(3, int(4 * (0.6 + 0.4 * vis)))
        for ent in entities:
            # traces
            for tx, ty, a in ent['trace']:
                px = off_x + int(tx * scale_x)
                py = off_y + int(ty * scale_y)
                if 0 <= px < screen.get_width() and 0 <= py < screen.get_height():
                    s = pygame.Surface((size, size), pygame.SRCALPHA)
                    s.fill((110, 90, 70, int((140 + 40*min(1.0, vis)) * a)))
                    screen.blit(s, (px - size//2, py - size//2))

            st = ent.get('state', 'calm')
            if st == 'calm': base_col = (140, 255, 140)
            elif st == 'work': base_col = (120, 180, 255)
            elif st == 'danger': base_col = (255, 100, 100)
            else: base_col = (240, 240, 240)
            d = ent.get('demog', 'man')
            if d == 'man': tint = (80, 160, 255)
            elif d == 'woman': tint = (255, 120, 180)
            elif d == 'elder': tint = (200, 200, 200)
            else: tint = (255, 220, 120)
            col = tuple(int(0.6*b + 0.4*t) for b, t in zip(base_col, tint))

            px = off_x + int(ent['x'] * scale_x)
            py = off_y + int(ent['y'] * scale_y)
            vx = ent.get('vx', 0.0); vy = ent.get('vy', 0.0)
            ang = math.atan2(vy, vx) if (vx or vy) else 0.0
            tri_r = size
            d = ent.get('demog', 'man')
            if d == 'elder':
                pygame.draw.rect(screen, col, pygame.Rect(px - size//2, py - size//2, size, size))
            elif d == 'child':
                pygame.draw.circle(screen, col, (px, py), max(2, size//2))
            else:
                p1 = (px + int(math.cos(ang) * tri_r), py + int(math.sin(ang) * tri_r))
                p2 = (px + int(math.cos(ang + 2.5) * tri_r * 0.7), py + int(math.sin(ang + 2.5) * tri_r * 0.7))
                p3 = (px + int(math.cos(ang - 2.5) * tri_r * 0.7), py + int(math.sin(ang - 2.5) * tri_r * 0.7))
                pygame.draw.polygon(screen, col, [p1, p2, p3])
            # carry icon
            carry = ent.get('carry')
            if carry == 'wood':
                pygame.draw.rect(screen, (130, 80, 50), pygame.Rect(px - size//2, py - size - 2, size, max(3, size//2)))
            elif carry == 'food':
                pygame.draw.rect(screen, (180, 220, 100), pygame.Rect(px - size//2, py - size - 2, size, max(3, size//2)))

    # Ecology helpers and demographic overlay
    animals = []
    plants = []

    def _spawn_ecology():
        for i in range(8):
            animals.append({'x': np.random.uniform(0, W), 'y': np.random.uniform(0, H), 'v': np.random.uniform(8, 16), 'dir': np.random.uniform(0, 2*math.pi), 'color': (180,120,80) if i%2==0 else (180,200,240)})
        for i in range(12):
            plants.append({'x': np.random.uniform(0, W), 'y': np.random.uniform(0, H), 'stage': np.random.uniform(0.2, 0.8)})

    _spawn_ecology()

    def update_ecology(dt):
        if not SHOW_ECOLOGY:
            return
        for a in animals:
            if np.random.rand() < 0.02:
                a['dir'] += np.random.uniform(-0.5, 0.5)
            a['x'] = (a['x'] + math.cos(a['dir']) * a['v'] * dt) % W
            a['y'] = (a['y'] + math.sin(a['dir']) * a['v'] * dt) % H
        for p in plants:
            p['stage'] = min(1.0, p['stage'] + dt*0.01)

    def draw_ecology(off_x, off_y, scale_x, scale_y):
        if not SHOW_ECOLOGY:
            return
        for p in plants:
            px = off_x + int(p['x'] * scale_x); py = off_y + int(p['y'] * scale_y)
            r = max(2, int(6 * p['stage']))
            pygame.draw.circle(screen, (100,200,120), (px, py), r, 1)
        for a in animals:
            px = off_x + int(a['x'] * scale_x); py = off_y + int(a['y'] * scale_y)
            ang = a['dir']; r = 6
            p1 = (px + int(math.cos(ang)*r), py + int(math.sin(ang)*r))
            p2 = (px + int(math.cos(ang+2.6)*r*0.6), py + int(math.sin(ang+2.6)*r*0.6))
            p3 = (px + int(math.cos(ang-2.6)*r*0.6), py + int(math.sin(ang-2.6)*r*0.6))
            pygame.draw.polygon(screen, a['color'], [p1,p2,p3])

    def overlay_demog_shapes(off_x, off_y, scale_x, scale_y):
        # draw extra silhouettes/badges over base entities
        for ent in entities:
            px = off_x + int(ent['x'] * scale_x)
            py = off_y + int(ent['y'] * scale_y)
            vx = ent.get('vx', 0.0); vy = ent.get('vy', 0.0)
            ang = math.atan2(vy, vx) if (vx or vy) else 0.0
            size = max(3, int(4 * (0.6 + 0.4 * max(scale_x, scale_y))))
            st = ent.get('state','calm')
            base_col = (140,255,140) if st=='calm' else (120,180,255) if st=='work' else (255,100,100) if st=='danger' else (240,240,240)
            dmo = ent.get('demog','man')
            tint = (80,160,255) if dmo=='man' else (255,120,180) if dmo=='woman' else (200,200,200) if dmo=='elder' else (255,220,120)
            mode = PALETTE_CYCLE[PALETTE_INDEX]
            if mode=='pastel':
                col = tuple(min(255,int(0.6*b+0.4*t+20)) for b,t in zip(base_col,tint))
            elif mode=='vivid':
                col = tuple(min(255,int(0.5*b+0.5*t+10)) for b,t in zip(base_col,tint))
            else:
                g = int(0.299*base_col[0]+0.587*base_col[1]+0.114*base_col[2])
                col = (g,g,g)
            if dmo=='elder':
                pygame.draw.line(screen, (120,90,60), (px + size//2, py - size//3), (px + size//2, py + size//2), 2)
            elif dmo=='child':
                bx, by = px + size//2 + 2, py - size - 4
                pygame.draw.line(screen, (200,200,200), (px, py - size//2), (bx, by), 1)
                pygame.draw.circle(screen, (255,210,120), (bx, by), 3)
                sx, sy = px - size//2 - 4, py - size - 4
                star = [(sx, sy-3), (sx+2, sy-1), (sx+4, sy-3), (sx+3, sy), (sx+5, sy+2), (sx+2, sy+1), (sx-1, sy+2), (sx+1, sy)]
                pygame.draw.polygon(screen, (255,240,150), star)

    # ---- Speech visualization (bubbles + dialog links + log) ----
    speech_items = []  # {src:int, dst:int|None, text:str, emo:str, ttl:float}
    EMO_COL = {
        'joy':   (255, 200, 80),
        'calm':  (180, 220, 255),
        'anger': (255, 120, 120),
        'fear':  (180, 180, 255),
        'surprise': (255, 240, 150),
        'sadness': (160, 180, 220),
        'neutral': (220, 220, 230),
    }

    def add_speech(src_idx, text, emotion='neutral', dst_idx=None, ttl=2.5):
        if 0 <= src_idx < len(entities):
            speech_items.append({'src': src_idx, 'dst': dst_idx, 'text': text, 'emo': emotion, 'ttl': ttl})

    def update_speech(dt):
        if not SHOW_SPEECH:
            return
        for it in speech_items:
            it['ttl'] -= dt
        while speech_items and speech_items[0]['ttl'] <= 0:
            # drop expired from front; keep recent at end
            speech_items.pop(0)

    def draw_speech(off_x, off_y, scale_x, scale_y):
        if not SHOW_SPEECH:
            return
        # bubbles near entities
        max_show = 12
        for it in speech_items[-max_show:]:
            src = it['src']; dst = it.get('dst')
            if not (0 <= src < len(entities)):
                continue
            sx = off_x + int(entities[src]['x'] * scale_x)
            sy = off_y + int(entities[src]['y'] * scale_y)
            alpha = max(40, int(200 * max(0.0, it['ttl'] / 2.5)))
            col = EMO_COL.get(it['emo'], EMO_COL['neutral'])
            # bubble panel
            text = it['text'][:24]
            t_surf = font.render(text, True, (20, 20, 24))
            pad = 4
            bw, bh = t_surf.get_width() + pad*2, t_surf.get_height() + pad*2
            bx, by = sx + 8, sy - 18 - bh
            panel = pygame.Surface((bw, bh), pygame.SRCALPHA)
            panel.fill((*col, alpha))
            screen.blit(panel, (bx, by))
            screen.blit(t_surf, (bx + pad, by + pad))
            # tail
            pygame.draw.line(screen, (*col, alpha), (sx, sy - 6), (bx, by + bh), 2)
            # dialog link
            if dst is not None and 0 <= dst < len(entities):
                dx = off_x + int(entities[dst]['x'] * scale_x)
                dy = off_y + int(entities[dst]['y'] * scale_y)
                pygame.draw.line(screen, col, (sx, sy), (dx, dy), 1)
        # left log panel
        logs = speech_items[-6:]
        if logs:
            rows = [f"{it['emo']}: {it['text'][:18]}" for it in logs]
            lw = max(font.size(r)[0] for r in rows) + 12
            lh = (font.get_height() + 2) * len(rows) + 8
            bg = pygame.Surface((lw, lh), pygame.SRCALPHA)
            bg.fill((0,0,0,120))
            screen.blit(bg, (10, 60))
            y = 64
            for r in rows:
                screen.blit(font.render(r, True, (220,220,230)), (16, y))
                y += font.get_height() + 2

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEWHEEL:
                # 留덉슦????以? 而ㅼ꽌 湲곗? ?듭빱 以????뺣?/?꾨옒 異뺤냼) + ??蹂댁젙
                mx, my = pygame.mouse.get_pos()
                # ?꾩옱 ?ㅼ????ㅽ봽?뗭쑝濡?而ㅼ꽌 ?꾨옒 ?붾뱶 醫뚰몴 怨꾩궛
                scaled_w0 = int(VIEW * scale)
                scaled_h0 = int(VIEW * scale)
                off_x0 = (VIEW - scaled_w0) // 2
                off_y0 = (VIEW - scaled_h0) // 2
                px_off_x0 = off_x0 + int(cam_x)
                px_off_y0 = off_y0 + int(cam_y)
                scale_x0 = (VIEW * scale) / W
                scale_y0 = (VIEW * scale) / H
                wx = (mx - px_off_x0) / max(1e-6, scale_x0)
                wy = (my - px_off_y0) / max(1e-6, scale_y0)

                # ???ㅼ???                factor = 1.1 if e.y > 0 else 0.9
                new_scale = max(min_scale, min(max_scale, scale * factor))
                # ???ㅽ봽?뗭씠 而ㅼ꽌 ?꾩튂?먯꽌 媛숈? ?붾뱶 醫뚰몴瑜?媛由ы궎?꾨줉 pan 蹂댁젙
                scaled_w1 = int(VIEW * new_scale)
                scaled_h1 = int(VIEW * new_scale)
                off_x1 = (VIEW - scaled_w1) // 2
                off_y1 = (VIEW - scaled_h1) // 2
                scale_x1 = (VIEW * new_scale) / W
                scale_y1 = (VIEW * new_scale) / H
                desired_px_off_x1 = mx - wx * scale_x1
                desired_px_off_y1 = my - wy * scale_y1
                pan_x = desired_px_off_x1 - off_x1
                pan_y = desired_px_off_y1 - off_y1
                # 利됱떆 諛섏쁺媛??뺣낫: 移대찓?쇰룄 ?숈씪 媛믪쑝濡??명똿
                cam_x, cam_y = pan_x, pan_y
                scale = new_scale
                # Shift + ?? 諛곗냽 議곗젅
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    if e.y > 0:
                        tps = min(240.0, tps * 1.2)
                    elif e.y < 0:
                        tps = max(0.5, tps / 1.2)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 2:  # middle button start pan
                    panning = True
                    pan_last = pygame.mouse.get_pos()
                elif e.button == 1:  # left: begin gesture (concentrate)
                    gest_active = True
                    brush_hold_t = 0.0
                    last_world_xy = None
                elif e.button == 3:  # right: begin gesture (disperse)
                    gest_active = True
                    brush_hold_t = 0.0
                    last_world_xy = None
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 2:
                    panning = False
                elif e.button in (1, 3):
                    # spawn burst effect on release
                    mx, my = pygame.mouse.get_pos()
                    fx_events.append({"type": "burst", "x": mx, "y": my, "t0": time.time(), "age": 0.0})
                    gest_active = False
            elif e.type == pygame.MOUSEMOTION:
                if panning:
                    mx, my = e.pos
                    lx, ly = pan_last
                    pan_x += (mx - lx)
                    pan_y += (my - ly)
                    pan_last = (mx, my)
                # Shift + drag: time accel by horizontal delta
                mods = pygame.key.get_mods()
                if (mods & pygame.KMOD_SHIFT) and any(pygame.mouse.get_pressed()):
                    dx = e.rel[0]
                    if dx != 0:
                        tps = min(240.0, max(0.5, tps * (1.0 + 0.01 * dx)))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_l:
                    show_legend = not show_legend
                elif e.key == pygame.K_h:
                    show_help = not show_help
                elif e.key == pygame.K_g:
                    SHOW_OUTLINE = not SHOW_OUTLINE
                # Shift+H: ?щ엺 愿李?紐⑤뱶 ?좉?
                elif (pygame.key.get_mods() & pygame.KMOD_SHIFT) and e.key == pygame.K_h:
                    HUMAN_VIEW = not HUMAN_VIEW
                # 鍮좊Ⅸ 以??? Z(異뺤냼)/X(?뺣?), F(?붾㈃ 留욎땄), 0(湲곕낯 諛곗쑉)
                elif e.key == pygame.K_z:
                    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, {'y': -1}))
                elif e.key == pygame.K_x:
                    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, {'y': 1}))
                elif e.key == pygame.K_0:
                    scale = 1.0
                    cam_x = cam_y = 0.0
                    pan_x = pan_y = 0
                elif e.key == pygame.K_f:
                    # ?붾㈃ 苑?梨꾩슦湲?湲곕낯 諛곗쑉怨??좎궗)
                    scale = 1.0
                    cam_x = cam_y = 0.0
                    pan_x = pan_y = 0
                elif e.key == pygame.K_m:
                    SHOW_MANA = not SHOW_MANA
                elif e.key == pygame.K_n:
                    SHOW_AETHER = not SHOW_AETHER
                elif e.key == pygame.K_o:
                    SHOW_ECOLOGY = not SHOW_ECOLOGY
                elif e.key == pygame.K_p:
                    PALETTE_INDEX = (PALETTE_INDEX + 1) % len(PALETTE_CYCLE)
                elif e.key == pygame.K_v:
                    SHOW_SPEECH = not SHOW_SPEECH
                elif e.key == pygame.K_k:
                    # demo utterance
                    if entities:
                        import random
                        src = random.randrange(len(entities))
                        dst = random.randrange(len(entities)) if len(entities) > 1 else None
                        phrase = random.choice(["불 피우자", "나무 더 가져와", "휴식하자", "함께 짓자"])
                        emo = random.choice(["joy","calm","surprise","neutral"]) 
                        add_speech(src, phrase, emo, dst)
                elif e.key == pygame.K_e:
                    SHOW_ENTITIES = not SHOW_ENTITIES
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    scale = min(max_scale, scale + 0.25)
                elif e.key == pygame.K_MINUS:
                    scale = max(min_scale, scale - 0.25)
                elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                    tps = min(120.0, tps * 1.5)
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                    tps = max(1.0, tps / 1.5)
                # ?レ옄?? 1x/2x/4x/8x
                elif e.key == pygame.K_1:
                    tps = base_tps * 1.0
                elif e.key == pygame.K_2:
                    tps = base_tps * 2.0
                elif e.key == pygame.K_3:
                    tps = base_tps * 4.0
                elif e.key == pygame.K_4:
                    tps = base_tps * 8.0
                elif e.key == pygame.K_LEFT:
                    focus_x = max(0, focus_x - 2)
                elif e.key == pygame.K_RIGHT:
                    focus_x = min(W - 1, focus_x + 2)
                elif e.key == pygame.K_UP:
                    focus_y = max(0, focus_y - 2)
                elif e.key == pygame.K_DOWN:
                    focus_y = min(H - 1, focus_y + 2)
                # WASD ?대룞 + A??Shift+A硫??댄뀗???좉?)
                elif e.key == pygame.K_w:
                    focus_y = max(0, focus_y - 2)
                elif e.key == pygame.K_s:
                    focus_y = min(H - 1, focus_y + 2)
                elif e.key == pygame.K_a:
                    # Shift+A -> ?댄뀗???좉?, a -> 醫뚮줈 ?대룞
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        show_attention = not show_attention
                    else:
                        focus_x = max(0, focus_x - 2)
                elif e.key == pygame.K_d:
                    focus_x = min(W - 1, focus_x + 2)
        # time stepping
        dt = clock.tick(60) / 1000.0
        if not paused:
            acc += dt
            step_interval = 1.0 / max(0.001, tps)
            while acc >= step_interval:
                world.update_fields()
                biome = classify_biome(world.height, world.moisture, world.temp)
                acc -= step_interval
                world_ticks = (world_ticks + 1) % 1000000000
            # brush hold timer (for pulse size)
            if gest_active:
                brush_hold_t = min(1.0, brush_hold_t + dt)
            # carriers transport update (visible hauling)
            for S in settlements:
                update_carriers(S, dt, ((world_ticks % daylen) > int(daylen * 0.7)))
            # ecology update
            update_ecology(dt)
            # speech ttl update
            update_speech(dt)
            # entities update (L3 life patterns)
            update_entities(dt, ((world_ticks % daylen) > int(daylen * 0.7)))

        # base field render (grayscale biome scaled with zoom)
        img = np.uint8(255 * (biome / (biome.max() or 1)))
        rgb = np.dstack([img, img, img]).swapaxes(0, 1)
        base = pygame.surfarray.make_surface(rgb)
        scaled_w = int(VIEW * scale)
        scaled_h = int(VIEW * scale)
        base = pygame.transform.smoothscale(base, (scaled_w, scaled_h))
        off_x = (VIEW - scaled_w) // 2
        off_y = (VIEW - scaled_h) // 2
        # camera smoothing (lerp)
        cam_x += (pan_x - cam_x) * 0.15
        cam_y += (pan_y - cam_y) * 0.15
        # apply pan (in pixels)
        px_off_x = off_x + int(cam_x)
        px_off_y = off_y + int(cam_y)
        # 寃쎄퀎 ?대옩?? 酉?諛뽰쑝濡??대?吏媛 怨쇰룄?섍쾶 踰쀬뼱?섏? ?딅룄濡??쒗븳
        min_x = VIEW - scaled_w
        min_y = VIEW - scaled_h
        px_off_x = max(min_x, min(0, px_off_x))
        px_off_y = max(min_y, min(0, px_off_y))
        # ?대옩??寃곌낵瑜???쑝濡?pan?먮룄 諛섏쁺(?먯쭊)
        pan_x = px_off_x - off_x
        pan_y = px_off_y - off_y
        screen.fill(COL["bg"]) 
        if HUMAN_VIEW:
            render_human_view(screen, world, scaled_w, scaled_h, px_off_x, px_off_y)
        else:
            screen.blit(base, (px_off_x, px_off_y))

        # Optional overlays: Aether (heatmap) and Mana (particles)
        if SHOW_AETHER:
            a = (world.aether * 255.0).astype(np.uint8)
            a_rgb = np.dstack([
                (a * 0.31).astype(np.uint8),
                (a * 0.70).astype(np.uint8),
                a
            ]).swapaxes(0, 1)
            a_surf = pygame.surfarray.make_surface(a_rgb)
            a_surf = pygame.transform.smoothscale(a_surf, (scaled_w, scaled_h))
            a_surf.set_alpha(90)
            screen.blit(a_surf, (px_off_x, px_off_y))

        if SHOW_MANA:
            step = max(1, int(4 / max(0.25, scale)))
            m = world.mana
            for yy in range(0, H, step):
                row = m[yy]
                for xx in range(0, W, step):
                    v = float(row[xx])
                    if v < 0.25:
                        continue
                    sx = px_off_x + int(xx * scale_x)
                    sy = px_off_y + int(yy * scale_y)
                    r = 1 + int(2 * v)
                    color = (60, 200, 255) if v < 0.6 else (120, 240, 255)
                    pygame.draw.circle(screen, color, (sx, sy), r, 0)

        # QOE tile overlay based on focus and budget
        tiles_x = max(1, W // tile_w)
        tiles_y = max(1, H // tile_h)
        tx_focus = focus_x // tile_w
        ty_focus = focus_y // tile_h
        visible = []
        for ty in range(max(0, ty_focus - radius), min(tiles_y, ty_focus + radius + 1)):
            for tx in range(max(0, tx_focus - radius), min(tiles_x, tx_focus + radius + 1)):
                visible.append((tx, ty))
        visible = visible[:budget]
        visible_set = set(visible)

        # dim entire screen to indicate dormant baseline
        dim = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        screen.blit(dim, (0, 0))

        # draw active tiles in green
        scale_x = (VIEW * scale) / W
        scale_y = (VIEW * scale) / H
        for (tx, ty) in visible_set:
            rx = px_off_x + int(tx * tile_w * scale_x)
            ry = px_off_y + int(ty * tile_h * scale_y)
            rw = int(tile_w * scale_x)
            rh = int(tile_h * scale_y)
            # Draw outline only (thickness 2) to avoid occluding view
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(rx, ry, rw, rh), 2)

        # focus dot (white)
        fx = px_off_x + int(focus_x * scale_x)
        fy = px_off_y + int(focus_y * scale_y)
        pygame.draw.circle(screen, (255, 255, 255), (fx, fy), 3)

        # attention overlay (toggle with 'A')
        tiles_x = max(1, W // tile_w)
        tiles_y = max(1, H // tile_h)
        active_mask = np.zeros((tiles_y, tiles_x), dtype=bool)
        for (tx, ty) in visible_set:
            if 0 <= ty < tiles_y and 0 <= tx < tiles_x:
                active_mask[ty, tx] = True
        if show_attention:
            scores = compute_attention_scores(tiles_x, tiles_y, (fx, fy), scale_x, scale_y, tile_w, tile_h, active_mask, radius)
            # semi-transparent heat overlay
            heat = pygame.Surface((scaled_w, scaled_h), pygame.SRCALPHA)
            # draw per tile rectangles on heat, then blit
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    s = float(scores[ty, tx])
                    color = color_from_score(s)
                    rx = int(tx * tile_w * scale_x)
                    ry = int(ty * tile_h * scale_y)
                    rw = int(tile_w * scale_x)
                    rh = int(tile_h * scale_y)
                    pygame.draw.rect(heat, color + (120,), pygame.Rect(rx, ry, rw, rh), 0)
            screen.blit(heat, (px_off_x, px_off_y))
            # beams to top-K tiles
            flat = [((tx, ty), scores[ty, tx]) for ty in range(tiles_y) for tx in range(tiles_x)]
            flat.sort(key=lambda x: x[1], reverse=True)
            for i, ((tx, ty), s) in enumerate(flat[:TOPK]):
                cx = px_off_x + int(tx * tile_w * scale_x + (tile_w * scale_x) // 2)
                cy = px_off_y + int(ty * tile_h * scale_y + (tile_h * scale_y) // 2)
                pygame.draw.line(screen, COL["beam"], (fx, fy), (cx, cy), 1)

        # settlements: update and draw
        is_night = ((world_ticks % daylen) > int(daylen * 0.7))
        world.is_night = bool(is_night)
        for S in settlements:
            tick_settlement(S, is_night)
            draw_settlement(S, px_off_x, px_off_y, scale_x, scale_y)
            draw_people(S, world, px_off_x, px_off_y, scale_x, scale_y, is_night)
            draw_carriers(S, px_off_x, px_off_y, scale_x, scale_y)
        # ecology + entities (top)
        draw_ecology(px_off_x, px_off_y, scale_x, scale_y)
        render_entities(px_off_x, px_off_y, scale_x, scale_y)
        overlay_demog_shapes(px_off_x, px_off_y, scale_x, scale_y)
        # speech bubbles and dialog links
        draw_speech(px_off_x, px_off_y, scale_x, scale_y)

        # legend, help, HUD
        legend_h = 0
        if show_legend:
            legend_h = draw_legend(screen, font)
        if show_help:
            draw_help(screen, font, legend_h)
        # Interaction: gesture stream to manipulate mana (continuous)
        mx, my = pygame.mouse.get_pos()
        # map to world coords
        wx = int((mx - px_off_x) / max(1e-6, scale_x))
        wy = int((my - px_off_y) / max(1e-6, scale_y))
        if 0 <= wx < W and 0 <= wy < H:
            buttons = pygame.mouse.get_pressed()
            r = max(1, int(5 / max(0.5, scale)))
            if buttons[0] or buttons[2]:
                kind = 'concentrate' if buttons[0] else 'disperse'
                if last_world_xy is None:
                    last_world_xy = (wx, wy)
                    _apply_brush(world, kind, wx, wy, r, 0.02)
                else:
                    lx, ly = last_world_xy
                    for px, py in _line_points(lx, ly, wx, wy, step=max(1, r)):
                        _apply_brush(world, kind, px, py, r, 0.02)
                    last_world_xy = (wx, wy)
            else:
                last_world_xy = None

        # visual feedback: pulse on hold and rainbow ring near focus
        if gest_active:
            _draw_pulse(screen, mx, my, base_r=6 + int(6 * brush_hold_t), color=(255, 255, 255))
        # resonance when cursor near focus
        if math.hypot(mx - fx, my - fy) < 30:
            _draw_rainbow_ring(screen, mx, my, 26)
            # nudge aether positively at world point
            if 0 <= wx < W and 0 <= wy < H:
                world.aether[max(0, wy - 2):min(H, wy + 2), max(0, wx - 2):min(W, wx + 2)] = np.clip(
                    world.aether[max(0, wy - 2):min(H, wy + 2), max(0, wx - 2):min(W, wx + 2)] + 0.003, 0.0, 1.0)

        # draw effect bursts (expanding rings)
        now = time.time()
        alive = []
        for fxev in fx_events:
            age = now - fxev["t0"]
            if age > 0.6:
                continue
            rad = int(10 + 80 * age)
            alpha = max(0, 180 - int(300 * age))
            ring = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)
            pygame.draw.circle(ring, (120, 240, 255, alpha), (fxev["x"], fxev["y"]), rad, 2)
            screen.blit(ring, (0, 0))
            alive.append(fxev)
        fx_events = alive
        phase = 'night' if ((world_ticks % daylen) > int(daylen * 0.7)) else 'day'
        hud_text = f"{phase} | focus=({focus_x},{focus_y}) tiles(full)={len(visible_set)} scale={scale:.2f} tps={tps:.1f}"
        hud = font.render(hud_text, True, (220, 220, 230))
        # draw HUD at bottom-left with a translucent background bar
        pad = 8
        bg_w = hud.get_width() + pad * 2
        bg_h = hud.get_height() + pad * 2
        y_bottom = screen.get_height() - bg_h - 10
        hud_bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 120))
        screen.blit(hud_bg, (10, y_bottom))
        screen.blit(hud, (10 + pad, y_bottom + pad))

        pygame.display.flip()
        # cap to display fps, simulation runs via tps
        # we already called clock.tick at top of loop via dt
        # keep UI responsive

    pygame.quit(); sys.exit()


if __name__ == "__main__":
    main()
