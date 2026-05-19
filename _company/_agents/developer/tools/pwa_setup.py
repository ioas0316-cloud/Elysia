#!/usr/bin/env python3
# version: pwa_setup_v1
"""웹사이트를 PWA(모바일 앱처럼)로 변환.

config:
  PROJECT_PATH — 대상 폴더 (web_init 결과 자동 사용)
  APP_NAME — 앱 이름 (홈화면에 표시)
  APP_SHORT_NAME — 짧은 이름 (12자 이하)
  THEME_COLOR — 상단 바 색 (예: #667eea)
  BACKGROUND_COLOR — 스플래시 배경
  ICON_EMOJI — 아이콘 자동 생성에 쓸 이모지 (예: 📚)

생성 파일:
  public/manifest.json — PWA 메타
  public/sw.js — 서비스 워커 (오프라인)
  public/icon-192.png + icon-512.png — 자동 생성 (이모지 기반)
  index.html (또는 public/index.html) — meta·link·script 자동 주입

설치 방법 (사용자):
  사파리·크롬에서 사이트 접속 → "홈 화면에 추가" → 풀스크린 앱 작동
"""
import os, sys, json, base64, re


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "pwa_setup.json")
WEB_INIT_CONFIG = os.path.join(HERE, "web_init.json")


def _log(msg, kind="info"):
    prefix = {"info": "💻", "ok": "✅", "warn": "⚠️ ", "err": "❌", "step": "▸"}.get(kind, "•")
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _load(p):
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _generate_icon_svg(emoji, bg_color, size=512):
    """이모지 기반 SVG 아이콘 생성 (라운드 코너 + 그라데이션)."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{bg_color}" stop-opacity="1"/>
      <stop offset="100%" stop-color="{bg_color}" stop-opacity="0.7"/>
    </linearGradient>
  </defs>
  <rect width="{size}" height="{size}" rx="{size//8}" ry="{size//8}" fill="url(#g)"/>
  <text x="50%" y="50%" font-size="{int(size*0.55)}" text-anchor="middle" dominant-baseline="central" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">{emoji}</text>
</svg>
'''


def _find_html(project_path):
    """프로젝트의 메인 HTML 찾기 (index.html, public/index.html, app/layout.tsx 등)."""
    candidates = [
        os.path.join(project_path, "index.html"),
        os.path.join(project_path, "public", "index.html"),
        os.path.join(project_path, "public", "manifest.json"),  # 이미 있으면 표시
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _find_public_dir(project_path):
    """public 디렉토리 찾기 또는 만들기."""
    public = os.path.join(project_path, "public")
    if os.path.exists(public):
        return public
    # Vite는 public/, Next.js도 public/
    os.makedirs(public, exist_ok=True)
    return public


def main():
    cfg = _load(CONFIG)
    init_cfg = _load(WEB_INIT_CONFIG)

    project_path = (cfg.get("PROJECT_PATH") or "").strip()
    if not project_path:
        project_path = (init_cfg.get("LAST_PROJECT") or "").strip()
    if not project_path:
        _log("PROJECT_PATH가 비어있고 web_init 기록도 없음", "err")
        sys.exit(1)

    project_path = os.path.expanduser(project_path)
    if not os.path.isdir(project_path):
        _log(f"폴더 없음: {project_path}", "err")
        sys.exit(1)

    app_name = (cfg.get("APP_NAME") or "").strip() or os.path.basename(project_path)
    short_name = (cfg.get("APP_SHORT_NAME") or "").strip() or app_name[:12]
    theme = (cfg.get("THEME_COLOR") or "").strip() or "#667eea"
    bg = (cfg.get("BACKGROUND_COLOR") or "").strip() or "#ffffff"
    icon_emoji = (cfg.get("ICON_EMOJI") or "").strip() or "✦"

    _log(f"PWA 셋업 시작 → {project_path}", "info")

    public = _find_public_dir(project_path)

    # 1. manifest.json
    manifest = {
        "name": app_name,
        "short_name": short_name,
        "description": f"{app_name} — Connect AI로 만들어진 PWA",
        "start_url": "/",
        "display": "standalone",
        "orientation": "portrait",
        "theme_color": theme,
        "background_color": bg,
        "icons": [
            {"src": "/icon-192.svg", "sizes": "192x192", "type": "image/svg+xml", "purpose": "any maskable"},
            {"src": "/icon-512.svg", "sizes": "512x512", "type": "image/svg+xml", "purpose": "any maskable"},
        ],
    }
    manifest_path = os.path.join(public, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _log(f"manifest.json 생성: {manifest_path}", "ok")

    # 2. 아이콘 (SVG로 — 모든 기기에서 잘 보임 + 작은 사이즈)
    for size in (192, 512):
        icon_path = os.path.join(public, f"icon-{size}.svg")
        with open(icon_path, "w", encoding="utf-8") as f:
            f.write(_generate_icon_svg(icon_emoji, theme, size))
        _log(f"icon-{size}.svg 생성", "ok")

    # 3. service worker
    sw_path = os.path.join(public, "sw.js")
    sw_content = f'''// Connect AI PWA Service Worker
// version: pwa_v1 — auto-generated
const CACHE = "{short_name}-v1";
const ASSETS = ["/", "/manifest.json", "/icon-192.svg", "/icon-512.svg"];

self.addEventListener("install", e => {{
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS).catch(()=>{{/* offline OK */}})));
  self.skipWaiting();
}});
self.addEventListener("activate", e => {{
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
}});
self.addEventListener("fetch", e => {{
  const req = e.request;
  if (req.method !== "GET") return;
  e.respondWith(
    caches.match(req).then(hit => hit || fetch(req).then(res => {{
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(req, copy)).catch(()=>{{/* ignore */}});
      return res;
    }}).catch(() => caches.match("/") || new Response("offline", {{ status: 503 }})))
  );
}});
'''
    with open(sw_path, "w", encoding="utf-8") as f:
        f.write(sw_content)
    _log(f"sw.js 생성: {sw_path}", "ok")

    # 4. HTML에 meta + link + script 주입
    # 후보: index.html, public/index.html, app/layout.tsx (Next.js)
    html_candidates = [
        os.path.join(project_path, "index.html"),
        os.path.join(project_path, "public", "index.html"),
    ]
    html_file = None
    for c in html_candidates:
        if os.path.exists(c):
            html_file = c
            break

    pwa_head = f'''
    <!-- PWA: 자동 생성 — pwa_setup.py -->
    <meta name="theme-color" content="{theme}">
    <link rel="manifest" href="/manifest.json">
    <link rel="icon" type="image/svg+xml" href="/icon-512.svg">
    <link rel="apple-touch-icon" href="/icon-512.svg">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-title" content="{short_name}">
    <meta name="mobile-web-app-capable" content="yes">'''
    pwa_script = '''
    <script>
      if ("serviceWorker" in navigator) {
        window.addEventListener("load", () => {
          navigator.serviceWorker.register("/sw.js").catch(()=>{/* ignore */});
        });
      }
    </script>'''

    if html_file:
        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()
        if "manifest.json" in html:
            _log(f"HTML 이미 PWA 메타 있음. 스킵: {html_file}", "warn")
        else:
            # </head> 직전에 head 삽입
            new_html = re.sub(r"</head>", pwa_head + "\n  </head>", html, count=1, flags=re.IGNORECASE)
            # </body> 직전에 script 삽입
            new_html = re.sub(r"</body>", pwa_script + "\n  </body>", new_html, count=1, flags=re.IGNORECASE)
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(new_html)
            _log(f"HTML 메타·script 주입: {html_file}", "ok")
    else:
        # Next.js app/layout.tsx 안내만
        next_layout = os.path.join(project_path, "src", "app", "layout.tsx")
        next_layout_alt = os.path.join(project_path, "app", "layout.tsx")
        layout = next_layout if os.path.exists(next_layout) else (next_layout_alt if os.path.exists(next_layout_alt) else None)
        if layout:
            _log(f"Next.js 감지 — {layout} 의 metadata에 manifest 추가하세요", "warn")
            _log(f"  export const metadata = {{ ... manifest: '/manifest.json' }}", "info")
        else:
            _log("HTML 파일을 찾지 못함. PWA 메타·script 수동 추가 필요.", "warn")
            _log(f"head: {pwa_head.strip()}", "info")
            _log(f"body: {pwa_script.strip()}", "info")

    # 결과 저장
    cfg["LAST_PROJECT"] = project_path
    cfg["LAST_APP_NAME"] = app_name
    cfg["LAST_THEME"] = theme
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print()
    _log(f"PWA 셋업 완료: {app_name}", "ok")
    _log("테스트:", "info")
    _log("  1. dev server 또는 배포된 URL을 모바일 브라우저로 열기", "info")
    _log("  2. iOS Safari: 공유 → 홈 화면에 추가", "info")
    _log("  3. Android Chrome: 우측 ⋮ → 홈 화면에 추가", "info")
    _log("  4. 풀스크린·아이콘·오프라인 작동 확인", "info")


if __name__ == "__main__":
    main()
