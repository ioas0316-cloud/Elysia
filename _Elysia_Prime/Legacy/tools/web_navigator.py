# [Genesis: 2025-12-02] Purified by Elysia
"""
Optional sandboxed web navigator (dynamic). This module prefers Playwright if
available; otherwise, returns a friendly message. All requests should still be
vetted by WebSanctum beforehand.
"""
from typing import Dict, Any, List


def navigate(url: str, steps: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return {"error": "Playwright not installed. Install and run in sandbox.", "url": url}

    steps = steps or []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            # Block resource types to reduce risk
            def route_handler(route, request):
                rtype = request.resource_type
                if rtype in ("script", "image", "media", "font", "stylesheet", "xhr", "fetch"):
                    return route.abort()
                return route.continue_()
            page.route("**/*", route_handler)
            page.goto(url, wait_until='domcontentloaded', timeout=5000)
            html = page.content()
            title = page.title()
            ctx.close(); browser.close()
            return {
                'url': url,
                'title': title,
                'content_snippet': html[:5000]
            }
    except Exception as e:
        return {'error': str(e), 'url': url}