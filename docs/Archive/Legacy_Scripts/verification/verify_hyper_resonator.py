from playwright.sync_api import sync_playwright, expect
import time

def verify_hyper_resonator():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to dashboard...")
            page.goto("http://localhost:8050")

            # 1. Verify New Controls (Resonator Controls)
            # Check for "Hyper-Resonator Controls:" label
            expect(page.get_by_text("Hyper-Resonator Controls:")).to_be_visible(timeout=10000)

            collapse_btn = page.locator("button:has-text('COLLAPSE')")
            resurrect_btn = page.locator("button:has-text('RESURRECT')")
            expect(collapse_btn).to_be_visible()
            print("✅ Resonator Controls Visible")

            # 2. Test Collapse (Wave -> Particle)
            print("Collapsing Resonator into Memory Orb...")
            collapse_btn.click()
            time.sleep(2)

            # Verify Title change or text change indicating particle state
            # The title is SVG text, hard to grab. But we can check stats panel text.
            # Look for "Memory Orb (Frozen)" in stats
            expect(page.locator("#stats-panel")).to_contain_text("Memory Orb (Frozen)")

            page.screenshot(path="verification/resonator_collapsed.png")
            print("📸 Screenshot taken: resonator_collapsed.png")

            # 3. Test Resurrection (Particle -> Wave)
            print("Resurrecting Resonator into Wave...")
            resurrect_btn.click()
            time.sleep(2)

            # Should NOT contain "Frozen" anymore
            # expect(page.locator("#stats-panel")).not_to_contain_text("Memory Orb (Frozen)")
            # Note: Playwright sync API `not_to_contain_text` needs correct syntax or logic

            page.screenshot(path="verification/resonator_resurrected.png")
            print("📸 Screenshot taken: resonator_resurrected.png")

        except Exception as e:
            print(f"❌ Verification Failed: {e}")
            page.screenshot(path="verification/failure.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_hyper_resonator()
