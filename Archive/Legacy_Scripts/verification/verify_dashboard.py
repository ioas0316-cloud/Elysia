from playwright.sync_api import sync_playwright, expect
import time

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to dashboard...")
            page.goto("http://localhost:8050")

            # 1. Verify Title (City of Logic)
            expect(page.get_by_role("heading", name="Elysia: City of Logic")).to_be_visible()
            print("✅ Title Visible")

            # 2. Verify Graph Container exists (Allow time for Dash to load React components)
            # Increase timeout to 10s for initial render
            expect(page.locator("#city-graph")).to_be_visible(timeout=10000)
            print("✅ Graph Container Visible")

            # 3. Verify Control Buttons (Quantum Coin)
            expect(page.get_by_role("button", name="ALERT")).to_be_visible()
            expect(page.get_by_role("button", name="CALM")).to_be_visible()
            print("✅ Buttons Visible")

            # 4. Trigger Alert Mode (Quantum Jump)
            print("Clicking ALERT button...")
            page.get_by_role("button", name="ALERT").click()
            time.sleep(3) # Wait for callback & transition

            # Verify background color change (Indirectly via screenshot or style check)
            # We'll take a screenshot to confirm visually
            page.screenshot(path="verification/dashboard_alert.png")
            print("📸 Screenshot taken: dashboard_alert.png")

            # 5. Trigger Calm Mode
            print("Clicking CALM button...")
            page.get_by_role("button", name="CALM").click()
            time.sleep(3)
            page.screenshot(path="verification/dashboard_calm.png")
            print("📸 Screenshot taken: dashboard_calm.png")

        except Exception as e:
            print(f"❌ Verification Failed: {e}")
            page.screenshot(path="verification/failure.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_dashboard()
