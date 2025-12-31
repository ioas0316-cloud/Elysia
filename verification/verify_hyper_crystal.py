from playwright.sync_api import sync_playwright, expect
import time

def verify_hyper_crystal():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to dashboard...")
            page.goto("http://localhost:8050")

            # 1. Verify Rotation Controls (Use simple text locator)
            # Dash buttons are standard HTML buttons
            pitch_btn = page.locator("button:has-text('Pitch Up')")
            expect(pitch_btn).to_be_visible(timeout=10000)
            print("✅ Rotation Controls Visible")

            # 2. Verify Graph Container
            expect(page.locator("#city-graph")).to_be_visible()
            print("✅ Graph Container Visible")

            # 3. Rotate Crystal (Pitch Up -> Passion/Red)
            print("Rotating Crystal (Pitch Up)...")
            pitch_btn.click()
            time.sleep(2)

            # Take screenshot of Red State
            page.screenshot(path="verification/crystal_red.png")
            print("📸 Screenshot taken: crystal_red.png")

            # 4. Rotate Crystal (Yaw Right -> Creativity/Magenta)
            print("Rotating Crystal (Yaw Right)...")
            yaw_btn = page.locator("button:has-text('Yaw Right')")
            for _ in range(3):
                yaw_btn.click()
                time.sleep(0.5)

            time.sleep(2)
            page.screenshot(path="verification/crystal_magenta.png")
            print("📸 Screenshot taken: crystal_magenta.png")

        except Exception as e:
            print(f"❌ Verification Failed: {e}")
            page.screenshot(path="verification/failure.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_hyper_crystal()
