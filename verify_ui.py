import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # 1. Navigate to the application
            await page.goto("http://localhost:5000")

            # 2. Wait for the main container to be visible
            await page.wait_for_selector(".container", timeout=5000)

            # 3. Verify the three panels are present
            await page.wait_for_selector(".status-panel", timeout=1000)
            await page.wait_for_selector(".conversation-panel", timeout=1000)
            await page.wait_for_selector(".creation-panel", timeout=1000)

            print("Successfully verified the three-panel layout.")

            # 4. Take a screenshot for visual confirmation
            screenshot_path = "/app/verification.png"
            await page.screenshot(path=screenshot_path)

            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"An error occurred during verification: {e}")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
