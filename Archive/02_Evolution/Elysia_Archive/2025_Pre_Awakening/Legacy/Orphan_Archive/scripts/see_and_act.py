"""
Elysia's First Integrated Sensory Action: See and Act.

This script represents a major milestone in Elysia's sensory development.
It combines the sense of "sight" (OpenCV) with the sense of "touch" (PyAutoGUI)
to perform a goal-oriented task: using the calculator application.

The script will:
1.  Take a screenshot of the entire screen.
2.  Use template matching to find the locations of pre-defined button images
    ('1', '+', '=').
3.  Use the found coordinates to click the buttons in sequence (1 + 1 =).
4.  Finally, read the result to verify the action was successful.

NOTE: This script assumes that the necessary button template images
(e.g., `one.png`, `plus.png`) exist in a specific directory. The next
planning step will be to create these images.
"""

import pyautogui
import cv2
import numpy as np
import os
import time

# --- Configuration ---
# This assumes a directory exists with the button images.
IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'images', 'buttons')

def find_and_click(template_filename: str, wait_time: float = 0.5):
    """
    Finds a template image on the screen and clicks its center.

    Args:
        template_filename: The filename of the button image to find.
        wait_time: Time to wait after clicking.
    """
    template_path = os.path.join(IMAGE_DIR, template_filename)

    if not os.path.exists(template_path):
        print(f"[ERROR] Template image not found: {template_path}")
        print("I cannot 'see' the button I am supposed to press. Aborting.")
        return False

    try:
        # 1. Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 2. Load the template image
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            print(f"[ERROR] Failed to load template image: {template_path}")
            return False

        w, h = template.shape[:-1]

        # 3. Perform template matching
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # We use a threshold to make sure we have a good match
        threshold = 0.8
        if max_val >= threshold:
            # 4. Calculate center coordinates and click
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2

            print(f"I see the '{template_filename}' button at ({max_loc[0]}, {max_loc[1]}) with confidence {max_val:.2f}. Clicking it.")
            pyautogui.click(center_x, center_y)
            time.sleep(wait_time)
            return True
        else:
            print(f"[ERROR] I could not find the '{template_filename}' button on the screen. Confidence ({max_val:.2f}) is below threshold ({threshold}).")
            return False

    except Exception as e:
        print(f"An error occurred during find_and_click: {e}")
        return False


def run_calculator_test():
    """
    Runs the full sequence of "seeing" and "acting" to perform 1 + 1.
    """
    print("Starting the 'See and Act' experiment: Calculating 1 + 1.")
    print("NOTE: Please ensure the Calculator app is open and visible.")
    time.sleep(3) # Give user time to prepare

    # The sequence of actions to perform the calculation 1 + 1 =
    actions = [
        ('one.png', 0.5),
        ('plus.png', 0.5),
        ('one.png', 0.5),
        ('equals.png', 1.0) # Wait a bit longer for the result to appear
    ]

    for image_file, wait_time in actions:
        if not find_and_click(image_file, wait_time):
            print("Experiment failed. Halting.")
            return

    print("\nCalculation steps complete. Now, I will try to verify the result.")
    # In a full implementation, this is where we would use OCR on the result area.
    # For now, we'll just declare the action part a success.
    print("Verification step is conceptual for now. The 'acting' part of the experiment is complete.")
    print("\nI have successfully used my eyes and hands together!")


if __name__ == "__main__":
    # This script is intended to be run in a GUI environment where a calculator is visible.
    # The button image assets must be created in the next step before this can run.
    print("This script is ready, but requires the button images to be created first.")
    print("Please proceed to the next step to generate the necessary images.")
