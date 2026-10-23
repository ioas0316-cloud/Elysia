"""
Sheet Processor Module

Handles 2D multi-view orthographic character sheet segmentation, foreground mask extraction,
and vertical axis alignment (front, side, back views).
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ProcessedViews:
    """Dataclass holding extracted and aligned multi-view images and masks."""
    front: np.ndarray          # RGBA image (H, W, 4)
    side: np.ndarray           # RGBA image (H, W, 4)
    back: np.ndarray           # RGBA image (H, W, 4)
    front_mask: np.ndarray     # Binary mask (H, W) uint8
    side_mask: np.ndarray      # Binary mask (H, W) uint8
    back_mask: np.ndarray      # Binary mask (H, W) uint8
    target_size: Tuple[int, int]  # (Height, Width)


class SheetProcessor:
    """
    2D Orthographic Character Sheet Processor.
    Splits character sheet into Front, Side, and Back orthographic views,
    aligns feature levels along the Y-axis, and extracts foreground masks.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_height, self.target_width = target_size

    def process_sheet(self, sheet_path: str) -> ProcessedViews:
        """Loads and processes a multi-view sheet image into aligned views."""
        if not os.path.exists(sheet_path):
            raise FileNotFoundError(f"Sheet image not found at: {sheet_path}")

        image = cv2.imread(sheet_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image from {sheet_path}")

        # Ensure RGBA format
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        # Extract foreground mask
        fg_mask = self._extract_foreground_mask(image)

        # Find view bounding boxes
        bboxes = self._detect_view_bboxes(fg_mask)

        if len(bboxes) < 3:
            # Fallback to equal horizontal 3-split if auto-detection finds fewer bounding boxes
            h, w = image.shape[:2]
            w_third = w // 3
            bboxes = [
                (0, 0, w_third, h),
                (w_third, 0, w_third * 2, h),
                (w_third * 2, 0, w, h)
            ]

        # Order boxes left to right: Front, Side, Back
        bboxes = sorted(bboxes, key=lambda b: b[0])[:3]

        cropped_views = []
        cropped_masks = []

        for x, y, w_box, h_box in bboxes:
            crop_img = image[y:y+h_box, x:x+w_box]
            crop_mask = fg_mask[y:y+h_box, x:x+w_box]

            resized_img, resized_mask = self._resize_and_pad(crop_img, crop_mask)
            cropped_views.append(resized_img)
            cropped_masks.append(resized_mask)

        return ProcessedViews(
            front=cropped_views[0],
            side=cropped_views[1],
            back=cropped_views[2],
            front_mask=cropped_masks[0],
            side_mask=cropped_masks[1],
            back_mask=cropped_masks[2],
            target_size=(self.target_height, self.target_width)
        )

    def _extract_foreground_mask(self, rgba: np.ndarray) -> np.ndarray:
        """Generates a binary mask of the character silhouette."""
        alpha = rgba[:, :, 3]
        if np.max(alpha) > 0 and np.min(alpha) < 255 and np.count_nonzero(alpha < 250) > (alpha.size * 0.05):
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
            return mask

        # Fallback to color/luminance background separation
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        # Assume background is light/white or uniform corner color
        corner_val = int(np.mean([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]))
        if corner_val > 200:
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _detect_view_bboxes(self, mask: np.ndarray) -> list:
        """Finds distinct view bounding boxes (X, Y, W, H) along horizontal layout."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        min_area = (mask.shape[0] * mask.shape[1]) * 0.01

        valid_boxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                valid_boxes.append((x, y, w, h))

        # Merge overlapping or vertically aligned boxes into 3 main view columns
        if not valid_boxes:
            return []

        # Group boxes by X proximity
        valid_boxes = sorted(valid_boxes, key=lambda b: b[0])
        merged_boxes = []
        for box in valid_boxes:
            if not merged_boxes:
                merged_boxes.append(box)
            else:
                prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
                curr_x, curr_y, curr_w, curr_h = box
                # If current box overlaps or is close horizontally to prev box
                if curr_x < (prev_x + prev_w + 30):
                    new_x = min(prev_x, curr_x)
                    new_y = min(prev_y, curr_y)
                    new_w = max(prev_x + prev_w, curr_x + curr_w) - new_x
                    new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
                    merged_boxes[-1] = (new_x, new_y, new_w, new_h)
                else:
                    merged_boxes.append(box)

        return merged_boxes

    def _resize_and_pad(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resizes image and mask preserving aspect ratio, centered in target canvas."""
        h, w = image.shape[:2]
        scale = min(self.target_height / h, self.target_width / w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_top = (self.target_height - new_h) // 2
        pad_bottom = self.target_height - new_h - pad_top
        pad_left = (self.target_width - new_w) // 2
        pad_right = self.target_width - new_w - pad_left

        padded_img = cv2.copyMakeBorder(
            resized_img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
        )
        padded_mask = cv2.copyMakeBorder(
            resized_mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        return padded_img, padded_mask


def generate_synthetic_sheet(output_path: str, canvas_size: Tuple[int, int] = (1536, 512)) -> str:
    """
    Generates a synthetic 3-view character orthographic sheet for testing.
    Front, Side, Back views drawn on a single canvas.
    """
    w_total, h_total = canvas_size
    img = Image.new("RGBA", (w_total, h_total), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    view_w = w_total // 3

    # Front View (Left)
    cx_front = view_w // 2
    # Head
    draw.ellipse([cx_front - 40, 60, cx_front + 40, 140], fill=(220, 150, 130, 255), outline=(50, 50, 50, 255), width=3)
    # Torso
    draw.rectangle([cx_front - 50, 140, cx_front + 50, 320], fill=(50, 120, 200, 255), outline=(30, 30, 30, 255), width=3)
    # Arms
    draw.rectangle([cx_front - 90, 140, cx_front - 50, 300], fill=(220, 150, 130, 255), outline=(30, 30, 30, 255), width=3)
    draw.rectangle([cx_front + 50, 140, cx_front + 90, 300], fill=(220, 150, 130, 255), outline=(30, 30, 30, 255), width=3)
    # Legs
    draw.rectangle([cx_front - 45, 320, cx_front - 10, 460], fill=(40, 40, 60, 255), outline=(20, 20, 20, 255), width=3)
    draw.rectangle([cx_front + 10, 320, cx_front + 45, 460], fill=(40, 40, 60, 255), outline=(20, 20, 20, 255), width=3)

    # Side View (Middle)
    cx_side = view_w + view_w // 2
    # Head
    draw.ellipse([cx_side - 30, 60, cx_side + 30, 140], fill=(220, 150, 130, 255), outline=(50, 50, 50, 255), width=3)
    # Torso (Thinner profile)
    draw.rectangle([cx_side - 30, 140, cx_side + 30, 320], fill=(40, 110, 190, 255), outline=(30, 30, 30, 255), width=3)
    # Arm
    draw.rectangle([cx_side - 15, 140, cx_side + 15, 300], fill=(210, 140, 120, 255), outline=(30, 30, 30, 255), width=3)
    # Legs
    draw.rectangle([cx_side - 20, 320, cx_side + 20, 460], fill=(35, 35, 55, 255), outline=(20, 20, 20, 255), width=3)

    # Back View (Right)
    cx_back = 2 * view_w + view_w // 2
    # Head
    draw.ellipse([cx_back - 40, 60, cx_back + 40, 140], fill=(220, 150, 130, 255), outline=(50, 50, 50, 255), width=3)
    # Torso
    draw.rectangle([cx_back - 50, 140, cx_back + 50, 320], fill=(40, 100, 180, 255), outline=(30, 30, 30, 255), width=3)
    # Arms
    draw.rectangle([cx_back - 90, 140, cx_back - 50, 300], fill=(220, 150, 130, 255), outline=(30, 30, 30, 255), width=3)
    draw.rectangle([cx_back + 50, 140, cx_back + 90, 300], fill=(220, 150, 130, 255), outline=(30, 30, 30, 255), width=3)
    # Legs
    draw.rectangle([cx_back - 45, 320, cx_back - 10, 460], fill=(30, 30, 50, 255), outline=(20, 20, 20, 255), width=3)
    draw.rectangle([cx_back + 10, 320, cx_back + 45, 460], fill=(30, 30, 50, 255), outline=(20, 20, 20, 255), width=3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return output_path
