"""
Color and contrast analysis for text-bearing UI components.

Implements OCR-based text detection and WCAG AA color contrast compliance checking.
Uses K-means clustering to extract dominant foreground and background colors from
detected component regions.

CHANGES FROM PREVIOUS VERSION:
  _rgb_to_hex() [NEW]:
    - Static helper that converts an (R, G, B) tuple to a '#RRGGBB' hex string.
      Used to produce frontend-ready color codes instead of raw tuples.

  assess_contrast_compliance():
    - Now builds a "per_component_data" list in addition to the existing
      "failed_components" list. Every text-bearing detection that was processed
      gets one entry in per_component_data, whether or not text was found and
      whether or not it failed the WCAG check. This lets analyzer.py populate
      per-component color detail (hex codes, contrast ratio, wcag_compliant flag)
      for every text component in the "components" array of the final report.
    - failed_components entries now include "component_id", "foreground_hex", and
      "background_hex" in addition to the existing fields.
    - "per_component_data" is returned in the result dict. analyzer.py consumes it
      via dict.pop() so it does NOT appear in the client-facing JSON.
"""

import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple
from pathlib import Path

from sklearn.cluster import KMeans

from utils.logger_utils import Logger
from utils.path_utils import PathUtils


class ColorAnalyzer:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

        # Initialize EasyOCR (downloads models on first run — takes ~30 s once)
        self.logger.info("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True if available
        self.logger.info("EasyOCR initialized")

        # WCAG AA minimum contrast threshold
        self.CONTRAST_THRESHOLD = 4.5

    # ------------------------------------------------------------------
    # New helper
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_hex(rgb_tuple) -> str:
        """
        Converts an (R, G, B) tuple (0–255 range) to a '#RRGGBB' hex string.

        Values are clamped to 0–255 to guard against floating-point K-means
        cluster centres that may be slightly out of range.

        Args:
            rgb_tuple: Iterable of three numeric values (R, G, B).

        Returns:
            str: Hex colour string, e.g. '#A3B2C1'.
        """
        r = max(0, min(255, int(rgb_tuple[0])))
        g = max(0, min(255, int(rgb_tuple[1])))
        b = max(0, min(255, int(rgb_tuple[2])))
        return "#{:02X}{:02X}{:02X}".format(r, g, b)

    # ------------------------------------------------------------------
    # Core extraction methods (unchanged)
    # ------------------------------------------------------------------

    def extract_component_colors(
        self, image_path: str, detection: Dict
    ) -> Dict:
        """
        Extracts foreground (text) and background colors from a component region
        using K-means clustering. Assumes darker cluster = text, lighter = background.

        Args:
            image_path (str): Path to the full webpage screenshot.
            detection (Dict): Component detection with keys: class, bbox.

        Returns:
            Dict: {
                "foreground_rgb": (R, G, B),
                "background_rgb": (R, G, B),
                "has_text": bool,
                "text_content": str,
                "method": "kmeans"
            }
            Returns None if extraction fails.
        """
        self.logger.info("inside extract_component_colors method..........")

        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None

            bbox = detection.get("bbox", {})
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("width", 0))
            h = int(bbox.get("height", 0))

            img_h, img_w = image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))

            region = image[y: y + h, x: x + w]

            if region.size == 0:
                self.logger.warning(f"Empty region for bbox: {bbox}")
                return None

            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            pixels = region_rgb.reshape(-1, 3).astype(np.float32)

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_

            brightness = [np.mean(color) for color in colors]
            if brightness[0] < brightness[1]:
                foreground = colors[0]
                background = colors[1]
            else:
                foreground = colors[1]
                background = colors[0]

            has_text, text_content = self._detect_text_ocr(region_rgb)

            return {
                "foreground_rgb": tuple(foreground.astype(int)),
                "background_rgb": tuple(background.astype(int)),
                "has_text": has_text,
                "text_content": text_content,
                "method": "kmeans",
            }

        except Exception as e:
            self.logger.exception(f"Error extracting colors: {e}")
            return None

    def _detect_text_ocr(self, region_rgb: np.ndarray) -> Tuple[bool, str]:
        """
        Runs EasyOCR on a component region to detect if it contains text.

        Args:
            region_rgb (np.ndarray): RGB image of component region.

        Returns:
            Tuple[bool, str]: (has_text, text_content)
        """
        try:
            results = self.ocr_reader.readtext(region_rgb, detail=0)
            if results:
                return True, " ".join(results)
            return False, ""
        except Exception as e:
            self.logger.exception(f"OCR detection error: {e}")
            return False, ""

    def calculate_contrast_ratio(
        self,
        fg_rgb: Tuple[int, int, int],
        bg_rgb: Tuple[int, int, int],
    ) -> float:
        """
        Calculates WCAG contrast ratio between foreground and background colors.

        Formula:
        1. Normalise RGB values to [0, 1]
        2. Apply gamma correction per WCAG standard
        3. Calculate relative luminance: L = 0.2126R + 0.7152G + 0.0722B
        4. Contrast ratio = (L1 + 0.05) / (L2 + 0.05) where L1 > L2

        Args:
            fg_rgb (Tuple[int, int, int]): Foreground RGB (0–255 range).
            bg_rgb (Tuple[int, int, int]): Background RGB (0–255 range).

        Returns:
            float: Contrast ratio (≥ 1.0). WCAG AA requires ≥ 4.5:1.
        """
        try:
            fg_norm = np.array(fg_rgb) / 255.0
            bg_norm = np.array(bg_rgb) / 255.0

            def gamma_correct(channel):
                if channel <= 0.03928:
                    return channel / 12.92
                return ((channel + 0.055) / 1.055) ** 2.4

            fg_corrected = np.array([gamma_correct(c) for c in fg_norm])
            bg_corrected = np.array([gamma_correct(c) for c in bg_norm])

            luminance_fg = (
                0.2126 * fg_corrected[0]
                + 0.7152 * fg_corrected[1]
                + 0.0722 * fg_corrected[2]
            )
            luminance_bg = (
                0.2126 * bg_corrected[0]
                + 0.7152 * bg_corrected[1]
                + 0.0722 * bg_corrected[2]
            )

            lighter = max(luminance_fg, luminance_bg)
            darker = min(luminance_fg, luminance_bg)
            contrast = (lighter + 0.05) / (darker + 0.05)
            return float(contrast)

        except Exception as e:
            self.logger.exception(f"Error calculating contrast ratio: {e}")
            return 1.0

    # ------------------------------------------------------------------
    # Main compliance assessment (modified)
    # ------------------------------------------------------------------

    def assess_contrast_compliance(
        self, image_path: str, detections: List[Dict]
    ) -> Dict:
        """
        Assesses WCAG AA color contrast compliance for all text-bearing components.

        For each detection this method:
          1. Extracts foreground / background colors via K-means (extract_component_colors).
          2. Runs EasyOCR to confirm text is present (inside extract_component_colors).
          3. Calculates WCAG contrast ratio if text is found.
          4. Records a per_component_data entry for every detection processed
             (not just failures), so that analyzer.py can populate color detail
             for ALL text components in the final report.

        Args:
            image_path (str): Path to the full webpage screenshot.
            detections (List[Dict]): Text-bearing component detections. Each dict
                should carry "component_id" (assigned by analyzer.py), "class",
                and "bbox".

        Returns:
            Dict:
                "average_contrast"      — float, mean contrast ratio of text components
                "compliant_count"       — int
                "total_text_components" — int  (components where OCR found text)
                "compliance_rate"       — float 0–1
                "failed_components"     — List[Dict] with keys:
                                            component_id, class, text_content,
                                            contrast_ratio, threshold,
                                            foreground_hex, background_hex,
                                            foreground_rgb, background_rgb, issue
                "suggestions"           — List[str]
                "per_component_data"    — List[Dict] INTERNAL — consumed + popped by
                                          analyzer.py, NOT in the final JSON.
                                          Each entry: component_id, foreground_hex,
                                          background_hex, contrast_ratio,
                                          wcag_compliant, has_text, text_content.
                                          foreground_hex / background_hex are None
                                          when OCR found no text or extraction failed.
        """
        self.logger.info("inside assess_contrast_compliance method..........")

        failed_components = []
        suggestions = []
        contrast_ratios = []

        # ── CHANGE ───────────────────────────────────────────────────────────
        # per_component_data collects one entry per detection regardless of
        # whether OCR found text. analyzer.py pops this from the result and
        # uses it to populate the "colors" field in the components array.
        # ─────────────────────────────────────────────────────────────────────
        per_component_data = []

        for det in detections:
            component_id = det.get("component_id")

            # ── Step A: Extract K-means colors + run OCR ──────────────────
            color_data = self.extract_component_colors(image_path, det)

            if not color_data:
                # Image crop failed (image unreadable, zero-size bbox, etc.)
                # Still record the component so analyzer knows it was attempted.
                per_component_data.append(
                    {
                        "component_id": component_id,
                        "foreground_hex": None,
                        "background_hex": None,
                        "contrast_ratio": None,
                        "wcag_compliant": None,
                        "has_text": False,
                        "text_content": "",
                    }
                )
                continue

            has_text = color_data.get("has_text", False)

            if not has_text:
                # OCR found no text — record it but skip contrast calculation.
                per_component_data.append(
                    {
                        "component_id": component_id,
                        "foreground_hex": None,
                        "background_hex": None,
                        "contrast_ratio": None,
                        "wcag_compliant": None,
                        "has_text": False,
                        "text_content": "",
                    }
                )
                continue

            # ── Step B: Calculate WCAG contrast ───────────────────────────
            fg_rgb = color_data["foreground_rgb"]
            bg_rgb = color_data["background_rgb"]
            contrast_ratio = self.calculate_contrast_ratio(fg_rgb, bg_rgb)
            contrast_ratios.append(contrast_ratio)

            fg_hex = self._rgb_to_hex(fg_rgb)
            bg_hex = self._rgb_to_hex(bg_rgb)
            is_compliant = contrast_ratio >= self.CONTRAST_THRESHOLD

            # ── Step C: Record per-component color data ───────────────────
            per_component_data.append(
                {
                    "component_id": component_id,
                    "foreground_hex": fg_hex,
                    "background_hex": bg_hex,
                    "contrast_ratio": round(contrast_ratio, 2),
                    "wcag_compliant": is_compliant,
                    "has_text": True,
                    "text_content": color_data.get("text_content", ""),
                }
            )

            # ── Step D: Track failures ─────────────────────────────────────
            if not is_compliant:
                failed_components.append(
                    {
                        # ── CHANGE: component_id and hex codes now included ──
                        "component_id": component_id,
                        "class": det.get("class"),
                        "text_content": color_data.get("text_content", ""),
                        "contrast_ratio": round(contrast_ratio, 2),
                        "threshold": self.CONTRAST_THRESHOLD,
                        "foreground_hex": fg_hex,
                        "background_hex": bg_hex,
                        "foreground_rgb": fg_rgb,
                        "background_rgb": bg_rgb,
                        "issue": (
                            f"Text color {fg_hex} on background {bg_hex} "
                            f"(contrast {contrast_ratio:.2f}:1 "
                            f"< {self.CONTRAST_THRESHOLD}:1)"
                        ),
                    }
                )
                suggestions.append(
                    f"Increase contrast for '{det.get('class')}' component "
                    f"(#{component_id}): current {contrast_ratio:.2f}:1, "
                    f"need {self.CONTRAST_THRESHOLD}:1"
                )

        total = len(contrast_ratios)
        compliant_count = sum(
            1 for ratio in contrast_ratios if ratio >= self.CONTRAST_THRESHOLD
        )
        average_contrast = float(np.mean(contrast_ratios)) if contrast_ratios else 0.0
        compliance_rate = compliant_count / total if total > 0 else 0.0

        self.logger.info(
            f"Contrast assessment: {compliant_count}/{total} compliant, "
            f"average ratio {average_contrast:.2f}:1"
        )

        return {
            "average_contrast": round(average_contrast, 2),
            "compliant_count": compliant_count,
            "total_text_components": total,
            "compliance_rate": round(compliance_rate, 3),
            "failed_components": failed_components,
            "suggestions": suggestions,
            # ── CHANGE: per_component_data added ──────────────────────────
            # analyzer.py pops this key after reading it, so it is absent
            # from the client-facing JSON report.
            "per_component_data": per_component_data,
        }