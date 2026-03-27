"""
Preprocessing utilities for detected UI components before layout analysis.

Handles filtering of structural containers and preparing component data for
metric calculations. Critical for preventing noise from full-page wrapper divs
that would artificially inflate clutter scores and skew alignment metrics.
"""

from pathlib import Path
from typing import List, Dict

from utils.logger_utils import Logger


class ComponentPreprocessor:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def filter_structural_containers(
        self, detections: List[Dict], viewport_width: int, viewport_height: int
    ) -> List[Dict]:
        """
        Removes large structural containers that are clearly page-layout wrappers
        rather than semantic UI components. Keeps smaller containers that represent
        actual UI elements like cards, panels, or sections.

        A container is considered structural (and filtered out) if it covers more
        than 80% of the viewport area. This threshold was chosen empirically to
        catch full-page wrappers while preserving semantic containers.

        Args:
            detections (List[Dict]): List of detected components with keys:
                - class (str): Component class name
                - bbox (Dict): Bounding box with x, y, width, height
            viewport_width (int): Viewport width in pixels (e.g., 1920)
            viewport_height (int): Viewport height in pixels (e.g., 1080)

        Returns:
            List[Dict]: Filtered detections with structural containers removed
        """
        self.logger.info("inside filter_structural_containers method..........")
        filtered = []
        viewport_area = viewport_width * viewport_height
        removed_count = 0

        try:
            for det in detections:
                # Non-containers always kept
                if det.get("class") != "container":
                    filtered.append(det)
                    continue

                # Calculate container area
                bbox = det.get("bbox", {})
                bbox_area = bbox.get("width", 0) * bbox.get("height", 0)
                coverage_ratio = bbox_area / viewport_area if viewport_area > 0 else 0

                # Filter out massive full-page wrappers (>80% viewport)
                if coverage_ratio > 0.80:
                    removed_count += 1
                    self.logger.debug(
                        f"Filtered structural container: {coverage_ratio:.2%} viewport coverage"
                    )
                    continue

                # Keep semantic containers (cards, panels, sections)
                filtered.append(det)

            self.logger.info(
                f"Filtered {removed_count} structural containers, kept {len(filtered)} components"
            )

        except Exception as e:
            self.logger.exception(f"Error filtering containers: {e}")
            # On error, return original detections to avoid data loss
            return detections

        return filtered

    def filter_text_bearing_components(self, detections: List[Dict]) -> List[Dict]:
        """
        Identifies components that should undergo OCR and color contrast analysis.
        Only text-bearing component types need text extraction and WCAG compliance checks.

        Text-bearing classes: heading, text, button, link, input, navigation
        Non-text classes: image, table, list, sidebar, dialog, header, footer, container

        Args:
            detections (List[Dict]): List of detected components with 'class' key

        Returns:
            List[Dict]: Components that should be processed for text and color analysis
        """
        self.logger.info("inside filter_text_bearing_components method..........")

        # Classes that typically contain text and need contrast checks
        text_bearing_classes = {
            "heading",
            "text",
            "button",
            "link",
            "input",
            "navigation",
        }

        text_components = [
            det for det in detections if det.get("class") in text_bearing_classes
        ]

        self.logger.info(
            f"Identified {len(text_components)} text-bearing components "
            f"from {len(detections)} total"
        )

        return text_components