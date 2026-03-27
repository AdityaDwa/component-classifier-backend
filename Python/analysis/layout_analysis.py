"""
Layout quality analysis: Clutter Score and Alignment Consistency.

Implements geometric layout metrics based on component positions and spatial
distribution. Both metrics are computed independently and combined with
diagnostic messages for actionable feedback.

CHANGES FROM PREVIOUS VERSION:
  calculate_clutter_score():
    - overlaps.append() now stores "component1" and "component2" as the full
      detection dicts (not just class name / bbox). Because analyzer.py assigns
      component_id to every detection before calling this function, those dicts
      already carry their IDs. This lets analyzer.py link overlap issues back to
      specific components.
    - Return dict now includes an "overlap_pairs" key containing the full list of
      overlap dicts. analyzer.py consumes this key (via dict.pop()) and does NOT
      forward it to the final JSON report.
  Everything else is unchanged.
"""

import math
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from analysis.metrics import LayoutMetrics
from constants.component_weights import COMPONENT_WEIGHTS


class LayoutAnalyzer:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.metrics_helper = LayoutMetrics()

        # Empirical constants
        self.N_MAX = 80  # Maximum components before layout is excessively populated
        self.NEAR_OVERLAP_DISTANCE = 20  # Pixels — threshold for near-overlap penalty

        # Clutter weights (equal importance per report)
        self.CLUTTER_WEIGHTS = {
            "density": 0.25,
            "area_ratio": 0.25,
            "spacing_variance": 0.25,
            "overlap_penalty": 0.25,
        }

        # Classification thresholds
        self.CLUTTER_THRESHOLDS = {
            "clean": 40,      # 0–40
            "moderate": 70,   # 40–70
            # crowded: 70–100
        }

        self.ALIGNMENT_THRESHOLDS = {
            "poor": 0.5,        # 0–0.5
            "acceptable": 0.75, # 0.5–0.75
            # excellent: 0.75–1.0
        }

    def calculate_clutter_score(
        self, detections: List[Dict], viewport_width: int, viewport_height: int
    ) -> Dict:
        """
        Calculates clutter score as weighted combination of four sub-criteria:
        1. Component Density  (N / N_max)
        2. Component Area Ratio  (total bbox area / viewport area)
        3. Spacing Variance  (consistency of whitespace distribution)
        4. Overlap Penalty  (IoU-based detection of overlapping components)

        Final score is scaled to 0–100. Component-type-aware weights are applied
        via COMPONENT_WEIGHTS so that intentional clustering (e.g. nav items) is
        penalised less than accidental overlap (e.g. two buttons).

        Args:
            detections (List[Dict]): Filtered detections — each dict must have
                at minimum: class, bbox. If component_id is present (set by
                analyzer.py before calling this), it is stored in overlap_pairs
                so analyzer.py can link issues back to individual components.
            viewport_width (int): Viewport width in pixels.
            viewport_height (int): Viewport height in pixels.

        Returns:
            Dict:
                "score"          — float 0–100
                "category"       — "clean" | "moderate" | "crowded"
                "breakdown"      — {density, area_ratio, spacing_variance,
                                    overlap_penalty} each in 0–1 range
                "issues"         — List[str] viewport-region-level descriptions
                "suggestions"    — List[str] actionable fixes
                "overlap_pairs"  — List[Dict] internal data consumed by
                                   analyzer.py; each entry has:
                                     "component1" : full detection dict (with id)
                                     "component2" : full detection dict (with id)
                                     "component1_class", "component2_class"
                                     "bbox1", "bbox2", "iou"
                                   analyzer.py pops this key before building the
                                   final JSON — it is NOT in the client response.
        """
        self.logger.info("inside calculate_clutter_score method..........")

        N = len(detections)
        W = viewport_width
        H = viewport_height
        viewport_area = W * H

        issues = []
        suggestions = []

        try:
            # ------------------------------------------------------------------
            # Sub-criterion 1: Component Density
            # D = N / N_max
            # ------------------------------------------------------------------
            density = N / self.N_MAX if self.N_MAX > 0 else 0.0
            density = min(density, 1.0)

            if N > 60:
                issues.append(
                    f"High component density ({N} components, threshold {self.N_MAX})"
                )
                suggestions.append(
                    "Consider consolidating related elements or using tabs/accordions"
                )

            # ------------------------------------------------------------------
            # Sub-criterion 2: Component Area Ratio
            # A_ratio = (sum of bbox areas) / viewport area
            # ------------------------------------------------------------------
            total_area = 0.0
            for det in detections:
                bbox = det.get("bbox", {})
                total_area += bbox.get("width", 0) * bbox.get("height", 0)

            area_ratio = total_area / viewport_area if viewport_area > 0 else 0.0
            area_ratio = min(area_ratio, 1.0)

            # ------------------------------------------------------------------
            # Sub-criterion 3: Spacing Variance
            # σ²_normalised = σ²(nearest_neighbour_distances) / σ²_max
            # ------------------------------------------------------------------
            distances = self.metrics_helper.nearest_neighbor_distance(detections)

            if len(distances) > 1:
                viewport_diagonal = math.sqrt(W ** 2 + H ** 2)
                max_variance = (viewport_diagonal / 4) ** 2
                spacing_variance = self.metrics_helper.variance_normalized(
                    distances, max_variance
                )
            else:
                spacing_variance = 0.0

            # ------------------------------------------------------------------
            # Sub-criterion 4: Overlap and Near-Overlap Penalty
            # P_normalised = (weighted overlap count) / N
            #
            # CHANGE: overlaps list now stores the full detection dicts for each
            # pair (component1 / component2). This lets analyzer.py read the
            # component_id from each dict and attach per-component issues.
            # ------------------------------------------------------------------
            overlaps = []
            total_penalty = 0.0

            for i, det_i in enumerate(detections):
                for j, det_j in enumerate(detections):
                    if i >= j:
                        continue

                    bbox_i = det_i.get("bbox", {})
                    bbox_j = det_j.get("bbox", {})

                    iou = self.metrics_helper.calculate_iou(bbox_i, bbox_j)

                    class_i = det_i.get("class", "container")
                    class_j = det_j.get("class", "container")

                    weight_i = COMPONENT_WEIGHTS.get(class_i, {}).get(
                        "overlap_penalty", 1.0
                    )
                    weight_j = COMPONENT_WEIGHTS.get(class_j, {}).get(
                        "overlap_penalty", 1.0
                    )
                    avg_weight = (weight_i + weight_j) / 2

                    if iou > 0.1:
                        total_penalty += 1.0 * avg_weight
                        # ── CHANGE ───────────────────────────────────────────
                        # Store full detection dicts (component1 / component2)
                        # in addition to the existing class/bbox fields.
                        # analyzer.py uses det_i["component_id"] and
                        # det_j["component_id"] to attach overlap issues back to
                        # the correct entries in the components array.
                        # ─────────────────────────────────────────────────────
                        overlaps.append(
                            {
                                "component1": det_i,
                                "component2": det_j,
                                "component1_class": class_i,
                                "component2_class": class_j,
                                "bbox1": bbox_i,
                                "bbox2": bbox_j,
                                "iou": round(iou, 3),
                            }
                        )
                    else:
                        # Check near-overlap (close proximity)
                        center_i_x = (
                            bbox_i.get("x", 0) + bbox_i.get("width", 0) / 2
                        )
                        center_i_y = (
                            bbox_i.get("y", 0) + bbox_i.get("height", 0) / 2
                        )
                        center_j_x = (
                            bbox_j.get("x", 0) + bbox_j.get("width", 0) / 2
                        )
                        center_j_y = (
                            bbox_j.get("y", 0) + bbox_j.get("height", 0) / 2
                        )

                        distance = math.sqrt(
                            (center_i_x - center_j_x) ** 2
                            + (center_i_y - center_j_y) ** 2
                        )

                        if distance < self.NEAR_OVERLAP_DISTANCE:
                            total_penalty += 0.5 * avg_weight

            overlap_penalty = total_penalty / N if N > 0 else 0.0
            overlap_penalty = min(overlap_penalty, 1.0)

            # Generate overlap diagnostics (viewport-region level for aggregate)
            if overlaps:
                overlap_regions = self._group_overlaps_by_region(overlaps, W, H)
                for region, region_overlaps in overlap_regions.items():
                    count = len(region_overlaps)
                    issues.append(
                        f"{count} component overlap{'s' if count > 1 else ''} "
                        f"detected in {region.replace('-', ' ')} region"
                    )
                    for overlap in region_overlaps[:2]:
                        suggestions.append(
                            f"Separate overlapping "
                            f"'{overlap['component1_class']}' and "
                            f"'{overlap['component2_class']}' components in "
                            f"{region.replace('-', ' ')}"
                        )

            # ------------------------------------------------------------------
            # Final Clutter Score
            # Clutter = 100 × (w1·D + w2·A_ratio + w3·S + w4·P)
            # ------------------------------------------------------------------
            clutter_score = 100 * (
                self.CLUTTER_WEIGHTS["density"] * density
                + self.CLUTTER_WEIGHTS["area_ratio"] * area_ratio
                + self.CLUTTER_WEIGHTS["spacing_variance"] * spacing_variance
                + self.CLUTTER_WEIGHTS["overlap_penalty"] * overlap_penalty
            )

            if clutter_score < self.CLUTTER_THRESHOLDS["clean"]:
                category = "clean"
            elif clutter_score < self.CLUTTER_THRESHOLDS["moderate"]:
                category = "moderate"
            else:
                category = "crowded"

            self.logger.info(
                f"Clutter score: {clutter_score:.2f} ({category}) — "
                f"D={density:.3f}, A={area_ratio:.3f}, "
                f"S={spacing_variance:.3f}, P={overlap_penalty:.3f}"
            )

            # ── CHANGE ────────────────────────────────────────────────────────
            # "overlap_pairs" is added to the return dict.
            # analyzer.py calls clutter_result.pop("overlap_pairs", []) to
            # consume this list and attach per-component issues. The key is
            # therefore absent from the final client-facing JSON.
            # ─────────────────────────────────────────────────────────────────
            return {
                "score": round(clutter_score, 2),
                "category": category,
                "breakdown": {
                    "density": round(density, 3),
                    "area_ratio": round(area_ratio, 3),
                    "spacing_variance": round(spacing_variance, 3),
                    "overlap_penalty": round(overlap_penalty, 3),
                },
                "issues": issues,
                "suggestions": suggestions,
                "overlap_pairs": overlaps,  # consumed + popped by analyzer.py
            }

        except Exception as e:
            self.logger.exception(f"Error calculating clutter score: {e}")
            return {
                "score": 0.0,
                "category": "error",
                "breakdown": {},
                "issues": [f"Calculation error: {str(e)}"],
                "suggestions": [],
                "overlap_pairs": [],
            }

    def _group_overlaps_by_region(
        self, overlaps: List[Dict], W: int, H: int
    ) -> Dict[str, List[Dict]]:
        """
        Groups overlapping component pairs by viewport region for spatial diagnostics.

        Args:
            overlaps (List[Dict]): List of overlap dicts with bbox1, bbox2.
            W (int): Viewport width.
            H (int): Viewport height.

        Returns:
            Dict[str, List[Dict]]: Overlaps grouped by region name.
        """
        regions = {}

        for overlap in overlaps:
            bbox1 = overlap["bbox1"]
            bbox2 = overlap["bbox2"]

            center_x = (
                bbox1.get("x", 0) + bbox1.get("width", 0) / 2
                + bbox2.get("x", 0) + bbox2.get("width", 0) / 2
            ) / 2

            center_y = (
                bbox1.get("y", 0) + bbox1.get("height", 0) / 2
                + bbox2.get("y", 0) + bbox2.get("height", 0) / 2
            ) / 2

            region = self.metrics_helper.get_region_name(center_x, center_y, W, H)

            if region not in regions:
                regions[region] = []
            regions[region].append(overlap)

        return regions

    def calculate_alignment_score(
        self, detections: List[Dict], viewport_width: int, viewport_height: int
    ) -> Dict:
        """
        Calculates alignment consistency as the average of three sub-criteria:
        1. Left Edge Alignment  (variance of left positions)
        2. Center Alignment     (variance of horizontal centers)
        3. Baseline Alignment   (variance of bottom edge positions)

        Each variance is normalised and subtracted from 1.0 so that lower variance
        (better alignment) yields higher scores.

        Args:
            detections (List[Dict]): Filtered detections with keys: class, bbox.
            viewport_width (int): Viewport width in pixels.
            viewport_height (int): Viewport height in pixels.

        Returns:
            Dict:
                "score"      — float 0–1
                "category"   — "poor" | "acceptable" | "excellent"
                "breakdown"  — {left_edge, center, baseline} each 0–1
                "issues"     — List[str]
                "suggestions"— List[str]
        """
        self.logger.info("inside calculate_alignment_score method..........")

        W = viewport_width
        H = viewport_height
        issues = []
        suggestions = []

        if len(detections) < 2:
            self.logger.warning("Fewer than 2 components — alignment score set to 1.0")
            return {
                "score": 1.0,
                "category": "excellent",
                "breakdown": {"left_edge": 1.0, "center": 1.0, "baseline": 1.0},
                "issues": [],
                "suggestions": [],
            }

        try:
            # ------------------------------------------------------------------
            # Sub-criterion 1: Left Edge Alignment
            # A_L = 1 - (σ²(left_positions) / σ²_L,max)
            # ------------------------------------------------------------------
            left_positions = []
            for det in detections:
                bbox = det.get("bbox", {})
                class_name = det.get("class", "container")
                weight = COMPONENT_WEIGHTS.get(class_name, {}).get(
                    "alignment_weight", 1.0
                )
                left_pos = bbox.get("x", 0)
                left_positions.extend([left_pos] * int(weight * 10))

            max_variance_left = (W / 2) ** 2
            variance_left = self.metrics_helper.variance_normalized(
                left_positions, max_variance_left
            )
            alignment_left = 1.0 - variance_left

            # ------------------------------------------------------------------
            # Sub-criterion 2: Center Alignment
            # A_C = 1 - (σ²(center_positions) / σ²_C,max)
            # ------------------------------------------------------------------
            center_positions = []
            for det in detections:
                bbox = det.get("bbox", {})
                class_name = det.get("class", "container")
                weight = COMPONENT_WEIGHTS.get(class_name, {}).get(
                    "alignment_weight", 1.0
                )
                center_pos = bbox.get("x", 0) + bbox.get("width", 0) / 2
                center_positions.extend([center_pos] * int(weight * 10))

            max_variance_center = (W / 2) ** 2
            variance_center = self.metrics_helper.variance_normalized(
                center_positions, max_variance_center
            )
            alignment_center = 1.0 - variance_center

            # ------------------------------------------------------------------
            # Sub-criterion 3: Baseline Alignment
            # A_B = 1 - (σ²(baseline_positions) / σ²_B,max)
            # ------------------------------------------------------------------
            baseline_positions = []
            for det in detections:
                bbox = det.get("bbox", {})
                class_name = det.get("class", "container")
                weight = COMPONENT_WEIGHTS.get(class_name, {}).get(
                    "alignment_weight", 1.0
                )
                baseline_pos = bbox.get("y", 0) + bbox.get("height", 0)
                baseline_positions.extend([baseline_pos] * int(weight * 10))

            max_variance_baseline = (H / 2) ** 2
            variance_baseline = self.metrics_helper.variance_normalized(
                baseline_positions, max_variance_baseline
            )
            alignment_baseline = 1.0 - variance_baseline

            # ------------------------------------------------------------------
            # Final Alignment Score
            # Alignment = (A_L + A_C + A_B) / 3
            # ------------------------------------------------------------------
            alignment_score = (
                alignment_left + alignment_center + alignment_baseline
            ) / 3

            if alignment_baseline < self.ALIGNMENT_THRESHOLDS["acceptable"]:
                issues.append(
                    f"Baseline alignment below threshold "
                    f"({alignment_baseline:.2f} < 0.75)"
                )
                suggestions.append(
                    "Align bottom edges of components on the same row"
                )

            if alignment_left < self.ALIGNMENT_THRESHOLDS["acceptable"]:
                issues.append(
                    f"Left edge alignment inconsistent "
                    f"({alignment_left:.2f} < 0.75)"
                )
                suggestions.append(
                    "Align left edges to a consistent grid column"
                )

            if alignment_score < self.ALIGNMENT_THRESHOLDS["poor"]:
                category = "poor"
            elif alignment_score < self.ALIGNMENT_THRESHOLDS["acceptable"]:
                category = "acceptable"
            else:
                category = "excellent"

            self.logger.info(
                f"Alignment score: {alignment_score:.3f} ({category}) — "
                f"Left={alignment_left:.3f}, Center={alignment_center:.3f}, "
                f"Baseline={alignment_baseline:.3f}"
            )

            return {
                "score": round(alignment_score, 3),
                "category": category,
                "breakdown": {
                    "left_edge": round(alignment_left, 3),
                    "center": round(alignment_center, 3),
                    "baseline": round(alignment_baseline, 3),
                },
                "issues": issues,
                "suggestions": suggestions,
            }

        except Exception as e:
            self.logger.exception(f"Error calculating alignment score: {e}")
            return {
                "score": 0.0,
                "category": "error",
                "breakdown": {},
                "issues": [f"Calculation error: {str(e)}"],
                "suggestions": [],
            }