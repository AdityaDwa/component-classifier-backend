"""
Layout quality analysis: Clutter Score and Alignment Consistency.

Implements geometric layout metrics based on component positions and spatial
distribution. Both metrics are computed independently and combined with
diagnostic messages for actionable feedback.

CHANGES IN THIS VERSION:
  - Added semantic nesting filter in calculate_clutter_score():
    Overlaps with IoU > 0.85 are still counted in the overlap_penalty
    (affecting the clutter score) but are NOT added to the overlaps list
    (so they don't appear as per-component issues).
    
    Rationale: IoU > 0.85 typically indicates semantic parent-child nesting
    (e.g., sidebar inside container, nav inside header) rather than broken
    layout. These are expected structural relationships, not layout problems.
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
        self.NEAR_OVERLAP_DISTANCE = 20  # Pixels - threshold for near-overlap penalty

        # Semantic nesting threshold
        # Overlaps with IoU above this are considered parent-child nesting,
        # not layout problems
        self.SEMANTIC_NESTING_THRESHOLD = 0.85

        # Clutter weights (equal importance per report)
        self.CLUTTER_WEIGHTS = {
            "density": 0.25,
            "area_ratio": 0.25,
            "spacing_variance": 0.25,
            "overlap_penalty": 0.25,
        }

        # Classification thresholds
        self.CLUTTER_THRESHOLDS = {
            "clean": 40,      # 0-40
            "moderate": 70,   # 40-70
            # crowded: 70-100
        }

        self.ALIGNMENT_THRESHOLDS = {
            "poor": 0.5,        # 0-0.5
            "acceptable": 0.75, # 0.5-0.75
            # excellent: 0.75-1.0
        }

    def calculate_clutter_score(
        self, detections: List[Dict], viewport_width: int, viewport_height: int
    ) -> Dict:
        """
        Calculates clutter score as weighted combination of four sub-criteria:
        1. Component Density (N/N_max)
        2. Component Area Ratio (total area / viewport area)
        3. Spacing Variance (consistency of whitespace distribution)
        4. Overlap Penalty (IoU-based detection of overlapping components)

        Final score scaled to 0-100 range with component-type-aware weighting.

        SEMANTIC NESTING FILTER:
          Overlaps with IoU > 0.85 are counted in the penalty score but NOT
          reported as individual component issues. This prevents false positives
          from parent-child nesting (e.g., sidebar in container, nav in header).

        Args:
            detections (List[Dict]): Filtered detections with keys: class, bbox, component_id
            viewport_width (int): Viewport width in pixels
            viewport_height (int): Viewport height in pixels

        Returns:
            Dict: {
                "score": float (0-100),
                "category": str ("clean" | "moderate" | "crowded"),
                "breakdown": {
                    "density": float (0-1),
                    "area_ratio": float (0-1),
                    "spacing_variance": float (0-1),
                    "overlap_penalty": float (0-1)
                },
                "overlap_pairs": [  # For analyzer.py to attach per-component issues
                    {
                        "component1_id": int,
                        "component2_id": int,
                        "component1_class": str,
                        "component2_class": str,
                        "iou": float
                    }
                ],
                "issues": List[str],
                "suggestions": List[str]
            }
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
            density = min(density, 1.0)  # Cap at 1.0

            if N > 60:
                issues.append(f"High component density ({N} components, threshold {self.N_MAX})")
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
            # σ²_normalized = σ²(nearest_neighbor_distances) / σ²_max
            # ------------------------------------------------------------------
            distances = self.metrics_helper.nearest_neighbor_distance(detections)

            if len(distances) > 1:
                # Max variance = (viewport diagonal / 4)²
                viewport_diagonal = math.sqrt(W**2 + H**2)
                max_variance = (viewport_diagonal / 4) ** 2

                spacing_variance = self.metrics_helper.variance_normalized(
                    distances, max_variance
                )
            else:
                spacing_variance = 0.0

            # ------------------------------------------------------------------
            # Sub-criterion 4: Overlap and Near-Overlap Penalty
            # P_normalized = (weighted overlap count) / N
            #
            # CRITICAL CHANGE: Semantic nesting filter applied here
            # ------------------------------------------------------------------
            overlaps = []  # Only overlaps with IoU < 0.85 (real layout problems)
            total_penalty = 0.0

            for i, det_i in enumerate(detections):
                for j, det_j in enumerate(detections):
                    if i >= j:
                        continue

                    bbox_i = det_i.get("bbox", {})
                    bbox_j = det_j.get("bbox", {})

                    iou = self.metrics_helper.calculate_iou(bbox_i, bbox_j)

                    # Apply component-type-aware overlap penalty
                    class_i = det_i.get("class", "container")
                    class_j = det_j.get("class", "container")
                    
                    weight_i = COMPONENT_WEIGHTS.get(class_i, {}).get("overlap_penalty", 1.0)
                    weight_j = COMPONENT_WEIGHTS.get(class_j, {}).get("overlap_penalty", 1.0)
                    avg_weight = (weight_i + weight_j) / 2

                    if iou > 0.1:
                        # ══════════════════════════════════════════════════════
                        # CHANGE APPLIED HERE (Lines 142-163)
                        # ══════════════════════════════════════════════════════
                        
                        # Always count overlap in penalty (affects clutter score)
                        total_penalty += 1.0 * avg_weight
                        
                        # SEMANTIC NESTING FILTER:
                        # Only add to overlaps list if IoU < 0.85
                        # High IoU (≥0.85) indicates parent-child nesting, not layout problem
                        if iou < self.SEMANTIC_NESTING_THRESHOLD:
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
                            # Log semantic nesting (for debugging, not reported as issue)
                            self.logger.debug(
                                f"Semantic nesting detected (IoU={iou:.2f}): "
                                f"{class_i} (#{det_i.get('component_id')}) contains "
                                f"{class_j} (#{det_j.get('component_id')})"
                            )
                        
                        # ══════════════════════════════════════════════════════
                        # END OF CHANGE
                        # ══════════════════════════════════════════════════════
                    else:
                        # Check near-overlap (close proximity)
                        center_i_x = bbox_i.get("x", 0) + bbox_i.get("width", 0) / 2
                        center_i_y = bbox_i.get("y", 0) + bbox_i.get("height", 0) / 2
                        center_j_x = bbox_j.get("x", 0) + bbox_j.get("width", 0) / 2
                        center_j_y = bbox_j.get("y", 0) + bbox_j.get("height", 0) / 2

                        distance = math.sqrt(
                            (center_i_x - center_j_x) ** 2 + (center_i_y - center_j_y) ** 2
                        )

                        if distance < self.NEAR_OVERLAP_DISTANCE:
                            total_penalty += 0.5 * avg_weight

            overlap_penalty = total_penalty / N if N > 0 else 0.0
            overlap_penalty = min(overlap_penalty, 1.0)

            # Generate overlap diagnostics (only for real layout problems, IoU < 0.85)
            if overlaps:
                overlap_regions = self._group_overlaps_by_region(
                    overlaps, W, H
                )
                for region, region_overlaps in overlap_regions.items():
                    count = len(region_overlaps)
                    issues.append(
                        f"{count} component overlap{'s' if count > 1 else ''} "
                        f"detected in {region.replace('-', ' ')} region"
                    )

                    # List specific overlapping components (max 2 per region)
                    for overlap in region_overlaps[:2]:
                        suggestions.append(
                            f"Separate overlapping '{overlap['component1_class']}' and "
                            f"'{overlap['component2_class']}' components in {region.replace('-', ' ')}"
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

            # Classify
            if clutter_score < self.CLUTTER_THRESHOLDS["clean"]:
                category = "clean"
            elif clutter_score < self.CLUTTER_THRESHOLDS["moderate"]:
                category = "moderate"
            else:
                category = "crowded"

            self.logger.info(
                f"Clutter score: {clutter_score:.2f} ({category}) - "
                f"D={density:.3f}, A={area_ratio:.3f}, S={spacing_variance:.3f}, P={overlap_penalty:.3f}"
            )

            return {
                "score": round(clutter_score, 2),
                "category": category,
                "breakdown": {
                    "density": round(density, 3),
                    "area_ratio": round(area_ratio, 3),
                    "spacing_variance": round(spacing_variance, 3),
                    "overlap_penalty": round(overlap_penalty, 3),
                },
                "overlap_pairs": overlaps,  # For analyzer.py to attach per-component issues
                "issues": issues,
                "suggestions": suggestions,
            }

        except Exception as e:
            self.logger.exception(f"Error calculating clutter score: {e}")
            return {
                "score": 0.0,
                "category": "error",
                "breakdown": {},
                "overlap_pairs": [],
                "issues": [f"Calculation error: {str(e)}"],
                "suggestions": [],
            }

    def _group_overlaps_by_region(
        self, overlaps: List[Dict], W: int, H: int
    ) -> Dict[str, List[Dict]]:
        """
        Groups overlapping component pairs by viewport region for spatial diagnostics.

        Args:
            overlaps (List[Dict]): List of overlap dicts with bbox1, bbox2
            W (int): Viewport width
            H (int): Viewport height

        Returns:
            Dict[str, List[Dict]]: Overlaps grouped by region name
        """
        regions = {}

        for overlap in overlaps:
            bbox1 = overlap["bbox1"]
            bbox2 = overlap["bbox2"]

            # Center point of overlap (midpoint between two bbox centers)
            center_x = (
                bbox1.get("x", 0) + bbox1.get("width", 0) / 2 +
                bbox2.get("x", 0) + bbox2.get("width", 0) / 2
            ) / 2

            center_y = (
                bbox1.get("y", 0) + bbox1.get("height", 0) / 2 +
                bbox2.get("y", 0) + bbox2.get("height", 0) / 2
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
        1. Left Edge Alignment (variance of left positions)
        2. Center Alignment (variance of horizontal centers)
        3. Baseline Alignment (variance of bottom edge positions)

        Each variance is normalized and subtracted from 1.0 so that lower variance
        (better alignment) yields higher scores.

        Args:
            detections (List[Dict]): Filtered detections with keys: class, bbox, component_id
            viewport_width (int): Viewport width in pixels
            viewport_height (int): Viewport height in pixels

        Returns:
            Dict: {
                "score": float (0-1),
                "category": str ("poor" | "acceptable" | "excellent"),
                "breakdown": {
                    "left_edge": float (0-1),
                    "center": float (0-1),
                    "baseline": float (0-1)
                },
                "issues": List[str],
                "suggestions": List[str]
            }
        """
        self.logger.info("inside calculate_alignment_score method..........")

        W = viewport_width
        H = viewport_height
        issues = []
        suggestions = []

        if len(detections) < 2:
            self.logger.warning("Fewer than 2 components, alignment score set to 1.0")
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
                
                # Apply component-type-aware weight
                weight = COMPONENT_WEIGHTS.get(class_name, {}).get("alignment_weight", 1.0)
                
                # Weight influences how much this component affects variance
                # Higher weight = component's misalignment penalized more
                left_pos = bbox.get("x", 0)
                left_positions.extend([left_pos] * int(weight * 10))  # Repeat based on weight

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
                weight = COMPONENT_WEIGHTS.get(class_name, {}).get("alignment_weight", 1.0)
                
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
                weight = COMPONENT_WEIGHTS.get(class_name, {}).get("alignment_weight", 1.0)
                
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
            alignment_score = (alignment_left + alignment_center + alignment_baseline) / 3

            # Generate diagnostics
            if alignment_baseline < self.ALIGNMENT_THRESHOLDS["acceptable"]:
                issues.append(
                    f"Baseline alignment below threshold ({alignment_baseline:.2f} < 0.75)"
                )
                suggestions.append("Align bottom edges of components on the same row")

            if alignment_left < self.ALIGNMENT_THRESHOLDS["acceptable"]:
                issues.append(
                    f"Left edge alignment inconsistent ({alignment_left:.2f} < 0.75)"
                )
                suggestions.append("Align left edges to a consistent grid column")

            # Classify
            if alignment_score < self.ALIGNMENT_THRESHOLDS["poor"]:
                category = "poor"
            elif alignment_score < self.ALIGNMENT_THRESHOLDS["acceptable"]:
                category = "acceptable"
            else:
                category = "excellent"

            self.logger.info(
                f"Alignment score: {alignment_score:.3f} ({category}) - "
                f"Left={alignment_left:.3f}, Center={alignment_center:.3f}, Baseline={alignment_baseline:.3f}"
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