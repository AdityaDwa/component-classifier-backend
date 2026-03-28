"""
Mathematical helper functions for layout quality metrics.

Implements core geometric and statistical calculations used across
clutter, alignment, and overlap analysis. All functions are stateless
and can be called independently.
"""

import math
from typing import List, Dict, Tuple

import numpy as np

from utils.logger_utils import Logger


class LayoutMetrics:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calculates Intersection over Union (IoU) between two bounding boxes.
        
        IoU = Area of Intersection / Area of Union
        
        Used to detect overlapping components. IoU > 0.1 indicates significant overlap.

        Args:
            box1 (Dict): First bounding box with keys: x, y, width, height
            box2 (Dict): Second bounding box with keys: x, y, width, height

        Returns:
            float: IoU value in range [0, 1]. 0 = no overlap, 1 = perfect overlap
        """
        try:
            # Extract coordinates
            x1_min = box1["x"]
            y1_min = box1["y"]
            x1_max = box1["x"] + box1["width"]
            y1_max = box1["y"] + box1["height"]

            x2_min = box2["x"]
            y2_min = box2["y"]
            x2_max = box2["x"] + box2["width"]
            y2_max = box2["y"] + box2["height"]

            # Calculate intersection rectangle
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            # Check if there's actual intersection
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0

            # Calculate areas
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            box1_area = box1["width"] * box1["height"]
            box2_area = box2["width"] * box2["height"]
            union_area = box1_area + box2_area - inter_area

            if union_area == 0:
                return 0.0

            iou = inter_area / union_area
            return float(iou)

        except Exception as e:
            self.logger.exception(f"Error calculating IoU: {e}")
            return 0.0

    def nearest_neighbor_distance(self, detections: List[Dict]) -> List[float]:
        """
        Calculates the minimum distance to the nearest neighboring component
        for each detection. Used to assess whitespace distribution consistency.

        Distance is measured as Euclidean distance between bounding box centers.

        Args:
            detections (List[Dict]): List of components with bbox data

        Returns:
            List[float]: Minimum distance to nearest neighbor for each component.
                        Empty list if fewer than 2 components.
        """
        self.logger.info("inside nearest_neighbor_distance method..........")
        
        if len(detections) < 2:
            return []

        distances = []

        try:
            for i, det_i in enumerate(detections):
                bbox_i = det_i.get("bbox", {})
                center_x_i = bbox_i.get("x", 0) + bbox_i.get("width", 0) / 2
                center_y_i = bbox_i.get("y", 0) + bbox_i.get("height", 0) / 2

                min_distance = float("inf")

                for j, det_j in enumerate(detections):
                    if i == j:
                        continue

                    bbox_j = det_j.get("bbox", {})
                    center_x_j = bbox_j.get("x", 0) + bbox_j.get("width", 0) / 2
                    center_y_j = bbox_j.get("y", 0) + bbox_j.get("height", 0) / 2

                    # Euclidean distance between centers
                    distance = math.sqrt(
                        (center_x_i - center_x_j) ** 2 + (center_y_i - center_y_j) ** 2
                    )

                    if distance < min_distance:
                        min_distance = distance

                distances.append(min_distance if min_distance != float("inf") else 0.0)

            self.logger.info(f"Calculated distances for {len(distances)} components")

        except Exception as e:
            self.logger.exception(f"Error calculating nearest neighbor distances: {e}")
            return []

        return distances

    def variance_normalized(
        self, values: List[float], max_variance: float
    ) -> float:
        """
        Calculates variance of a list of values and normalizes it by a maximum threshold.
        
        Normalized variance = σ² / σ²_max
        
        Used for alignment consistency (variance of positions) and spacing consistency
        (variance of distances). Lower variance = more consistent layout.

        Args:
            values (List[float]): List of numeric values (positions, distances, etc.)
            max_variance (float): Maximum expected variance for normalization

        Returns:
            float: Normalized variance in range [0, 1]. 0 = perfectly consistent,
                   1 = maximum inconsistency
        """
        if len(values) < 2 or max_variance == 0:
            return 0.0

        try:
            variance = float(np.var(values))
            normalized = min(variance / max_variance, 1.0)  # Cap at 1.0
            return normalized

        except Exception as e:
            self.logger.exception(f"Error calculating normalized variance: {e}")
            return 0.0

    def get_viewport_regions(
        self, viewport_width: int, viewport_height: int
    ) -> Dict[str, Dict]:
        """
        Divides viewport into a 3x3 grid of named regions for spatial diagnostics.
        
        Grid layout:
        +------------------+------------------+------------------+
        |   top-left       |   top-center     |   top-right      |
        +------------------+------------------+------------------+
        |   middle-left    |   center         |   middle-right   |
        +------------------+------------------+------------------+
        |   bottom-left    |  bottom-center   |   bottom-right   |
        +------------------+------------------+------------------+

        Args:
            viewport_width (int): Viewport width in pixels
            viewport_height (int): Viewport height in pixels

        Returns:
            Dict[str, Dict]: Region names mapped to coordinate bounds:
                {
                    "top-left": {"x": (x_min, x_max), "y": (y_min, y_max)},
                    ...
                }
        """
        W = viewport_width
        H = viewport_height

        return {
            "top-left": {"x": (0, W / 3), "y": (0, H / 3)},
            "top-center": {"x": (W / 3, 2 * W / 3), "y": (0, H / 3)},
            "top-right": {"x": (2 * W / 3, W), "y": (0, H / 3)},
            "middle-left": {"x": (0, W / 3), "y": (H / 3, 2 * H / 3)},
            "center": {"x": (W / 3, 2 * W / 3), "y": (H / 3, 2 * H / 3)},
            "middle-right": {"x": (2 * W / 3, W), "y": (H / 3, 2 * H / 3)},
            "bottom-left": {"x": (0, W / 3), "y": (2 * H / 3, H)},
            "bottom-center": {"x": (W / 3, 2 * W / 3), "y": (2 * H / 3, H)},
            "bottom-right": {"x": (2 * W / 3, W), "y": (2 * H / 3, H)},
        }

    def get_region_name(
        self, x: float, y: float, viewport_width: int, viewport_height: int
    ) -> str:
        """
        Determines which viewport region a point falls into.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            viewport_width (int): Viewport width in pixels
            viewport_height (int): Viewport height in pixels

        Returns:
            str: Region name (e.g., "top-left", "center", "bottom-right")
        """
        regions = self.get_viewport_regions(viewport_width, viewport_height)

        for name, bounds in regions.items():
            x_min, x_max = bounds["x"]
            y_min, y_max = bounds["y"]

            if x_min <= x < x_max and y_min <= y < y_max:
                return name

        # Fallback for edge cases (coordinates exactly on boundary)
        return "center"