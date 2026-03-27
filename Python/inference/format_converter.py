"""
YOLO format to analysis format converter.

Handles conversion of YOLO Results objects to the dictionary format expected
by the UIAnalyzer. Extracts bounding boxes, class names, and confidence scores.
"""

from typing import List, Dict

from utils.logger_utils import Logger


class YOLOFormatConverter:
    """
    Converts YOLO prediction results to analysis-ready format.
    
    YOLO outputs boxes in xyxy format (x1, y1, x2, y2).
    Analysis expects xywh format (x, y, width, height).
    """

    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def convert(self, yolo_results) -> List[Dict]:
        """
        Converts YOLO Results object to analysis format.

        Args:
            yolo_results: YOLO Results object from model.predict()
                         (ultralytics.engine.results.Results)

        Returns:
            List[Dict]: Detections in format:
                [
                    {
                        "class": "button",
                        "confidence": 0.92,
                        "bbox": {"x": 100, "y": 200, "width": 150, "height": 80}
                    },
                    ...
                ]
                Note: component_id is NOT set here — it's assigned by analyzer.py
        """
        self.logger.info("inside convert method..........")

        detections = []

        try:
            # YOLO results is a list when predicting single image
            # Extract first result (single image)
            result = yolo_results[0] if isinstance(yolo_results, list) else yolo_results

            # Iterate through detected boxes
            for box in result.boxes:
                # Extract xyxy coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Convert to xywh format
                x = x1
                y = y1
                width = x2 - x1
                height = y2 - y1

                # Get class name
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # Get confidence
                confidence = float(box.conf[0])

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                })

            self.logger.info(f"Converted {len(detections)} YOLO detections to analysis format")

        except Exception as e:
            self.logger.exception(f"Error converting YOLO format: {e}")
            raise

        return detections