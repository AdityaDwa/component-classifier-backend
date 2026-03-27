"""
Main inference orchestrator - YOLO detection + UI quality analysis.

This is the primary entry point for running complete webpage analysis.
Loads the trained YOLO model, runs detection, converts format, and calls
the UIAnalyzer to generate comprehensive quality metrics.

USAGE MODES:
  1. Python API: Import UIPredictor and call predict_and_analyze()
  2. Command Line: python predictor.py <image_path>
  3. Flask API: Use flask_api.py wrapper (see setup guide)
  4. FastAPI: Use fastapi_server.py wrapper (see setup guide)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from analysis.analyzer import UIAnalyzer
from inference.format_converter import YOLOFormatConverter


class UIPredictor:
    """
    Unified interface for UI component detection and analysis.
    
    Combines YOLO object detection with layout quality metrics, color contrast
    analysis, and accessibility compliance checking.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor with trained YOLO model.

        Args:
            model_path (str, optional): Path to trained YOLO weights (.pt file).
                                       If None, defaults to checkpoints/train/weights/best.pt
        """
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.path_utils = PathUtils()

        # Load YOLO model
        if model_path is None:
            model_path = str(
                self.path_utils.get_base_path().joinpath(
                    "checkpoints/train/weights/best.pt"
                )
            )

        self.logger.info(f"Loading YOLO model from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Ensure training is complete and best.pt exists."
            )

        self.model = YOLO(model_path)
        self.logger.info("YOLO model loaded successfully")

        # Initialize analyzer and converter
        self.analyzer = UIAnalyzer()
        self.converter = YOLOFormatConverter()

    def predict_and_analyze(
        self,
        image_path: str,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        confidence_threshold: float = 0.25,
        save_visualization: bool = False,
    ) -> Dict:
        """
        Complete inference pipeline: detect components + analyze layout quality.

        Pipeline:
          1. Run YOLO prediction on image
          2. Convert YOLO format → analysis format
          3. Call UIAnalyzer for comprehensive quality assessment
          4. Return complete JSON report

        Args:
            image_path (str): Absolute path to webpage screenshot
            viewport_width (int): Screenshot width in pixels (default 1920)
            viewport_height (int): Screenshot height in pixels (default 1080)
            confidence_threshold (float): YOLO confidence threshold (default 0.25)
            save_visualization (bool): Save annotated image with bboxes (default False)

        Returns:
            Dict: Complete analysis report with structure:
                {
                    "metadata": {...},
                    "components": [...],  # Per-component details with IDs, colors, issues
                    "clutter": {...},
                    "alignment": {...},
                    "contrast": {...},
                    "overall_grade": "B",
                    "summary": "..."
                }

        Raises:
            FileNotFoundError: If image_path doesn't exist
            Exception: If YOLO prediction or analysis fails
        """
        self.logger.info("inside predict_and_analyze method..........")
        self.logger.info(f"Processing image: {image_path}")

        # Validate image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # ------------------------------------------------------------------
            # Step 1: YOLO Prediction
            # ------------------------------------------------------------------
            self.logger.info("Running YOLO prediction...")
            
            yolo_results = self.model.predict(
                source=image_path,
                save=save_visualization,  # Save annotated image if requested
                conf=confidence_threshold,
                verbose=False  # Suppress YOLO console output
            )

            # ------------------------------------------------------------------
            # Step 2: Format Conversion
            # ------------------------------------------------------------------
            self.logger.info("Converting YOLO format to analysis format...")
            detections = self.converter.convert(yolo_results)

            self.logger.info(f"Detected {len(detections)} components")

            # Log detected classes for debugging
            class_counts = {}
            for det in detections:
                class_name = det["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            self.logger.info(f"Component breakdown: {class_counts}")

            # ------------------------------------------------------------------
            # Step 3: UI Quality Analysis
            # ------------------------------------------------------------------
            self.logger.info("Running UI quality analysis...")
            
            report = self.analyzer.analyze_layout(
                image_path=image_path,
                yolo_detections=detections,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )

            self.logger.info(
                f"Analysis complete - Grade: {report['overall_grade']}, "
                f"Clutter: {report['clutter']['score']:.1f}, "
                f"Alignment: {report['alignment']['score']:.2f}"
            )

            return report

        except Exception as e:
            self.logger.exception(f"Error in predict_and_analyze: {e}")
            raise


def main():
    """
    Command-line interface for running inference.

    Usage:
        python predictor.py <image_path> [--model <model_path>] [--save-viz]

    Examples:
        python predictor.py D:/screenshots/homepage.webp
        python predictor.py test.png --model custom_model.pt --save-viz
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run UI component detection and quality analysis"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to webpage screenshot"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model weights (default: checkpoints/train/weights/best.pt)"
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save annotated image with bounding boxes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)"
    )

    args = parser.parse_args()

    # Run prediction
    predictor = UIPredictor(model_path=args.model)
    result = predictor.predict_and_analyze(
        image_path=args.image_path,
        save_visualization=args.save_viz
    )

    # Output result
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        # Print to stdout for backend to capture
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()