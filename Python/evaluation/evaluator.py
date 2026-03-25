import csv
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from ultralytics import YOLO

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from constants.json_constants import label_to_id_dict

class Evaluator:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.class_names: list = [
            name for name, _ in sorted(label_to_id_dict.items(), key=lambda x: x[1])
        ]

    def load_model(self) -> YOLO:
        """
        Loads the best trained YOLO model from checkpoints.

        Args:
            None

        Returns:
            YOLO: Loaded YOLO model instance.

        Raises:
            FileNotFoundError: If best.pt does not exist at expected path.
            Exception: If model fails to load.
        """
        self.logger.info("inside load_model method..........")
        try:
            model_path: Path = (
                PathUtils().get_checkpoints_path().joinpath("train", "weights", "best.pt")
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            model: YOLO = YOLO(str(model_path))
            self.logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.exception(f"Error loading model: {e}")
            raise

    def run_on_test_set(self, model: YOLO):
        """
        Runs the loaded model against the test split and returns the YOLO metrics object.

        Args:
            model (YOLO): Loaded YOLO model instance.

        Returns:
            metrics: YOLO DetMetrics object containing all evaluation results.

        Raises:
            Exception: If evaluation fails.
        """
        self.logger.info("inside run_on_test_set method..........")
        try:
            yaml_path: Path = PathUtils().get_yaml_path()
            eval_dir: Path = PathUtils().get_base_path().joinpath("evaluation_results")
            eval_dir.mkdir(parents=True, exist_ok=True)

            # split="test" uses the test path from data.yaml — images never seen during training
            # plots=True auto-saves confusion matrix and PR curve to evaluation_results/test_evaluation/
            metrics = model.val(
                data=str(yaml_path),
                split="test",
                plots=True,
                project=str(eval_dir),
                name="test_evaluation",
                exist_ok=True,
            )
            self.logger.info("Test set evaluation completed successfully")
            return metrics
        except Exception as e:
            self.logger.exception(f"Error running test set evaluation: {e}")
            raise

    def extract_per_class_metrics(self, metrics) -> list:
        """
        Extracts per-class precision, recall, F1, mAP50 and mAP50-95 from the YOLO metrics object.

        Args:
            metrics: YOLO DetMetrics object returned by model.val().

        Returns:
            list: List of dicts, one per class, with keys:
                  class, precision, recall, f1, mAP50, mAP50_95
        """
        self.logger.info("inside extract_per_class_metrics method..........")
        per_class_data: list = []
        try:
            precision: np.ndarray = metrics.box.p
            recall: np.ndarray = metrics.box.r
            ap50: np.ndarray = metrics.box.ap50
            ap: np.ndarray = metrics.box.ap

            for i, class_name in enumerate(self.class_names):
                p: float = float(precision[i]) if i < len(precision) else 0.0
                r: float = float(recall[i]) if i < len(recall) else 0.0

                # 1e-9 added to denominator to prevent division by zero for undetected classes
                f1: float = (2 * p * r) / (p + r + 1e-9)

                per_class_data.append(
                    {
                        "class": class_name,
                        "precision": round(p, 4),
                        "recall": round(r, 4),
                        "f1": round(f1, 4),
                        "mAP50": round(float(ap50[i]) if i < len(ap50) else 0.0, 4),
                        "mAP50_95": round(float(ap[i]) if i < len(ap) else 0.0, 4),
                    }
                )

            self.logger.info(f"Per-class metrics extracted for {len(per_class_data)} classes")
        except Exception as e:
            self.logger.exception(f"Error extracting per-class metrics: {e}")

        return per_class_data

    def save_metrics_to_csv(self, per_class_metrics: list, metrics) -> None:
        """
        Saves per-class and overall metrics to timestamped CSV files inside evaluation_results/.

        Args:
            per_class_metrics (list): List of per-class metric dicts.
            metrics: YOLO DetMetrics object for overall scores.

        Returns:
            None
        """
        self.logger.info("inside save_metrics_to_csv method..........")
        try:
            eval_dir: Path = PathUtils().get_base_path().joinpath("evaluation_results")
            eval_dir.mkdir(parents=True, exist_ok=True)

            timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Per-class CSV — 14 rows, one per class
            per_class_path: Path = eval_dir.joinpath(f"per_class_metrics_{timestamp}.csv")
            fieldnames: list = ["class", "precision", "recall", "f1", "mAP50", "mAP50_95"]
            with open(per_class_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_class_metrics)
            self.logger.info(f"Per-class metrics saved to {per_class_path}")

            # Overall metrics CSV — 4 rows, averaged across all classes
            overall_path: Path = eval_dir.joinpath(f"overall_metrics_{timestamp}.csv")
            with open(overall_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["mAP50_overall", round(float(metrics.box.map50), 4)])
                writer.writerow(["mAP50_95_overall", round(float(metrics.box.map), 4)])
                writer.writerow(["precision_overall", round(float(metrics.box.mp), 4)])
                writer.writerow(["recall_overall", round(float(metrics.box.mr), 4)])
            self.logger.info(f"Overall metrics saved to {overall_path}")

        except Exception as e:
            self.logger.exception(f"Error saving metrics to CSV: {e}")

    def print_summary(self, per_class_metrics: list, metrics) -> None:
        """
        Prints a formatted per-class and overall metrics table to the terminal.

        Args:
            per_class_metrics (list): List of per-class metric dicts.
            metrics: YOLO DetMetrics object for overall scores.

        Returns:
            None
        """
        self.logger.info("inside print_summary method..........")
        try:
            separator: str = "=" * 75
            thin_separator: str = "-" * 75

            print(f"\n{separator}")
            print("  EVALUATION RESULTS — TEST SET (never seen during training)")
            print(separator)
            print(f"  Overall mAP@0.5      : {float(metrics.box.map50):.4f}")
            print(f"  Overall mAP@0.5:0.95 : {float(metrics.box.map):.4f}")
            print(f"  Overall Precision    : {float(metrics.box.mp):.4f}")
            print(f"  Overall Recall       : {float(metrics.box.mr):.4f}")
            print(thin_separator)
            print(
                f"  {'Class':<14} {'Precision':>10} {'Recall':>8} "
                f"{'F1':>8} {'mAP50':>8} {'mAP50-95':>10}"
            )
            print(thin_separator)
            for row in per_class_metrics:
                print(
                    f"  {row['class']:<14} {row['precision']:>10.4f} {row['recall']:>8.4f} "
                    f"{row['f1']:>8.4f} {row['mAP50']:>8.4f} {row['mAP50_95']:>10.4f}"
                )
            print(f"{separator}\n")

        except Exception as e:
            self.logger.exception(f"Error printing summary: {e}")

    def evaluator_main(self) -> None:
        """
        Main orchestrator for post-training evaluation on the test set.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: Propagates any critical errors after logging.
        """
        self.logger.info("inside evaluator_main method..........")
        try:
            model: YOLO = self.load_model()
            metrics = self.run_on_test_set(model)
            per_class_metrics: list = self.extract_per_class_metrics(metrics)
            self.save_metrics_to_csv(per_class_metrics, metrics)
            self.print_summary(per_class_metrics, metrics)
        except Exception as e:
            self.logger.exception(f"Error in evaluator_main: {e}")
            raise

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluator_main()