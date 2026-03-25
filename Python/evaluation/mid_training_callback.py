import csv
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from constants.json_constants import label_to_id_dict


class MidTrainingCallback:
    """
    Registers two callbacks into YOLO training to capture sub-epoch loss
    visibility and per-class metrics at every epoch.

    Callback 1 — on_batch_end: logs box/cls/dfl loss every N iterations.
    Callback 2 — on_epoch_end: logs per-class precision/recall/F1/mAP50 every epoch.
    """

    def __init__(self, eval_interval: int = 300):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.eval_interval: int = eval_interval
        self.global_iteration: int = 0
        self.class_names: list = [
            name for name, _ in sorted(label_to_id_dict.items(), key=lambda x: x[1])
        ]
        self.iteration_csv_path: Path = None
        self.epoch_csv_path: Path = None
        self._setup_csv_files()

    def _setup_csv_files(self) -> None:
        """
        Creates both CSV files with headers before training starts.

        Args:
            None

        Returns:
            None
        """
        self.logger.info("inside _setup_csv_files method..........")
        try:
            eval_dir: Path = PathUtils().get_base_path().joinpath("evaluation_results")
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Timestamp in filename ensures each training run gets its own files
            timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Iteration-level loss CSV
            self.iteration_csv_path = eval_dir.joinpath(
                f"iteration_losses_{timestamp}.csv"
            )
            with open(self.iteration_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["global_iteration", "epoch", "box_loss", "cls_loss", "dfl_loss"]
                )
            self.logger.info(f"Iteration loss CSV created at {self.iteration_csv_path}")

            # Epoch per-class metrics CSV — columns: heading_precision, heading_recall, etc.
            self.epoch_csv_path = eval_dir.joinpath(
                f"epoch_per_class_metrics_{timestamp}.csv"
            )
            header: list = ["epoch"]
            for class_name in self.class_names:
                header.extend(
                    [
                        f"{class_name}_precision",
                        f"{class_name}_recall",
                        f"{class_name}_f1",
                        f"{class_name}_mAP50",
                    ]
                )
            with open(self.epoch_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self.logger.info(f"Epoch per-class CSV created at {self.epoch_csv_path}")

        except Exception as e:
            self.logger.exception(f"Error setting up CSV files: {e}")

    def on_batch_end(self, trainer) -> None:
        """
        Logs training losses to CSV every N iterations. Called automatically by YOLO after every batch.

        Args:
            trainer: YOLO BaseTrainer instance passed automatically by YOLO.

        Returns:
            None
        """
        self.global_iteration += 1

        # Only write every eval_interval iterations — no performance hit otherwise
        if self.global_iteration % self.eval_interval != 0:
            return

        self.logger.info(
            f"inside on_batch_end method - global iteration {self.global_iteration}.........."
        )
        try:
            loss_items = trainer.loss_items
            box_loss: float = round(float(loss_items[0]), 5) if len(loss_items) > 0 else 0.0
            cls_loss: float = round(float(loss_items[1]), 5) if len(loss_items) > 1 else 0.0
            dfl_loss: float = round(float(loss_items[2]), 5) if len(loss_items) > 2 else 0.0

            # trainer.epoch is 0-indexed, adding 1 for human-readable output
            with open(self.iteration_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.global_iteration,
                        trainer.epoch + 1,
                        box_loss,
                        cls_loss,
                        dfl_loss,
                    ]
                )
            self.logger.info(
                f"Iteration {self.global_iteration} losses logged — "
                f"box: {box_loss}, cls: {cls_loss}, dfl: {dfl_loss}"
            )
        except Exception as e:
            self.logger.exception(
                f"Error in on_batch_end at iteration {self.global_iteration}: {e}"
            )

    def on_epoch_end(self, trainer) -> None:
        """
        Logs per-class precision, recall, F1 and mAP50 to CSV after each epoch. Called automatically by YOLO after each epoch's validation completes.

        Args:
            trainer: YOLO BaseTrainer instance passed automatically by YOLO.

        Returns:
            None
        """
        current_epoch: int = trainer.epoch + 1
        self.logger.info(
            f"inside on_epoch_end method - epoch {current_epoch}.........."
        )
        try:
            metrics = trainer.validator.metrics
            precision: np.ndarray = metrics.box.p
            recall: np.ndarray = metrics.box.r
            ap50: np.ndarray = metrics.box.ap50

            row: list = [current_epoch]
            for i, class_name in enumerate(self.class_names):
                p: float = float(precision[i]) if i < len(precision) else 0.0
                r: float = float(recall[i]) if i < len(recall) else 0.0
                # F1 computed manually — more reliable than YOLO internal across versions
                f1: float = (2 * p * r) / (p + r + 1e-9)
                ap: float = float(ap50[i]) if i < len(ap50) else 0.0

                row.extend(
                    [
                        round(p, 4),
                        round(r, 4),
                        round(f1, 4),
                        round(ap, 4),
                    ]
                )

            with open(self.epoch_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            self.logger.info(f"Epoch {current_epoch} per-class metrics logged")

        except Exception as e:
            self.logger.exception(f"Error in on_epoch_end at epoch {current_epoch}: {e}")

    def register_callbacks(self, model) -> None:
        """
        Registers both callbacks to the YOLO model before training starts.

        Args:
            model: YOLO model instance (from ultralytics import YOLO).

        Returns:
            None

        Raises:
            Exception: If callback registration fails.
        """
        self.logger.info("inside register_callbacks method..........")
        try:
            model.add_callback("on_train_batch_end", self.on_batch_end)
            model.add_callback("on_fit_epoch_end", self.on_epoch_end)
            self.logger.info(
                f"Callbacks registered — iteration logging every {self.eval_interval} iterations"
            )
        except Exception as e:
            self.logger.exception(f"Error registering callbacks: {e}")
            raise