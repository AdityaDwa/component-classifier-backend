import csv
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from constants.json_constants import label_to_id_dict


class MidTrainingCallback:
    """
    Registers two callbacks into YOLO training:

    Callback 1 — on_batch_end (every N iterations):
        Fires after every training batch. Every `eval_interval` iterations,
        writes box_loss, cls_loss, dfl_loss to iteration_losses_TIMESTAMP.csv.
        This gives you sub-epoch loss visibility. If your model peaks at
        iteration 1800 of epoch 3 and overfits by epoch end, you will see
        the loss rise again in this file. YOLO's results.csv never shows this.

    Callback 2 — on_epoch_end (every epoch):
        Fires after each epoch's validation completes. Extracts per-class
        precision, recall, F1, mAP50 for all 14 classes and appends one row
        to epoch_per_class_metrics_TIMESTAMP.csv.
        YOLO's results.csv only stores overall mAP. This gives you per-class
        breakdown at every epoch so you can track if rare classes like
        sidebar or dialog ever improve.
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
        Creates both CSV files with their headers at callback initialization
        time (before training starts). Timestamp in filename ensures each
        training run gets its own files and previous runs are not overwritten.

        Args:
            None

        Returns:
            None
        """
        self.logger.info("inside _setup_csv_files method..........")
        try:
            eval_dir: Path = PathUtils().get_base_path().joinpath("evaluation_results")
            eval_dir.mkdir(parents=True, exist_ok=True)

            timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # --- Iteration-level loss CSV ---
            self.iteration_csv_path = eval_dir.joinpath(
                f"iteration_losses_{timestamp}.csv"
            )
            with open(self.iteration_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["global_iteration", "epoch", "box_loss", "cls_loss", "dfl_loss"]
                )
            self.logger.info(f"Iteration loss CSV created at {self.iteration_csv_path}")

            # --- Epoch per-class metrics CSV ---
            # One column per class per metric: heading_precision, heading_recall, etc.
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
        Called by YOLO after every training batch automatically.
        Increments the global iteration counter and on every Nth iteration
        reads the training losses from the trainer object and appends one
        row to iteration_losses CSV.

        trainer.loss_items is a tensor with [box_loss, cls_loss, dfl_loss].
        trainer.epoch is 0-indexed so we add 1 for human-readable output.

        Args:
            trainer: YOLO BaseTrainer instance passed automatically by YOLO.

        Returns:
            None
        """
        self.global_iteration += 1

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
        Called by YOLO after each epoch's validation run completes automatically.
        Reads per-class precision and recall from trainer.validator.metrics,
        computes F1 manually, and appends one row to epoch_per_class_metrics CSV.

        F1 is computed here instead of reading from YOLO directly because
        YOLO's internal F1 computation can differ across ultralytics versions.
        Computing it manually as harmonic mean of P and R is always reliable.

        If a class has no detections its precision and recall are 0 so F1
        will also be 0 — this is correct behavior, not a bug.

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
        Registers both callbacks to the YOLO model. Must be called AFTER
        creating the YOLO model instance and BEFORE calling model.train().
        Order matters — YOLO attaches callbacks before training loop starts.

        Args:
            model: YOLO model instance (from ultralytics import YOLO).

        Returns:
            None
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