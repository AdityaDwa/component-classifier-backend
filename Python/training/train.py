from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from training.config import TRAINING_CONFIG
from training.yaml_generator import YAMLGenerator
from evaluation.mid_training_callback import MidTrainingCallback


class ModelTrainer:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.config = TRAINING_CONFIG

    def prepare_training(self) -> Path:
        """
        Prepares necessary files before training starts.
        Calls YAMLGenerator to create data.yaml and classes.txt.

        Args:
            None

        Returns:
            Path: Path to the generated data.yaml file.
        """
        self.logger.info("inside prepare_training method..........")
        generator = YAMLGenerator()
        yaml_path: Path = generator.yaml_generator_main()
        self.logger.info(f"Training preparation complete. data.yaml at {yaml_path}")
        return yaml_path

    def train_model(self, data_yaml_path: Path):
        """
        Loads YOLOv11s model, registers mid-training callbacks, and starts
        training. Callbacks are registered AFTER model is created but BEFORE
        model.train() is called — this is the only valid window for registration.

        The MidTrainingCallback is initialized with eval_interval=300 meaning
        training losses are logged every 300 batches. At 2307 iterations per
        epoch this gives approximately 7 log points per epoch which is enough
        to detect mid-epoch peaks that epoch-level logging would miss.

        Args:
            data_yaml_path (Path): Path to data.yaml YOLO configuration file.

        Returns:
            YOLO training results object.

        Raises:
            Exception: Logs and re-raises any training errors.
        """
        self.logger.info("inside train_model method..........")
        self.logger.info(f"Training configuration: {self.config}")

        model: YOLO = YOLO(self.config["model"])
        self.logger.info(f"Loaded model: {self.config['model']}")

        callback: MidTrainingCallback = MidTrainingCallback(eval_interval=300)
        callback.register_callbacks(model)
        self.logger.info("Mid-training callbacks registered")

        project_path: Path = PathUtils().get_base_path().joinpath(self.config["project"])

        results = model.train(
            data=str(data_yaml_path),
            epochs=self.config["epochs"],
            imgsz=self.config["imgsz"],
            batch=self.config["batch"],
            patience=self.config["patience"],
            device=self.config["device"],
            workers=self.config["workers"],
            project=str(project_path),
            name=self.config["name"],
            verbose=True,
        )

        self.logger.info("Training completed successfully")
        return results

    def training_main(self) -> None:
        """
        Main orchestrator for training. Prepares config files, registers
        mid-training callbacks, runs training, logs total elapsed time.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: Propagates any critical errors after logging.
        """
        self.logger.info("inside training_main method..........")
        start_time = datetime.now(timezone.utc)

        try:
            yaml_path: Path = self.prepare_training()
            results = self.train_model(yaml_path)
            self.logger.info(f"Training results: {results}")
        except Exception as e:
            self.logger.exception(f"Error during training: {e}")
            raise

        end_time = datetime.now(timezone.utc)
        total_time = end_time - start_time
        self.logger.info(f"Total training time: {total_time}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.training_main()