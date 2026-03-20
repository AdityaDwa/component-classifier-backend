from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from training.config import TRAINING_CONFIG, CLASS_WEIGHTS
from training.yaml_generator import YAMLGenerator


class ModelTrainer:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.config = TRAINING_CONFIG

    def prepare_training(self) -> Path:
        """
        Prepares necessary files before training.

        Returns:
            Path: Path to data.yaml file
        """
        self.logger.info("inside prepare_training method..........")
        
        # Generate data.yaml
        generator = YAMLGenerator()
        yaml_path = generator.yaml_generator_main()
        
        self.logger.info(f"Training preparation complete. data.yaml at {yaml_path}")
        return yaml_path

    def train_model(self, data_yaml_path: Path) -> dict:
        """
        Trains YOLOv11s model on UI components dataset.

        Args:
            data_yaml_path: Path to data.yaml configuration file

        Returns:
            dict: Training results
        """
        self.logger.info("inside train_model method..........")
        self.logger.info(f"Training configuration: {self.config}")
        
        # Initialize YOLO model
        model = YOLO(self.config["model"])
        self.logger.info(f"Loaded model: {self.config['model']}")
        
        # Train with class weights for imbalance
        results = model.train(
            data=str(data_yaml_path),
            epochs=self.config["epochs"],
            imgsz=self.config["imgsz"],
            batch=self.config["batch"], 
            patience=self.config["patience"],
            device=self.config["device"],
            workers=self.config["workers"],
            project=self.config["project"],
            name=self.config["name"],
            verbose=True,
        )
        
        self.logger.info("Training completed successfully")
        return results

    def training_main(self) -> None:
        """
        Main training orchestrator method.
        """
        self.logger.info("inside training_main method..........")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Prepare config files
            yaml_path = self.prepare_training()
            
            # Train model
            results = self.train_model(yaml_path)
            
            # Log results
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