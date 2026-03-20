import yaml
from pathlib import Path

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from constants.json_constants import label_to_id_dict


class YAMLGenerator:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def create_classes_txt(self) -> None:
        """
        Creates classes.txt file with class names in order.
        """
        self.logger.info("inside create_classes_txt method..........")
        
        # Sort by ID to maintain order
        sorted_classes = sorted(label_to_id_dict.items(), key=lambda x: x[1])
        class_names = [name for name, _ in sorted_classes]
        
        classes_file = PathUtils().get_txt_path().joinpath("classes.txt")
        with open(classes_file, "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        self.logger.info(f"Created classes.txt with {len(class_names)} classes")

    def create_data_yaml(self) -> Path:
        """
        Creates data.yaml config file for YOLO training.

        Returns:
            Path: Path to created data.yaml file
        """
        self.logger.info("inside create_data_yaml method..........")
        
        data_dir = PathUtils().get_split_data_path()
        
        # Read class names from classes.txt
        classes_file = PathUtils().get_txt_path().joinpath("classes.txt")
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        # Create YAML structure
        data_config = {
            "path": str(data_dir.absolute()),  # Absolute path to data directory
            "train": "train/images",           # Relative to 'path'
            "val": "validation/images",
            "test": "test/images",             # NEW: test set
            "nc": len(classes),                # Number of classes
            "names": classes                   # Class names list
        }
        
        # Write YAML
        yaml_path = PathUtils().get_yaml_path()
        with open(yaml_path, "w") as f:
            yaml.dump(data_config, f, sort_keys=False)
        
        self.logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path

    def yaml_generator_main(self) -> Path:
        """
        Main method to generate YOLO config files.

        Returns:
            Path: Path to data.yaml
        """
        self.logger.info("inside yaml_generator_main method..........")
        self.create_classes_txt()
        yaml_path = self.create_data_yaml()
        return yaml_path


if __name__ == "__main__":
    generator = YAMLGenerator()
    generator.yaml_generator_main()