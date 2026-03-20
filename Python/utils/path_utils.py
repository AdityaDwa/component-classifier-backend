from pathlib import Path

import yaml


class PathUtils:
    __base_path: Path = Path(__file__).resolve().parent.parent

    def get_base_path(self) -> Path:
        """
        Generate the value of the private attribute `__base_path`.

        Return:
            Path: The value of the private attribute `__base_path`.
        """
        return self.__base_path

    def get_configuration(self) -> dict:
        """
        Reads and loads a YAML configuration file located at a specified path and returns the configuration data.

        Return:
            dict: Configuration data loaded from the "config.yml" file located in the "config" directory
        """
        with open(self.__base_path.joinpath("config", "config.yml"), "r") as file:
            config: dict = yaml.safe_load(file)
        return config

    @classmethod
    def get_log_path(cls, filename: str) -> Path:
        """
        Generate the full path of a log file within a specified directory.

        Args:
            filename (str): String that represents the name of the log file that will be created or accessed.

        Return:
            Path: Full path to a log file by joining the base path with a "logs" directory and the provided filename.
        """
        log_dir: Path = cls().__base_path.joinpath("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path: Path = log_dir.joinpath(filename)
        return log_file_path

    def ensure_folder_exists(self, folder_path: Path) -> Path:
        """
        Creates folder if it doesn't exist.

        Args:
            folder_path (Path): Path object representing the folder to create.

        Returns:
            Path: The same folder path after ensuring it exists.
        """
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def get_raw_data_path(self) -> Path:
        """
        Returns the absolute path of the raw data folder.

        Args:
            None

        Returns:
            Path: Absolute path to the raw data directory.
        """
        return self.__base_path.joinpath("data/raw_data")

    def get_txt_path(self) -> Path:
        """
        Returns the absolute path of the txt folder, creating it if it doesn't exist.

        Args:
            None

        Returns:
            Path: Absolute path to the txt directory.
        """
        txt_path = self.__base_path.joinpath("txt")
        return self.ensure_folder_exists(txt_path)

    def get_image_dataset_path(self) -> Path:
        """
        Returns the absolute path of the image dataset folder, creating it if it doesn't exist.

        Args:
            None

        Returns:
            Path: Absolute path to the image dataset directory.
        """
        image_path = self.__base_path.joinpath("datasets/images")
        return self.ensure_folder_exists(image_path)

    def get_label_dataset_path(self, file_name: str) -> Path:
        """
        Returns the absolute path of the label dataset file.

        Args:
            file_name (str): Name of the label file.

        Returns:
            Path: Absolute path to the label dataset file.
        """
        return self.__base_path.joinpath("datasets", "labels", file_name)

    def get_label_dataset_dir(self) -> Path:
        """
        Returns the absolute path of the label dataset directory, creating it if it doesn't exist.

        Args:
            None

        Returns:
            Path: Absolute path to the label dataset directory.
        """
        label_dir = self.__base_path.joinpath("datasets", "labels")
        return self.ensure_folder_exists(label_dir)

    def get_split_data_path(self) -> Path:
        """
        Returns the absolute path of the split_data directory (train/val/test splits).

        Args:
            None

        Returns:
            Path: Absolute path to the split_data directory.
        """
        return self.__base_path.joinpath("split_data")

    def get_yaml_path(self) -> Path:
        """
        Returns the absolute path of the data.yaml file.

        Args:
            None

        Returns:
            Path: Absolute path to the data.yaml configuration file.
        """
        return self.__base_path.joinpath("data.yaml")

    def get_checkpoints_path(self) -> Path:
        """
        Returns the absolute path of the checkpoints directory (model weights storage).

        Args:
            None

        Returns:
            Path: Absolute path to the checkpoints directory.
        """
        return self.__base_path.joinpath("checkpoints")