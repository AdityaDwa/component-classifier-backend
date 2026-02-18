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

        Arge:
            filename (str): String that represents the name of the log file that will be created or accessed.

        Return:
            Path: Full path to a log file by joining the base path with a "logs" directory and the provided filename.
        """
        log_dir: Path = cls().__base_path.joinpath("logs")
        log_file_path: Path = log_dir.joinpath(filename)
        return log_file_path

    def get_raw_data_path(self) -> Path:
        """
        Returns the absolute path of the raw data folder.

        Args:
            None

        Returns:
            Path: Absolute path to the raw data directory.
        """
        return self.__base_path.joinpath("data/raw_data")
