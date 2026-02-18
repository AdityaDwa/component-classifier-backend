from pathlib import Path

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from constants.constants import FILE_COUNT

config: dict = PathUtils().get_configuration()
preprocessing_detail: dict = config["PREPROCESSING"]


class DirectoryCleanup:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.allowed_prefixes: str = preprocessing_detail["ALLOWED_PREFIX"]
        self.remove_suffixes: str = preprocessing_detail["REMOVE_SUFFIX"]

    def remove_unwanted_files(self, file_path: Path) -> None:
        """
        Function that removes unwanted files based on filename prefix and suffix rules.

        Args:
            file_path (Path): Path object representing the file to evaluate.

        Returns:
            None: This function does not return a value. It performs file deletion as a side effect.

        Raises:
            Exception: Logs any exception raised during file deletion.
        """
        self.logger.info(f"inside remove_unwanted_files method..........")
        if any(file_path.name.startswith(prefix) for prefix in self.allowed_prefixes):
            if any(file_path.name.endswith(suffix) for suffix in self.remove_suffixes):
                try:
                    file_path.unlink()
                    self.logger.info(f"Deleted file due to unwanted suffix: {file_path}")
                except Exception as e:
                    self.logger.exception(f"Failed to delete {file_path}: {e}")
            return
        else:
            try:
                file_path.unlink()
                self.logger.info(f"Deleted file due to disallowed prefix: {file_path}")
            except Exception as e:
                self.logger.exception(f"Failed to delete {file_path}: {e}")

    def load_and_preprocess(self) -> None:
        """
        Function that loads raw data directories and applies preprocessing by removing unwanted files.

        Args:
            None

        Returns:
            None: This function does not return a value. It performs preprocessing operations and logs progress as side
            effects.
        """
        self.logger.info(f"inside load_and_preprocess method..........")
        total_file: int = 0
        raw_data_path = PathUtils().get_raw_data_path()
        for folder in raw_data_path.iterdir():
            if folder.is_dir():
                self.logger.info(f"Processing folder: {folder}")
                for file in folder.iterdir():
                    if file.is_file():
                        self.remove_unwanted_files(file)
                        total_file += 1
            self.logger.info(f"Removed unwanted files from: {folder}")
        self.logger.info(f"Total files processed: {total_file}")

    def count_files_in_folder(self) -> None:
        """
        Function that checks each subfolder in the raw data directory and validates the number of files inside.

        Args:
            None

        Returns:
            None: This function does not return a value. It performs validation and folder deletion as side effects.
        """
        self.logger.info(f"inside count_files_in_folder method..........")
        raw_data_path = PathUtils().get_raw_data_path()
        for folder in raw_data_path.iterdir():
            if folder.is_dir():
                total_files = sum(1 for item in folder.iterdir() if item.is_file())
                if total_files == FILE_COUNT:
                    return
                else:
                    folder.unlink()
                    self.logger.info(f"Folder deleted due to less files: {folder}")


if __name__ == "__main__":
    dir_cleanup = DirectoryCleanup()
    dir_cleanup.load_and_preprocess()
    dir_cleanup.count_files_in_folder()
