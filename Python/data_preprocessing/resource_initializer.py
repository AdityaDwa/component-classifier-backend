import gzip
import shutil
from pathlib import Path

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from utils.generic_utils import remove_file
from constants.constants import IMAGE_EXTENSION


class ResourceInitializer:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def move_image_to_dataset(self) -> None:
        """
        Moves image files from the raw data directory to the dataset directory.
        Only files with extensions listed in IMAGE_EXTENSION are moved.
        After copying each image, the original file is removed. Logs all major steps and exceptions encountered
        during processing.

        Args:
            None

        Returns:
            None
        """
        self.logger.info(f"inside move_image_to_dataset method..........")
        raw_data_path: Path = PathUtils().get_raw_data_path()
        try:
            for folder in raw_data_path.iterdir():
                if folder.is_dir():
                    self.logger.info(f"Processing folder: {folder}")
                    for file in folder.iterdir():
                        try:
                            if file.is_file() and any(
                                file.name.endswith(suffix) for suffix in IMAGE_EXTENSION
                            ):
                                self.logger.info(f"Processing file: {file}")
                                image_path: Path = (
                                    PathUtils().get_image_dataset_path().joinpath(folder.name)
                                )
                                new_image_path: Path = image_path.with_suffix(file.suffix)
                                shutil.copy2(str(file), str(new_image_path))
                                self.logger.info(f"Image file: {file} moved to {new_image_path}")
                                remove_file(file)
                                self.logger.info(f"File: {file} removed")

                        except Exception as e:
                            self.logger.exception(f"Failed processing file {file}: {e}")
        except Exception as e:
            self.logger.exception(f"Failed scanning raw data path {raw_data_path}: {e}")

    def unzip_raw_data(self) -> None:
        """
        Unzips all `.gz` files in the raw data directory and removes the original compressed files.
        Iterates through each folder in the raw data path, extracts `.gz` files, and logs the progress and any
        errors encountered.

        Args:
            None

        Returns:
            None
        """
        self.logger.info("inside unzip_raw_data method..........")
        raw_data_path: Path = PathUtils().get_raw_data_path()
        try:
            for folder in raw_data_path.iterdir():
                if not folder.is_dir():
                    continue
                self.logger.info(f"Processing folder: {folder}")
                for file in folder.iterdir():
                    try:
                        if file.is_file() and file.suffix == ".gz":
                            self.logger.info(f"Processing file: {file}")
                            output_file = file.with_suffix("")
                            with gzip.open(file, "rb") as file_in:
                                with open(output_file, "wb") as file_out:
                                    shutil.copyfileobj(file_in, file_out)
                            self.logger.info(f"Extracted {file} -> {output_file}")
                            remove_file(file)
                            self.logger.info(f"File: {file} removed")
                    except Exception as e:
                        self.logger.exception(f"Failed unzipping file {file}: {e}")
        except Exception as e:
            self.logger.exception(f"Failed scanning raw data path {raw_data_path}: {e}")

    def resource_initializer_main(self) -> None:
        """
        Main method to initialize resources by preparing image and raw data files.
        Calls methods to move image files into the dataset directory and unzip raw `.gz` files.

        Args:
            None

        Returns:
            None
        """
        self.logger.info("inside resource_initializer_main method..........")
        self.move_image_to_dataset()
        self.unzip_raw_data()


if __name__ == "__main__":
    resource_initializer = ResourceInitializer()
    resource_initializer.resource_initializer_main()
