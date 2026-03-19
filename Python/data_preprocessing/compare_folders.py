from pathlib import Path

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from utils.generic_utils import remove_file


class CompareFolders:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        dataset_path = PathUtils().get_dataset_path()
        self.image_folder = dataset_path / "images"
        self.label_folder = dataset_path / "labels"

    def _get_files(self, folder_path: Path) -> dict:
        """
        Retrieve all files in the given folder and map them by filename stem.

        Args:
            folder_path (Path): Path to the directory containing files.

        Returns:
            dict[str, Path]: A dictionary mapping file stems to their full Path objects.
        """
        self.logger.info(f"inside _get_files method..........folder_path: {folder_path}")
        return {f.stem: f for f in folder_path.iterdir() if f.is_file()}

    def _count_files(self, folder_path: Path) -> int:
        """
        Count the number of files in the given folder.

        Args:
            folder_path (Path): Path to the directory to be scanned.

        Returns:
            int: Total number of files in the folder.
        """
        self.logger.info(f"inside _count_files method..........folder_path: {folder_path}")
        return sum(1 for f in folder_path.iterdir() if f.is_file())

    def check_files_in_folder(self) -> None:
        """
        Ensure consistency between image and label folders by matching file names.

        Workflow:
            1. Load files from image and label folders.
            2. Extract filename stems and compare them.
            3. Identify files missing in either folder.
            4. Remove unmatched files from both folders.
            5. Log final file counts after cleanup.

        Returns:
            None

        Note:
            - Matching is based only on filename (stem), not file content.
            - Use caution as file deletions are permanent.
        """
        self.logger.info(f"inside check_files_in_folder method..........")
        image_files = self._get_files(self.image_folder)
        label_files = self._get_files(self.label_folder)

        image_keys = set(image_files)
        label_keys = set(label_files)

        missing_in_labels = image_keys - label_keys
        missing_in_images = label_keys - image_keys

        self.logger.info(f"Missing labels count: {len(missing_in_labels)}")
        self.logger.info(f"Missing images count: {len(missing_in_images)}")

        # Remove unmatched files
        for key in missing_in_labels:
            remove_file(image_files[key])

        for key in missing_in_images:
            remove_file(label_files[key])

        # Final counts
        image_count = self._count_files(self.image_folder)
        label_count = self._count_files(self.label_folder)

        self.logger.info(f"Final image count: {image_count}")
        self.logger.info(f"Final label count: {label_count}")


if __name__ == "__main__":
    CompareFolders().check_files_in_folder()
